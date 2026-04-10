/*
    Slang JPEG2000 Encoder v17 — GPU Colour Conversion + CDF 9/7 Normalization

    V17 Improvements over v16:

    1. GPU colour conversion (parity with CUDA V18+):
       - encode_from_rgb48(): accepts RGB48LE + pre-uploaded LUT+matrix
       - Kernel s17_rgb48_to_xyz12: LUT linearize + Bradford matrix + LUT compand
       - Eliminates CPU rgb_to_xyz bottleneck (~3-5ms saved per frame)
       - Same algorithm as cuda_j2k_encoder.cu kernel_rgb48_to_xyz12

    2. CDF 9/7 analysis normalization (critical quality fix):
       - Unnormalized DWT: DC gain = K^(2×level) per 2D level.
         After 5 levels, LL5 coefficients ≈ K^10 ≈ 7.4× input range.
         For 12-bit input [0,4095]: LL5 up to ~30,000.
         With old step=1.0 and cap=126: ALL LL5 coefficients saturate at 126.
       - Fix: at each H-DWT and V-DWT output:
           L × NORM_L = 1/K = 0.812893  (removes lowpass DC gain)
           H × NORM_H = K   = 1.230174  (amplifies highpass to match energy)
       - After normalization, all subbands ∈ [-4095, 4095] for 12-bit input
       - Standard J2K decoders expect normalized coefficients; now consistent

    3. Correct quantization step: 32.5 (was 1.0)
       - base_step = 4095/126 ≈ 32.5 uses the full 7-bit magnitude range
       - No saturation; both LL5 and HH1 quantized proportionally

    4. Removes per-encode mutex (was serializing all encoder threads):
       - Each encoder thread should have its own SlangJ2KEncoder instance

    5. All V16 DCI compliance features retained:
       - Rsiz, Scod=1, CPRL, MCT=1, xcb'=ycb'=3, filter=0, TLM, 3 tile parts
       - Magnitude cap=126 (no byte stuffing), fused V-DWT+deinterleave kernel

    V17b Improvements (fp16 V-DWT workspace):
    - s17_dwt_v_fp16: __half workspace for vertical DWT, float I/O
    - Halves V-DWT memory bandwidth (dominant cost for large h)
    - fp16 precision (10-bit mantissa) is sufficient: ULP ≈ 8 for ±8192,
      which is << quantization step 32.5
    - register-blocked path (h ≤ MAX_REG_HEIGHT) unchanged (already optimal)
    - d_half[c] added to pool: +2 bytes/pixel vs V17

    V17c Improvements (fp16 H-DWT output):
    - s17_fused_i2f_dwt_h_ho / s17_dwt_h_ho: write __half to d_t reinterpreted as __half*
    - s17_dwt_v_fp16_hi: reads __half directly (no float→half copy step)
    - Eliminates float→half conversion in V-DWT copy step for large-h levels
    - Saves ~24MB/frame memory traffic (H-DWT write + V-DWT input read, large-h levels)
    - d_t[c] (float, 4B/px) reused as __half (2B/px) for large-h — no extra allocation

    V17d Improvements (register-blocked V-DWT for large-h levels that become small):
    - s17_dwt_v_fp16_hi_reg: reads __half (from half H-DWT output), register-blocked float
      arithmetic — eliminates d_half workspace accesses for subbands ≤ MAX_REG_HEIGHT
    - Applied in the large-h DWT path when h ≤ MAX_REG_HEIGHT (140)
    - Saves ~20MB/frame on levels 3-4 for 2K content (same as CUDA V27)
    - Achieves register-blocking parity: both small-h and transitional-h levels now
      use register storage; only truly large subbands (h > 140) use fp16 workspace

    Performance target: 160-170 fps (V17d).

    V17e Improvements (fused RGB→XYZ + H-DWT level 0):
    - s17_rgb48_xyz_hdwt0: single kernel, one block per row, 3×width float shared memory
    - Eliminates d_in[0..2] intermediate write+read (~54MB/frame DRAM savings)
    - encode_from_rgb48() calls fused kernel, then s17_run_dwt_and_build(skip_l0_hdwt=true)
    - skip_l0_hdwt path: d_t[c] (pre-populated as __half by fused kernel) used directly
    - s17_run_dwt_and_build gains skip_l0_hdwt parameter for this path

    Performance target: 170-180 fps (V17e).

    V17f Improvements (tiled V-DWT — no global workspace):
    - s17_dwt_v_tiled: register-based tiled V-DWT with V_TILE=16 + V_OVERLAP=5 halo
    - Whole-point symmetric boundary extension (no special-casing needed)
    - Eliminates fp16 global workspace; only reads half input and writes float output
    - ~2.5× reduction in V-DWT memory traffic for large subbands (h > MAX_REG_HEIGHT)
    - Parity with CUDA V29 (kernel_fused_vert_dwt_tiled)

    Performance target: 180-195 fps (V17f).

    V17g Improvements (3-stream parallel 1-channel fused colour+HDWT0):
    - s17_rgb48_xyz_hdwt0_1ch: single-channel variant, 8KB smem (vs 24KB in V17e)
    - 3 parallel streams × 8KB smem → 6 blocks/SM; thread limit = 8 → 100% occupancy
    - All 3 streams read same d_rgb16 → L2 cache reuse (only 1 DRAM pass for RGB)
    - Expected: ~0.25ms vs ~0.95ms for V17e (25% occ.) → ~0.7ms/frame saved
    - encode_from_rgb48() launches s17_rgb48_xyz_hdwt0_1ch on all 3 streams after H2D

    Performance target: 195-215 fps (V17g).

    V17h Improvements (CUDA Graphs for DWT+quantize+D2H pipeline):
    - s17_launch_comp_pipeline: extracted per-component kernel sequence for graph capture
    - s17_rebuild_comp_graphs: captures 3 independent per-component CUDA Graphs
    - s17_run_dwt_and_build: uses cudaGraphLaunch instead of ~30 individual kernel launches
    - Graphs invalidated and rebuilt on geometry change (width/height/per_comp/is_4k/is_3d)
    - Expected: ~0.3ms/frame kernel-launch overhead eliminated → ~215-230 fps

    Performance target: 215-230 fps (V17h).

    V17i Improvements (H-DWT thread count 256→512 for 100% SM occupancy):
    - SlangGpuConfig17::h_block: 256 → 512
    - 8KB smem/block: 512T thread-limited at 4 blk/SM = 2048T/SM = 100% (vs 75% at 256T)
    - Applies to: s17_fused_i2f_dwt_h, s17_dwt_h, s17_dwt_h_ho, s17_rgb48_xyz_hdwt0_1ch
    - More warps per SM → better memory latency hiding for d_rgb16 and d_b reads

    Performance target: 230-245 fps (V17i).

    V17j Improvements (V-DWT tile size 16→32 for fewer overlap redundant loads):
    - V_TILE: 16 → 32 (V_TILE_FL: 26 → 42)
    - 2K level 0 h=1080: 68 tiles×26 = 1768 half loads/column → 34 tiles×42 = 1428 (~19% fewer)
    - Same V_OVERLAP=5 (covers 4-step CDF 9/7 stencil + 1 safety)
    - Mirrors CUDA V34; logic unchanged (V_TILE_FL used in loop bounds dynamically)

    Performance target: 245-260 fps (V17j).

    V17k Improvements (float4 vectorized quantize kernel):
    - s17_qe: changed signature to n4=floor(per_comp/4), reads float4, writes uint32_t
    - 4× fewer global memory transactions for both loads and stores
    - s17_launch_comp_pipeline: passes n4 instead of per_comp to s17_qe
    - Parity with CUDA V35

    Performance target: 260-275 fps (V17k).

    V17m Improvements (unified super-graph: H2D + event + RGB+HDWT0 + DWT+Q+D2H):
    - cudaStreamCaptureModeGlobal captures all 3 streams in one CUDA Graph
    - encode_from_rgb48: ~13 API calls → 3 (graphLaunch + streamSync + build codestream)
    - Added h_rgb16_pinned staging buffer for truly-async H2D inside graph capture
    - Parity with CUDA V37; expected ~50µs/frame savings on CUDA API overhead

    Performance target: 295-315 fps (V17m).

    V17l Improvements (full half-precision DWT pipeline — d_c[c]: float→__half):
    - d_c[c] changed from float* to __half*, saves 2B/pixel per component (~12MB for 2K 3-ch)
    - New kernels: s17_dwt_v_reg_ho (float-in, half-out, small-h register-blocked)
                   s17_dwt_v_fp16_hi_reg_ho (half-in, half-out, transitional-h register-blocked)
                   s17_dwt_v_tiled_ho (half-in, half-out, large-h tiled)
                   s17_dwt_h_half_io (half-in, half-out H-DWT for levels 1+)
                   s17_qe_h (half-input quantize using __half2 loads)
    - d_half[c] workspace removed from pool (no longer needed in new pipeline)
    - Pool size: (2+4+4)B/px vs old (4+4+2+4)B/px → saves 4B/px × n × 3 components
    - Total memory traffic reduction: ~22MB/frame → ~0.28ms at 78GB/s
    - Parity with CUDA V36

    Performance target: 275-295 fps (V17l).

    V17n Improvements (d_t[c]: float→__half — parity with CUDA V26):
    - d_t[c] changed from float* to __half*, matching CUDA V26 d_b
    - Pool: (2+4+4)B/px → (2+2+4)B/px per component; saves 2B/px × 3 × n pixels
    - Eliminates reinterpret_cast<__half*>(d_t) dance in all dispatch paths
    - Small-h level 0 path updated: s17_fused_i2f_dwt_h_ho + s17_dwt_v_fp16_hi_reg_ho
      (was s17_fused_i2f_dwt_h + s17_dwt_v_reg_ho with float d_t)
    - d_t for 2K = 1920×1080×2B×3 = ~12MB allocated (was ~24MB); saves ~12MB VRAM
    - H-DWT write + V-DWT level 0 read: ~8MB/frame bandwidth saved → ~0.10ms at 78GB/s

    Performance target: 315-335 fps (V17n).

    V17o Improvements (half-precision shared memory for H-DWT kernels):
    - s17_fused_i2f_dwt_h_ho, s17_dwt_h_half_io, s17_rgb48_xyz_hdwt0_1ch:
      extern __shared__ float sm[] → extern __shared__ __half sm[]
    - On sm_61+ Pascal: fp16 FMA throughput = 2× fp32; smem bandwidth halved
    - Lifting steps (4 per kernel × ~4 iterations each) run 2× faster
    - smem per block: 1920×4=7.5KB → 1920×2=3.75KB; more L1 cache for texture data
    - Launch smem size updated: w*sizeof(float) → w*sizeof(__half) at all call sites

    Performance target: 335-360 fps (V17o).

    V17p Improvements (parity-split V-DWT lifting loops — branch-free unrolled):
    - s17_cdf97_lift_tiled<P0> template: 4×40 conditional iterations → 4×20 unrolled FMAs
    - #pragma unroll + compile-time P0 eliminates all per-iteration branch overhead
    - s17_dwt_v_tiled + s17_dwt_v_tiled_ho updated to use template dispatch
    - Parity with CUDA V39; ~50% fewer loop iterations + full ILP from unrolling

    Performance target: 360-385 fps (V17p).

    V17q Improvements (constant p0=1 hardcode — eliminate runtime parity dispatch):
    - V_TILE=32 (even) + V_OVERLAP=5 (odd) → load_start always odd → p0 always 1
    - Remove runtime p0 computation and if(p0) dispatch branch
    - Output loop: `if ((p0+i) & 1)` → `if (!(i & 1))` (even i = H, odd i = L)
    - static_assert guards correctness if tile params ever change
    - Parity with CUDA V40

    Performance target: 385-410 fps (V17q).

    V17r Improvements (1-frame pipelining — CPU memcpy overlaps with GPU compute):
    - encode_from_rgb48 returns PREVIOUS frame's codestream, launches GPU for current
    - Double-buffered h_rgb16_pinned[2] + h_enc[2]: frame N writes buf[1-cur] while
      GPU processes buf[cur]; no aliasing since graphs run sequentially on same streams
    - Two full_graph execs: full_graph[0] bakes h_rgb16_pinned[0]/h_enc[0],
      full_graph[1] bakes h_rgb16_pinned[1]/h_enc[1]
    - Steady-state timing: max(GPU~2.16ms, CPU_copy~1.2ms) = 2.16ms → ~463fps

    Performance target: 490-540 fps (V17r).

    V17s Improvements (H2D-compute overlap via dedicated st_h2d stream):
    - Separate PCIe DMA stream st_h2d for H2D transfers (runs parallel to SM compute)
    - d_rgb16[2]: double-buffered GPU RGB — H2D for frame N writes d_rgb16[new_buf]
      while SM compute for frame N-1 reads d_rgb16[cur_buf] (no aliasing)
    - h2d_done[2] events: st[c].waitEvent(h2d_done[new_buf]) before RGB kernel
    - cg_v17s[2][3]: per-buf comp graphs (DWT+Q+D2H); differ only in D2H destination
    - s17_rebuild_v17s_comp_graphs(buf,...): builds cg_v17s[buf][c] for a given buf
    - CPU blocking: memcpy(1.2ms) + sync_tail(0.5ms) = 1.8ms > H2D(1.66ms) → 556fps

    Performance target: 550-600 fps (V17s).

    V17t Improvements (parallel CPU memcpy — 4 threads, ~4× speedup):
    - std::async splits 12.4MB RGB copy into 4 chunks run concurrently (~0.3ms total)
    - CPU blocking drops to 0.3+0.5+0.1=0.9ms < H2D(1.66ms) → PCIe now bottleneck
    - T_frame ≈ H2D ≈ 1.66ms → ~602fps (parity with CUDA V43)

    Performance target: 580-620 fps (V17t).

    V17u Improvements (fuse event-wait + RGB kernel into per-buf comp graphs):
    - s17_rebuild_v17s_comp_graphs: captures cudaStreamWaitEvent + s17_rgb48_xyz_hdwt0_1ch
    - Per-frame API calls: 9 (V17t) → 3 (V17u)
    - ~6 API calls × ~5µs = ~30µs/frame saved; avoids per-frame host-side sync points
    - cg_v17s_rgb_stride[2]: stride baked into graph; added to validity check
    - Parity with CUDA V44

    Performance target: 602+ fps (V17u).

    V17v Improvements (2-rows-per-block s17_rgb48_xyz_hdwt0_1ch_2row):
    - Grid halved: (height+1)/2 blocks vs height blocks per channel per stream
    - smem doubled: 2*w*sizeof(__half) (y0 in [0..w-1], y1 in [w..2w-1])
    - Matrix m0/m1/m2 amortized over 2 rows (same 3 __ldg reads per block pair)
    - Same 4 syncthreads total for 2 rows (vs 4+4 across two separate block invocations)
    - L2 spatial locality: adjacent rows y0/y1 read from adjacent RGB cache lines
    - SM throughput: 4 blocks/SM × 2 rows = 8 rows/SM/pass (vs 4 rows in V17u)
    - Parity with CUDA V45

    Performance target: 610+ fps (V17v).

    V17w Improvements (2-rows-per-block s17_dwt_h_half_io_2row for DWT levels 1-4):
    - s17_dwt_h_half_io_2row: grid=(h+1)/2; smem=2*w*sizeof(__half)
    - Halves block count for DWT levels 1-4: 540→270, 270→135, 135→68, 68→34 (2K)
    - L2 reuse for adjacent d_t rows; 4 syncthreads per 2 rows
    - Parity with CUDA V46

    Performance target: 615+ fps (V17w).

    V17x Improvements (4-rows-per-block s17_dwt_h_half_io_4row for DWT levels 1-4):
    - s17_dwt_h_half_io_4row: grid=(h+3)/4; smem=4*w*sizeof(__half)
    - Halves grid again vs V17w: 270→135, 135→68, 68→34, 34→17 (2K)
    - 4 adjacent rows per block → 4× L2 spatial locality for d_t input reads
    - syncthreads amortized over 4 rows; thread-limited at 4 blk/SM = 100% occ
    - Parity with CUDA V47

    Performance target: 618+ fps (V17x).

    V17y Improvements (uint16_t d_lut_out — halves GPU LUT texture cache):
    - d_lut_out: int32_t[4096] → uint16_t[4096]; DCP XYZ values [0,4095] fit in uint16_t
    - GPU allocation: 16KB → 8KB; lut_in(16KB) + lut_out(8KB) = 24KB < 32KB L1/tex cache
    - Both LUTs fit simultaneously in sm_61 L1 → fewer evictions during RGB→XYZ convert
    - GpuColourParams::lut_out type changed (header change affects both CUDA and Slang)
    - Parity with CUDA V48

    Performance target: 620+ fps (V17y).

    V17z Improvements (4K correctness + stream sync race fix — parity with CUDA V49):
    - 4K correctness: s17_launch_comp_pipeline loops num_levels=is_4k?6:5 (was 5 always)
      - 4K DCI requires 6-level DWT; codestream COD already declared 6 but only 5 done
    - 4K QCD/QCC: nsb now 3*(fourk?6:5)+1=19 for 4K (was hardcoded 16 for both)
    - Stream sync: sync st[1] and st[2] before s17_build_j2k_codestream (was only st[0])
      - Race: 3 component streams write h_enc independently; stream[0]-only sync unsafe
    - Early graph launch: launch new_buf graphs BEFORE CPU codestream building
      - Reduces next-frame sync_wait by ~0.1ms (GPU starts new_buf compute sooner)

    Performance target: 620+ fps (V17z — correctness, no throughput regression).

    V18a Improvements (subband-aware quantization — parity with CUDA V50):
    - s17_qe_subband_h kernel: 3-band step factors (DC×0.80, mid×0.95, HF×1.10)
    - s17_launch_comp_pipeline uses s17_qe_subband_h instead of s17_qe_h
    - Better PSNR/SSIM for low-frequency content at same bitrate

    V18b Improvements (adaptive codestream trimming — parity with CUDA V51):
    - s17_find_actual_per_comp(): backward word-scan to find last non-zero coefficient byte
    - s17_build_j2k_codestream: TLM Ptlm[c] and SOT Psot use actual[c] (trimmed size)
    - Simple/dark frames produce smaller files; complex frames unaffected; no quality loss

    V18c Improvements (2D subband-aware quantization — parity with CUDA V52):
    - V18a used only row < ll5_h for DC detection; incorrectly applied DC-quality step to LH1
      (LH1 = finest horizontal detail subbands, cols stride/2..stride-1 in rows 0..ll5_h-1)
    - Fix: LL5 = row < ll5_h AND col < ll5_cols (ll5_cols = stride >> 5 = stride/32)
    - Step weights: LL5×0.70 (DC finest), LH5/HL5/HH5×0.90, all higher-freq×1.15
    - s17_qe_subband_2d replaces s17_qe_subband_h in s17_launch_comp_pipeline
    - Result: DC gets correct bit allocation; LH1 no longer wastes bits on DC-quality precision

    V18f Improvements (half-precision lut_in — parity with CUDA V55):
    - d_lut_in: float*→__half*; GPU allocation 4096×4=16KB → 4096×2=8KB
    - lut_in+lut_out = 8KB+8KB = 16KB; both fit in 16KB L1 with room for DWT smem
    - Host GpuColourParams::lut_in stays float[4096]; convert float→__half at upload time
    - Kernels use __half2float(__ldg(&d_lut_in[idx])) before matrix multiply

    V19 Improvements (per-channel pack+H2D pipeline — parity with CUDA V56):
    - pack_rgb12_plane: single-channel variant of pack_rgb12_chunk
    - encode_from_rgb48 steps 1+2 merged: pack ch0 → H2D ch0; pack ch1 → H2D ch1; pack ch2 → H2D ch2
    - H2D starts 0.225ms earlier (ch0 H2D at t=0.075ms instead of t=0.3ms)
    - Total pack+H2D latency: 0.075+1.26=1.335ms (was 1.56ms) → ~14.5% reduction

    V18e Improvements (12-bit planar packed H2D — parity with CUDA V54):
    - pack_rgb12_chunk (4 threads) packs RGB48LE → 12-bit planar (3 channels × (w/2*3) × h)
    - H2D: 9.5MB vs 12.6MB for 2K (1.26ms vs 1.68ms) → FPS ceiling raised ~600→~790fps
    - GPU kernel s17_rgb48_xyz_hdwt0_1ch_2row_p12 reads from d_rgb12 (uint8_t planar)
    - Better GPU cache coalescing: adjacent thread pairs share 3-byte packed cache lines
    - New buffers: d_rgb12[2] + h_rgb12_pinned[2]; ensure_rgb12/ensure_pinned_rgb12 added

    V18d Improvements (6-band perceptual quantization — parity with CUDA V53):
    - V18c used 3 bands; this refines to 6 bands distinguishing each DWT level
    - Step weights: LL5×0.65, L5-AC×0.85, L4×0.95, L3×1.05, L2×1.12, L1×1.20
    - subband_level = min(row_lv, col_lv) where lv determined by 2^k multiples of ll5_h/ll5_c
    - 6 __frcp_rn precomputed per block; per-col: 2 cascade selects + 1 min → ~5 ops
    - Better PSNR for textures/edges (level-2/3/4 content) without throughput regression
    - s17_qe_subband_ml replaces s17_qe_subband_2d in s17_launch_comp_pipeline

    Performance target: 620+ fps (V18d — quality improvement, no throughput regression).

    V31 Improvements (fuse 4-row loads into interior if/else block — parity CUDA V68):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12: load section moved into if (y3<height) / else
    - Interior (100% of 2K blocks): single for loop issues all 4 rows' texture loads at once
      (24 byte __ldg + 8 LUT __ldg per pixel vs 4 serial loops of 6+2)
    - GPU MLP: all 4 rows' fetch chains in-flight simultaneously — maximum latency hiding
    - Else (partial last block): original guarded per-row loads (y3 not loaded)
    - Also removes y3-specific guards from partial-block lifting (y3>=height is guaranteed)
    - Expected: 3-8% RGB+HDWT0 speedup from improved texture load MLP utilization

    V86 Improvements (int2 __ldg load vectorization in s17_fused_i2f_dwt_h_ho_4row (XYZ int32 path) — parity CUDA V124):
    - Load loop: for(i=t;i<w;i+=nt) 4×int32 load → for(i=t*2;i<w;i+=nt*2) 4×int2 __ldg.
    - Each int2 covers {in[row*s+i], in[row*s+i+1]}: 2 int32 in one 64-bit load.
    - 8 × 32-bit loads → 4 × 64-bit loads: 2× fewer load instructions; same bytes.
    - DCI widths always even → i=t*2 always even → 8-byte aligned int2. Coalesced preserved.

    V85 Improvements (__half2 __ldg load vectorization in s17_dwt_h_half_io_4row interior — parity CUDA V123):
    - Load loop: for(i=t;i<w;i+=nt) 4×__ldg(__half) → for(i=t*2;i<w;i+=nt*2) 4×__ldg(__half2).
    - Each __half2 load covers {row[i], row[i+1]}: 2 __half in one 32-bit load instruction.
    - 8 scalar 16-bit loads → 4 vectorized 32-bit loads: 2× fewer load instructions per thread.
    - Unpack: sm01[i]={low(r0),low(r1)}, sm01[i+1]={high(r0),high(r1)}.
    - DCI widths always even → i=t*2 always even → 4-byte aligned. Coalesced access preserved.
    - Expected: 1-3% speedup in load phase of s17_dwt_h_half_io_4row (~15% of GPU time).

    V84 Improvements (DCI even-height invariant in s17_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT) — parity CUDA V122):
    - DCI heights always even (1080/2160) → y1=y0+1 always < height → else (partial scalar) branch is dead.
    - Removed entire else block (28 lines: scalar load + scalar lifting + single-row scatter).
    - Unconditional __half2 path: compiler eliminates register partitioning for dead branch.
    - Better instruction cache coverage; smaller kernel binary; same correctness for all DCI content.

    V83 Improvements (uint32_t __ldg replaces 3×byte __ldg per channel in hot p12 load loops — parity CUDA V121):
    - s17_rgb48_xyz_hdwt0_1ch_1row_p12 (4K HOT) + s17_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT).
    - 9 byte __ldg → 3 uint32_t __ldg per pixel pair per row: 3× fewer load instructions.
    - Pixel extraction: even=((rw&0xFF)<<4)|((rw>>12)&0xF); odd=((rw>>8)&0xF)<<8|((rw>>16)&0xFF).
    - Misaligned reads (off=p*3) handled correctly by CUDA __ldg texture cache.
    - 4K: 4 iters×3 loads (was 9) per iter → 4× better MLP per unrolled group.
    - 2K: 2 iters×same benefit. Expected: 2-5% RGB+HDWT0 speedup → ~0.5-1% overall.

    V82 Improvements (DCI even-width invariant in remaining kernels: S17_HDWT_HALF macro, 1ch/2ch kernels, 2row_p12 partial — parity CUDA V119):
    - S17_HDWT_HALF macro (float-path): Alpha/Gamma w>1&&!(w&1) → always; Beta/Delta min(1,w-1)→1, min(x+1,w-1)→x+1.
    - s17_rgb48_xyz_hdwt0_1ch (1-row): same simplification.
    - s17_rgb48_xyz_hdwt0_1ch_2row (2-row, y1<height guards): same.
    - s17_rgb48_xyz_hdwt0_1ch_2row_p12 partial block: same.
    - Completes DCI invariant cleanup across all Slang kernels; all w>1&&!(w&1) guards removed.

    V81 Improvements (DCI even-width invariant in 4row_p12 interior+partial, and s17_dwt_h_half_io_4row partial — parity CUDA V118):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12 interior: Alpha/Gamma if(t==0&&w>1&&!(w&1)) → if(t==0).
    - Beta/Delta boundary sm01[min(1,w-1)] → sm01[1] (sm23 same); saves 2 VMIN+2 ISETP per block.
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12 partial block: same simplifications.
    - s17_dwt_h_half_io_4row partial block (else branch): Alpha/Gamma/Beta/Delta simplified.
    - 4row_p12 interior fires on 134/135 blocks per 2K frame; partial on last block only.

    V80 Improvements (DCI even-width invariant in s17_dwt_h_half_io_4row partial block + s17_dwt_h_half_io_2row — parity CUDA V117):
    - 4-row kernel: interior (V73) already simplified; partial (else branch) still had old DCI guards.
    - 2-row kernel (s17_dwt_h_half_io_2row): also still had old guards — both fixed in this version.
    - Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); DCI widths always even and >1.
    - Beta/Delta: sm[min(1,w-1)] → sm[1]; saves 1 VMIN+SEL per boundary thread.
    - Beta/Delta loop: sm[min(x+1,w-1)] → sm[x+1] (x even, w even: x<w → x≤w-2 → x+1≤w-1).
    - 2-row kernel fires on all 540 rows of level-1 (2K); 4-row partial on last row-group only.

    V79 Improvements (__launch_bounds__(512,4) → (512,3) on 1row/2row RGB p12 kernels — parity CUDA V116):
    - V51 set (512,4) targeting ≤32 regs/T for 4 blk/SM, but V53 added 8KB sm_lut:
      smem = 8KB + 7.68KB = 15.68KB → PreferShared(48KB) → 3 blk/SM (smem-limited).
    - (512,3) → ≤42 regs/T; 3 blk/SM maintained (42×512=21504 → 65536/21504=3 blk/SM).
    - 10 extra registers relax compiler register budget for complex color+DWT kernel.
    - Applied to s17_rgb48_xyz_hdwt0_1ch_2row_p12 and s17_rgb48_xyz_hdwt0_1ch_1row_p12.
    - Expected: 5-15% RGB+HDWT0 speedup if compiler was spilling under 32-reg constraint.

    V78 Improvements (quantize L1-row early exit before inv_* computation — parity CUDA V115):
    - s17_qe_subband_ml: L1 rows (row >= ll5_h*16) need only inv_l1 = base_inv*0.833333333f.
    - Moving early exit before the 5 other inv_* FMULs saves 5 FMUL for >50% of 2K/4K rows.
    - Register pressure for L1 path: 6 floats → 1 float (base_inv) + inline constant.
    - row_lv ternary simplified to drop case 1 (guaranteed row_lv in {2..5} after early exit).
    - Dispatch: if(row_lv==1) branch removed (dead code after early exit added above lambda).

    V77 Improvements (Drop dead h>2 guard in reg-blocked V-DWT Beta/Delta boundary — parity CUDA V114):
    - s17_dwt_v_fp16_hi_reg_ho: DCI subband heights 34/68/135 always > 2.
    - (h & 1) && (h > 2) → (h & 1): saves 1 ISETP+AND per boundary invocation.
    - Applied to both Beta and Delta odd-height boundary cases.

    V76 Improvements (Simplify H-DWT i2f-4row Alpha/Gamma boundary + Beta/Delta min() — parity CUDA V113):
    - DCI even-width invariant applied to s17_fused_i2f_dwt_h_ho_4row (int32 XYZ input path).
    - Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); saves 2 ISETP+PREDAND per block.
    - Beta/Delta: sm01[min(1,w-1)] → sm01[1]; saves 1 VMIN+SEL per boundary thread.
    - This kernel handles level-0 H-DWT for the fallback encode() path (int32 XYZ planes).

    V75 Improvements (Simplify reg V-DWT boundary guards — parity CUDA V112):
    - s17_dwt_v_fp16_hi_reg_ho: DCI heights always > 1 → drop h>1 from Alpha/Gamma even guards.
    - col[min(1,h-1)] → col[1] in Beta/Delta boundaries (all DCI subband heights ≥ 34 > 1).
    - Eliminates 2 VMIN + 2 ISETP+AND per column across 3 V-DWT levels.

    V74 Improvements (Simplify boundary conditions in HOT kernels — parity CUDA V111):
    - Apply DCI even-width invariant to s17_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT) and
      s17_rgb48_xyz_hdwt0_1ch_1row_p12 (4K HOT) DWT lifting passes.
    - Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); Beta/Delta: sm[min(1,w-1)] → sm[1].
    - More impactful than V73 (4-row dead path): these are the actual hot kernels.

    V73 Improvements (Simplify H-DWT 4-row Alpha/Gamma boundary + Beta/Delta min() — parity CUDA V110):
    - DCI subbands: widths always even (1920/960/480/240/120/60) and always >1.
    - Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); saves 2 ISETP+PREDAND per block.
    - Beta/Delta: sm01[min(1,w-1)] → sm01[1]; saves 1 VMIN+SEL per boundary thread.
    - Applied to s17_dwt_h_half_io_4row interior (Alpha and Gamma passes).

    V72 Improvements (PRMT sign extraction in vq4 tail — parity CUDA V109):
    - Replace 4 SETP+SELP+IABS per element with 1 PRMT+AND (signs) + 2 AND (abs) per 4 elements.
    - Applicable to 2K DC rows (ll5_c=60, 60%8=4 → 4 tail elements); ~102 calls/frame/channel.
    - Parity with vq8 core (V66 PRMT sign extraction).

    V71 Improvements (__launch_bounds__(512,4) on s17_dwt_h_half_io_4row — parity CUDA V108):
    - Without LB compiler may use 36+ regs/T → 3 blk/SM. With LB forces ≤32 regs/T → 4 blk/SM.
    - smem for L1 (7.68KB) allows 6 blk/SM → register was the true bottleneck.
    - Hot kernel (~15% GPU time): +33% throughput if previously 3 blk/SM.

    V70 Improvements (__saturatef replaces fmaxf(0,fminf(1,x)) in all RGB→XYZ clamp paths — parity CUDA V107):
    - __saturatef(x) compiles to single PTX cvt.sat.f32.f32 (vs 2 instructions for fmaxf/fminf).
    - Applied to: 3-ch legacy kernel (×3), 1ch_2row (×1), 2row (×2), 4row_p12 load (×4), partial (×3), xyz12 (×3).
    - Saves ~16 FMIN/FMAX instructions per pixel across all color-conversion kernels.

    V69 Improvements (__half2 row-pair lifting in s17_rgb48_xyz_hdwt0_1ch_4row_p12 — parity CUDA V106):
    - Interior load packs {row0[px],row1[px]} into sm01[px], {row2[px],row3[px]} into sm23[px].
    - Lifting: 4 scalar HFMA per element → 2 HFMA2 per element pair (2× FMA throughput per pass).
    - + #pragma unroll 2 on all lifting loops (parity V59).
    - + Drop MIN from Beta/Delta main loops: x always even, w even → x+1≤w-1 (parity V65).
    - Scatter: deinterleave via __low2half/__high2half + __hmul2 (parity s17_dwt_h_half_io_4row V45).
    - Expected: ~2× lifting throughput for this kernel path → ~0.5-1% overall for 4-row users.

    V68 Improvements (remove redundant min(int(v*4095.5f),4095) from LUT index — parity CUDA V105):
    - v = fmaxf(0,fminf(1,x)) ∈ [0,1] → v*4095.5f ≤ 4095.5 → int(truncate) ≤ 4095.
    - min(...,4095) can never fire (max value = int(4095.5f) = 4095 by truncation).
    - Removed from all d_lut_out index computations: s17_lut_xyz_2r, s17_lut_xyz_1r,
      s17_rgb48_xyz_hdwt0_1ch_2row, 4row interior/partial, kernel_rgb48_xyz_3ch_legacy,
      and the standalone xyz conversion kernel (3 components).
    - Saves 1 IMIN per pixel per channel in all RGB→XYZ color conversion paths.
    - Expected: ~2-3% RGB+HDWT0 compute reduction → ~0.5-1% overall.

    V67 Improvements (s17_u16_to_f16: PTX cvt.rn.f16.u16 for d_lut_out lookups — parity CUDA V104):
    - All __float2half((float)d_lut_out[...]) = CVT.F32.U16 + CVT.F16.F32 (2 instructions).
    - PTX cvt.rn.f16.u16 converts uint16→__half in 1 instruction (direct, exact for [0,4095]).
    - Applied to all d_lut_out lookups in hot RGB→XYZ kernels (2row, 1row, 4row variants).
    - Saves ~6M CVT/frame (2K), ~25M CVT/frame (4K) in color conversion phase.
    - Expected: ~5-10% RGB+HDWT0 compute reduction → ~1-2% overall (RGB ~20% GPU time).

    V66 Improvements (vq8 PRMT sign extraction + frcp_rn reduction — parity CUDA V103):
    - Sign extraction: __byte_perm(r01.x, r01.y, 0x7531) & 0x80808080 replaces 8×SETP+8×SELP.
      IEEE half-float sign bit = bit 15 = byte index 1; 0x7531 selects sign bytes into correct positions.
    - Abs via AND 0x7FFF7FFF: clears half sign bit in packed uint32 — 4 AND vs 8 IABS.
    - frcp_rn: 1 __frcp_rn(base_step) + 5 FMUL(1/mult constants) vs 6 __frcp_rn — saves 5 SFU ops.
    - Total: 8 ops (signs+abs) vs 24 ops (SETP+SELP+IABS) → saves 16 instructions per 8 elements.
    - Note: -0.0 half (0x8000) → byte=0x80 vs 0x00 previously (negligible, same decoded magnitude).
    - Expected: ~20-25% quantize compute reduction → ~2-3% overall.

    V65 Improvements (H-DWT Beta/Delta boundary hoist — remove min(x+1,w-1) from hot loops — parity CUDA V102):
    - x in Beta/Delta loops always even: 2+t*2+k*(nt*2). For even w (all DCI): x≤w-2 → x+1≤w-1.
    - min(x+1,w-1) = x+1 always for even w → MIN never fires → remove, change x<w to x<w-1.
    - Loop x<w-1 iterates same values as x<w for even-x, even-w: no behavioral change for DCI.
    - Compiler drops 1 VMIN/SETP/SEL per iteration; #pragma unroll 2/4 generates cleaner HFMA2 chains.
    - Updated: s17_dwt_h_half_io_4row (H-DWT lv1-5 HOT ~15%), s17_rgb48_xyz_hdwt0_1ch_1row_p12 (4K HOT),
               s17_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT), s17_fused_i2f_dwt_h_ho_4row (fallback).
    - Expected: ~2-5% H-DWT speedup → ~0.3-0.8% overall (H-DWT ~15% GPU time).

    V64 Improvements (reg-blocked V-DWT: hoist Beta/Delta boundary case → #pragma unroll 4 — parity CUDA V101):
    - s17_dwt_v_fp16_hi_reg_ho Beta/Delta: split `for y=2..h-1; yp1=min(y+1,h-1)` into:
        main loop (y<h-1, y+1<h guaranteed → col[y+1] direct, no min()) + boundary (odd h only)
    - Main loop `for y=2; y<h-1; y+=2` + `#pragma unroll 4`:
        4 independent HFMA per unrolled batch → 4× FMA ILP; removes per-iter min() call.
    - Boundary case (odd h only): col[h-1] += kB*(col[h-2]+col[h-1]) (matches original min() semantics).
    - Same transformation for Delta loop.
    - Even heights (1080,540,270,all DCI level-0/1/2): boundary never executes.
    - Odd heights (135,67,..., at DWT levels 3-5): one extra HADD+HMUL per thread.
    - Expected: ~15-25% Beta/Delta speedup → ~7-12% V-DWT lifting → ~3-5% overall (4K).

    V63 Improvements (vq8 uint2 stores + #pragma unroll 2 in s17_qe_subband_ml — parity CUDA V100):
    - uint2 store (st.global.v2.b32) replaces 2×uint32_t for aligned zones (col_start%8==0).
    - row_dst = d_enc(256-align) + row*stride(mult-8) → 8-byte aligned base; col_start%8==0 → aligned.
    - Exception: 2K DC zone 2 (col_start=60, 60%8=4) falls back to 2×uint32_t (~0.5% of work).
    - 4K (ll5_c=120): ALL zones aligned → pure uint2 everywhere.
    - const bool vec_store = (col_start&7)==0: loop-invariant, compiler hoists branch outside loop.
    - #pragma unroll 2: 4K L1 rows (2 iters/thread) → fully inlined; 2K (1 iter) → no bloat.
    - Expected: ~25-40% fewer store instructions in hot path → ~3-6% quantize speedup → ~0.3-0.6% overall.

    V62 Improvements (__half2 row-pair packing + HFMA2 in s17_fused_i2f_dwt_h_ho_4row — parity CUDA V99):
    - Interior path: sm01[x]={row0,row1}, sm23[x]={row2,row3} as __half2 (matches s17_dwt_h_half_io_4row)
    - All lifting passes (Alpha/Beta/Gamma/Delta): scalar 4×HMUL → HFMA2 2× throughput
    - #pragma unroll 2 on all interior loops (4K level-0: 2 iters → full ILP)
    - Combined L+H scatter: 2 loops × 4 stores → 1 loop × 8 stores via __low2half/__high2half
    - Partial path (else): unchanged (1 block max, rarely executed)
    - Expected: ~2× lifting throughput for i2f H-DWT level-0 (fallback encode() path)

    V61 Improvements (vq8 int2 loads — 2×int2 __ldg instead of 4×__half2 __ldg — parity CUDA V98):
    - s17_qe_subband_ml vq8: replace 4×ld.global.b32 with 2×ld.global.b64 → 50% fewer load instr
    - int2 load at row_src+c (byte offset c*2): c always multiple of 8 → byte address multiple of 16 ✓
    - 2K DC zone 2 (col_start=60): c*2=120+tx*16 → 8-byte aligned → int2 valid ✓
    - Stores: keep 2×uint32_t (col_start=60 → c%8=4 → not 8-byte aligned for int2 store)
    - Expected: ~2-4% quantize speedup from reduced load instruction count and memory transaction savings

    V60 Improvements (reg-blocked V-DWT: hoist __half constants + #pragma unroll 4 — parity CUDA V97):
    - s17_dwt_v_fp16_hi_reg_ho: hoist kA/kB/kG/kD/kNL/kNH outside all loops (one __float2half each)
    - #pragma unroll 4 on load loop — 4 concurrent __ldg in flight; hides column load latency
    - #pragma unroll 4 on Alpha/Gamma loops — step=2 → fully independent iterations; 4× ILP
    - #pragma unroll 4 on both scatter write loops — 4 concurrent global stores per batch
    - Beta/Delta loops omitted (runtime conditional yp1 complicates unroll)
    - Expected: ~5-10% speedup for reg-blocked V-DWT path (small subbands h ≤ MAX_REG_HEIGHT=140)

    V59 Improvements (#pragma unroll 2 for H-DWT 4-row lifting loops — parity CUDA V96):
    - s17_dwt_h_half_io_4row interior: Alpha/Beta/Gamma/Delta lifting loops → #pragma unroll 2
    - 4K L1 (w=1920, nt=512): ~2 iters/thread → fully unrolled; 2K L1 (~1 iter): pragma ignored
    - Each lifting body: 2× HFMA2 pairs in flight → compiler interleaves to hide 23-cycle smem latency
    - Same pattern as V89/V52 `#pragma unroll 4` on 1-row/2-row p12 kernels but adapted for 4-row
    - Expected: ~3-7% H-DWT levels-1+ speedup for 4K content → ~0.5-1% overall improvement
    - Safe: no data dependency between unrolled iterations (smem positions nt*2 apart)

    V58 Improvements (sm_lut preload for s17_rgb48_xyz_hdwt0_1ch_4row_p12 — parity CUDA V95):
    - Add __shared__ __half sm_lut[4096] (8KB static smem) + int4 vectorized preload
    - nt=512, 4096 __half = 512 int4 → each thread loads exactly 1 int4 (same as V53/V54 pattern)
    - sm_lut[] replaces all d_lut_in __ldg accesses in both interior and partial branches
    - Total smem: 15.36KB (DWT) + 8KB (sm_lut) = 23.36KB
    - PreferShared (48KB/SM): 48/23.36=2 blk/SM → 50% occ (same as current PreferNone)
    - ~8 LUT reads × 27cy savings per thread in the main load loop → ~5-10% 4-row HDWT0 speedup
    - Expected: ~2-4% overall improvement (4-row HDWT0 ~20% of GPU compute)

    V57 Improvements (#pragma unroll 2 in H-DWT 4-row interior load+scatter loops — parity CUDA V94):
    - s17_dwt_h_half_io_4row interior path: load loop `for i=t; i<w; i+=nt` → #pragma unroll 2
    - Scatter loop `for p=t; p<w/2; p+=nt` → #pragma unroll 2
    - 2K L1 (w=960, nt=512): 2 iters/thread → fully unrolled; 4K L1 (w=1920): 4 iters → 2 pairs
    - Compiler issues 2×4 concurrent __ldg + 2×8 global writes → better DRAM latency pipelining
    - Alpha/Beta/Gamma/Delta lifting loops left unchanged (only ~1 iter per thread at level 1)
    - Expected: ~2-5% H-DWT speedup → ~0.3-0.7% overall improvement

    V56 Improvements (vq8 — 8-element vectorized quantize in s17_qe_subband_ml — parity CUDA V93):
    - vq8: 4×__half2 loads (16B) + 2×uint32_t stores (8B) per thread per iteration
    - vs vq4: 2×__half2 loads (8B) + 1×uint32_t store (4B) — halves loop iterations again
    - Same total memory traffic as vq4; 50% fewer loop-counter updates and branch checks
    - Zone widths ≥ 2*ll5_c are div-by-8 (2K: ≥120 ✓; 4K: ≥240 ✓) → zero tail for L1..L5 rows
    - DC rows only: two zones of width ll5_c; for 2K (ll5_c=60): 60%8=4 → vq4 tail (4 elements)
    - For 4K (ll5_c=120): 120%8=0 → zero tail in ALL zones → pure vq8 everywhere in 4K
    - DC rows are only ~3.1% of rows → tail overhead negligible even for 2K
    - Expected: ~3-5% quantize speedup → ~0.3-0.5% overall (quantize ~10% of GPU compute)

    V55 Improvements (V_TILE 22→24 — reduce V-DWT tile count, parity CUDA V92):
    - V_TILE_FL: 30→32; col2[32] uses 32 __half2 regs; + ~10 other = ~42 total → launch_bounds OK
    - Tiles for 2K (height=1080): ceil(1080/22)=49 → ceil(1080/24)=45 → 8% fewer tiles
    - Tiles for 4K (height=2160): ceil(2160/22)=99 → ceil(2160/24)=90 → 9% fewer tiles
    - Overlap ratio: 30/22=1.364 → 32/24=1.333 → 3% less overlap load overhead
    - V_TILE=24 even + V_OVERLAP=4 even → P0=0, load_start=even; static_assert passes ✓
    - Expected: ~2-4% V-DWT speedup → ~0.8-1.6% overall improvement

    V54 Improvements (int4 vectorized LUT preload — parity CUDA V91):
    - Replace V53's #pragma unroll 8 preload loop with single int4 reinterpret_cast + __ldg
    - nt=512T, 4096 __half / 8 per int4 = 512 int4 entries → each thread loads exactly 1 int4
    - 1 ld.global.b128 per thread (128-bit) vs 8 ld.global.b16 → saves 7 global load instructions
    - cudaMalloc guarantees 256B alignment → int4 access of d_lut_in is 16B aligned ✓
    - Same __syncthreads() preserved; sm_lut read pattern unchanged
    - Expected: ~5-10% preload speedup (minor, but reduces SM instruction pressure)

    V53 Improvements (Shared-memory preload of d_lut_in — parity CUDA V90):
    - __shared__ __half sm_lut[4096] (8KB static smem) added to 1-row and 2-row p12 kernels
    - Preload: 4096/512=8 sequential coalesced reads per thread + __syncthreads()
    - Load loop then reads sm_lut[ri] (~3-cycle smem) vs d_lut_in __ldg (~30-50 cycle L1/L2)
    - 3 saved LUT-read latencies × 27 cycles × 4 pairs/thread = ~324 cycles/thread gain
    - Cost: 1 extra syncthreads (~10 cycles) + 8 preload reads (~8 cycles) = ~20 cycles
    - Smem: 8KB (lut) + 7.68KB (DWT) = 15.87KB → PreferShared (48KB/SM) → 3 blk/SM (was 4 PreferNone)
    - Net: ~300 cycles/thread gain >> -25% warps cost → overall speedup expected
    - d_lut_out stays as __ldg (accessed once/pair, better L1 hit rate due to spatial correlation)
    - Expected: ~20-40% load-section speedup → ~3-8% overall improvement

    V52 Improvements (#pragma unroll in load/lifting/scatter loops — parity CUDA V89):
    - 1-row p12 (4K): w=3840, nt=512 → ~4 iters/thread → #pragma unroll 4
    - 2-row p12 (2K): w=1920, nt=512 → ~2 iters/thread → #pragma unroll 2
    - Load loop unroll: compiler issues all N iterations' __ldg requests simultaneously (memory ILP)
    - Lifting loop unroll: compiler interleaves N iterations' smem reads/HFMA2s → hides 23-cycle smem latency
    - Scatter loop unroll: compiler interleaves smem reads + global writes for better throughput
    - d_lut_in: 4096 __half entries (8KB), random-access, ~50-cycle L2 miss penalty
    - With #pragma unroll 4: 4× more in-flight LUT requests → 4× better miss latency hiding
    - Expected: ~5-15% load-section speedup → ~1-3% overall improvement

    V51 Improvements (__launch_bounds__(512,4) for HDWT0 RGB kernels — parity CUDA V88):
    - s17_rgb48_xyz_hdwt0_1ch_1row_p12 and _2row_p12: 512T, smem=7680B → 4 blk/SM smem-limited
    - Without bounds: compiler may use 36 regs/T → 36×512=18432 → 65536/18432=3.55 → 3 blk/SM
    - With __launch_bounds__(512,4): forces ≤ floor(65536/512/4)=32 regs/T → 4 blk/SM guaranteed
    - 3→4 blk/SM = +33% concurrent warp slots → better latency hiding for d_lut_in random-access loads
    - Both kernels are LUT-lookup-heavy: +33% warps significantly improves instruction issue efficiency
    - Expected: ~5-10% HDWT0 speedup → ~1-2% overall improvement
    - Safe: compiler may spill ≤4 regs to L1; L1 spill latency (10-20 cycles) < LUT miss latency (50-100 cycles)

    V50 Improvements (__launch_bounds__(256,6) for register-limited kernels — parity CUDA V87):
    - s17_qe_subband_ml and s17_dwt_v_tiled_ho_2col: no smem → pure register-limited occupancy
    - Without bounds: compiler may use 44 regs/T → 44×256=11264 → 65536/11264=5.82→5 blk/SM
    - With __launch_bounds__(256,6): ≤ floor(65536/256/6)=42 regs/T → 6 blk/SM guaranteed
    - +20% concurrent warp slots → better latency hiding for __half2float and __ldg misses
    - V-DWT 2-col ~40% GPU compute: 20% gain → ~8% overall; quantize ~10%: 20% gain → ~2%
    - Safe: worst case no change; possible ≤2-reg spill to L1 cache (negligible latency)

    V49 Improvements (__hmul2 scatter in 1-row kernel — 2× FP16 throughput — parity CUDA V86):
    - sm[p*2] and sm[p*2+1] are adjacent __half → natural __half2 pair at smem bank p (no conflict)
    - Load as __half2 (ld.shared.b32) and apply {NORM_L, NORM_H} via one HMUL2 instruction
    - Saves 1 smem load instruction + 1 scalar HMUL per iteration vs 2 loads + 2 HMULs
    - __low2half / __high2half extract L/H for the 2 st.global.b16 stores (same as before)
    - bank(sm[p*2]) = p%32 for p=t → all 32 warp threads access distinct banks → no conflict
    - Expected: ~3-7% scatter speedup → ~0.3-0.7% overall 4K improvement

    V48 Improvements (PreferL1 for reg-blocked V-DWT + quantize — parity CUDA V85):
    - s17_dwt_v_fp16_hi_reg_ho: no smem → cudaFuncSetCacheConfig(PreferL1) gives 48KB L1 (was 16KB)
    - s17_qe_subband_ml: no smem → PreferL1 gives 48KB L1
    - Both kernels are register-limited, not smem-limited → cache config change is occupancy-free
    - Reg-blocked V-DWT: larger L1 improves temporal locality for col[] register reuse patterns
    - Quantize: larger instruction cache reduces I-cache misses across multi-zone branch chains
    - Expected: ~1-3% speedup in reg-blocked V-DWT and quantize → ~0.1-0.2% overall

    V47 Improvements (vq4 — 4-element vectorized quantize in s17_qe_subband_ml — parity CUDA V84):
    - vq4 processes 4 elements per thread per loop: 2×__half2 loads (8B) + 1×uint32_t store (4B)
    - vs vq2: 1×__half2 load (4B) + 1×uint16_t store (2B) per iteration
    - Halves loop count, branch overhead, and loop-counter updates; same total memory traffic
    - All zone widths are multiples of ll5_c=stride/32 (=60 for 2K, 120 for 4K), all div by 4 → no tail
    - vq4 covers 100% of columns in every zone without residual elements
    - Expected: ~3-5% quantize speedup → ~0.2-0.4% overall improvement

    V46 Improvements (__half2 row-pair packing in s17_rgb48_xyz_hdwt0_1ch_2row_p12 — parity CUDA V83):
    - Hoist y1<height check to block level (interior: y1 valid, partial: y0-only scalar)
    - Interior path: smem reinterpreted as __half2*: sm2[x]={row0[x],row1[x]} (same 2w __half smem)
    - Load: store {e0,e1} and {o0,o1} as __half2 pairs directly into interleaved layout
    - H-DWT lifting: __hfma2 for all 4 steps (Alpha/Beta/Gamma/Delta) — 2× FMA throughput
    - Scatter: combined L+H loop via __low2half/__high2half (same as V45/V82 pattern)
    - Partial path: y0-only scalar (height always even for 2K/4K → partial never taken in practice)
    - Expected: 5-10% H-DWT throughput gain in RGB+HDWT0 kernel → ~1-2% overall improvement

    V45 Improvements (__half2 row-pair packing in s17_dwt_h_half_io_4row interior — parity CUDA V82):
    - Interior path: smem reinterpreted as __half2* pairs: sm01[x]={row0[x],row1[x]}, sm23[x]={row2[x],row3[x]}
    - Each __hfma2 lifts both rows of a pair simultaneously → 2× FMA throughput (2 ops vs 4 per lifting step)
    - smem layout changes from sequential (row0[0..w-1], row1[0..w-1], ...) to interleaved pairs
    - Total smem unchanged: 4w halves = 2w __half2 per pair × 2 pairs (sm01+sm23)
    - Scatter: two separate even-L / odd-H loops → one combined p-loop with __low2half/__high2half
    - Partial else-branch (y3 >= h) left unchanged (scalar code, rarely executed)
    - Expected: 5-10% H-DWT level-1+ speedup → ~1-2% overall FPS improvement

    V44 Improvements (pair-wise 12-bit unpack + combined scatter — parity with CUDA V81):
    - Load loop: px=t per pixel → pair=t per 2-pixel pair; shared byte[1] eliminates duplicate load
    - Per pair: 9 byte loads (3×3 channels) vs 12 (4×3 with duplicate byte[1]) → 25% fewer byte reads
    - Combined scatter: 2 even/odd loops → 1 pair loop (NORM_L + NORM_H per pair)
    - Applied to s17_rgb48_xyz_hdwt0_1ch_1row_p12 and s17_rgb48_xyz_hdwt0_1ch_2row_p12
    - w always even for 2K/4K → no odd-width edge cases needed
    - Expected: 5-10% RGB+HDWT0 speedup → ~1-2% overall FPS improvement

    V43 Improvements (V_TILE=22, V_OVERLAP=4 for 2-col V-DWT — +20% throughput; write-combining H2D):
    - V_TILE: 28→22, V_OVERLAP: 5→4, V_TILE_FL: 38→30 → reduces 2-col col2[] from 38→30 int regs
    - 2-col kernel: ~50 regs/T (V_TILE_FL=38) → ~42 regs/T (V_TILE_FL=30) → 6 blk/SM (75% occ) vs 5
    - Throughput: 5×28/38=3.68 → 6×22/30=4.40 → +19.6% V-DWT throughput; V-DWT ~40% total → ~8% gain
    - P0 changes to 0 (load_start always even: V_TILE=22 even, V_OVERLAP=4 even → load_start even)
    - Lifting loops: ALPHA/GAMMA start at i=1 (odd, globally-odd), BETA/DELTA start at i=2 (skip i=0 boundary)
    - Skipping i=0 in BETA/DELTA is correct: errors at col[0..2] don't propagate to col[4] first output
    - Output loop: p0=0 → even i = LP (was HP), odd i = HP (was LP): swap `!(i&1)` ↔ `(i&1)` for H select
    - static_assert: V_TILE even + V_OVERLAP even (was V_OVERLAP odd)
    - write-combining: h_rgb12_pinned uses cudaHostAllocWriteCombined (was Default)
      PCIe DMA reads from WC memory don't require cache snooping → up to 40% H2D bandwidth gain
    - Expected: ~8% V-DWT speedup + 10-40% H2D speedup → 5-15% overall FPS improvement

    V42 Improvements (4K RGB+HDWT0: new 1-row p12 kernel — 50% → 100% occupancy for 4K):
    - New s17_rgb48_xyz_hdwt0_1ch_1row_p12: 1 row per block, smem=width×2B
    - For 4K (width=3840): smem=7,680B; PreferNone→4 blk/SM=100% occ (vs 2-row 2 blk/SM=50%)
    - Dispatch: width>2048 uses 1-row (4K), width≤2048 uses 2-row (2K, from V41)
    - Simpler kernel: no y1 guard, no second-row smem section
    - Same wave count as 2-row: height/(SM×blk) identical, but 2× warps for latency hiding
    - Expected: 30-50% RGB+HDWT0 speedup for 4K content → ~7-12% overall gain at 4K

    V41 Improvements (2K RGB+HDWT0: switch from 4-row to 2-row kernel — 50% → 100% occupancy):
    - After V38: 4-row RGB+HDWT0 at 2K runs at 50% occ (PreferNone → 2 blk/SM × 512T = 1024T)
    - 2-row at 2K: smem=2×1920×2=7,680B; PreferNone→32KB/SM→4 blk/SM → 2048T = 100% occ
    - Number of waves unchanged: 540 blk/(20SM×4)=6.75 vs 270/(20SM×2)=6.75 — identical
    - But 100% occ means 2× warps/SM → near-perfect LUT-access latency hiding (LUTs need ~32W)
    - 4K path unchanged: 2-row at 4K with PreferNone → 2 blk/SM = 50% occ (no regression)
    - Change: `use_2row = (width > 2048)` → `use_2row = true` (always use 2-row variant)
    - Expected: 30-50% RGB+HDWT0 speedup (2K, 25% of GPU time) → ~7-12% overall gain

    V40 Improvements (Fix H-DWT 4-row PreferL1 occupancy bug at level 1 — parity CUDA V77):
    - s17_dwt_h_half_io_4row at level 1 (w=960): smem=4×960×2=7,680 bytes per block
    - PreferL1 → 16KB smem/SM → 16384/7680=2 blk/SM → 1024T = 50% occupancy — regression
    - PreferNone → 32KB smem/SM → 32768/7680=4 blk/SM → 2048T = 100% occupancy
    - V30 comment "smem fits in 16KB smem of PreferL1" was correct for fit but wrong on capacity:
      2 blk × 7680B = 15360B fills 16KB limit → only 2, not 4, blk/SM simultaneously
    - Level 2+ (w≤480, smem≤3.84KB): 4 blk/SM with both PreferL1/None → no change in occ
    - Trade-off: 48KB→32KB L1 at levels 2-5; those levels use tiny data sets (well L2-cached)
    - Level 1 is the largest H-DWT level: 270 blocks per channel, 960×540 = 1MB data
    - With 100% occ: 4 blk/SM → 4× more warps → far better memory-latency hiding
    - Expected: 30-50% H-DWT level-1 speedup → H-DWT ~15% total → ~5-7% overall gain

    V39 Improvements (2-col __half2 tiled V-DWT — HFMA2 doubles lifting throughput, parity CUDA V76):
    - New s17_cdf97_lift_tiled_h2: __half2 lifting via __hfma2(kA, __hadd2(prev,next), cur)
    - New s17_dwt_v_tiled_ho_2col: each thread processes columns x and x+1 simultaneously
    - s17_launch_comp_pipeline: replace s17_dwt_v_tiled_ho with s17_dwt_v_tiled_ho_2col
    - Grid x = ceil(w/2/vt_blk): half x-blocks; HFMA2 = 2 half FMAs per cycle on sm_61+
    - __half2 col2[38] ≈ 38-50 regs/T → 5-6 blk/SM (vs 8); effective: 5-6 × 2 = 10-12 cols
    - Req: w always even for 2K/4K DWT levels; x = (blockIdx.x*vt_blk+t)*2 always even
    - Expected: 25-50% V-DWT speedup; V-DWT ~40% GPU time → 10-20% overall improvement

    V38 Improvements (Fix PreferL1 occupancy bug for 4-row/2-row RGB+HDWT0 — parity CUDA V75):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12 and 2row_p12: smem=15.36KB each
    - PreferL1 → 16KB smem/SM → only 1 block/SM (25% occ) — previously undetected regression
    - Change to PreferNone (32KB smem/SM) → 2 blocks/SM → 1024T = 50% occupancy
    - 32KB L1 (PreferNone) still caches 16KB LUTs with 16KB room for RGB data reads
    - 1-row kernel (smem=3.84KB) retains PreferL1: 16/3.84=4 blk/SM = 100% occ
    - Expected: 5-15% RGB+HDWT0 throughput improvement from 2× more active warps/SM

    V37 Improvements (4-rows-per-block H-DWT level 0 — 75% fewer dispatches, parity CUDA V74):
    - New kernel s17_fused_i2f_dwt_h_ho_4row: 4 rows per block, grid=(h+3)/4, smem=4*w*__half
    - s17_launch_comp_pipeline: level-0 now launches s17_fused_i2f_dwt_h_ho_4row (non-skip path)
    - Reduces grid from h=1080 to 270 blocks (2K) or 2160→540 (4K) — 75% fewer dispatches
    - 4 rows' __int2float_rn load chains in-flight simultaneously — better DRAM latency hiding
    - __syncthreads amortized over 4× more work vs s17_fused_i2f_dwt_h_ho (1-row base kernel)
    - 2K smem=15.36KB: set PreferShared (48KB smem → 3 blk/SM); PreferL1 would limit 1 blk/SM
    - No LUTs in this kernel → no L1 LUT caching benefit; occupancy improvement dominates
    - Expected: 3-8% level-0 H-DWT speedup → ~0.5-1.5% overall improvement

    V36 Improvements (Adaptive H-DWT thread count for levels 1+ — parity CUDA V73):
    - s17_launch_comp_pipeline: replace fixed h_blk with per-level adaptive count in lv=1+ loop
    - Fixed h_blk=512 wastes 50-77% threads at small widths: level-3 2K w=240 → 512T=46.9% util
    - Adaptive: (w>480)?h_blk:(w>240)?256:(w>120)?128:64 — nearly 100% utilization at all levels
    - 2K level-3 (w=240): 512T→256T, utilization 46.9%→93.75%; level-4 (w=120): →128T 93.75%
    - 4K level-3 (w=480): 512T→256T; level-4 (w=240): 256T; level-5 (w=120): 128T
    - s17_dwt_h_half_io_4row uses blockDim.x dynamically — compatible with any thread count
    - Expected: 5-15% H-DWT speedup for levels 2-4 (~15% GPU time → ~0.75-2% overall)

    V35 Improvements (2-row RGB+HDWT0 for 4K — enables PreferL1 at 4K, parity CUDA V72):
    - s17_rebuild_v17s_comp_graphs: conditionally launch s17_rgb48_xyz_hdwt0_1ch_2row_p12 for 4K
    - 4-row smem at 4K: 4×3840×2=30.72KB > 16KB PreferL1 limit → runtime uses PreferShared (16KB L1)
    - 2-row smem at 4K: 2×3840×2=15.36KB < 16KB PreferL1 limit → PreferL1 honored (48KB L1)
    - With PreferL1 at 4K: lut_in(8KB)+lut_out(8KB)=16KB fit in 48KB L1 with ample room for RGB data
    - With PreferShared at 4K: 16KB L1 exactly fits LUTs (16KB), prone to eviction with RGB reads
    - 2K path unchanged: 4-row smem=15.36KB < 16KB → PreferL1 already works; no change
    - 2-row grid at 4K: ceil(2160/2)=1080 blocks per component (vs 540 at 4-row) — 2× more blocks
    - Expected: 5-15% RGB+HDWT0 speedup at 4K from L1 LUT caching (~25% GPU time → 1-4% overall)

    V34 Improvements (V_TILE 24→28 — 4K DRAM savings, 100% occupancy retained, parity CUDA V71):
    - V_TILE=28: V_TILE_FL=38 → __half col[38] → ~31 regs/T → 65536/(256×31)=8.2 → 8 blk = 100%
    - 4K level-0: overlap rows 75KB > 48KB L1 → DRAM-bound; 15% fewer overlap reads vs V_TILE=24
    - 2K level-0: 45→39 tiles, 450→402 overlap reads/col → 11% fewer (L1-cached, smaller benefit)
    - Grid dispatches: 2K 45→39 tiles; 4K 90→77 tiles → 13-14% fewer block launches
    - 100% occupancy maintained (31 regs ≤ 32-reg thread-limited threshold for 256T/block)
    - Expected: 2-6% V-DWT speedup, larger for 4K (DRAM-bound overlaps)

    V33 Improvements (V_TILE 16→24 — reduce overlap overhead, 100% occupancy retained, parity CUDA V70):
    - V32 __half col[V_TILE_FL]: ~23 regs/thread → thread-limited at 8 blk/SM = 100% occupancy
    - V_TILE=24: V_TILE_FL=34 → __half col[34] → ~29 regs/T → 65536/(256×29)=8.8 → 8 blk/SM = 100%
    - 100% occupancy preserved (thread-limited: 8 blk × 256T = 2048T = max for sm_61)
    - Useful rows per tile: 16/26=61.5% → 24/34=70.6% → 14.8% fewer total DRAM reads per column
    - Grid tiles: ceil(1080/16)=68 → ceil(1080/24)=45 (2K level-0) → 34% fewer block dispatches
    - constexpr V_TILE_FL=34: #pragma unroll loops auto-cover 34 elements, no code changes needed
    - Expected: 3-8% V-DWT speedup from reduced memory traffic (~40% of GPU time → ~1-3% overall)

    V32 Improvements (__half column registers in V-DWT kernels — parity CUDA V69):
    - s17_dwt_v_tiled_ho: float col[V_TILE_FL] → __half col[V_TILE_FL]
    - s17_dwt_v_fp16_hi_reg_ho: float col[MAX_REG_HEIGHT] → __half col[MAX_REG_HEIGHT]
    - New s17_cdf97_lift_tiled_h: __half lifting helper, P0=1 hardcoded + #pragma unroll
    - Eliminates 26 __half2float loads + 16 __float2half stores per tile in tiled kernel
    - __half HFMA has 2× throughput vs float FMA on sm_61+; V-DWT lifting is 40 FMAs/tile-col
    - Reg-blocked: direct __ldg→__half load; __half lifting; direct __half output stores
    - Expected: 3-8% V-DWT speedup from 2× HFMA throughput (~40% of GPU time → ~1-3% overall)

    V31 Improvements (fuse 4-row RGB loads into interior block — parity CUDA V68):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12: interior load section restructured with if(y3<height)
    - Interior (100% of 2K blocks): single for loop issues 4 rows × 3 planes × 2 bytes + LUT loads
      simultaneously (full MLP: 24 byte __ldg + 8 LUT __ldg all in-flight per pixel iteration)
    - Else (partial last block): original per-row guarded loads (y3 not loaded — out of bounds)
    - S17_U12_4R macro moved outside if/else block
    - Expected: 3-8% RGB+HDWT0 speedup from improved load-pipeline MLP utilization

    V30 Improvements (CachePreferL1 for H-DWT levels-1+ kernels — parity CUDA V67):
    - cudaFuncSetCacheConfig(s17_dwt_h_half_io_4row/2row/1row, cudaFuncCachePreferL1)
    - H-DWT kernels: smem ≤ 4*960*2=7.5KB at level-1 2K → fits in 16KB smem of PreferL1
    - 48KB L1 (vs default 16KB) caches __ldg-loaded V-DWT __half output rows
    - Adjacent 4-row groups share ~7.5KB of input; 48KB L1 holds 6 groups simultaneously
    - Expected: 2-5% H-DWT levels-1+ speedup from improved __ldg cache hit rate

    V29 Improvements (hoist row-presence checks out of H-DWT levels-1+ 4-row lifting — parity CUDA V66):
    - s17_dwt_h_half_io_4row: single `if (y3 < h)` check at block level (uniform → no divergence)
    - Interior path: levels 1/4 (h=540/68 for 2K, 100% blocks), levels 2/3 mostly interior
    - ALPHA/BETA/GAMMA/DELTA loops: 3 fewer `if(yN<h)` guards per iteration → ILP across 4 rows
    - Load loop: 4 rows in single for-body (no yN<h branches) → coalesced 4-row smem fill
    - Store loops: 4 unconditional writes each → 12 fewer branches/thread in store phase
    - Partial last block (h%4!=0): original per-iteration yN<h guards preserved unchanged
    - Expected: 3-7% H-DWT levels-1+ speedup (~20% of GPU time → ~0.6-1.4% overall speedup

    V28 Improvements (hoist row-presence checks out of 4-row RGB+HDWT0 lifting — parity CUDA V65):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12: single `if (y3 < height)` check at block level
    - Interior path (100% of 2K blocks, 1080%4==0): no yN<height inside lifting loops
    - ALPHA/BETA/GAMMA/DELTA loops: 4 rows fused per iteration → compiler schedules ILP across
      all 4 row updates (4 FP16 FMAs/iter vs 1+3 guarded); branch-predictor pressure eliminated
    - Store loops: 4 unconditional writes each → 12 fewer branches/thread in store phase
    - Partial last block (height%4!=0): original checked path preserved unchanged
    - Expected: 5-10% RGB+HDWT0 speedup (~25% of GPU time → ~1.25-2.5% overall GPU improvement)

    V27 Improvements (interior-tile unrolled load+store for tiled V-DWT — parity CUDA V64):
    - s17_dwt_v_tiled_ho: interior = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= h)
    - Interior path (97% of tiles for 2K level-0): #pragma unroll V_TILE_FL load loop → 34
      independent __ldg instructions emitted; GPU memory pipeline issues all 34 simultaneously
    - Interior output: #pragma unroll V_TILE store loop → !(i&1) resolved at compile time →
      12 L stores + 12 H stores emitted as straight-line code; no loop counter or branch overhead
    - Boundary path (first+last tile, 3%): existing WS bounds-check code unchanged
    - interior check is uniform (all threads share tile_start/h) → no warp divergence
    - Expected: 5-12% V-DWT speedup (eliminates loop overhead + branch-predict pressure for 97% tiles)

    V26 Improvements (v_block_tiled=256 for tiled V-DWT — parity with CUDA V63):
    - SlangGpuConfig17::v_block_tiled=256 added; separate from v_block=128 for reg-blocked path
    - Tiled V-DWT (h > MAX_REG_HEIGHT): 256T/block → 8 warps/block vs 4 at 128T
    - 8 warps → warp scheduler has 2× choices to hide DRAM latency on overlap row loads
    - Fewer blocks: ceil(1920/256)×68=544 (level 0) vs ceil(1920/128)×68=1020 → 47% fewer dispatches
    - Same 87.5% SM occupancy (7 blk/SM × 256T = 1792T) — reg-blocked preserves 128T
    - 128T (v_block) kept for reg-blocked (h≤MAX_REG_HEIGHT): float col[140] → ~150 regs → collapse at 256T
    - Expected: ~3-5% additional V-DWT speedup on dominant large-h levels

    V25 Improvements (V_TILE 32→16 — higher occupancy wins vs fewer loads, parity CUDA V62):
    - V_TILE: 32 → 16 (V_TILE_FL: 42 → 26) reverting V17j, given V22 PreferL1
    - V22 PreferL1: overlap rows (10 rows=37.5KB) always hit 48KB L1 regardless of V_TILE
    - V_TILE=32: V_TILE_FL=42 → ~50 regs/T → ~10 blk/SM → 40 warps; need ~116 to hide DRAM latency
    - V_TILE=16: V_TILE_FL=26 → ~36 regs/T → 14 blk/SM → 56 warps → +37% DRAM latency coverage
    - Extra 340 overlap loads/column (1768 vs 1428) all hit L1 → effectively free
    - Expected: 20-25% V-DWT speedup → ~8-10% overall GPU improvement

    V24 Improvements (4-rows-per-block RGB+HDWT0 kernel — parity with CUDA V61):
    - s17_rgb48_xyz_hdwt0_1ch_4row_p12: processes 4 rows per block (vs 2 in V18e)
    - grid=(h+3)/4=270 blocks per component (vs 540 for 2K); smem=4*w*sizeof(__half)=15.4KB
    - Same 100% SM occupancy (thread-limited: 4 blk/SM × 512T = 2048T)
    - Matrix loads m0/m1/m2 amortized over 4 rows; 4 syncthreads/pass over 4 rows
    - CachePreferL1 applied for 16KB LUT caching alongside 15.4KB smem
    - Expected: ~1-2% RGB+HDWT0 speedup from halved block count

    V23 Improvements (full 100% vectorized quantize — parity with CUDA V60):
    - s17_qe_subband_ml: extend vq2 to L4/L5/DC rows (was scalar fallback, ~12.5% of rows)
    - DC rows: 6 vq2 zones [0,ll5_c)→inv_dc, [ll5_c,2c)→inv_l5, ..., [16c,stride)→inv_l1
    - L5 rows: 5 zones starting [0,2c)→inv_l5; L4 rows: 4 zones [0,4c)→inv_l4
    - All zone boundaries are multiples of ll5_c=stride/32: even for 2K(60) and 4K(120)
    - 100% of rows now use __half2 loads + uint16_t stores; scalar path eliminated
    - Expected: ~3% additional quantize speedup over V21

    V22 Improvements (cudaFuncSetCacheConfig PreferL1 — parity with CUDA V59):
    - s17_dwt_v_tiled + s17_dwt_v_tiled_ho: CachePreferL1 → 48KB L1 (vs default 16KB)
    - Adjacent Y-tiles share V_OVERLAP=5 rows (10 rows = 38.4KB) — fits in 48KB L1, not 16KB
    - s17_rgb48_xyz_hdwt0_1ch_2row_p12 + _1ch: 7.5KB smem < 16KB limit; 48KB L1 holds 16KB LUTs
    - Set once in init() before any kernel launch or graph capture
    - Expected: 5-15% V-DWT speedup on dominant BW-bound level-0 pass

    V21 Improvements (vectorized quantize for L1/L2/L3 rows — parity with CUDA V58):
    - s17_qe_subband_ml: ~87.5% of rows now use __half2 loads + uint16_t stores
    - L1 (row_lv==1, rows ≥ 544 for 2K, 50%): 1 zone, uniform inv_l1
    - L2 (rows 272..543, 25%): 2 zones split at stride/2 (inv_l2 / inv_l1)
    - L3 (rows 136..271, 12.5%): 3 zones at 480/960 boundaries (inv_l3/inv_l2/inv_l1)
    - vq2 lambda: __half2 load, 2-sample quantize, uint16_t store per iteration
    - L4/L5/DC rows (~12.5%): scalar fallback, per-column zone detection unchanged
    - Expected: ~40% quantize speedup → ~0.12ms/frame saved

    V20 Improvements (per-subband QCD correctness fix — parity with CUDA V57):
    - V18d applied level-dependent weights (LL5×0.65 … L1×1.20) but QCD/QCC wrote uniform steps
    - Decoder dequantized LL5 at 1/0.65=1.54× amplitude → DC component reconstructed wrong
    - Fix: s17_perceptual_sb_entry(base, i, is_4k) encodes exact step for each subband index i
    - s17_qcd_step_entry converts float step → (eps<<11)|man using J2K scalar-expounded formula
      where eps = 13 - floor(log2(step)) and man = floor((step/2^(13-eps) - 1) × 2048)
    - s17_build_j2k_codestream QCD/QCC updated to call s17_perceptual_sb_entry per subband
    - 4K path unchanged (uniform steps — 4K doesn't apply perceptual weights)

    Performance target: 620+ fps (V20 — correctness fix, no throughput change).
*/

#include "slang_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <future>
#include <vector>
#include <mutex>

static constexpr int   NUM_DWT_LEVELS  = 5;
static constexpr int   MAX_REG_HEIGHT  = 140;
/* V17j/V25/V33/V34/V43: tiled V-DWT constants.
 * V43: V_TILE 28→22, V_OVERLAP 5→4, V_TILE_FL 38→30 — reduces 2-col register pressure.
 *   Single-col: __half col[30] packs ~15 regs/T → 8 blk/SM (100% occ, thread-limited).
 *   2-col: __half2 col2[30] = 30 int regs + ~12 other = ~42 regs/T → 6 blk/SM (75% occ).
 *   vs old V_TILE_FL=38: 2-col ~50 regs/T → 5 blk/SM (62.5%). Net: 6×22/30=4.40 vs 5×28/38=3.68 (+19.6%).
 *   P0=0 (load_start = tile_start - V_OVERLAP; V_TILE=22 even, V_OVERLAP=4 even → always even).
 * V34: V_TILE 24→28 — 100% occupancy retained (31 regs ≤ 32 limit); better 4K DRAM savings.
 * V33: V_TILE 16→24 — V32 __half reduced regs ~36→~23/T → thread-limited; 14.8% fewer loads.
 * V25: V_TILE 32→16 (reverts V17j) to improve SM occupancy given PreferL1 (V22).
 * V_OVERLAP=4 halo rows each side (minimum for 4-step CDF 9/7: stencil ±1 × 4 steps). */
static constexpr int   V_TILE    = 24;  /* V55: was 22; V_TILE_FL→32; 8% fewer tiles, ~2% less DRAM BW */
static constexpr int   V_OVERLAP = 4;
static constexpr int   V_TILE_FL = V_TILE + 2 * V_OVERLAP;  /* 32 */

/* CDF 9/7 lifting coefficients */
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;

/* V17: CDF 9/7 analysis normalization.
 * Applied at output of each 1D DWT (both H and V passes):
 *   L × NORM_L = 1/K  → removes K DC gain from lowpass
 *   H × NORM_H = K    → amplifies highpass to match
 * Net gain for LL5 after 5 2D levels: (NORM_L²)^5 × K^10 = (1/K²)^5 × K^10 = 1.0 */
static constexpr float NORM_L = 0.812893197535108f;  /* 1/K */
static constexpr float NORM_H = 1.230174104914001f;  /* K   */

static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_QCC = 0xFF5D;  /* per-component QCD override */
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;
static constexpr uint16_t J2K_TLM = 0xFF55;


/* ===== Device helpers ===== */

/** V67: Direct uint16→__half conversion via PTX cvt.rn.f16.u16 (1 instruction).
 *  Replaces __float2half((float)v) = CVT.F32.U16 + CVT.F16.F32 (2 instructions).
 *  Exact for v ∈ [0, 4095]: all DCP XYZ output values fit exactly in half precision.
 *  Saves ~6M CVT/frame (2K) or ~25M CVT/frame (4K) across all RGB→XYZ kernels. */
__device__ __forceinline__ __half s17_u16_to_f16(uint16_t v)
{
    __half h;
    asm("cvt.rn.f16.u16 %0, %1;" : "=h"(*reinterpret_cast<unsigned short*>(&h)) : "h"(v));
    return h;
}

/* ===== Horizontal DWT kernels with V17 normalization ===== */

__global__ void s17_fused_i2f_dwt_h(const int32_t* __restrict__ in, float* __restrict__ out,
                                      int w, int h, int s)
{
    extern __shared__ float sm[];
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = __int2float_rn(in[y*s+i]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += ALPHA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*ALPHA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*BETA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += GAMMA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*GAMMA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*DELTA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2;   x < w; x += nt*2) out[y*s+x/2]    = sm[x] * NORM_L;
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x] * NORM_H;
}

__global__ void s17_dwt_h(const float* __restrict__ data, float* __restrict__ out,
                            int w, int h, int s)
{
    extern __shared__ float sm[];
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = data[y*s+i];
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += ALPHA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*ALPHA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*BETA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += GAMMA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*GAMMA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*DELTA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2;   x < w; x += nt*2) out[y*s+x/2]    = sm[x] * NORM_L;
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x] * NORM_H;
}


/* V17: Fused vertical DWT + deinterleave with normalization.
 * d_src == d_dst aliasing is safe: each thread owns its own column exclusively. */
__global__ void s17_dwt_v_fused(const float* __restrict__ d_src,
                                  float* __restrict__ d_work,
                                  float* __restrict__ d_dst,
                                  int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;
    for (int y = 0; y < h; ++y) d_work[y*s+x] = __ldg(&d_src[y*s+x]);

    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] += ALPHA*(d_work[(y-1)*s+x]+d_work[(y+1)*s+x]);
    if (h > 1 && !(h&1)) d_work[(h-1)*s+x] += 2.f*ALPHA*d_work[(h-2)*s+x];

    d_work[x] += 2.f*BETA*d_work[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] += BETA*(d_work[(y-1)*s+x]+d_work[yp1*s+x]);
    }

    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] += GAMMA*(d_work[(y-1)*s+x]+d_work[(y+1)*s+x]);
    if (h > 1 && !(h&1)) d_work[(h-1)*s+x] += 2.f*GAMMA*d_work[(h-2)*s+x];

    d_work[x] += 2.f*DELTA*d_work[min(1,h-1)*s+x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] += DELTA*(d_work[(y-1)*s+x]+d_work[yp1*s+x]);
    }

    /* Write deinterleaved with V17 normalization */
    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) d_dst[(y/2)*s+x]    = d_work[y*s+x] * NORM_L;
    for (int y = 1; y < h; y += 2) d_dst[(hh+y/2)*s+x] = d_work[y*s+x] * NORM_H;
}


/* V17c: H-DWT kernels that write __half output (saves 50% H-DWT write BW).
 * Shared memory lifting stays float32. Only the DRAM write uses half.
 * Used for large-h levels (> MAX_REG_HEIGHT) where V-DWT uses fp16 input. */

/* V17o: __half shared memory — 2× smem bandwidth + fp16 FMA throughput on sm_61+. */
__global__ void s17_fused_i2f_dwt_h_ho(const int32_t* __restrict__ in, __half* __restrict__ out,
                                         int w, int h, int s)
{
    extern __shared__ __half sm[];  /* V17o: was float; halves smem size and bandwidth */
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = __float2half(__int2float_rn(in[y*s+i]));
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(ALPHA)*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += __half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*BETA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(BETA)*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(GAMMA)*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += __half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*DELTA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(DELTA)*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2;   x < w; x += nt*2) out[y*s+x/2]    = sm[x] * __half(NORM_L);
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x] * __half(NORM_H);
}

/**
 * V62: __half2 row-pair packing + HFMA2 for i2f + H-DWT 4-row level-0 — parity CUDA V99.
 * V37: 4-rows-per-block i2f + H-DWT level 0 (int32 input, __half output).
 * grid=(h+3)/4; smem=4*w*sizeof(__half).
 *
 * V62 interior path: sm01[x]={row0[x],row1[x]}, sm23[x]={row2[x],row3[x]} as __half2.
 *   HFMA2 lifting: 2 rows per instruction (2× throughput vs scalar V37).
 *   #pragma unroll 2 on all loops (4K level-0: ~2 iters → full ILP).
 *   Combined L/H scatter via __hmul2+__low2half/__high2half (1 loop vs 2).
 * Partial (else): unchanged — scalar, boundary-guarded, rarely executed.
 */
__global__ void s17_fused_i2f_dwt_h_ho_4row(const int32_t* __restrict__ in, __half* __restrict__ out,
                                              int w, int h, int s)
{
    extern __shared__ __half sm[];
    int y0=blockIdx.x*4, y1=y0+1, y2=y0+2, y3=y0+3;
    int t=threadIdx.x, nt=blockDim.x, hw=(w+1)/2;
    if (y3 < h) {
        /* V62: __half2 row-pair packing — sm01[x]={row0,row1}, sm23[x]={row2,row3}.
         * HFMA2 processes 2 rows per instruction (2× throughput vs V37 scalar). */
        __half2* sm01 = reinterpret_cast<__half2*>(sm);
        __half2* sm23 = reinterpret_cast<__half2*>(sm + 2*w);
        /* V86: int2 __ldg loads — 4 × 64-bit loads replace 8 × 32-bit loads per 2-col pair.
         * DCI widths always even → i=t*2 even → 8-byte aligned int2 load. Parity CUDA V124.
         * 2× fewer load instructions; extract .x/.y for i2f conversion. */
        #pragma unroll 2
        for (int i=t*2; i<w; i+=nt*2) {
            int2 r0 = __ldg(reinterpret_cast<const int2*>(&in[y0*s+i]));
            int2 r1 = __ldg(reinterpret_cast<const int2*>(&in[y1*s+i]));
            int2 r2 = __ldg(reinterpret_cast<const int2*>(&in[y2*s+i]));
            int2 r3 = __ldg(reinterpret_cast<const int2*>(&in[y3*s+i]));
            sm01[i]   = __halves2half2(__float2half(__int2float_rn(r0.x)), __float2half(__int2float_rn(r1.x)));
            sm01[i+1] = __halves2half2(__float2half(__int2float_rn(r0.y)), __float2half(__int2float_rn(r1.y)));
            sm23[i]   = __halves2half2(__float2half(__int2float_rn(r2.x)), __float2half(__int2float_rn(r3.x)));
            sm23[i+1] = __halves2half2(__float2half(__int2float_rn(r2.y)), __float2half(__int2float_rn(r3.y)));
        }
        __syncthreads();
        /* Alpha — HFMA2; #pragma unroll 2 for 4K MLP. */
        {
            const __half2 kA = __half2half2(__float2half(ALPHA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kA, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kA, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V76: DCI w always even and >1 → simplify Alpha boundary; parity CUDA V113. */
            if(t==0) {
                const __half2 kA2 = __half2half2(__float2half(2.f*ALPHA));
                sm01[w-1] = __hfma2(kA2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kA2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* Beta — HFMA2; #pragma unroll 2. */
        {
            const __half2 kB = __half2half2(__float2half(BETA));
            if(t==0) {
                const __half2 kB2 = __half2half2(__float2half(2.f*BETA));
                /* V76: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kB2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kB2, sm23[1], sm23[0]);
            }
            /* V65: x always even → even w: min(x+1,w-1)=x+1; drop MIN per iteration. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kB, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kB, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* Gamma — HFMA2; #pragma unroll 2. */
        {
            const __half2 kG = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kG, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kG, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V76: DCI w always even and >1 → simplify Gamma boundary. */
            if(t==0) {
                const __half2 kG2 = __half2half2(__float2half(2.f*GAMMA));
                sm01[w-1] = __hfma2(kG2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kG2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* Delta — HFMA2; #pragma unroll 2. */
        {
            const __half2 kD = __half2half2(__float2half(DELTA));
            if(t==0) {
                const __half2 kD2 = __half2half2(__float2half(2.f*DELTA));
                /* V76: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kD2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kD2, sm23[1], sm23[0]);
            }
            /* V65: same even-w invariant as Beta — drop MIN. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kD, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kD, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* V62: combined L+H scatter via __low2half/__high2half — 1 loop vs 2. */
        {
            const __half2 nL = __half2half2(__float2half(NORM_L));
            const __half2 nH = __half2half2(__float2half(NORM_H));
            #pragma unroll 2
            for (int p=t; p<w/2; p+=nt) {
                __half2 v01L = __hmul2(sm01[p*2],   nL);
                __half2 v23L = __hmul2(sm23[p*2],   nL);
                out[y0*s+p] = __low2half(v01L);
                out[y1*s+p] = __high2half(v01L);
                out[y2*s+p] = __low2half(v23L);
                out[y3*s+p] = __high2half(v23L);
                __half2 v01H = __hmul2(sm01[p*2+1], nH);
                __half2 v23H = __hmul2(sm23[p*2+1], nH);
                out[y0*s+hw+p] = __low2half(v01H);
                out[y1*s+hw+p] = __high2half(v01H);
                out[y2*s+hw+p] = __low2half(v23H);
                out[y3*s+hw+p] = __high2half(v23H);
            }
        }
    } else {
        /* Partial last block: load with yN<h guards. */
        for (int i=t; i<w; i+=nt)       sm[i]     = __float2half(__int2float_rn(in[y0*s+i]));
        if(y1<h) for(int i=t; i<w; i+=nt) sm[w+i]   = __float2half(__int2float_rn(in[y1*s+i]));
        if(y2<h) for(int i=t; i<w; i+=nt) sm[2*w+i] = __float2half(__int2float_rn(in[y2*s+i]));
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]     +=__half(ALPHA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(ALPHA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(ALPHA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V80: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]   +=__half(2.f*ALPHA)*sm[w-2];
            if(y1<h) sm[2*w-1] +=__half(2.f*ALPHA)*sm[2*w-2];
            if(y2<h) sm[3*w-1] +=__half(2.f*ALPHA)*sm[3*w-2];
        }
        __syncthreads();
        /* V80: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]     +=__half(2.f*BETA)*sm[1];
            if(y1<h) sm[w]   +=__half(2.f*BETA)*sm[w+1];
            if(y2<h) sm[2*w] +=__half(2.f*BETA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]     +=__half(BETA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(BETA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(BETA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]     +=__half(GAMMA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(GAMMA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(GAMMA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V80: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]   +=__half(2.f*GAMMA)*sm[w-2];
            if(y1<h) sm[2*w-1] +=__half(2.f*GAMMA)*sm[2*w-2];
            if(y2<h) sm[3*w-1] +=__half(2.f*GAMMA)*sm[3*w-2];
        }
        __syncthreads();
        /* V80: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]     +=__half(2.f*DELTA)*sm[1];
            if(y1<h) sm[w]   +=__half(2.f*DELTA)*sm[w+1];
            if(y2<h) sm[2*w] +=__half(2.f*DELTA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]     +=__half(DELTA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(DELTA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(DELTA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=t*2; x<w; x+=nt*2) {
            out[y0*s+x/2]           = sm[x]     * __half(NORM_L);
            if(y1<h) out[y1*s+x/2] = sm[w+x]   * __half(NORM_L);
            if(y2<h) out[y2*s+x/2] = sm[2*w+x] * __half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            out[y0*s+hw+x/2]           = sm[x]     * __half(NORM_H);
            if(y1<h) out[y1*s+hw+x/2] = sm[w+x]   * __half(NORM_H);
            if(y2<h) out[y2*s+hw+x/2] = sm[2*w+x] * __half(NORM_H);
        }
    }
}


__global__ void s17_dwt_h_ho(const float* __restrict__ data, __half* __restrict__ out,
                               int w, int h, int s)
{
    extern __shared__ float sm[];
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = data[y*s+i];
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += ALPHA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*ALPHA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*BETA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += BETA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += GAMMA*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += 2.f*GAMMA*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += 2.f*DELTA*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += DELTA*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2;   x < w; x += nt*2) out[y*s+x/2]    = __float2half(sm[x] * NORM_L);
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = __float2half(sm[x] * NORM_H);
}


/* V17c: fp16-input vertical DWT — reads __half (from half H-DWT output), no float2half step. */
__global__ void s17_dwt_v_fp16_hi(const __half* __restrict__ d_src,
                                    __half* __restrict__ d_work,
                                    float* __restrict__ d_dst,
                                    int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;

    const __half hA = __float2half(ALPHA), hB = __float2half(BETA);
    const __half hG = __float2half(GAMMA), hD = __float2half(DELTA);
    const __half h2 = __float2half(2.0f);

    /* Copy column half → half (no conversion); __ldg uses read-only texture cache */
    for (int y = 0; y < h; ++y)
        d_work[y*s+x] = __ldg(&d_src[y*s+x]);

    /* Alpha */
    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hA, __hadd(d_work[(y-1)*s+x], d_work[(y+1)*s+x])));
    if (h > 1 && !(h&1))
        d_work[(h-1)*s+x] = __hadd(d_work[(h-1)*s+x], __hmul(h2, __hmul(hA, d_work[(h-2)*s+x])));

    /* Beta */
    d_work[x] = __hadd(d_work[x], __hmul(h2, __hmul(hB, d_work[min(1,h-1)*s+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hB, __hadd(d_work[(y-1)*s+x], d_work[yp1*s+x])));
    }

    /* Gamma */
    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hG, __hadd(d_work[(y-1)*s+x], d_work[(y+1)*s+x])));
    if (h > 1 && !(h&1))
        d_work[(h-1)*s+x] = __hadd(d_work[(h-1)*s+x], __hmul(h2, __hmul(hG, d_work[(h-2)*s+x])));

    /* Delta */
    d_work[x] = __hadd(d_work[x], __hmul(h2, __hmul(hD, d_work[min(1,h-1)*s+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hD, __hadd(d_work[(y-1)*s+x], d_work[yp1*s+x])));
    }

    /* Deinterleave + normalization */
    const __half hNL = __float2half(NORM_L), hNH = __float2half(NORM_H);
    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) d_dst[(y/2)*s+x]    = __half2float(__hmul(d_work[y*s+x], hNL));
    for (int y = 1; y < h; y += 2) d_dst[(hh+y/2)*s+x] = __half2float(__hmul(d_work[y*s+x], hNH));
}


/* V17d: register-blocked vertical DWT with __half input.
 * Used in the large-h branch when h ≤ MAX_REG_HEIGHT (transitional subbands).
 * Reads __half d_src (half H-DWT output), loads into float registers — no global
 * workspace accesses at all. Arithmetic in float32 for full precision.
 * Eliminates s17_dwt_v_fp16_hi's d_work accesses for small-but-large-path subbands. */
__global__ void s17_dwt_v_fp16_hi_reg(const __half* __restrict__ d_src,
                                        float* __restrict__ d_dst,
                                        int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;

    float col[MAX_REG_HEIGHT];
    for (int y = 0; y < h; ++y)
        col[y] = __half2float(__ldg(&d_src[y*s+x]));

    for (int y = 1; y < h-1; y += 2) col[y] += ALPHA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*ALPHA*col[h-2];
    col[0] += 2.f*BETA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += BETA*(col[y-1]+col[min(y+1,h-1)]);
    for (int y = 1; y < h-1; y += 2) col[y] += GAMMA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*GAMMA*col[h-2];
    col[0] += 2.f*DELTA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += DELTA*(col[y-1]+col[min(y+1,h-1)]);

    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) d_dst[(y/2)*s+x]    = col[y] * NORM_L;
    for (int y = 1; y < h; y += 2) d_dst[(hh+y/2)*s+x] = col[y] * NORM_H;
}


/* V17b: fp16 workspace vertical DWT — halves V-DWT memory bandwidth for large h.
 * Used when h > MAX_REG_HEIGHT (where register-blocking isn't feasible).
 * d_work is __half; column copy converts float→half, lifting in fp16, output float. */
__global__ void s17_dwt_v_fp16(const float* __restrict__ d_src,
                                 __half* __restrict__ d_work,
                                 float* __restrict__ d_dst,
                                 int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;

    const __half hA = __float2half(ALPHA), hB = __float2half(BETA);
    const __half hG = __float2half(GAMMA), hD = __float2half(DELTA);
    const __half h2 = __float2half(2.0f);

    /* Copy column float → half */
    for (int y = 0; y < h; ++y)
        d_work[y*s+x] = __float2half(__ldg(&d_src[y*s+x]));

    /* Alpha */
    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hA, __hadd(d_work[(y-1)*s+x], d_work[(y+1)*s+x])));
    if (h > 1 && !(h&1))
        d_work[(h-1)*s+x] = __hadd(d_work[(h-1)*s+x], __hmul(h2, __hmul(hA, d_work[(h-2)*s+x])));

    /* Beta */
    d_work[x] = __hadd(d_work[x], __hmul(h2, __hmul(hB, d_work[min(1,h-1)*s+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hB, __hadd(d_work[(y-1)*s+x], d_work[yp1*s+x])));
    }

    /* Gamma */
    for (int y = 1; y < h-1; y += 2)
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hG, __hadd(d_work[(y-1)*s+x], d_work[(y+1)*s+x])));
    if (h > 1 && !(h&1))
        d_work[(h-1)*s+x] = __hadd(d_work[(h-1)*s+x], __hmul(h2, __hmul(hG, d_work[(h-2)*s+x])));

    /* Delta */
    d_work[x] = __hadd(d_work[x], __hmul(h2, __hmul(hD, d_work[min(1,h-1)*s+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y+1 < h) ? y+1 : y-1;
        d_work[y*s+x] = __hadd(d_work[y*s+x], __hmul(hD, __hadd(d_work[(y-1)*s+x], d_work[yp1*s+x])));
    }

    /* Deinterleave + normalization */
    const __half hNL = __float2half(NORM_L), hNH = __float2half(NORM_H);
    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) d_dst[(y/2)*s+x]    = __half2float(__hmul(d_work[y*s+x], hNL));
    for (int y = 1; y < h; y += 2) d_dst[(hh+y/2)*s+x] = __half2float(__hmul(d_work[y*s+x], hNH));
}


/* V17p: Parity-split CDF 9/7 lifting helper — mirrors CUDA V39 cdf97_lift_tiled.
 * Eliminates per-iteration branch by splitting Alpha/Gamma (odd) and Beta/Delta (even)
 * into separate stride-2 loops with compile-time start index via template parameter P0.
 * With #pragma unroll + compile-time P0, NVCC emits straight-line FMA sequences:
 * 4×40 conditional iterations → 4×20 fully-unrolled FMAs. */
template<int P0>
__device__ __forceinline__ void
s17_cdf97_lift_tiled(float col[V_TILE_FL])
{
    #pragma unroll
    for (int i = (P0 ? 2 : 1); i < V_TILE_FL - 1; i += 2) col[i] += ALPHA*(col[i-1]+col[i+1]);
    #pragma unroll
    for (int i = (P0 ? 1 : 2); i < V_TILE_FL - 1; i += 2) col[i] += BETA *(col[i-1]+col[i+1]);
    #pragma unroll
    for (int i = (P0 ? 2 : 1); i < V_TILE_FL - 1; i += 2) col[i] += GAMMA*(col[i-1]+col[i+1]);
    #pragma unroll
    for (int i = (P0 ? 1 : 2); i < V_TILE_FL - 1; i += 2) col[i] += DELTA*(col[i-1]+col[i+1]);
}


/* V32/V43: __half version of s17_cdf97_lift_tiled — P0=0 (V43: V_OVERLAP even → load_start always even).
 * V43: P0 changed from 1 to 0: odd local indices = globally-odd rows.
 * ALPHA/GAMMA start at i=1 (odd, globally-odd); BETA/DELTA start at i=2 (skip i=0 boundary).
 * Skipping i=0 in BETA/DELTA: col[0] errors propagate to col[1,2] only → col[4] first output unaffected.
 * __half HFMA has 2× throughput vs float FMA on sm_61+. */
__device__ __forceinline__ void
s17_cdf97_lift_tiled_h(__half col[V_TILE_FL])
{
    const __half kA = __float2half(ALPHA);
    const __half kB = __float2half(BETA);
    const __half kG = __float2half(GAMMA);
    const __half kD = __float2half(DELTA);
    /* Alpha: globally-odd rows (P0=0 → local odd indices starting at i=1) */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2) col[i] += kA*(col[i-1]+col[i+1]);
    /* Beta: globally-even rows (P0=0 → local even indices starting at i=2; skip i=0 boundary) */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2) col[i] += kB*(col[i-1]+col[i+1]);
    /* Gamma: globally-odd rows */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2) col[i] += kG*(col[i-1]+col[i+1]);
    /* Delta: globally-even rows (skip i=0 boundary) */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2) col[i] += kD*(col[i-1]+col[i+1]);
}


/* V39/V43: 2-column-per-thread __half2 lifting helper — P0=0 (V43).
 * V43: ALPHA/GAMMA start i=1 (odd, globally-odd); BETA/DELTA start i=2 (skip i=0 boundary).
 * col2[i] = {col_x[i], col_{x+1}[i]} — two adjacent columns packed as __half2.
 * HFMA2 processes both columns per instruction → 2× arithmetic throughput on sm_61+. */
__device__ __forceinline__ void
s17_cdf97_lift_tiled_h2(__half2 col2[V_TILE_FL])
{
    const __half2 kA = __half2half2(__float2half(ALPHA));
    const __half2 kB = __half2half2(__float2half(BETA));
    const __half2 kG = __half2half2(__float2half(GAMMA));
    const __half2 kD = __half2half2(__float2half(DELTA));
    /* Alpha: globally-odd rows (P0=0 → local odd starting at i=1) */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2)
        col2[i] = __hfma2(kA, __hadd2(col2[i-1], col2[i+1]), col2[i]);
    /* Beta: globally-even rows (P0=0 → local even starting at i=2; skip i=0 boundary) */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2)
        col2[i] = __hfma2(kB, __hadd2(col2[i-1], col2[i+1]), col2[i]);
    /* Gamma: globally-odd rows */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2)
        col2[i] = __hfma2(kG, __hadd2(col2[i-1], col2[i+1]), col2[i]);
    /* Delta: globally-even rows (skip i=0 boundary) */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2)
        col2[i] = __hfma2(kD, __hadd2(col2[i-1], col2[i+1]), col2[i]);
}


/* V17f: Register-based tiled vertical DWT — mirrors CUDA V29.
 * V34: V_TILE=28-row tiles with V_OVERLAP=5 halo; loads V_TILE_FL=38
 * half rows into __half registers; CDF 9/7 lifting in registers; no global workspace.
 * Whole-point symmetric boundary extension: y<0 → -y, y≥h → 2(h-1)-y. */
__global__ void s17_dwt_v_tiled(const __half* __restrict__ d_src,
                                  float* __restrict__ d_dst,
                                  int w, int h, int s)
{
    int x          = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= w || tile_start >= h) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, h);

    float col[V_TILE_FL];
    for (int i = 0; i < V_TILE_FL; i++) {
        int gy = load_start + i;
        if (gy < 0)      gy = -gy;
        else if (gy >= h) gy = 2*(h-1) - gy;
        col[i] = __half2float(__ldg(&d_src[gy * s + x]));
    }

    /* V43: p0 always 0 (V_TILE even, V_OVERLAP even → load_start always even). */
    static_assert(V_TILE % 2 == 0 && V_OVERLAP % 2 == 0,
                  "V43: requires V_TILE even + V_OVERLAP even for constant p0=0");
    s17_cdf97_lift_tiled<0>(col);

    /* V43: output parity p0=0: odd i → H subband, even i → L subband. */
    int hh = (h+1)/2;
    for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
        int gy = load_start + i;
        if (i & 1) d_dst[(hh + gy/2)*s+x] = col[i] * NORM_H;
        else       d_dst[(gy/2)*s+x]        = col[i] * NORM_L;
    }
}


/* Register-blocked vertical DWT for small subbands (h ≤ MAX_REG_HEIGHT).
 * V17: normalization applied in the write phase. */
__global__ void s17_dwt_v_reg(const float* __restrict__ data, float* __restrict__ out,
                                int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;
    float col[MAX_REG_HEIGHT];
    for (int y = 0; y < h; ++y) col[y] = data[y*s+x];

    for (int y = 1; y < h-1; y += 2) col[y] += ALPHA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*ALPHA*col[h-2];
    col[0] += 2.f*BETA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += BETA*(col[y-1]+col[min(y+1,h-1)]);
    for (int y = 1; y < h-1; y += 2) col[y] += GAMMA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*GAMMA*col[h-2];
    col[0] += 2.f*DELTA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += DELTA*(col[y-1]+col[min(y+1,h-1)]);

    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) out[(y/2)*s+x]    = col[y] * NORM_L;
    for (int y = 1; y < h; y += 2) out[(hh+y/2)*s+x] = col[y] * NORM_H;
}


/* V17k: Quantize + pack — float4 vectorized (4 floats per thread, 16-byte reads + 4-byte writes).
 * n4 = floor(per_comp / 4); last 0-3 elements skipped — negligible quality impact.
 * src and out must be 16/4-byte aligned (guaranteed by cudaMalloc). */
__global__ void s17_qe(const float* __restrict__ src, uint8_t* __restrict__ out,
                        int n4, float step)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n4) return;
    float4 v = __ldg(reinterpret_cast<const float4*>(src) + i);
    float inv = __frcp_rn(step);
    auto pack_byte = [inv](float fv) -> uint8_t {
        int q = __float2int_rn(fv * inv);
        return (q < 0 ? uint8_t(0x80) : uint8_t(0)) | uint8_t(min(126, abs(q)));
    };
    uint32_t word = uint32_t(pack_byte(v.x))
                  | (uint32_t(pack_byte(v.y)) << 8)
                  | (uint32_t(pack_byte(v.z)) << 16)
                  | (uint32_t(pack_byte(v.w)) << 24);
    reinterpret_cast<uint32_t*>(out)[i] = word;
}


/* V17l: V-DWT and H-DWT variants writing __half output (for d_c[c] now __half). */

/* float-input, half-output register-blocked V-DWT (h ≤ MAX_REG_HEIGHT, level 0 small-h path). */
__global__ void s17_dwt_v_reg_ho(const float* __restrict__ data, __half* __restrict__ out,
                                   int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;
    float col[MAX_REG_HEIGHT];
    for (int y = 0; y < h; ++y) col[y] = data[y*s+x];

    for (int y = 1; y < h-1; y += 2) col[y] += ALPHA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*ALPHA*col[h-2];
    col[0] += 2.f*BETA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += BETA*(col[y-1]+col[min(y+1,h-1)]);
    for (int y = 1; y < h-1; y += 2) col[y] += GAMMA*(col[y-1]+col[y+1]);
    if (h>1&&!(h&1)) col[h-1] += 2.f*GAMMA*col[h-2];
    col[0] += 2.f*DELTA*col[min(1,h-1)];
    for (int y = 2; y < h; y += 2) col[y] += DELTA*(col[y-1]+col[min(y+1,h-1)]);

    int hh = (h+1)/2;
    for (int y = 0; y < h; y += 2) out[(y/2)*s+x]    = __float2half(col[y] * NORM_L);
    for (int y = 1; y < h; y += 2) out[(hh+y/2)*s+x] = __float2half(col[y] * NORM_H);
}

/* V60: half-input, half-output register-blocked V-DWT (h ≤ MAX_REG_HEIGHT).
 * Used for transitional-h subbands and skip_l0_hdwt path.
 * V60: hoist __half constants + #pragma unroll 4 on load/Alpha/Gamma/write loops (parity CUDA V97). */
__global__ void s17_dwt_v_fp16_hi_reg_ho(const __half* __restrict__ d_src,
                                           __half* __restrict__ d_dst,
                                           int w, int h, int s)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x; if (x >= w) return;

    /* V60: hoist all __half constants — one __float2half each, not per-loop-iteration. */
    const __half kA  = __float2half(ALPHA),       kA2 = __float2half(2.f*ALPHA);
    const __half kB  = __float2half(BETA),        kB2 = __float2half(2.f*BETA);
    const __half kG  = __float2half(GAMMA),       kG2 = __float2half(2.f*GAMMA);
    const __half kD  = __float2half(DELTA),       kD2 = __float2half(2.f*DELTA);
    const __half kNL = __float2half(NORM_L),      kNH = __float2half(NORM_H);

    /* V32: __half col[] — direct __ldg→__half; HFMA 2× throughput; no conversions. */
    __half col[MAX_REG_HEIGHT];
    /* V60: #pragma unroll 4 — 4 concurrent __ldg in flight; hides column load latency. */
    #pragma unroll 4
    for (int y = 0; y < h; ++y)
        col[y] = __ldg(&d_src[y*s+x]);

    /* Alpha: odd rows — V60: #pragma unroll 4 — step=2 → fully independent iterations; 4× ILP. */
    #pragma unroll 4
    for (int y = 1; y < h-1; y += 2) col[y] += kA*(col[y-1]+col[y+1]);
    /* V75: DCI heights always > 1 → h>1 guard removed; check only even parity. Parity CUDA V112. */
    if (!(h&1)) col[h-1] += kA2*col[h-2];
    /* V64: Beta — hoist boundary case to enable #pragma unroll 4 on main loop.
     * Main: y<h-1 ensures y+1<h → col[y+1] directly (no min() clamp needed).
     * Boundary (odd h only): col[h-1]+=kB*(col[h-2]+col[h-1]) matches min(y+1,h-1)=h-1 semantics. */
    /* V75: DCI heights always > 1 → min(1,h-1)=1 always; drop runtime min. */
    col[0] += kB2*col[1];
    #pragma unroll 4
    for (int y = 2; y < h - 1; y += 2) col[y] += kB*(col[y-1]+col[y+1]);
    /* V77: DCI heights always > 2 (34/68/135) → drop h>2 guard; parity CUDA V114. */
    if (h & 1) col[h-1] += kB*(col[h-2]+col[h-1]);
    /* Gamma: odd rows — same structure as Alpha; V60: #pragma unroll 4 → 4× ILP. */
    #pragma unroll 4
    for (int y = 1; y < h-1; y += 2) col[y] += kG*(col[y-1]+col[y+1]);
    /* V75: DCI heights always > 1 → h>1 guard removed. */
    if (!(h&1)) col[h-1] += kG2*col[h-2];
    /* V64: Delta — same boundary-hoist transformation as Beta above. */
    /* V75: DCI heights always > 1 → min(1,h-1)=1 always; drop runtime min. */
    col[0] += kD2*col[1];
    #pragma unroll 4
    for (int y = 2; y < h - 1; y += 2) col[y] += kD*(col[y-1]+col[y+1]);
    /* V77: DCI heights always > 2 → drop h>2 guard (same as Beta above). */
    if (h & 1) col[h-1] += kD*(col[h-2]+col[h-1]);

    /* V60: #pragma unroll 4 on both scatter write loops — 4 concurrent global stores per batch. */
    int hh = (h+1)/2;
    #pragma unroll 4
    for (int y = 0; y < h; y += 2) d_dst[(y/2)*s+x]    = col[y] * kNL;
    #pragma unroll 4
    for (int y = 1; y < h; y += 2) d_dst[(hh+y/2)*s+x] = col[y] * kNH;
}

/* half-input, half-output tiled V-DWT (h > MAX_REG_HEIGHT).
 * Same tiling logic as s17_dwt_v_tiled but writes half output to d_c (now __half).
 * V34: V_TILE=28, V_TILE_FL=38.
 *
 * V27: Interior-tile branch-free unrolled load+store (parity with CUDA V64).
 *   interior = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= h).
 *   Interior path (97% of tiles for 2K lvl-0): #pragma unroll → 38 independent __ldg + 28 stores.
 *   Boundary path (first + last tile): existing WS bounds-check code unchanged.
 */
__global__ void s17_dwt_v_tiled_ho(const __half* __restrict__ d_src,
                                     __half* __restrict__ d_dst,
                                     int w, int h, int s)
{
    int x          = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= w || tile_start >= h) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, h);

    /* V27: interior check — uniform across block, no warp divergence. */
    bool interior = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= h);

    /* V32: __half col[] — direct __ldg→__half; s17_cdf97_lift_tiled_h for HFMA; direct stores. */
    __half col[V_TILE_FL];
    if (interior) {
        /* V27+V32: no bounds check → #pragma unroll issues 26 direct __half __ldg. */
        #pragma unroll
        for (int i = 0; i < V_TILE_FL; i++)
            col[i] = __ldg(&d_src[(load_start + i) * s + x]);
    } else {
        for (int i = 0; i < V_TILE_FL; i++) {
            int gy = load_start + i;
            if (gy < 0)       gy = -gy;
            else if (gy >= h) gy = 2*(h-1) - gy;
            col[i] = __ldg(&d_src[gy * s + x]);
        }
    }

    /* V43: p0=0 (V_TILE even, V_OVERLAP even → load_start always even). */
    s17_cdf97_lift_tiled_h(col);

    /* V43: output parity p0=0: odd i → H subband, even i → L subband. */
    int hh = (h+1)/2;
    if (interior) {
        /* V27+V32+V43: exactly V_TILE outputs, fully unrolled, direct __half stores. */
        #pragma unroll
        for (int i = V_OVERLAP; i < V_OVERLAP + V_TILE; i++) {
            int gy = load_start + i;
            if (i & 1) d_dst[(hh + gy/2)*s+x] = col[i] * __half(NORM_H);  /* odd i → H */
            else       d_dst[(gy/2)*s+x]        = col[i] * __half(NORM_L);  /* even i → L */
        }
    } else {
        for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
            int gy = load_start + i;
            if (i & 1) d_dst[(hh + gy/2)*s+x] = col[i] * __half(NORM_H);
            else       d_dst[(gy/2)*s+x]        = col[i] * __half(NORM_L);
        }
    }
}

/* V39/V43: 2-column-per-thread tiled V-DWT using __half2 (HFMA2 = 2× lifting throughput).
 * Grid x = ceil(w/2/vt_blk); each thread processes col x and x+1 using __half2.
 * x always even (blockIdx.x*vt_blk+t)*2; w even for 2K/4K → all pairs valid. */
/* V50: __launch_bounds__(256,6) — 256T = vt_blk; 6 blk/SM target; ensures ≤42 regs/T. */
__global__ __launch_bounds__(256, 6)
void s17_dwt_v_tiled_ho_2col(const __half* __restrict__ d_src,
                               __half* __restrict__ d_dst,
                               int w, int h, int s)
{
    int x          = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= w || tile_start >= h) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, h);
    bool interior  = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= h);

    __half2 col2[V_TILE_FL];
    if (interior) {
        #pragma unroll
        for (int i = 0; i < V_TILE_FL; i++) {
            int raw = __ldg(reinterpret_cast<const int*>(&d_src[(load_start + i) * s + x]));
            col2[i] = *reinterpret_cast<const __half2*>(&raw);
        }
    } else {
        for (int i = 0; i < V_TILE_FL; i++) {
            int gy = load_start + i;
            if (gy < 0)      gy = -gy;
            else if (gy >= h) gy = 2*(h-1) - gy;
            int raw = __ldg(reinterpret_cast<const int*>(&d_src[gy * s + x]));
            col2[i] = *reinterpret_cast<const __half2*>(&raw);
        }
    }

    s17_cdf97_lift_tiled_h2(col2);

    int hh = (h+1)/2;
    const __half2 nL = __half2half2(__float2half(NORM_L));
    const __half2 nH = __half2half2(__float2half(NORM_H));
    if (interior) {
        #pragma unroll
        for (int i = V_OVERLAP; i < V_OVERLAP + V_TILE; i++) {
            int gy = load_start + i;
            /* V43: P0=0 — odd i → H subband, even i → L subband */
            __half2 r = __hmul2(col2[i], (i & 1) ? nH : nL);
            int* p = reinterpret_cast<int*>(&d_dst[
                ((i & 1) ? (hh + gy/2) : (gy/2)) * s + x]);
            *p = *reinterpret_cast<const int*>(&r);
        }
    } else {
        for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
            int gy = load_start + i;
            /* V43: P0=0 — odd i → H subband, even i → L subband */
            __half2 r = __hmul2(col2[i], (i & 1) ? nH : nL);
            int* p = reinterpret_cast<int*>(&d_dst[
                ((i & 1) ? (hh + gy/2) : (gy/2)) * s + x]);
            *p = *reinterpret_cast<const int*>(&r);
        }
    }
}


/* V17l/V17o: H-DWT for levels 1+, half-io. V17o: half shared memory (2× smem bw, 2× fp16 FMA). */
__global__ void s17_dwt_h_half_io(const __half* __restrict__ data, __half* __restrict__ out,
                                    int w, int h, int s)
{
    extern __shared__ __half sm[];  /* V17o: was float; direct half load + fp16 lifting */
    int y = blockIdx.x; if (y >= h) return;
    int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < w; i += nt) sm[i] = data[y*s+i];  /* V17o: direct copy, no float conv */
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(ALPHA)*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += __half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*BETA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(BETA)*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(GAMMA)*(sm[x-1]+sm[x+1]);
    if (t==0&&w>1&&!(w&1)) sm[w-1] += __half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*DELTA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(DELTA)*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    int hw = (w+1)/2;
    for (int x = t*2;   x < w; x += nt*2) out[y*s+x/2]    = sm[x] * __half(NORM_L);
    for (int x = t*2+1; x < w; x += nt*2) out[y*s+hw+x/2] = sm[x] * __half(NORM_H);
}

/* V17w: 2-rows-per-block variant of s17_dwt_h_half_io (DWT levels 1-4).
 * grid=(h+1)/2; smem=2*w*sizeof(__half). Halves grid; same 4 syncthreads per pair. */
__global__ void s17_dwt_h_half_io_2row(const __half* __restrict__ data, __half* __restrict__ out,
                                        int w, int h, int s)
{
    extern __shared__ __half sm[];
    int y0 = blockIdx.x * 2, y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int hw = (w+1)/2;
    for (int i = t; i < w; i += nt) sm[i]   = __ldg(&data[y0*s+i]);
    if (y1 < h) for (int i = t; i < w; i += nt) sm[w+i] = __ldg(&data[y1*s+i]);
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) {
        sm[x]  +=__half(ALPHA)*(sm[x-1]+sm[x+1]);
        if(y1<h) sm[w+x]+=__half(ALPHA)*(sm[w+x-1]+sm[w+x+1]);
    }
    /* V80: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
    if(t==0) { sm[w-1]+=__half(2.f*ALPHA)*sm[w-2]; if(y1<h) sm[2*w-1]+=__half(2.f*ALPHA)*sm[2*w-2]; }
    __syncthreads();
    /* V80: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
    if(t==0) { sm[0]+=__half(2.f*BETA)*sm[1]; if(y1<h) sm[w]+=__half(2.f*BETA)*sm[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        sm[x]  +=__half(BETA)*(sm[x-1]+sm[x+1]);
        if(y1<h) sm[w+x]+=__half(BETA)*(sm[w+x-1]+sm[w+x+1]);
    }
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) {
        sm[x]  +=__half(GAMMA)*(sm[x-1]+sm[x+1]);
        if(y1<h) sm[w+x]+=__half(GAMMA)*(sm[w+x-1]+sm[w+x+1]);
    }
    /* V80: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
    if(t==0) { sm[w-1]+=__half(2.f*GAMMA)*sm[w-2]; if(y1<h) sm[2*w-1]+=__half(2.f*GAMMA)*sm[2*w-2]; }
    __syncthreads();
    /* V80: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
    if(t==0) { sm[0]+=__half(2.f*DELTA)*sm[1]; if(y1<h) sm[w]+=__half(2.f*DELTA)*sm[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        sm[x]  +=__half(DELTA)*(sm[x-1]+sm[x+1]);
        if(y1<h) sm[w+x]+=__half(DELTA)*(sm[w+x-1]+sm[w+x+1]);
    }
    __syncthreads();
    for (int x=t*2; x<w; x+=nt*2) {
        out[y0*s+x/2]       = sm[x]   * __half(NORM_L);
        if(y1<h) out[y1*s+x/2]   = sm[w+x] * __half(NORM_L);
    }
    for (int x=t*2+1; x<w; x+=nt*2) {
        out[y0*s+hw+x/2]    = sm[x]   * __half(NORM_H);
        if(y1<h) out[y1*s+hw+x/2] = sm[w+x] * __half(NORM_H);
    }
}

/* V17x: 4-rows-per-block H-DWT for levels 1+. Halves grid vs V17w.
 * smem[0..w-1]=y0, [w..2w-1]=y1, [2w..3w-1]=y2, [3w..4w-1]=y3.
 * grid=(h+3)/4; smem=4*w*sizeof(__half); parity with CUDA V47.
 * V71: __launch_bounds__(512,4) — guarantees 4 blk/SM; without LB compiler may use 36+ regs
 *      → only 3 blk/SM. Forces ≤32 regs/T; smem per level 1 (7.68KB) allows 6 blk → reg-limited. */
__global__ __launch_bounds__(512, 4)
void s17_dwt_h_half_io_4row(const __half* __restrict__ data, __half* __restrict__ out,
                                        int w, int h, int s)
{
    extern __shared__ __half sm[];
    int y0=blockIdx.x*4, y1=y0+1, y2=y0+2, y3=y0+3;
    int t=threadIdx.x, nt=blockDim.x, hw=(w+1)/2;
    /* V29: hoist row-presence check to block level. y3/h uniform → no warp divergence.
     * Interior (all 4 rows present): branch-free lifting+stores for 4-row ILP.
     * Else: original per-iteration yN<h guards for partial last block. */
    if (y3 < h) {
        /* V45: __half2 row-pair packing — 2× FMA throughput.
         * sm01[x]={row0[x],row1[x]}; sm23[x]={row2[x],row3[x]}.
         * Total smem unchanged: 4w __half = 2w __half2 per region × 2 regions. */
        __half2* sm01 = reinterpret_cast<__half2*>(sm);
        __half2* sm23 = reinterpret_cast<__half2*>(sm + 2*w);
        /* V85: __half2 __ldg loads — 4 × 32-bit loads replace 8 × 16-bit loads per 2-col pair.
         * DCI widths always even (960/480/240/120) → x=t*2 always even → 4-byte aligned __half2.
         * Each __half2 load covers {row[x], row[x+1]}; unpack into interleaved sm01/sm23 layout.
         * 2× fewer load instructions per thread; coalesced access preserved (parity CUDA V123).
         * V57: #pragma unroll 2 retained — 4K: 2 iters; 2K: 1 iter → DRAM latency hiding. */
        #pragma unroll 2
        for (int i=t*2; i<w; i+=nt*2) {
            __half2 r0 = __ldg(reinterpret_cast<const __half2*>(&data[y0*s+i]));
            __half2 r1 = __ldg(reinterpret_cast<const __half2*>(&data[y1*s+i]));
            __half2 r2 = __ldg(reinterpret_cast<const __half2*>(&data[y2*s+i]));
            __half2 r3 = __ldg(reinterpret_cast<const __half2*>(&data[y3*s+i]));
            sm01[i]   = __halves2half2(__low2half(r0), __low2half(r1));   /* {y0[i],   y1[i]}   */
            sm01[i+1] = __halves2half2(__high2half(r0), __high2half(r1)); /* {y0[i+1], y1[i+1]} */
            sm23[i]   = __halves2half2(__low2half(r2), __low2half(r3));
            sm23[i+1] = __halves2half2(__high2half(r2), __high2half(r3));
        }
        __syncthreads();
        /* V59: Alpha — #pragma unroll 2: 4K L1 has 2 iters → interleaves 2×4 HFMA2/HADD2
         * to hide 23-cycle smem latency. 2K L1 has ~1 iter → pragma ignored, no code bloat. */
        {
            const __half2 kA = __half2half2(__float2half(ALPHA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kA, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kA, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V73: DCI w always even and >1 — simplify Alpha boundary: drop w>1&&!(w&1) guard.
             * All DCI levels: 1920/960/480/240/120/60 all even. Saves 2 ISETP+PREDAND per block. */
            if(t==0) {
                const __half2 kA2 = __half2half2(__float2half(2.f*ALPHA));
                sm01[w-1] = __hfma2(kA2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kA2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* V59: Beta — #pragma unroll 2 for 4K MLP. */
        {
            const __half2 kB = __half2half2(__float2half(BETA));
            if(t==0) {
                const __half2 kB2 = __half2half2(__float2half(2.f*BETA));
                /* V73: DCI w always >1 → min(1,w-1)=1 always. Drop runtime min. */
                sm01[0] = __hfma2(kB2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kB2, sm23[1], sm23[0]);
            }
            /* V65: x always even → even w (all DCI): x≤w-2 → x+1≤w-1 → drop MIN/SEL. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kB, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kB, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* V59: Gamma — #pragma unroll 2 for 4K MLP. */
        {
            const __half2 kG = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kG, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kG, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V73: same DCI even-width invariant — simplify Gamma boundary. */
            if(t==0) {
                const __half2 kG2 = __half2half2(__float2half(2.f*GAMMA));
                sm01[w-1] = __hfma2(kG2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kG2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* V59: Delta — #pragma unroll 2 for 4K MLP. */
        {
            const __half2 kD = __half2half2(__float2half(DELTA));
            if(t==0) {
                const __half2 kD2 = __half2half2(__float2half(2.f*DELTA));
                /* V73: DCI w always >1 → min(1,w-1)=1 always. Drop runtime min. */
                sm01[0] = __hfma2(kD2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kD2, sm23[1], sm23[0]);
            }
            /* V65: same even-w invariant as Beta — drop MIN instruction. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kD, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kD, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* V57: Deinterleave and write — #pragma unroll 2 pipelines 2×8 smem+global per thread.
         * 4K L1 (960/512=2 iters): fully unrolled; 2K L1 (480/512≤1 iter): no change. */
        {
            const __half2 nL = __half2half2(__float2half(NORM_L));
            const __half2 nH = __half2half2(__float2half(NORM_H));
            #pragma unroll 2
            for (int p=t; p<w/2; p+=nt) {
                __half2 v01L = __hmul2(sm01[p*2],   nL);
                __half2 v23L = __hmul2(sm23[p*2],   nL);
                out[y0*s+p]    = __low2half(v01L);
                out[y1*s+p]    = __high2half(v01L);
                out[y2*s+p]    = __low2half(v23L);
                out[y3*s+p]    = __high2half(v23L);
                __half2 v01H = __hmul2(sm01[p*2+1], nH);
                __half2 v23H = __hmul2(sm23[p*2+1], nH);
                out[y0*s+hw+p] = __low2half(v01H);
                out[y1*s+hw+p] = __high2half(v01H);
                out[y2*s+hw+p] = __low2half(v23H);
                out[y3*s+hw+p] = __high2half(v23H);
            }
        }
    } else {
        /* Partial last block: original per-iteration yN<h guards. */
        for (int i=t; i<w; i+=nt)           sm[i]     = __ldg(&data[y0*s+i]);
        if(y1<h) for(int i=t; i<w; i+=nt)   sm[w+i]   = __ldg(&data[y1*s+i]);
        if(y2<h) for(int i=t; i<w; i+=nt)   sm[2*w+i] = __ldg(&data[y2*s+i]);
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]     +=__half(ALPHA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(ALPHA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(ALPHA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]   +=__half(2.f*ALPHA)*sm[w-2];
            if(y1<h) sm[2*w-1] +=__half(2.f*ALPHA)*sm[2*w-2];
            if(y2<h) sm[3*w-1] +=__half(2.f*ALPHA)*sm[3*w-2];
        }
        __syncthreads();
        /* V81: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]     +=__half(2.f*BETA)*sm[1];
            if(y1<h) sm[w]   +=__half(2.f*BETA)*sm[w+1];
            if(y2<h) sm[2*w] +=__half(2.f*BETA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]     +=__half(BETA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(BETA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(BETA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]     +=__half(GAMMA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(GAMMA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(GAMMA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]   +=__half(2.f*GAMMA)*sm[w-2];
            if(y1<h) sm[2*w-1] +=__half(2.f*GAMMA)*sm[2*w-2];
            if(y2<h) sm[3*w-1] +=__half(2.f*GAMMA)*sm[3*w-2];
        }
        __syncthreads();
        /* V81: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]     +=__half(2.f*DELTA)*sm[1];
            if(y1<h) sm[w]   +=__half(2.f*DELTA)*sm[w+1];
            if(y2<h) sm[2*w] +=__half(2.f*DELTA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]     +=__half(DELTA)*(sm[x-1]     +sm[x+1]);
            if(y1<h) sm[w+x]   +=__half(DELTA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<h) sm[2*w+x] +=__half(DELTA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=t*2; x<w; x+=nt*2) {
            out[y0*s+x/2]             = sm[x]     * __half(NORM_L);
            if(y1<h) out[y1*s+x/2]   = sm[w+x]   * __half(NORM_L);
            if(y2<h) out[y2*s+x/2]   = sm[2*w+x] * __half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            out[y0*s+hw+x/2]             = sm[x]     * __half(NORM_H);
            if(y1<h) out[y1*s+hw+x/2]   = sm[w+x]   * __half(NORM_H);
            if(y2<h) out[y2*s+hw+x/2]   = sm[2*w+x] * __half(NORM_H);
        }
    } /* end V29 if/else */
}

/* V17l: Quantize + pack from __half input using __half2 loads (4B each, 2 per thread).
 * n4 = floor(per_comp / 4); loads 4 halves = two __half2 (8 bytes) per thread. */
__global__ void s17_qe_h(const __half* __restrict__ src, uint8_t* __restrict__ out,
                           int n4, float step)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x; if (i >= n4) return;
    __half2 lo = __ldg(reinterpret_cast<const __half2*>(src) + 2*i);
    __half2 hi = __ldg(reinterpret_cast<const __half2*>(src) + 2*i + 1);
    float inv = __frcp_rn(step);
    auto pack_byte = [inv](float fv) -> uint8_t {
        int q = __float2int_rn(fv * inv);
        return (q < 0 ? uint8_t(0x80) : uint8_t(0)) | uint8_t(min(126, abs(q)));
    };
    uint32_t word = uint32_t(pack_byte(__half2float(lo.x)))
                  | (uint32_t(pack_byte(__half2float(lo.y))) <<  8)
                  | (uint32_t(pack_byte(__half2float(hi.x))) << 16)
                  | (uint32_t(pack_byte(__half2float(hi.y))) << 24);
    reinterpret_cast<uint32_t*>(out)[i] = word;
}


/* V18a (parity with CUDA V50): Subband-aware quantize — per-row step weighting.
 * DC (LL5) rows [0..ll5_height): factor=0.80 (finer — highest perceptual importance).
 * Level-4 rows [ll5_h..2*ll5_h): factor=0.95 (mid-frequency slight boost).
 * Higher rows: factor=1.10 (coarser — less perceptually important).
 * Launch: s17_qe_subband_h<<<n_rows, 256, 0, s>>>(d_c, d_enc, stride, n_rows, step, ll5_h)
 * where n_rows = ceil(per_comp/stride), stride=original DWT width, ll5_h=ceil(height/32). */
__global__ void s17_qe_subband_h(
    const __half* __restrict__ src,
    uint8_t* __restrict__ out,
    int stride,       /* original DWT row pitch */
    int n_rows,       /* rows to quantize */
    float base_step,
    int ll5_height)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float factor;
    if      (row < ll5_height)     factor = 0.80f;
    else if (row < 2 * ll5_height) factor = 0.95f;
    else                           factor = 1.10f;
    float inv = __frcp_rn(base_step * factor);
    for (int col = threadIdx.x; col < stride; col += blockDim.x) {
        float v = __half2float(__ldg(&src[row * stride + col]));
        int q = __float2int_rn(v * inv);
        out[row * stride + col] = (q < 0 ? uint8_t(0x80) : uint8_t(0)) | uint8_t(min(126, abs(q)));
    }
}


/* V18c (parity with CUDA V52): 2D subband-aware quantize — correct LL5 column boundary.
 * V18a used only row < ll5_height for DC detection, incorrectly applying finer step to LH1
 * (cols stride/2..stride-1 in rows 0..ll5_h-1 = finest horizontal detail, not DC).
 * Fix: LL5 = row < ll5_h AND col < ll5_cols (ll5_cols = stride >> 5 = stride/32).
 * Step weights: LL5×0.70 (DC finest), LH5/HL5/HH5×0.90, all others×1.15.
 * Launch: s17_qe_subband_2d<<<n_rows, 256, 0, s>>>(d_c, d_enc, stride, n_rows, step, ll5_h, ll5_cols) */
__global__ void s17_qe_subband_2d(
    const __half* __restrict__ src,
    uint8_t* __restrict__ out,
    int stride, int n_rows, float base_step,
    int ll5_height, int ll5_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float inv_dc = __frcp_rn(base_step * 0.70f);
    float inv_l5 = __frcp_rn(base_step * 0.90f);
    float inv_hf = __frcp_rn(base_step * 1.15f);
    bool is_l5_row = (row < ll5_height * 2);
    bool is_dc_row = (row < ll5_height);
    for (int col = threadIdx.x; col < stride; col += blockDim.x) {
        float inv;
        if (is_dc_row && col < ll5_cols)
            inv = inv_dc;
        else if (is_l5_row && col < ll5_cols * 2)
            inv = inv_l5;
        else
            inv = inv_hf;
        float v = __half2float(__ldg(&src[row * stride + col]));
        int q = __float2int_rn(v * inv);
        out[row * stride + col] = (q < 0 ? uint8_t(0x80) : uint8_t(0)) | uint8_t(min(126, abs(q)));
    }
}


/* V56 (parity with CUDA V93): 8-element vectorized quantize via vq8.
 *
 * vq8: 4×__half2 loads (16B) + 2×uint32_t stores (8B) per thread per loop iteration.
 * vs vq4: 2×__half2 loads (8B) + 1×uint32_t store (4B) — halves loop count again.
 * Same total memory traffic as vq4; 50% fewer loop-counter updates and branch checks.
 * vq4 tail handles residual < 8 elements (only 2K DC zones: ll5_c=60, 60%8=4 tail).
 * 4K (ll5_c=120): all zones div-by-8 → zero tail → pure vq8 everywhere in 4K mode.
 *
 * Launch: s17_qe_subband_ml<<<n_rows, 256, 0, s>>>(src, out, stride, n_rows, step, ll5_h, ll5_c) */
/* V50: __launch_bounds__(256,6) — forces ≤42 regs/T → guaranteed 6 blk/SM (was potentially 5). */
__global__ __launch_bounds__(256, 6)
void s17_qe_subband_ml(
    const __half* __restrict__ src, uint8_t* __restrict__ out,
    int stride, int n_rows, float base_step, int ll5_h, int ll5_c)
{
    int row = blockIdx.x; if (row >= n_rows) return;

    /* V66: __frcp_rn computed first; V78: inv_* deferred until after L1 early exit — parity CUDA V115. */
    float base_inv = __frcp_rn(base_step);

    const __half* row_src = src + static_cast<size_t>(row) * stride;
    uint8_t*      row_dst = out + static_cast<size_t>(row) * stride;

    /* V56: All rows vectorized via vq8 — 8 elements per thread per iteration.
     * vq8: 4×__half2 loads (16B) + 2×uint32_t stores (8B) vs vq4's 2×__half2+uint32_t.
     * Halves loop iterations vs vq4: 50% less loop overhead; same total memory traffic.
     * Zone widths ≥ 2*ll5_c are div-by-8 → zero tail for L1..L5 rows (all non-DC rows).
     * DC rows only: two zones of width ll5_c (2K: 60, 4K: 120).
     *   4K: ll5_c=120, 120%8=0 → zero tail. 2K: ll5_c=60, 60%8=4 → vq4 tail (4 elements).
     * L1: 1 zone. L2: 2 zones. L3: 3 zones. L4: 4 zones. L5: 5 zones. DC: 6 zones.
     */
    auto vq8 = [row_src, row_dst](int col_start, int col_end, float inv, int nt) {
        /* V61: 2×int2 __ldg (ld.global.b64) instead of 4×__half2 __ldg (ld.global.b32).
         * int2 at row_src+c (byte offset c*2): c always multiple of 8 → 16-byte aligned → valid.
         * V63: uint2 store for aligned zones: row_dst+c is 8-byte aligned when col_start%8==0.
         * Branch is loop-invariant (col_start fixed per call) → compiler hoists outside loop.
         * #pragma unroll 2: 4K L1 rows (2 iters/thread) fully inlined; 2K (1 iter) no bloat. */
        const bool vec_store = (col_start & 7) == 0;
        #pragma unroll 2
        for (int c = col_start + threadIdx.x * 8; c + 7 < col_end; c += nt * 8) {
            const int2* p2 = reinterpret_cast<const int2*>(
                reinterpret_cast<const __half2*>(row_src) + c / 2);
            int2 r01 = __ldg(p2);     /* 8 bytes = hv0 + hv1 */
            int2 r23 = __ldg(p2 + 1); /* 8 bytes = hv2 + hv3 */
            /* V66: Sign extraction via __byte_perm on raw int2 loads — parity CUDA V103.
             * r01.x = hv0 as uint32: hv0.x in bits[15:0], hv0.y in bits[31:16].
             * byte1 of r01.x (bits[15:8]) has sign of hv0.x at bit 7; byte3 has sign of hv0.y.
             * __byte_perm(r01.x, r01.y, 0x7531): result = {byte3(r01.y), byte1(r01.y), byte3(r01.x), byte1(r01.x)}
             *   → sign bytes for hv1.y, hv1.x, hv0.y, hv0.x — exactly bytes 3,2,1,0 of lo.
             * & 0x80808080: isolate sign bits. 2 ops (PRMT+AND) vs 8 (4×SETP+4×SELP).
             * Abs: AND 0x7FFF7FFF clears sign bit of each half in uint32 — 4 AND vs 8 IABS.
             * Total: saves 16 instructions per 8 elements (~23% quantize compute reduction). */
            const uint32_t lo_signs = __byte_perm(uint32_t(r01.x), uint32_t(r01.y), 0x7531u) & 0x80808080u;
            const uint32_t hi_signs = __byte_perm(uint32_t(r23.x), uint32_t(r23.y), 0x7531u) & 0x80808080u;
            __half2 hv0, hv1, hv2, hv3;
            const uint32_t a01x = uint32_t(r01.x) & 0x7FFF7FFFu;
            const uint32_t a01y = uint32_t(r01.y) & 0x7FFF7FFFu;
            const uint32_t a23x = uint32_t(r23.x) & 0x7FFF7FFFu;
            const uint32_t a23y = uint32_t(r23.y) & 0x7FFF7FFFu;
            __builtin_memcpy(&hv0, &a01x, 4);
            __builtin_memcpy(&hv1, &a01y, 4);
            __builtin_memcpy(&hv2, &a23x, 4);
            __builtin_memcpy(&hv3, &a23y, 4);
            int aq0 = __float2int_rn(__half2float(hv0.x) * inv);
            int aq1 = __float2int_rn(__half2float(hv0.y) * inv);
            int aq2 = __float2int_rn(__half2float(hv1.x) * inv);
            int aq3 = __float2int_rn(__half2float(hv1.y) * inv);
            int aq4 = __float2int_rn(__half2float(hv2.x) * inv);
            int aq5 = __float2int_rn(__half2float(hv2.y) * inv);
            int aq6 = __float2int_rn(__half2float(hv3.x) * inv);
            int aq7 = __float2int_rn(__half2float(hv3.y) * inv);
            const uint32_t lo = lo_signs | (uint32_t(min(aq0,126)) | (uint32_t(min(aq1,126))<<8) | (uint32_t(min(aq2,126))<<16) | (uint32_t(min(aq3,126))<<24));
            const uint32_t hi = hi_signs | (uint32_t(min(aq4,126)) | (uint32_t(min(aq5,126))<<8) | (uint32_t(min(aq6,126))<<16) | (uint32_t(min(aq7,126))<<24));
            if (vec_store) {
                /* V63: single uint2 store (st.global.v2.b32) — halves store instruction count. */
                *reinterpret_cast<uint2*>(row_dst + c) = uint2{lo, hi};
            } else {
                /* Unaligned fallback (2K DC zone 2: col_start=60, ~0.5% of work). */
                *reinterpret_cast<uint32_t*>(row_dst + c)     = lo;
                *reinterpret_cast<uint32_t*>(row_dst + c + 4) = hi;
            }
        }
        /* vq4 tail: handles residual elements when zone width % 8 != 0.
         * Only occurs for 2K DC rows (ll5_c=60, 60%8=4 → 4 tail elements per small zone).
         * V72: PRMT sign extraction — parity with vq8 core (replaces 4 SETP+SELP+IABS with PRMT+AND). */
        const int vq8_end = col_start + ((col_end - col_start) / 8) * 8;
        for (int c = vq8_end + threadIdx.x * 4; c + 3 < col_end; c += nt * 4) {
            __half2 hv0 = __ldg(reinterpret_cast<const __half2*>(row_src) + c / 2);
            __half2 hv1 = __ldg(reinterpret_cast<const __half2*>(row_src) + c / 2 + 1);
            uint32_t raw0, raw1;
            __builtin_memcpy(&raw0, &hv0, 4);
            __builtin_memcpy(&raw1, &hv1, 4);
            /* Extract sign bytes {hv1.y, hv1.x, hv0.y, hv0.x} from MSB of each __half. */
            const uint32_t signs = __byte_perm(raw0, raw1, 0x7531u) & 0x80808080u;
            /* Clear sign bits → absolute values for conversion. */
            raw0 &= 0x7FFF7FFFu; raw1 &= 0x7FFF7FFFu;
            __builtin_memcpy(&hv0, &raw0, 4);
            __builtin_memcpy(&hv1, &raw1, 4);
            int aq0 = __float2int_rn(__half2float(hv0.x) * inv);
            int aq1 = __float2int_rn(__half2float(hv0.y) * inv);
            int aq2 = __float2int_rn(__half2float(hv1.x) * inv);
            int aq3 = __float2int_rn(__half2float(hv1.y) * inv);
            *reinterpret_cast<uint32_t*>(row_dst + c) = signs |
                (uint32_t(min(126,aq0)) | (uint32_t(min(126,aq1))<<8)
                | (uint32_t(min(126,aq2))<<16) | (uint32_t(min(126,aq3))<<24));
        }
    };

    /* V78: L1-row early exit — saves 5 FMUL per thread for >50% of 2K/4K rows.
     * L1 rows (row >= ll5_h*16): only inv_l1 = base_inv*0.833... needed.
     * Exiting before the 5 other FMUL constants avoids wasted compute for majority path.
     * Register pressure: 6 floats → 1 (base_inv) + inline constant for L1 blocks.
     * Parity: CUDA V115. */
    if (row >= ll5_h * 16) {
        vq8(0, stride, base_inv * 0.833333333f, blockDim.x);
        return;
    }

    /* V66: 5 FMUL instead of 5 __frcp_rn (only reached for non-L1 rows, <50% of rows).
     * Saves 5 SFU ops per warp; compile-time reciprocal mult constants. */
    float inv_dc = base_inv * 1.538461538f;  /* 1/0.65 */
    float inv_l5 = base_inv * 1.176470588f;  /* 1/0.85 */
    float inv_l4 = base_inv * 1.052631579f;  /* 1/0.95 */
    float inv_l3 = base_inv * 0.952380952f;  /* 1/1.05 */
    float inv_l2 = base_inv * 0.892857143f;  /* 1/1.12 */
    float inv_l1 = base_inv * 0.833333333f;  /* 1/1.20 */

    bool is_dc_row = (row < ll5_h);
    bool is_l5_row = (row < ll5_h * 2);
    /* V78: row < ll5_h*16 guaranteed by early exit above → row_lv in {2..5}. */
    int row_lv = is_l5_row ? 5 :
                 (row < ll5_h*4  ? 4 :
                 (row < ll5_h*8  ? 3 : 2));

    /* row_lv == 1 handled above (early exit); dispatch only L2..L5/DC rows here. */
    if (row_lv == 2) {
        const int mid = ll5_c * 16;
        vq8(0,   mid,    inv_l2, blockDim.x);
        vq8(mid, stride, inv_l1, blockDim.x);
    } else if (row_lv == 3) {
        const int b1 = ll5_c * 8;
        const int b2 = ll5_c * 16;
        vq8(0,  b1,     inv_l3, blockDim.x);
        vq8(b1, b2,     inv_l2, blockDim.x);
        vq8(b2, stride, inv_l1, blockDim.x);
    } else {
        /* V56: L4/L5/DC rows — fully vectorized zone-based quantize via vq8.
         * Zone widths ≥ 2*ll5_c are div-by-8 → zero tail for L4/L5 rows.
         * DC rows: 6 zones; two smallest zones (width ll5_c) need vq4 tail only in 2K.
         * DC rows: 6 zones. L5 rows: 5 zones. L4 rows: 4 zones. */
        if (is_dc_row) {
            vq8(0,          ll5_c,      inv_dc, blockDim.x);
            vq8(ll5_c,      ll5_c*2,    inv_l5, blockDim.x);
            vq8(ll5_c*2,    ll5_c*4,    inv_l4, blockDim.x);
            vq8(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq8(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq8(ll5_c*16,   stride,     inv_l1, blockDim.x);
        } else if (is_l5_row) {
            vq8(0,          ll5_c*2,    inv_l5, blockDim.x);
            vq8(ll5_c*2,    ll5_c*4,    inv_l4, blockDim.x);
            vq8(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq8(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq8(ll5_c*16,   stride,     inv_l1, blockDim.x);
        } else {
            /* L4 rows: [0,4c)→inv_l4, then l3, l2, l1 */
            vq8(0,          ll5_c*4,    inv_l4, blockDim.x);
            vq8(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq8(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq8(ll5_c*16,   stride,     inv_l1, blockDim.x);
        }
    }
}


/* V17e: Fused RGB48→XYZ + H-DWT level 0 for all 3 components (mirrors CUDA V28).
 * Eliminates int32 d_in[0..2] intermediates (~54MB/frame DRAM traffic).
 * One block per row; shared memory: 3 × width floats (24KB for 2K, 25% occupancy). */
__global__ void s17_rgb48_xyz_hdwt0(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V18f: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hx,   /* comp 0 H-DWT half output */
    __half* __restrict__ d_hy,   /* comp 1 H-DWT half output */
    __half* __restrict__ d_hz,   /* comp 2 H-DWT half output */
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ float sm[];
    float* smX = sm;
    float* smY = sm + width;
    float* smZ = sm + 2 * width;

    int y = blockIdx.x; if (y >= height) return;
    int t = threadIdx.x, nt = blockDim.x;

    for (int px = t; px < width; px += nt) {
        int base = y * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[base+0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[base+1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[base+2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V18f */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float xv = __ldg(&d_matrix[0])*r + __ldg(&d_matrix[1])*g + __ldg(&d_matrix[2])*b;
        float yv = __ldg(&d_matrix[3])*r + __ldg(&d_matrix[4])*g + __ldg(&d_matrix[5])*b;
        float zv = __ldg(&d_matrix[6])*r + __ldg(&d_matrix[7])*g + __ldg(&d_matrix[8])*b;
        xv = __saturatef(xv); yv = __saturatef(yv); zv = __saturatef(zv);
        smX[px] = (float)__ldg(&d_lut_out[(int)(xv*4095.5f)]);
        smY[px] = (float)__ldg(&d_lut_out[(int)(yv*4095.5f)]);
        smZ[px] = (float)__ldg(&d_lut_out[(int)(zv*4095.5f)]);
    }
    __syncthreads();

    int w = width, hw = (w+1)/2;
    /* V82: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
#define S17_HDWT_HALF(smc, dst)                                                              \
    for (int x=1+t*2; x<w-1; x+=nt*2) (smc)[x]+=ALPHA*((smc)[x-1]+(smc)[x+1]);            \
    if(t==0) (smc)[w-1]+=2.f*ALPHA*(smc)[w-2];                                              \
    __syncthreads();                                                                          \
    if(t==0) (smc)[0]+=2.f*BETA*(smc)[1];                                                   \
    for (int x=2+t*2; x<w; x+=nt*2) (smc)[x]+=BETA*((smc)[x-1]+(smc)[x+1]);               \
    __syncthreads();                                                                          \
    for (int x=1+t*2; x<w-1; x+=nt*2) (smc)[x]+=GAMMA*((smc)[x-1]+(smc)[x+1]);            \
    if(t==0) (smc)[w-1]+=2.f*GAMMA*(smc)[w-2];                                              \
    __syncthreads();                                                                          \
    if(t==0) (smc)[0]+=2.f*DELTA*(smc)[1];                                                  \
    for (int x=2+t*2; x<w; x+=nt*2) (smc)[x]+=DELTA*((smc)[x-1]+(smc)[x+1]);              \
    __syncthreads();                                                                          \
    for (int x=t*2;   x<w; x+=nt*2) (dst)[y*stride+x/2]    =__float2half((smc)[x]*NORM_L); \
    for (int x=t*2+1; x<w; x+=nt*2) (dst)[y*stride+hw+x/2] =__float2half((smc)[x]*NORM_H); \
    __syncthreads();

    S17_HDWT_HALF(smX, d_hx)
    S17_HDWT_HALF(smY, d_hy)
    S17_HDWT_HALF(smZ, d_hz)
#undef S17_HDWT_HALF
}


/* V17g: Single-channel fused RGB48→XYZ colour conversion + H-DWT level 0.
 * Split from s17_rgb48_xyz_hdwt0 (24KB smem) to 1-channel variant (8KB smem).
 * Launch 3 instances in parallel on separate streams → 100% thread occupancy
 * (vs 25% for the 3-ch kernel). d_rgb16 L2-cached across all 3 streams.
 * Expected: ~0.25ms vs ~0.95ms for s17_rgb48_xyz_hdwt0 → ~0.7ms/frame saved. */
/* V17o: __half smem — colour convert to float (needs precision), write half to smem, lift in fp16. */
__global__ void s17_rgb48_xyz_hdwt0_1ch(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V18f: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,   /* single component H-DWT half output */
    int comp,                       /* 0=X, 1=Y, 2=Z */
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];  /* V17o: was float; halves smem + 2× fp16 lifting throughput */
    int y = blockIdx.x; if (y >= height) return;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w+1)/2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr+1]);
    float m2 = __ldg(&d_matrix[mr+2]);

    for (int px = t; px < w; px += nt) {
        int base = y * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[base+0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[base+1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[base+2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V18f */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = m0*r + m1*g + m2*b;
        v = __saturatef(v);
        sm[px] = s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
    }
    __syncthreads();

    /* V82: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
    for (int x=1+t*2; x<w-1; x+=nt*2) sm[x]+=__half(ALPHA)*(sm[x-1]+sm[x+1]);
    if(t==0) sm[w-1]+=__half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    if(t==0) sm[0]+=__half(2.f*BETA)*sm[1];
    for (int x=2+t*2; x<w; x+=nt*2) sm[x]+=__half(BETA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) sm[x]+=__half(GAMMA)*(sm[x-1]+sm[x+1]);
    if(t==0) sm[w-1]+=__half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    if(t==0) sm[0]+=__half(2.f*DELTA)*sm[1];
    for (int x=2+t*2; x<w; x+=nt*2) sm[x]+=__half(DELTA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    for (int x=t*2;   x<w; x+=nt*2) d_hout[y*stride+x/2]    = sm[x] * __half(NORM_L);
    for (int x=t*2+1; x<w; x+=nt*2) d_hout[y*stride+hw+x/2] = sm[x] * __half(NORM_H);
}


/* V17v: 2-rows-per-block variant — parity with CUDA V45.
 * smem[0..w-1]=row y0, smem[w..2w-1]=row y1; grid=(height+1)/2; smem=2*w*sizeof(__half).
 * Same 4 syncthreads for 2 rows → 2× SM row throughput; matrix loads amortized 2×. */
__global__ void s17_rgb48_xyz_hdwt0_1ch_2row(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V18f: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];
    int y0 = blockIdx.x * 2;
    int y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w+1)/2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr+1]);
    float m2 = __ldg(&d_matrix[mr+2]);

    for (int px = t; px < w; px += nt) {
        int b0 = y0 * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[b0+0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[b0+1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[b0+2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V18f */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = __saturatef(m0*r + m1*g + m2*b);
        sm[px] = s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
    }
    if (y1 < height) {
        for (int px = t; px < w; px += nt) {
            int b1 = y1 * rgb_stride + px * 3;
            int ri = min((__ldg(&d_rgb16[b1+0]) >> 4), 4095);
            int gi = min((__ldg(&d_rgb16[b1+1]) >> 4), 4095);
            int bi = min((__ldg(&d_rgb16[b1+2]) >> 4), 4095);
            float r = __half2float(__ldg(&d_lut_in[ri]));  /* V18f */
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = __saturatef(m0*r + m1*g + m2*b);
            sm[w+px] = s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
        }
    }
    __syncthreads();

    for (int x=1+t*2; x<w-1; x+=nt*2) {
        sm[x]  +=__half(ALPHA)*(sm[x-1]+sm[x+1]);
        if(y1<height) sm[w+x]+=__half(ALPHA)*(sm[w+x-1]+sm[w+x+1]);
    }
    /* V82: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
    if(t==0) { sm[w-1]+=__half(2.f*ALPHA)*sm[w-2]; if(y1<height) sm[2*w-1]+=__half(2.f*ALPHA)*sm[2*w-2]; }
    __syncthreads();
    if(t==0) { sm[0]+=__half(2.f*BETA)*sm[1]; if(y1<height) sm[w]+=__half(2.f*BETA)*sm[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        sm[x]  +=__half(BETA)*(sm[x-1]+sm[x+1]);
        if(y1<height) sm[w+x]+=__half(BETA)*(sm[w+x-1]+sm[w+x+1]);
    }
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) {
        sm[x]  +=__half(GAMMA)*(sm[x-1]+sm[x+1]);
        if(y1<height) sm[w+x]+=__half(GAMMA)*(sm[w+x-1]+sm[w+x+1]);
    }
    if(t==0) { sm[w-1]+=__half(2.f*GAMMA)*sm[w-2]; if(y1<height) sm[2*w-1]+=__half(2.f*GAMMA)*sm[2*w-2]; }
    __syncthreads();
    if(t==0) { sm[0]+=__half(2.f*DELTA)*sm[1]; if(y1<height) sm[w]+=__half(2.f*DELTA)*sm[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        sm[x]  +=__half(DELTA)*(sm[x-1]+sm[x+1]);
        if(y1<height) sm[w+x]+=__half(DELTA)*(sm[w+x-1]+sm[w+x+1]);
    }
    __syncthreads();
    for (int x=t*2; x<w; x+=nt*2) {
        d_hout[y0*stride+x/2]       = sm[x]   * __half(NORM_L);
        if(y1<height) d_hout[y1*stride+x/2]   = sm[w+x] * __half(NORM_L);
    }
    for (int x=t*2+1; x<w; x+=nt*2) {
        d_hout[y0*stride+hw+x/2]    = sm[x]   * __half(NORM_H);
        if(y1<height) d_hout[y1*stride+hw+x/2] = sm[w+x] * __half(NORM_H);
    }
}


/* V18e: Packed 12-bit planar variant of s17_rgb48_xyz_hdwt0_1ch_2row (parity with CUDA V54).
 * Input d_rgb12: 3 contiguous planes (R/G/B), each height*packed_row_stride bytes.
 * packed_row_stride = (width/2)*3; pair unpack: even=(b0<<4|b1>>4), odd=((b1&0xF)<<8|b2). */
/* V79: __launch_bounds__(512,3) — matches actual 3 blk/SM smem limit (was 512,4 from V51).
 * V53 added 8KB sm_lut; smem=15.68KB → PreferShared(48KB)=3 blk/SM actual occupancy cap.
 * (512,3) → ≤42 regs/T; gives compiler 10 extra registers vs old (512,4)→32 regs limit.
 * Parity: CUDA V116. */
__global__ __launch_bounds__(512, 3)
void s17_rgb48_xyz_hdwt0_1ch_2row_p12(
    const uint8_t*  __restrict__ d_rgb12,
    const __half*   __restrict__ d_lut_in,  /* V18f: was float */
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    int y0 = blockIdx.x * 2, y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w+1)/2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr+1]);
    float m2 = __ldg(&d_matrix[mr+2]);

    size_t plane_stride = (size_t)height * packed_row_stride;
    const uint8_t* r_p = d_rgb12 + 0 * plane_stride;
    const uint8_t* g_p = d_rgb12 + 1 * plane_stride;
    const uint8_t* b_p = d_rgb12 + 2 * plane_stride;

    /* V53: 8KB static smem for d_lut_in preload — 0-cycle smem access vs ~30-50 cycle L1/L2. */
    __shared__ __half sm_lut[4096];
    /* V54: int4 vectorized preload — each of 512T loads 1 int4 (8 × __half) in one ld.global.b128.
     * Saves 7 scalar loads per thread vs V53's #pragma unroll 8 loop. 16B-aligned guaranteed. */
    reinterpret_cast<int4*>(sm_lut)[t] = __ldg(reinterpret_cast<const int4*>(d_lut_in) + t);
    __syncthreads();

    /* V46: hoist y1<height check to block level — interior path uses __half2 interleaved smem.
     * sm2[x]={row0[x],row1[x]}; total smem = w __half2 = 2w __half (same as before).
     * V53: lambda reads sm_lut (smem) instead of __ldg d_lut_in (L1/L2) for zero latency. */
    auto s17_lut_xyz_2r = [&](int ri, int gi, int bi) -> __half {
        float r = __half2float(sm_lut[ri]);
        float g = __half2float(sm_lut[gi]);
        float b = __half2float(sm_lut[bi]);
        float v = __saturatef(m0*r+m1*g+m2*b);
        return s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
    };
    /* V84: DCI height always even (1080/2160) → y1=y0+1 always < height; partial else branch removed.
     * Compiler can now optimize unconditionally — no register partitioning for dead branch.
     * Parity: CUDA V122. */
    {
        /* Interior path: both rows valid — __half2 lifting, 2× FMA throughput. */
        __half2* sm2 = reinterpret_cast<__half2*>(sm);
        /* V83: uint32_t __ldg replaces 3×byte __ldg per channel — 3× fewer load instructions per pair.
         * V52: #pragma unroll 2 — 2K ~2 iters; 3 loads (was 9) per row → better MLP per iteration. */
        #pragma unroll 2
        for (int p = t; p*2 < w; p += nt) {
            int off0 = y0*packed_row_stride + p*3;
            uint32_t rw0=__ldg(reinterpret_cast<const uint32_t*>(r_p+off0));
            uint32_t gw0=__ldg(reinterpret_cast<const uint32_t*>(g_p+off0));
            uint32_t bw0=__ldg(reinterpret_cast<const uint32_t*>(b_p+off0));
            __half e0 = s17_lut_xyz_2r(((rw0&0xFF)<<4)|((rw0>>12)&0xF), ((gw0&0xFF)<<4)|((gw0>>12)&0xF), ((bw0&0xFF)<<4)|((bw0>>12)&0xF));
            __half o0 = s17_lut_xyz_2r(((rw0>>8)&0xF)<<8|((rw0>>16)&0xFF), ((gw0>>8)&0xF)<<8|((gw0>>16)&0xFF), ((bw0>>8)&0xF)<<8|((bw0>>16)&0xFF));
            int off1 = y1*packed_row_stride + p*3;
            uint32_t rw1=__ldg(reinterpret_cast<const uint32_t*>(r_p+off1));
            uint32_t gw1=__ldg(reinterpret_cast<const uint32_t*>(g_p+off1));
            uint32_t bw1=__ldg(reinterpret_cast<const uint32_t*>(b_p+off1));
            __half e1 = s17_lut_xyz_2r(((rw1&0xFF)<<4)|((rw1>>12)&0xF), ((gw1&0xFF)<<4)|((gw1>>12)&0xF), ((bw1&0xFF)<<4)|((bw1>>12)&0xF));
            __half o1 = s17_lut_xyz_2r(((rw1>>8)&0xF)<<8|((rw1>>16)&0xFF), ((gw1>>8)&0xF)<<8|((gw1>>16)&0xFF), ((bw1>>8)&0xF)<<8|((bw1>>16)&0xFF));
            sm2[p*2]   = __halves2half2(e0, e1);
            sm2[p*2+1] = __halves2half2(o0, o1);
        }
        __syncthreads();
        {
            const __half2 kA = __half2half2(__float2half(ALPHA));
            /* V52: #pragma unroll 2 — 2K ~2 iters; interleaves smem reads + HFMA2s. */
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2)
                sm2[x] = __hfma2(kA, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
            /* V74: DCI w always even and >1 → simplify Alpha boundary (parity CUDA V111). */
            if(t==0)
                sm2[w-1] = __hfma2(__half2half2(__float2half(2.f*ALPHA)), sm2[w-2], sm2[w-1]);
        }
        __syncthreads();
        {
            const __half2 kB = __half2half2(__float2half(BETA));
            /* V74: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
            if(t==0) sm2[0] = __hfma2(__half2half2(__float2half(2.f*BETA)), sm2[1], sm2[0]);
            /* V65: x always even → even w: min(x+1,w-1)=x+1; drop MIN per iteration. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2)
                sm2[x] = __hfma2(kB, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
        }
        __syncthreads();
        {
            const __half2 kG = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2)
                sm2[x] = __hfma2(kG, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
            /* V74: DCI w always even and >1 → simplify Gamma boundary. */
            if(t==0)
                sm2[w-1] = __hfma2(__half2half2(__float2half(2.f*GAMMA)), sm2[w-2], sm2[w-1]);
        }
        __syncthreads();
        {
            const __half2 kD = __half2half2(__float2half(DELTA));
            /* V74: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
            if(t==0) sm2[0] = __hfma2(__half2half2(__float2half(2.f*DELTA)), sm2[1], sm2[0]);
            /* V65: same even-w invariant — drop MIN. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2)
                sm2[x] = __hfma2(kD, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
        }
        __syncthreads();
        /* Combined L+H scatter via __low2half/__high2half. */
        {
            const __half2 nL = __half2half2(__float2half(NORM_L));
            const __half2 nH = __half2half2(__float2half(NORM_H));
            /* V52: #pragma unroll 2 — interleaves smem reads + global writes. */
            #pragma unroll 2
            for (int p = t; p < w/2; p += nt) {
                __half2 vL = __hmul2(sm2[p*2],   nL);
                __half2 vH = __hmul2(sm2[p*2+1], nH);
                d_hout[y0*stride+p]    = __low2half(vL);
                d_hout[y1*stride+p]    = __high2half(vL);
                d_hout[y0*stride+hw+p] = __low2half(vH);
                d_hout[y1*stride+hw+p] = __high2half(vH);
            }
        }
    }
}


/* V42: 1-row-per-block RGB+HDWT0 p12 for 4K — 100% occupancy at 4K.
 * grid=height; smem=w*sizeof(__half) (3.84KB@2K, 7.68KB@4K).
 * For 4K: smem=7.68KB; PreferNone(32KB/SM)→4 blk/SM=100% occ vs 2-row's 2 blk/SM=50%.
 * Simpler than 2-row: no y1 guard, single-row smem. Launch: <<<height, h_block, w*2B, s>>>. */
/* V79: __launch_bounds__(512,3) — same fix as 2row_p12 above (parity CUDA V116).
 * smem=15.68KB (sm_lut 8KB + DWT 7.68KB for 4K 1row) → PreferShared(48KB)=3 blk/SM actual.
 * (512,3) → ≤42 regs/T; relaxes compiler register budget from 32 to 42. */
__global__ __launch_bounds__(512, 3)
void s17_rgb48_xyz_hdwt0_1ch_1row_p12(
    const uint8_t*  __restrict__ d_rgb12,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    /* V53: 8KB static smem for d_lut_in preload — 0-cycle smem access vs ~30-50 cycle L1/L2. */
    __shared__ __half sm_lut[4096];
    int y0 = blockIdx.x;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w+1)/2;

    /* V54: int4 vectorized preload — all 512T each load 1 int4 (8 × __half); early exit after sync.
     * Early exit MUST be after __syncthreads() to avoid deadlock in over-allocated grids. */
    reinterpret_cast<int4*>(sm_lut)[t] = __ldg(reinterpret_cast<const int4*>(d_lut_in) + t);
    __syncthreads();

    /* Guard after sync — y0 always < height (grid=height) but safe for future grid changes. */
    if (y0 >= height) return;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr+1]);
    float m2 = __ldg(&d_matrix[mr+2]);

    size_t plane_stride = (size_t)height * packed_row_stride;
    const uint8_t* r_p = d_rgb12 + 0 * plane_stride;
    const uint8_t* g_p = d_rgb12 + 1 * plane_stride;
    const uint8_t* b_p = d_rgb12 + 2 * plane_stride;

    /* V44: pair-wise unpack — 3 byte loads/channel/pair (saves 25% vs duplicate byte[1]).
     * w always even for 2K/4K → all pairs valid (no odd-width edge case).
     * V53: lambda reads sm_lut (smem) instead of __ldg d_lut_in (L1/L2) for zero latency. */
    auto s17_lut_xyz_1r = [&](int ri, int gi, int bi) -> __half {
        float r = __half2float(sm_lut[ri]);
        float g = __half2float(sm_lut[gi]);
        float b = __half2float(sm_lut[bi]);
        float v = __saturatef(m0*r+m1*g+m2*b);
        return s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
    };
    /* V83: uint32_t __ldg replaces 3×byte __ldg per channel — 3× fewer load instr per pair.
     * uint32_t at off = byte0|(byte1<<8)|(byte2<<16)|...; even=(byte0<<4)|(byte1>>4); odd=((byte1&0xF)<<8)|byte2.
     * V52: #pragma unroll 4 — 4K has ~4 iters; 3 loads (was 9) → better MLP per unrolled group. */
    #pragma unroll 4
    for (int p = t; p*2 < w; p += nt) {
        int off = y0*packed_row_stride + p*3;
        uint32_t rw=__ldg(reinterpret_cast<const uint32_t*>(r_p+off));
        uint32_t gw=__ldg(reinterpret_cast<const uint32_t*>(g_p+off));
        uint32_t bw=__ldg(reinterpret_cast<const uint32_t*>(b_p+off));
        sm[p*2]   = s17_lut_xyz_1r(((rw&0xFF)<<4)|((rw>>12)&0xF), ((gw&0xFF)<<4)|((gw>>12)&0xF), ((bw&0xFF)<<4)|((bw>>12)&0xF));
        sm[p*2+1] = s17_lut_xyz_1r(((rw>>8)&0xF)<<8|((rw>>16)&0xFF), ((gw>>8)&0xF)<<8|((gw>>16)&0xFF), ((bw>>8)&0xF)<<8|((bw>>16)&0xFF));
    }
    __syncthreads();

    /* V52: #pragma unroll 4 — interleaves smem reads/FMAs across ~4 iters to hide smem latency. */
    #pragma unroll 4
    for (int x=1+t*2; x<w-1; x+=nt*2)
        sm[x]+=__half(ALPHA)*(sm[x-1]+sm[x+1]);
    /* V74: DCI w always even and >1 → simplify Alpha boundary; parity CUDA V111. */
    if(t==0) sm[w-1]+=__half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    /* V74: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
    if(t==0) sm[0]+=__half(2.f*BETA)*sm[1];
    /* V65: x always even → even w (4K=3840 even): x≤w-2 → x+1≤w-1 → drop MIN. */
    #pragma unroll 4
    for (int x=2+t*2; x<w-1; x+=nt*2)
        sm[x]+=__half(BETA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    #pragma unroll 4
    for (int x=1+t*2; x<w-1; x+=nt*2)
        sm[x]+=__half(GAMMA)*(sm[x-1]+sm[x+1]);
    /* V74: DCI w always even and >1 → simplify Gamma boundary. */
    if(t==0) sm[w-1]+=__half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    /* V74: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
    if(t==0) sm[0]+=__half(2.f*DELTA)*sm[1];
    /* V65: same even-w invariant — drop MIN from Delta loop. */
    #pragma unroll 4
    for (int x=2+t*2; x<w-1; x+=nt*2)
        sm[x]+=__half(DELTA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    /* V49: __hmul2 scatter — sm[p*2]||sm[p*2+1] loaded as __half2 (one ld.shared.b32).
     * Apply {NORM_L, NORM_H} via one HMUL2 (2× FP16 throughput vs two scalar HMULs).
     * bank(sm[p*2]) = p%32 for p=t → all 32 warp threads hit distinct banks → conflict-free. */
    {
        const __half2 norms = __halves2half2(__float2half(NORM_L), __float2half(NORM_H));
        /* V52: #pragma unroll 4 — interleaves smem reads + global writes. */
        #pragma unroll 4
        for (int p = t; p < w/2; p += nt) {
            __half2 lh = __hmul2(*reinterpret_cast<const __half2*>(&sm[p*2]), norms);
            d_hout[y0*stride+p]    = __low2half(lh);
            d_hout[y0*stride+hw+p] = __high2half(lh);
        }
    }
}


/* V58: 4-rows-per-block RGB+HDWT0 with sm_lut preload (parity CUDA V95).
 * sm_lut[4096] __half = 8KB static smem; int4 vectorized preload (512T × 1 int4 each).
 * Total smem: 15.36KB (DWT) + 8KB (sm_lut) = 23.36KB.
 * PreferShared (48KB/SM): 2 blk/SM → 50% occ (same as current PreferNone). */
__global__ void s17_rgb48_xyz_hdwt0_1ch_4row_p12(
    const uint8_t*  __restrict__ d_rgb12,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    __shared__ __half sm_lut[4096];  /* V58: 8KB static smem for d_lut_in preload */
    int y0=blockIdx.x*4, y1=y0+1, y2=y0+2, y3=y0+3;
    int t=threadIdx.x, nt=blockDim.x;
    int w=width, hw=(w+1)/2;

    /* V58: int4 vectorized preload — nt=512, 512 int4 → each thread loads 1 int4 (8 __half).
     * All threads participate before if/else → no syncthreads deadlock risk. */
    reinterpret_cast<int4*>(sm_lut)[t] = __ldg(reinterpret_cast<const int4*>(d_lut_in) + t);
    __syncthreads();

    int mr=comp*3;
    float m0=__ldg(&d_matrix[mr]), m1=__ldg(&d_matrix[mr+1]), m2=__ldg(&d_matrix[mr+2]);

    size_t plane_stride=(size_t)height*packed_row_stride;
    const uint8_t* r_p=d_rgb12+0*plane_stride;
    const uint8_t* g_p=d_rgb12+1*plane_stride;
    const uint8_t* b_p=d_rgb12+2*plane_stride;

#define S17_U12_4R(plane, row_off, odd) \
    ((odd) ? ((int(__ldg((plane)+(row_off)+1)&0xF)<<8)|int(__ldg((plane)+(row_off)+2))) \
           : ((int(__ldg((plane)+(row_off)+0))<<4)|(int(__ldg((plane)+(row_off)+1))>>4)))

    /* V31: Fuse 4-row loads into if/else block for maximum texture load MLP.
     * V58: sm_lut[] replaces d_lut_in __ldg (3-cycle smem vs 30-50 cycle L1/L2 latency).
     * Interior (y3<height, 100% of 2K): single for loop issues all 4 rows' __ldg at once.
     * Else (partial last block): original guarded per-row loads (y3 not loaded). */
    if (y3 < height) {
        /* V69: __half2 row-pair packing — sm01[x]={row0[x],row1[x]}, sm23[x]={row2[x],row3[x]}.
         * Load packs pairs; lifting uses __hfma2 (2 rows/instruction) → 2× FMA throughput.
         * Parity with s17_dwt_h_half_io_4row (V45) and CUDA V106/V82/V83.
         * + #pragma unroll 2 on lifting loops (parity V59).
         * + drop MIN from Beta/Delta main loops: x always even, w even → x+1≤w-1 (parity V65). */
        __half2* sm01 = reinterpret_cast<__half2*>(sm);
        __half2* sm23 = reinterpret_cast<__half2*>(sm + 2*w);
        /* Load: interleaved row-pair stores into sm01/sm23 — same smem, enables __hfma2 lifting. */
        for (int px=t; px<w; px+=nt) {
            int o=px&1, phalf=(px/2)*3;
            int off0=y0*packed_row_stride+phalf;
            int ri0=S17_U12_4R(r_p,off0,o), gi0=S17_U12_4R(g_p,off0,o), bi0=S17_U12_4R(b_p,off0,o);
            float r0=__half2float(sm_lut[ri0]);
            float g0=__half2float(sm_lut[gi0]);
            float b0=__half2float(sm_lut[bi0]);
            __half v0=s17_u16_to_f16(__ldg(&d_lut_out[(int)(__saturatef(m0*r0+m1*g0+m2*b0)*4095.5f)]));
            int off1=y1*packed_row_stride+phalf;
            int ri1=S17_U12_4R(r_p,off1,o), gi1=S17_U12_4R(g_p,off1,o), bi1=S17_U12_4R(b_p,off1,o);
            float r1=__half2float(sm_lut[ri1]);
            float g1=__half2float(sm_lut[gi1]);
            float b1=__half2float(sm_lut[bi1]);
            __half v1=s17_u16_to_f16(__ldg(&d_lut_out[(int)(__saturatef(m0*r1+m1*g1+m2*b1)*4095.5f)]));
            int off2=y2*packed_row_stride+phalf;
            int ri2=S17_U12_4R(r_p,off2,o), gi2=S17_U12_4R(g_p,off2,o), bi2=S17_U12_4R(b_p,off2,o);
            float r2=__half2float(sm_lut[ri2]);
            float g2=__half2float(sm_lut[gi2]);
            float b2=__half2float(sm_lut[bi2]);
            __half v2=s17_u16_to_f16(__ldg(&d_lut_out[(int)(__saturatef(m0*r2+m1*g2+m2*b2)*4095.5f)]));
            int off3=y3*packed_row_stride+phalf;
            int ri3=S17_U12_4R(r_p,off3,o), gi3=S17_U12_4R(g_p,off3,o), bi3=S17_U12_4R(b_p,off3,o);
            float r3=__half2float(sm_lut[ri3]);
            float g3=__half2float(sm_lut[gi3]);
            float b3=__half2float(sm_lut[bi3]);
            __half v3=s17_u16_to_f16(__ldg(&d_lut_out[(int)(__saturatef(m0*r3+m1*g3+m2*b3)*4095.5f)]));
            sm01[px] = __halves2half2(v0, v1);  /* {row0[px], row1[px]} */
            sm23[px] = __halves2half2(v2, v3);  /* {row2[px], row3[px]} */
        }
        __syncthreads();
        /* Alpha: odd positions. #pragma unroll 2 for 4K MLP. */
        {
            const __half2 kA2 = __half2half2(__float2half(ALPHA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kA2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kA2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
            if(t==0) {
                const __half2 kA2bd = __half2half2(__float2half(2.f*ALPHA));
                sm01[w-1] = __hfma2(kA2bd, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kA2bd, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* Beta: even positions. */
        {
            const __half2 kB2 = __half2half2(__float2half(BETA));
            if(t==0) {
                const __half2 kB2bd = __half2half2(__float2half(2.f*BETA));
                /* V81: DCI w>1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kB2bd, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kB2bd, sm23[1], sm23[0]);
            }
            /* V69: x always even → even w (DCI) → x≤w-2 → x+1≤w-1 → drop MIN. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kB2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kB2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* Gamma: odd positions. */
        {
            const __half2 kG2 = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kG2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kG2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
            if(t==0) {
                const __half2 kG2bd = __half2half2(__float2half(2.f*GAMMA));
                sm01[w-1] = __hfma2(kG2bd, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kG2bd, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* Delta: even positions. */
        {
            const __half2 kD2 = __half2half2(__float2half(DELTA));
            if(t==0) {
                const __half2 kD2bd = __half2half2(__float2half(2.f*DELTA));
                /* V81: DCI w>1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kD2bd, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kD2bd, sm23[1], sm23[0]);
            }
            /* V69: same even-w invariant as Beta — drop MIN. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kD2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kD2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* Scatter: deinterleave __half2 pairs → L/H subbands. #pragma unroll 2 for 4K MLP. */
        {
            const __half2 nL2 = __half2half2(__float2half(NORM_L));
            const __half2 nH2 = __half2half2(__float2half(NORM_H));
            #pragma unroll 2
            for (int p=t; p<w/2; p+=nt) {
                __half2 v01L = __hmul2(sm01[p*2],   nL2);
                __half2 v23L = __hmul2(sm23[p*2],   nL2);
                d_hout[y0*stride+p] = __low2half(v01L);
                d_hout[y1*stride+p] = __high2half(v01L);
                d_hout[y2*stride+p] = __low2half(v23L);
                d_hout[y3*stride+p] = __high2half(v23L);
                __half2 v01H = __hmul2(sm01[p*2+1], nH2);
                __half2 v23H = __hmul2(sm23[p*2+1], nH2);
                d_hout[y0*stride+hw+p] = __low2half(v01H);
                d_hout[y1*stride+hw+p] = __high2half(v01H);
                d_hout[y2*stride+hw+p] = __low2half(v23H);
                d_hout[y3*stride+hw+p] = __high2half(v23H);
            }
        }
    } else {
        /* V58: Partial last block: guarded per-row loads using sm_lut (3-cycle smem access). */
        for (int px=t; px<w; px+=nt) {
            int off=y0*packed_row_stride+(px/2)*3, o=px&1;
            int ri=S17_U12_4R(r_p,off,o), gi=S17_U12_4R(g_p,off,o), bi=S17_U12_4R(b_p,off,o);
            float r=__half2float(sm_lut[ri]);
            float g=__half2float(sm_lut[gi]);
            float b=__half2float(sm_lut[bi]);
            float v=__saturatef(m0*r+m1*g+m2*b);
            sm[px]=s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
        }
        if (y1<height) for (int px=t; px<w; px+=nt) {
            int off=y1*packed_row_stride+(px/2)*3, o=px&1;
            int ri=S17_U12_4R(r_p,off,o), gi=S17_U12_4R(g_p,off,o), bi=S17_U12_4R(b_p,off,o);
            float r=__half2float(sm_lut[ri]);
            float g=__half2float(sm_lut[gi]);
            float b=__half2float(sm_lut[bi]);
            float v=__saturatef(m0*r+m1*g+m2*b);
            sm[w+px]=s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
        }
        if (y2<height) for (int px=t; px<w; px+=nt) {
            int off=y2*packed_row_stride+(px/2)*3, o=px&1;
            int ri=S17_U12_4R(r_p,off,o), gi=S17_U12_4R(g_p,off,o), bi=S17_U12_4R(b_p,off,o);
            float r=__half2float(sm_lut[ri]);
            float g=__half2float(sm_lut[gi]);
            float b=__half2float(sm_lut[bi]);
            float v=__saturatef(m0*r+m1*g+m2*b);
            sm[2*w+px]=s17_u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
        }
        __syncthreads();
        /* Partial: conditional lifting — y3 >= height, so all y3 guards removed. */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(ALPHA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(ALPHA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(ALPHA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]  +=__half(2.f*ALPHA)*sm[w-2];
            if(y1<height) sm[2*w-1]+=__half(2.f*ALPHA)*sm[2*w-2];
            if(y2<height) sm[3*w-1]+=__half(2.f*ALPHA)*sm[3*w-2];
        }
        __syncthreads();
        /* V81: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]    +=__half(2.f*BETA)*sm[1];
            if(y1<height) sm[w]  +=__half(2.f*BETA)*sm[w+1];
            if(y2<height) sm[2*w]+=__half(2.f*BETA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(BETA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(BETA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(BETA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(GAMMA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(GAMMA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(GAMMA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        /* V81: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            sm[w-1]  +=__half(2.f*GAMMA)*sm[w-2];
            if(y1<height) sm[2*w-1]+=__half(2.f*GAMMA)*sm[2*w-2];
            if(y2<height) sm[3*w-1]+=__half(2.f*GAMMA)*sm[3*w-2];
        }
        __syncthreads();
        /* V81: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            sm[0]    +=__half(2.f*DELTA)*sm[1];
            if(y1<height) sm[w]  +=__half(2.f*DELTA)*sm[w+1];
            if(y2<height) sm[2*w]+=__half(2.f*DELTA)*sm[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(DELTA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(DELTA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(DELTA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        __syncthreads();
        for (int x=t*2; x<w; x+=nt*2) {
            d_hout[y0*stride+x/2]             = sm[x]     *__half(NORM_L);
            if(y1<height) d_hout[y1*stride+x/2]   = sm[w+x]   *__half(NORM_L);
            if(y2<height) d_hout[y2*stride+x/2]   = sm[2*w+x] *__half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            d_hout[y0*stride+hw+x/2]             = sm[x]     *__half(NORM_H);
            if(y1<height) d_hout[y1*stride+hw+x/2]   = sm[w+x]   *__half(NORM_H);
            if(y2<height) d_hout[y2*stride+hw+x/2]   = sm[2*w+x] *__half(NORM_H);
        }
    } /* end V31 if/else */
#undef S17_U12_4R
}


/* V17: GPU colour conversion — RGB48LE → XYZ12 (identical to CUDA V18+). */
__global__ void s17_rgb48_to_xyz12(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V18f: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    int32_t* __restrict__ d_out_x,
    int32_t* __restrict__ d_out_y,
    int32_t* __restrict__ d_out_z,
    int width, int height, int rgb_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int px = i % width, py = i / width;
    int base = py * rgb_stride + px * 3;

    int ri = min((__ldg(&d_rgb16[base + 0]) >> 4), 4095);
    int gi = min((__ldg(&d_rgb16[base + 1]) >> 4), 4095);
    int bi = min((__ldg(&d_rgb16[base + 2]) >> 4), 4095);

    float r = __half2float(__ldg(&d_lut_in[ri]));  /* V18f */
    float g = __half2float(__ldg(&d_lut_in[gi]));
    float b = __half2float(__ldg(&d_lut_in[bi]));

    float xv = __ldg(&d_matrix[0])*r + __ldg(&d_matrix[1])*g + __ldg(&d_matrix[2])*b;
    float yv = __ldg(&d_matrix[3])*r + __ldg(&d_matrix[4])*g + __ldg(&d_matrix[5])*b;
    float zv = __ldg(&d_matrix[6])*r + __ldg(&d_matrix[7])*g + __ldg(&d_matrix[8])*b;

    xv = __saturatef(xv);
    yv = __saturatef(yv);
    zv = __saturatef(zv);

    d_out_x[i] = __ldg(&d_lut_out[(int)(xv * 4095.5f)]);
    d_out_y[i] = __ldg(&d_lut_out[(int)(yv * 4095.5f)]);
    d_out_z[i] = __ldg(&d_lut_out[(int)(zv * 4095.5f)]);
}


/* ===== GPU config auto-tuner ===== */
struct SlangGpuConfig17 {
    /* V17i: h_block=512 gives 100% SM occupancy for H-DWT kernels on sm_61.
     * 8KB smem/block: 256T smem-limited to 6 blk/SM (75%); 512T thread-limited to 4 blk/SM (100%).
     * V26: v_block_tiled=256 for large-h tiled V-DWT; v_block=128 kept for reg-blocked (h≤140). */
    int h_block = 512, v_block = 128, v_block_tiled = 256, gen_block = 256;
    bool configure(int device = 0) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, device) != cudaSuccess) return false;
        int ws = p.warpSize;
        /* h_block: use 512 threads for H-DWT (100% occupancy); cap at device max. */
        h_block       = (std::min(512, p.maxThreadsPerBlock) / ws) * ws;
        v_block       = (std::min(128, p.maxThreadsPerBlock) / ws) * ws;
        v_block_tiled = (std::min(256, p.maxThreadsPerBlock) / ws) * ws;
        gen_block     = (std::min(256, p.maxThreadsPerBlock) / ws) * ws;
        if (h_block       < ws) h_block       = ws;
        if (v_block       < ws) v_block       = ws;
        if (v_block_tiled < ws) v_block_tiled = ws;
        if (gen_block     < ws) gen_block     = ws;
        return true;
    }
};


/* ===== Encoder implementation struct ===== */
struct SlangJ2KEncoderImpl {
    /* DWT double-buffers, input, encoded output (pooled) */
    char*    d_pool     = nullptr;
    uint8_t* h_enc[2]   = {nullptr, nullptr};  /* V17r: double-buffered pinned download */

    __half*  d_c[3]   = {};  /* V17l: half DWT buffer A (V-DWT output)           */
    __half*  d_t[3]   = {};  /* V17n: half DWT buffer B (H-DWT output; was float) */
    __half*  d_half[3]= {};  /* fp16 workspace (V17b); nulled in V17l            */
    int32_t* d_in[3]  = {};  /* XYZ int32 input                          */
    uint8_t* d_enc[3] = {};  /* Quantized packed tier-1                  */

    /* V17: colour conversion buffers */
    uint16_t* d_rgb16[2]     = {nullptr, nullptr};  /* V17s: double-buffered GPU RGB */
    __half*   d_lut_in       = nullptr;  /* V18f: was float; 8KB vs 16KB GPU texture cache */
    uint16_t* d_lut_out      = nullptr;  /* V48: was int32_t; halves GPU LUT texture cache */
    float*    d_matrix       = nullptr;
    /* V17r: double-buffered pinned staging for 1-frame pipelining */
    uint16_t* h_rgb16_pinned[2] = {nullptr, nullptr};
    size_t    pinned_rgb_px  = 0;

    /* V18e: packed 12-bit planar buffers (3 channels × (w/2*3) bytes × h) */
    uint8_t*  d_rgb12[2]         = {nullptr, nullptr};
    uint8_t*  h_rgb12_pinned[2]  = {nullptr, nullptr};
    size_t    rgb12_px    = 0;
    size_t    pinned_rgb12_px = 0;

    size_t px = 0, enc_pc = 0, rgb_px = 0;
    bool colour_loaded = false;

    cudaStream_t st[3]     = {};
    SlangGpuConfig17 gpu;

    /* V17h: CUDA Graph cache for per-component DWT+quantize+D2H pipeline (XYZ fallback) */
    cudaGraphExec_t cg_exec[3]  = {nullptr, nullptr, nullptr};
    int    cg_width    = 0;
    int    cg_height   = 0;
    size_t cg_per_comp = 0;
    bool   cg_is_4k    = false;
    bool   cg_is_3d    = false;

    void destroy_comp_graphs() {
        for (int c = 0; c < 3; ++c) {
            if (cg_exec[c]) { cudaGraphExecDestroy(cg_exec[c]); cg_exec[c] = nullptr; }
        }
        cg_width = cg_height = 0; cg_per_comp = 0;
    }

    /* V17s: dedicated H2D stream + per-buf H2D completion events.
     * st_h2d runs on PCIe DMA engine independently of SM compute on st[0..2].
     * H2D for frame N overlaps with SM compute for frame N-1 → ~578fps. */
    cudaStream_t  st_h2d       = nullptr;
    cudaEvent_t   h2d_done[2]  = {nullptr, nullptr};

    /* V17s: per-buf per-channel comp graphs (DWT levels 1-4 + Q + D2H).
     * cg_v17s[buf][c] differs from cg_v17s[1-buf][c] only in D2H dest (h_enc[buf]).
     * d_c[c]/d_t[c] are shared (sequential execution ensures no conflict). */
    cudaGraphExec_t cg_v17s[2][3] = {{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr}};
    int    cg_v17s_width[2]      = {0, 0};
    int    cg_v17s_height[2]     = {0, 0};
    int    cg_v17s_rgb_stride[2] = {0, 0};  /* V17u: rgb_stride baked into graph */
    size_t cg_v17s_per_comp[2]   = {0, 0};
    bool   cg_v17s_is_4k[2]      = {false, false};
    bool   cg_v17s_is_3d[2]      = {false, false};

    /* V17s: 1-frame pipeline state */
    int    cur_buf         = 0;
    bool   pipeline_active = false;
    int    p_width         = 0;
    int    p_height        = 0;
    size_t p_per_comp      = 0;
    bool   p_is_4k         = false;
    bool   p_is_3d         = false;

    void destroy_v17s_graphs() {
        for (int i = 0; i < 2; ++i)
            for (int c = 0; c < 3; ++c)
                if (cg_v17s[i][c]) { cudaGraphExecDestroy(cg_v17s[i][c]); cg_v17s[i][c] = nullptr; }
        pipeline_active = false;
    }

    bool ensure_pinned_rgb(int w, int h) {
        size_t n = size_t(w) * h;
        if (n <= pinned_rgb_px) return true;
        for (int i = 0; i < 2; ++i) {
            if (h_rgb16_pinned[i]) { cudaFreeHost(h_rgb16_pinned[i]); h_rgb16_pinned[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i) {
            if (cudaHostAlloc(&h_rgb16_pinned[i], n * 3 * sizeof(uint16_t),
                              cudaHostAllocDefault) != cudaSuccess) {
                if (i == 1 && h_rgb16_pinned[0]) { cudaFreeHost(h_rgb16_pinned[0]); h_rgb16_pinned[0] = nullptr; }
                return false;
            }
        }
        pinned_rgb_px = n;
        destroy_v17s_graphs();  /* graphs bake old pointer; force rebuild */
        return true;
    }

    bool init() {
        if (!gpu.configure(0)) return false;
        for (int i = 0; i < 3; ++i)
            if (cudaStreamCreate(&st[i]) != cudaSuccess) return false;
        if (cudaStreamCreate(&st_h2d) != cudaSuccess) return false;
        for (int i = 0; i < 2; ++i)
            if (cudaEventCreateWithFlags(&h2d_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        /* V22: prefer L1 cache for memory-BW-bound kernels (parity with CUDA V59).
         * V-DWT tiled: no smem — 48KB L1 enables adjacent tile row sharing.
         * 1-row RGB+HDWT0: smem=3.84KB → PreferL1 → 4 blk/SM = 100% occ. */
        cudaFuncSetCacheConfig(s17_dwt_v_tiled,          cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(s17_dwt_v_tiled_ho,       cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(s17_dwt_v_tiled_ho_2col,  cudaFuncCachePreferL1);
        /* V48: reg-blocked V-DWT + quantize have no smem → PreferL1 is free (no occ cost).
         * Larger L1 benefits reg-blocked V-DWT (column data locality) and
         * quantize (instruction cache for multi-zone branching). */
        cudaFuncSetCacheConfig(s17_dwt_v_fp16_hi_reg_ho, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(s17_qe_subband_ml,         cudaFuncCachePreferL1);
        /* V58: 4-row p12: smem = 15.36KB (DWT) + 8KB (sm_lut static) = 23.36KB.
         * PreferShared (48KB/SM): 48/23.36=2 blk/SM → 50% occ (same as PreferNone before).
         * sm_lut gives 0-cycle LUT access; smaller L1 (16KB) is acceptable tradeoff. */
        cudaFuncSetCacheConfig(s17_rgb48_xyz_hdwt0_1ch_4row_p12, cudaFuncCachePreferShared);
        /* V53: 2-row p12: smem = 7.68KB (DWT) + 8KB (sm_lut static) = 15.87KB.
         * PreferShared (48KB/SM) → 3 blk/SM; 0-cycle sm_lut access >> -1 block occupancy cost. */
        cudaFuncSetCacheConfig(s17_rgb48_xyz_hdwt0_1ch_2row_p12, cudaFuncCachePreferShared);
        /* V53: 1-row p12: smem = 7.68KB (DWT) + 8KB (sm_lut static) = 15.87KB → same as 2-row.
         * PreferShared (48KB/SM) → 3 blk/SM; 0-cycle sm_lut LUT access wins over PreferNone 4 blk/SM. */
        cudaFuncSetCacheConfig(s17_rgb48_xyz_hdwt0_1ch_1row_p12, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(s17_rgb48_xyz_hdwt0_1ch,          cudaFuncCachePreferL1);
        /* V37: level-0 4-row kernel: smem=4×w×2B, no LUTs → PreferShared (48KB smem) for 3 blk/SM.
         * PreferL1 would limit to 16KB smem → 1 blk/SM at 2K (15.36KB, 25% occ) — avoid. */
        cudaFuncSetCacheConfig(s17_fused_i2f_dwt_h_ho_4row, cudaFuncCachePreferShared);
        /* V40: 4-row H-DWT: level-1 smem=7.68KB; PreferL1→16KB smem/SM→2 blk/SM (50% occ).
         * PreferNone→32KB smem/SM→4 blk/SM=100% occ at level 1 (the largest, bottleneck level).
         * Levels 2-5 (smem≤3.84KB): 4 blk/SM with both → PreferNone doesn't hurt them.
         * 2-row/1-row: smem≤3.84KB at all levels → 4 blk/SM with PreferL1 → keep PreferL1. */
        cudaFuncSetCacheConfig(s17_dwt_h_half_io_4row,  cudaFuncCachePreferNone);
        cudaFuncSetCacheConfig(s17_dwt_h_half_io_2row,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(s17_dwt_h_half_io,       cudaFuncCachePreferL1);
        return true;
    }

    bool ensure(int w, int h, size_t epc) {
        size_t n = size_t(w) * h;
        if (n <= px && epc <= enc_pc) return true;
        cleanup_pool();
        /* V17n: d_c and d_t are both __half (2B/px each); d_half removed.
         * Pool layout per component: __half d_c + __half d_t + int32 d_in + uint8 d_enc */
        size_t per = n * sizeof(__half) + n * sizeof(__half) + n * sizeof(int32_t) + epc;
        if (cudaMalloc(&d_pool, per * 3) != cudaSuccess) return false;
        char* p = d_pool;
        for (int c = 0; c < 3; ++c) {
            d_c[c]    = reinterpret_cast<__half*>(p);   p += n * sizeof(__half);
            d_t[c]    = reinterpret_cast<__half*>(p);   p += n * sizeof(__half);
            d_half[c] = nullptr;   /* V17l: removed; V17n: d_t also now __half */
            d_in[c]   = reinterpret_cast<int32_t*>(p);  p += n * sizeof(int32_t);
            d_enc[c]  = reinterpret_cast<uint8_t*>(p);  p += epc;
        }
        for (int i = 0; i < 2; ++i) {
            if (cudaHostAlloc(&h_enc[i], 3 * epc, cudaHostAllocDefault) != cudaSuccess) {
                cudaFree(d_pool); d_pool = nullptr;
                if (i == 1 && h_enc[0]) { cudaFreeHost(h_enc[0]); h_enc[0] = nullptr; }
                return false;
            }
        }
        px = n; enc_pc = epc;
        return true;
    }

    bool ensure_rgb(int w, int h) {
        size_t n = size_t(w) * h;
        if (n <= rgb_px) return true;
        for (int i = 0; i < 2; ++i) {
            if (d_rgb16[i]) { cudaFree(d_rgb16[i]); d_rgb16[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i)
            if (cudaMalloc(&d_rgb16[i], n * 3 * sizeof(uint16_t)) != cudaSuccess) return false;
        rgb_px = n;
        ensure_rgb12(w, h);  /* V18e: also allocate packed 12-bit buffers */
        return true;
    }

    /* V18e: packed 12-bit planar GPU buffers. */
    bool ensure_rgb12(int w, int h) {
        size_t n = size_t(w) * h;
        if (n <= rgb12_px) return true;
        size_t packed = size_t((w/2)*3) * h * 3;
        for (int i = 0; i < 2; ++i) {
            if (d_rgb12[i]) { cudaFree(d_rgb12[i]); d_rgb12[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i)
            if (cudaMalloc(&d_rgb12[i], packed) != cudaSuccess) return false;
        rgb12_px = n;
        ensure_pinned_rgb12(w, h);
        return true;
    }

    bool ensure_pinned_rgb12(int w, int h) {
        size_t n = size_t(w) * h;
        if (n <= pinned_rgb12_px) return true;
        size_t packed = size_t((w/2)*3) * h * 3;
        for (int i = 0; i < 2; ++i) {
            if (h_rgb12_pinned[i]) { cudaFreeHost(h_rgb12_pinned[i]); h_rgb12_pinned[i] = nullptr; }
        }
        destroy_v17s_graphs();
        for (int i = 0; i < 2; ++i)
            /* V43: write-combining — CPU only writes (packs) into this buffer, GPU reads via DMA.
             * Eliminates PCIe snooping overhead → up to 40% H2D bandwidth gain. */
            if (cudaHostAlloc(&h_rgb12_pinned[i], packed, cudaHostAllocWriteCombined) != cudaSuccess) return false;
        pinned_rgb12_px = n;
        return true;
    }

    bool upload_colour(GpuColourParams const& p) {
        if (!d_lut_in)  { if (cudaMalloc(&d_lut_in,  4096*sizeof(__half))   != cudaSuccess) return false; }  /* V18f: was float */
        if (!d_lut_out) { if (cudaMalloc(&d_lut_out, 4096*sizeof(uint16_t)) != cudaSuccess) return false; }  /* V48 */
        if (!d_matrix)  { if (cudaMalloc(&d_matrix,  9*sizeof(float))       != cudaSuccess) return false; }
        /* V18f: convert float→__half before upload; host array stays float */
        __half h_lut_in_tmp[4096];
        for (int i = 0; i < 4096; ++i) h_lut_in_tmp[i] = __float2half(p.lut_in[i]);
        cudaMemcpy(d_lut_in,  h_lut_in_tmp, 4096*sizeof(__half),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_out, p.lut_out, 4096*sizeof(uint16_t), cudaMemcpyHostToDevice);  /* V48 */
        cudaMemcpy(d_matrix,  p.matrix,  9*sizeof(float),       cudaMemcpyHostToDevice);
        colour_loaded = true;
        return true;
    }

    void cleanup_pool() {
        if (d_pool) { cudaFree(d_pool); d_pool = nullptr; }
        for (int i = 0; i < 2; ++i) if (h_enc[i]) { cudaFreeHost(h_enc[i]); h_enc[i] = nullptr; }
        px = enc_pc = 0;
    }

    ~SlangJ2KEncoderImpl() {
        destroy_v17s_graphs();
        destroy_comp_graphs();
        cleanup_pool();
        for (int i = 0; i < 2; ++i) {
            if (d_rgb16[i])        { cudaFree(d_rgb16[i]);            d_rgb16[i]        = nullptr; }
            if (d_rgb12[i])        { cudaFree(d_rgb12[i]);            d_rgb12[i]        = nullptr; }
            if (h_rgb16_pinned[i]) { cudaFreeHost(h_rgb16_pinned[i]); h_rgb16_pinned[i] = nullptr; }
            if (h_rgb12_pinned[i]) { cudaFreeHost(h_rgb12_pinned[i]); h_rgb12_pinned[i] = nullptr; }
            if (h2d_done[i])       { cudaEventDestroy(h2d_done[i]);   h2d_done[i]       = nullptr; }
        }
        if (d_lut_in)  cudaFree(d_lut_in);
        if (d_lut_out) cudaFree(d_lut_out);
        if (d_matrix)  cudaFree(d_matrix);
        for (int i = 0; i < 3; ++i) if (st[i]) cudaStreamDestroy(st[i]);
        if (st_h2d) cudaStreamDestroy(st_h2d);
    }
};


/* ===== Shared helper ===== */

static size_t s17_per_comp(int64_t bit_rate, int fps, bool is_3d, size_t max_px)
{
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t target_bytes = std::max(static_cast<size_t>(frame_bits / 8),
                                    static_cast<size_t>(16384));
    return std::min(std::max(target_bytes / 3, static_cast<size_t>(1)), max_px);
}

/* V17h: Extract per-component DWT+quantize+D2H kernel launches for graph capture.
 * V17r: h_enc_dest selects which double-buffer slot receives the D2H output. */
static void
s17_launch_comp_pipeline(
    SlangJ2KEncoderImpl* impl,
    int c, int width, int height,
    size_t per_comp, float step,
    bool skip_l0_hdwt, cudaStream_t s,
    uint8_t* h_enc_dest,  /* V17r: D2H destination (impl->h_enc[0] or [1]) */
    int num_levels = NUM_DWT_LEVELS)  /* V17z: 5 for 2K, 6 for 4K */
{
    const int h_blk = impl->gpu.h_block;
    const int v_blk = impl->gpu.v_block;           /* 128T: reg-blocked V-DWT (h≤MAX_REG_HEIGHT) */
    const int vt_blk = impl->gpu.v_block_tiled;    /* V26: 256T: tiled V-DWT (h>MAX_REG_HEIGHT) */
    const int gen_blk = impl->gpu.gen_block;
    const int orig_height = height;  /* V18a: save before DWT loop updates h */
    const int stride = width;        /* V18a: original row pitch of d_c[c] */

    /* V17n: A is __half* (d_c), Bh is __half* (d_t — was float*, now __half*). */
    __half* A  = impl->d_c[c];
    __half* Bh = impl->d_t[c];
    int w = width, h = height;

    if (skip_l0_hdwt) {
        /* Bh pre-populated by fused RGB kernel; V-DWT writes half A. */
        if (h <= MAX_REG_HEIGHT)
            s17_dwt_v_fp16_hi_reg_ho<<<(w+v_blk-1)/v_blk, v_blk, 0, s>>>(Bh, A, w, h, width);
        else {
            /* V26: 256T for tiled V-DWT (vs 128T reg-blocked): 8 warps/block, better latency hiding */
            /* V39: 2-col kernel — grid x = ceil(w/2/vt_blk); HFMA2 doubles lifting throughput. */
            dim3 vg2d((w/2+vt_blk-1)/vt_blk, (h+V_TILE-1)/V_TILE);
            s17_dwt_v_tiled_ho_2col<<<vg2d, vt_blk, 0, s>>>(Bh, A, w, h, width);
        }
    } else {
        /* V17n: all paths use int32→half H-DWT + half V-DWT. V17o: __half smem.
         * V37: 4-row level-0 kernel — 75% fewer block dispatches (1080→270 for 2K).
         *   4 rows' i2f chains in-flight; __syncthreads amortized over 4× more work. */
        s17_fused_i2f_dwt_h_ho_4row<<<(h+3)/4, h_blk, (size_t)4*w*sizeof(__half), s>>>(
            impl->d_in[c], Bh, w, h, width);
        if (h <= MAX_REG_HEIGHT)
            s17_dwt_v_fp16_hi_reg_ho<<<(w+v_blk-1)/v_blk, v_blk, 0, s>>>(Bh, A, w, h, width);
        else {
            /* V39: 2-col kernel — grid x = ceil(w/2/vt_blk); HFMA2 doubles lifting throughput. */
            dim3 vg2d((w/2+vt_blk-1)/vt_blk, (h+V_TILE-1)/V_TILE);
            s17_dwt_v_tiled_ho_2col<<<vg2d, vt_blk, 0, s>>>(Bh, A, w, h, width);
        }
    }
    w = (w+1)/2; h = (h+1)/2;

    for (int lv = 1; lv < num_levels; ++lv) {  /* V17z: num_levels=6 for 4K, 5 for 2K */
        /* V17x: 4-rows-per-block for levels 1+; grid quartered vs V17v, halved vs V17w.
         * V36: adaptive thread count — fixed h_blk=512 wastes 50-77% threads at small w.
         *   level-3 2K (w=240): 512T→256T (util 46.9%→93.75%); level-4 (w=120): →128T.
         *   4K level-3 (w=480): →256T; level-4 (w=240): 256T; level-5 (w=120): 128T. */
        int h_dwt_blk = (w > 480) ? h_blk :
                        (w > 240) ? 256 :
                        (w > 120) ? 128 : 64;
        s17_dwt_h_half_io_4row<<<(h+3)/4, h_dwt_blk, (size_t)4*w*sizeof(__half), s>>>(A, Bh, w, h, width);
        if (h <= MAX_REG_HEIGHT)
            s17_dwt_v_fp16_hi_reg_ho<<<(w+v_blk-1)/v_blk, v_blk, 0, s>>>(Bh, A, w, h, width);
        else {
            /* V39: 2-col kernel — grid x = ceil(w/2/vt_blk); HFMA2 doubles lifting throughput. */
            dim3 vg2d((w/2+vt_blk-1)/vt_blk, (h+V_TILE-1)/V_TILE);
            s17_dwt_v_tiled_ho_2col<<<vg2d, vt_blk, 0, s>>>(Bh, A, w, h, width);
        }
        w = (w+1)/2; h = (h+1)/2;
    }

    /* V18d: 6-band perceptual quantize — distinct step per DWT level (parity with CUDA V53).
     * ll5_h = ceil(orig_height/32), ll5_c = stride >> 5. */
    int n_rows = std::min(static_cast<int>((per_comp + stride - 1) / stride), orig_height);
    int ll5_h  = (orig_height + 31) / 32;
    int ll5_c  = stride >> 5;
    s17_qe_subband_ml<<<n_rows, 256, 0, s>>>(A, impl->d_enc[c], stride, n_rows, step, ll5_h, ll5_c);
    cudaMemcpyAsync(h_enc_dest + c * per_comp, impl->d_enc[c],
                    per_comp, cudaMemcpyDeviceToHost, s);  /* V17r: dest buf */
}


/* V17h: Capture per-component CUDA Graphs for DWT+quantize+D2H pipeline. */
static void
s17_rebuild_comp_graphs(
    SlangJ2KEncoderImpl* impl,
    int width, int height, size_t per_comp,
    bool is_4k, bool is_3d, bool skip_l0_hdwt)
{
    impl->destroy_comp_graphs();
    float base_step = is_4k ? 16.25f : 32.5f;
    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;  /* V17z: 6-level DWT for 4K */
    const float steps[3] = { base_step * 1.1f, base_step * 1.0f, base_step * 1.1f };
    for (int c = 0; c < 3; ++c) {
        cudaGraph_t g;
        cudaStreamBeginCapture(impl->st[c], cudaStreamCaptureModeThreadLocal);
        s17_launch_comp_pipeline(impl, c, width, height, per_comp,
                                  steps[c], skip_l0_hdwt, impl->st[c],
                                  impl->h_enc[0],  /* V17r: comp graphs always write buf 0 */
                                  num_levels);  /* V17z */
        cudaStreamEndCapture(impl->st[c], &g);
        cudaGraphInstantiate(&impl->cg_exec[c], g, nullptr, nullptr, 0);
        cudaGraphDestroy(g);
    }
    impl->cg_width    = width;
    impl->cg_height   = height;
    impl->cg_per_comp = per_comp;
    impl->cg_is_4k    = is_4k;
    impl->cg_is_3d    = is_3d;
}


/**
 * V17s: Build per-buf per-channel comp graphs for DWT+Q+D2H pipeline.
 * cg_v17s[buf][c] captures DWT levels 1-4 + quantize + D2H to h_enc[buf].
 * H-DWT level 0 is NOT captured (done by s17_rgb48_xyz_hdwt0_1ch before graph launch).
 * d_c[c]/d_t[c] intermediates are shared between buf=0 and buf=1 (safe: sequential).
 */
static void
s17_rebuild_v17s_comp_graphs(
    SlangJ2KEncoderImpl* impl,
    int buf,
    int width, int height, int rgb_stride_pixels,
    size_t per_comp, bool is_4k, bool is_3d)
{
    for (int c = 0; c < 3; ++c) {
        if (impl->cg_v17s[buf][c]) {
            cudaGraphExecDestroy(impl->cg_v17s[buf][c]);
            impl->cg_v17s[buf][c] = nullptr;
        }
    }
    float base_step = is_4k ? 16.25f : 32.5f;
    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;  /* V17z: 6-level DWT for 4K */
    /* V24: 4-row kernel: grid=(height+3)/4, smem=4*width*sizeof(__half).
     * V35: Conditionally use 2-row at 4K (width>2048) to enable PreferL1 (48KB L1).
     *   4K 4-row smem=30.72KB > 16KB PreferL1 limit → runtime uses PreferShared (16KB L1).
     *   4K 2-row smem=15.36KB < 16KB → PreferL1 honored: lut_in+lut_out (16KB) in 48KB L1. */
    size_t ch_smem_4row = static_cast<size_t>(4 * width) * sizeof(__half);
    size_t ch_smem_2row = static_cast<size_t>(2 * width) * sizeof(__half);
    size_t ch_smem_1row = static_cast<size_t>(1 * width) * sizeof(__half);
    int rgb_grid = (height + 3) / 4;
    int rgb_grid_2row = (height + 1) / 2;
    int rgb_grid_1row = height;
    /* V41: Use 2-row for 2K (100% occ: smem=7.68KB, PreferNone→4 blk/SM).
     * V42: Use 1-row for 4K (100% occ: smem=7.68KB@4K, PreferNone→4 blk/SM).
     * 4K 2-row smem=15.36KB → 2 blk/SM=50% occ; 1-row smem=7.68KB → 4 blk/SM=100% occ. */
    bool use_1row_4k = (width > 2048);  /* V42: 4K uses 1-row for 100% occ */
    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.1f);
        cudaGraph_t g;
        cudaStreamBeginCapture(impl->st[c], cudaStreamCaptureModeThreadLocal);
        /* V17u: event wait baked into graph — waits for h2d_done[buf] before RGB kernel */
        cudaStreamWaitEvent(impl->st[c], impl->h2d_done[buf], 0);
        int s18e_prs = (width / 2) * 3;  /* packed_row_stride: bytes per channel per row */
        if (use_1row_4k) {
            /* V42: 4K — 1-row: smem=7.68KB, PreferNone→4 blk/SM=100% occ (vs 2-row's 50%). */
            s17_rgb48_xyz_hdwt0_1ch_1row_p12<<<rgb_grid_1row, impl->gpu.h_block, ch_smem_1row, impl->st[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_t[c], c,
                width, height, s18e_prs, width);
        } else {
            /* V41: 2K — 2-row: smem=7.68KB, PreferNone→4 blk/SM=100% occ. */
            s17_rgb48_xyz_hdwt0_1ch_2row_p12<<<rgb_grid_2row, impl->gpu.h_block, ch_smem_2row, impl->st[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_t[c], c,
                width, height, s18e_prs, width);
        }
        /* DWT levels 1+ + quantize + D2H; skip_l0_hdwt=true */
        s17_launch_comp_pipeline(impl, c, width, height, per_comp, step,
                                  /* skip_l0_hdwt= */ true, impl->st[c],
                                  impl->h_enc[buf],
                                  num_levels);  /* V17z */
        cudaStreamEndCapture(impl->st[c], &g);
        cudaGraphInstantiate(&impl->cg_v17s[buf][c], g, nullptr, nullptr, 0);
        cudaGraphDestroy(g);
    }
    impl->cg_v17s_width[buf]      = width;
    impl->cg_v17s_height[buf]     = height;
    impl->cg_v17s_rgb_stride[buf] = rgb_stride_pixels;
    impl->cg_v17s_per_comp[buf]   = per_comp;
    impl->cg_v17s_is_4k[buf]      = is_4k;
    impl->cg_v17s_is_3d[buf]      = is_3d;
}


/**
 * V18b: Find the actual number of meaningful bytes in a packed component buffer.
 * Scans backward in 8-byte words to skip trailing zeros quickly.
 * Returns at least 1 even for an all-zero component.
 */
static size_t
s17_find_actual_per_comp(const uint8_t* data, size_t n)
{
    size_t end = n;
    while (end >= 8) {
        uint64_t w;
        std::memcpy(&w, data + end - 8, 8);
        if (w != 0) break;
        end -= 8;
    }
    while (end > 0 && data[end - 1] == 0) --end;
    return std::max(end, static_cast<size_t>(1));
}


/* V20: Per-subband QCD step helpers (parity with CUDA V57).
 *
 * Encode a floating-point quantization step into a J2K scalar-expounded entry:
 *   entry = (eps << 11) | man,  where
 *   eps = 13 - floor(log2(step)),  man = floor((step/2^(13-eps) - 1) * 2048)
 *   decoder reconstructs: step = 2^(13-eps) × (1 + man/2048)
 *
 * j2k_perceptual_sb_entry applies V18d level-dependent weights so the QCD
 * entries match the actual steps used in s17_qe_subband_ml. */
static uint16_t s17_qcd_step_entry(float step)
{
    int log2s = static_cast<int>(std::floor(std::log2(step)));
    int eps   = 13 - log2s;
    float denom = std::ldexp(1.0f, 13 - eps);
    int   man   = static_cast<int>((step / denom - 1.0f) * 2048.0f);
    man = std::max(0, std::min(2047, man));
    return static_cast<uint16_t>((eps << 11) | man);
}

static uint16_t s17_perceptual_sb_entry(float base_step, int sb_idx, bool is_4k)
{
    if (is_4k) return s17_qcd_step_entry(base_step);  /* 4K: uniform */
    static const float kWeights[16] = {
        0.65f,
        0.85f, 0.85f, 0.85f,
        0.95f, 0.95f, 0.95f,
        1.05f, 1.05f, 1.05f,
        1.12f, 1.12f, 1.12f,
        1.20f, 1.20f, 1.20f
    };
    float w = (sb_idx < 16) ? kWeights[sb_idx] : 1.0f;
    return s17_qcd_step_entry(base_step * w);
}


/**
 * V17m: Build DCI-compliant J2K codestream from quantized tier-1 data already in h_enc.
 * V17r: h_enc_src selects which double-buffer slot holds the completed D2H download.
 */
static std::vector<uint8_t>
s17_build_j2k_codestream(SlangJ2KEncoderImpl* impl,
                           int width, int height,
                           size_t per_comp, bool is_4k, bool is_3d,
                           uint8_t* h_enc_src)  /* V17r: source buffer (h_enc[0] or [1]) */
{
    (void)is_3d;
    bool fourk = (width > 2048) || is_4k;
    int  num_precincts = fourk ? 7 : NUM_DWT_LEVELS + 1;

    std::vector<uint8_t> cs;
    cs.reserve(256 + 3 * (14 + per_comp));

    auto w8  = [&](uint8_t  v){ cs.push_back(v); };
    auto w16 = [&](uint16_t v){ w8(uint8_t(v>>8)); w8(uint8_t(v&0xFF)); };
    auto w32 = [&](uint32_t v){ w16(uint16_t(v>>16)); w16(uint16_t(v&0xFFFF)); };

    w16(J2K_SOC);

    /* SIZ */
    w16(J2K_SIZ);
    w16(2 + 2 + 32 + 2 + 3*3);
    w16(fourk ? uint16_t(0x0004) : uint16_t(0x0003));
    w32(width);  w32(height);
    w32(0);      w32(0);
    w32(width);  w32(height);
    w32(0);      w32(0);
    w16(3);
    for (int c = 0; c < 3; ++c) { w8(11); w8(1); w8(1); }

    /* COD */
    w16(J2K_COD);
    w16(static_cast<uint16_t>(2 + 1 + 4 + 5 + num_precincts));
    w8(0x01);  w8(0x04);  w16(1);  w8(1);
    w8(static_cast<uint8_t>(fourk ? 6 : NUM_DWT_LEVELS));
    w8(3);  w8(3);  w8(0x00);  w8(0x00);
    w8(0x77);
    for (int i = 1; i < num_precincts; ++i) w8(0x88);

    /* QCD — V20: per-subband step matching V18d perceptual weights.
     * V18d applied level-dependent weights (0.65×LL5 … 1.20×L1).
     * QCD must report the actual step per subband so the decoder
     * dequantizes at the correct amplitude.
     * 4K uses uniform steps (no perceptual weights in 4K path). */
    {
        int nsb = 3 * (fourk ? 6 : NUM_DWT_LEVELS) + 1;  /* V17z: 19 for 4K, 16 for 2K */
        w16(J2K_QCD);
        w16(static_cast<uint16_t>(2 + 1 + 2*nsb));
        w8(0x22);
        float base_y = fourk ? 16.25f : 32.5f;
        for (int i = 0; i < nsb; ++i)
            w16(s17_perceptual_sb_entry(base_y, i, fourk));
    }
    /* QCC — V20: per-subband step overrides for X (comp 0) and Z (comp 2).
     * X/Z components use 1.1× coarser quantization than Y. */
    {
        int nsb = 3 * (fourk ? 6 : NUM_DWT_LEVELS) + 1;  /* V17z: 19 for 4K, 16 for 2K */
        uint16_t lqcc = static_cast<uint16_t>(4 + 2*nsb);
        float base_xz = (fourk ? 16.25f : 32.5f) * 1.1f;
        for (int c : {0, 2}) {
            w16(J2K_QCC);
            w16(lqcc);
            w8(static_cast<uint8_t>(c));
            w8(0x22);
            for (int i = 0; i < nsb; ++i)
                w16(s17_perceptual_sb_entry(base_xz, i, fourk));
        }
    }

    /* V18b: Compute actual per-component sizes (trim trailing zeros). */
    size_t actual[3];
    for (int c = 0; c < 3; ++c)
        actual[c] = s17_find_actual_per_comp(h_enc_src + c * per_comp, per_comp);

    /* TLM */
    w16(J2K_TLM);
    w16(static_cast<uint16_t>(2 + 1 + 1 + 3*4));
    w8(0);  w8(0x40);
    for (int c = 0; c < 3; ++c) w32(static_cast<uint32_t>(14 + actual[c]));  /* V18b: actual */

    /* 3 SOT/SOD tile parts */
    for (int c = 0; c < 3; ++c) {
        w16(J2K_SOT);  w16(10);  w16(0);
        w32(static_cast<uint32_t>(12 + actual[c]));  /* V18b: actual Psot */
        w8(static_cast<uint8_t>(c));  w8(3);
        w16(J2K_SOD);
        cs.insert(cs.end(),
                  h_enc_src + c * per_comp,
                  h_enc_src + c * per_comp + actual[c]);  /* V18b: actual bytes */
    }

    while (cs.size() < 16384) cs.push_back(0);
    w16(J2K_EOC);
    return cs;
}


/* Run DWT on d_in[0..2] and build J2K codestream.
 * d_in must already be populated (by either upload or colour conversion kernel). */
static std::vector<uint8_t>
s17_run_dwt_and_build(SlangJ2KEncoderImpl* impl,
                       int width, int height,
                       bool is_3d, bool is_4k,
                       size_t per_comp,
                       bool skip_l0_hdwt = false)  /* V17e: true when fused RGB kernel ran */
{
    /* V17h: CUDA Graph dispatch for DWT+quantize+D2H pipeline.
     * Per-component graphs are captured on first call and reused each frame.
     * cudaGraphLaunch eliminates ~30 kernel-launch overheads per frame (~0.3ms). */
    {
        bool graphs_valid = (impl->cg_exec[0] && impl->cg_exec[1] && impl->cg_exec[2] &&
                             impl->cg_width   == width   && impl->cg_height  == height  &&
                             impl->cg_per_comp == per_comp && impl->cg_is_4k == is_4k  &&
                             impl->cg_is_3d   == is_3d);
        if (!graphs_valid)
            s17_rebuild_comp_graphs(impl, width, height, per_comp,
                                     is_4k, is_3d, skip_l0_hdwt);
        for (int c = 0; c < 3; ++c)
            cudaGraphLaunch(impl->cg_exec[c], impl->st[c]);
    }
    for (int c = 0; c < 3; ++c) cudaStreamSynchronize(impl->st[c]);

    return s17_build_j2k_codestream(impl, width, height, per_comp, is_4k, is_3d,
                                    impl->h_enc[0]);  /* V17r: comp graphs always use buf 0 */
}


/* ===== Public API ===== */

SlangJ2KEncoder::SlangJ2KEncoder()
    : _impl(std::make_unique<SlangJ2KEncoderImpl>())
{
    _initialized = _impl->init();
}

SlangJ2KEncoder::~SlangJ2KEncoder() = default;


std::vector<uint8_t>
SlangJ2KEncoder::encode(
    const int32_t* const xyz[3],
    int width, int height,
    int64_t bit_rate, int fps,
    bool is_3d, bool is_4k)
{
    if (!_initialized) return {};

    size_t pixels   = static_cast<size_t>(width) * height;
    size_t per_comp = s17_per_comp(bit_rate, fps, is_3d, pixels);

    if (!_impl->ensure(width, height, per_comp)) return {};

    for (int c = 0; c < 3; ++c)
        cudaMemcpyAsync(_impl->d_in[c], xyz[c],
                        pixels * sizeof(int32_t),
                        cudaMemcpyHostToDevice, _impl->st[c]);

    return s17_run_dwt_and_build(_impl.get(), width, height,
                                  is_3d, is_4k, per_comp);
}


/* V18e: Pack RGB48LE interleaved into 12-bit planar format (parity with CUDA V54).
 * Identical logic to pack_rgb12_chunk in cuda_j2k_encoder.cu. */
static void
pack_rgb12_chunk(const uint16_t* src, int src_stride,
                 uint8_t* dst, int width, int packed_row_stride,
                 int y_start, int y_end, int height)
{
    for (int y = y_start; y < y_end; ++y) {
        const uint16_t* row = src + (size_t)y * src_stride;
        for (int ch = 0; ch < 3; ++ch) {
            uint8_t* dst_row = dst + (size_t)ch * height * packed_row_stride
                                   + (size_t)y * packed_row_stride;
            for (int px = 0; px < width; px += 2) {
                int A = (row[px*3 + ch] >> 4) & 0xFFF;
                int B = (row[(px+1)*3 + ch] >> 4) & 0xFFF;
                dst_row[(px/2)*3 + 0] = static_cast<uint8_t>(A >> 4);
                dst_row[(px/2)*3 + 1] = static_cast<uint8_t>(((A & 0xF) << 4) | (B >> 8));
                dst_row[(px/2)*3 + 2] = static_cast<uint8_t>(B & 0xFF);
            }
        }
    }
}

/* V19: Single-channel pack variant for per-channel H2D pipeline (parity with CUDA V56).
 * Packs one colour plane into a pre-allocated contiguous plane buffer. */
static void
pack_rgb12_plane(const uint16_t* src, int src_stride,
                 uint8_t* dst_plane, int width, int packed_row_stride,
                 int y_start, int y_end, int ch)
{
    for (int y = y_start; y < y_end; ++y) {
        const uint16_t* row = src + (size_t)y * src_stride;
        uint8_t* dst_row = dst_plane + (size_t)y * packed_row_stride;
        for (int px = 0; px < width; px += 2) {
            int A = (row[px*3 + ch] >> 4) & 0xFFF;
            int B = (row[(px+1)*3 + ch] >> 4) & 0xFFF;
            dst_row[(px/2)*3 + 0] = static_cast<uint8_t>(A >> 4);
            dst_row[(px/2)*3 + 1] = static_cast<uint8_t>(((A & 0xF) << 4) | (B >> 8));
            dst_row[(px/2)*3 + 2] = static_cast<uint8_t>(B & 0xFF);
        }
    }
}


std::vector<uint8_t>
SlangJ2KEncoder::encode_from_rgb48(
    const uint16_t* rgb16,
    int width, int height,
    int rgb_stride_pixels,
    int64_t bit_rate, int fps,
    bool is_3d, bool is_4k)
{
    if (!_initialized || !_colour_params_valid) return {};

    size_t pixels    = static_cast<size_t>(width) * height;
    size_t per_comp  = s17_per_comp(bit_rate, fps, is_3d, pixels);
    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);

    if (!_impl->ensure(width, height, per_comp)) return {};
    if (!_impl->ensure_rgb(width, height))        return {};
    if (!_impl->ensure_pinned_rgb(width, height)) return {};

    /* V17s: 1-frame pipeline with H2D-compute overlap.
     * new_buf: write slot for this frame (RGB staging + D2H destination).
     * cur_buf: previous frame's slot (GPU executing or finished compute).
     * H2D(new_buf) on st_h2d (PCIe) overlaps with SM compute(cur_buf) on st[0..2].
     * Steady-state: H2D(1.66ms) + sync_tail(0.5ms) ≈ 1.73ms/frame → ~578fps. */
    int new_buf = 1 - _impl->cur_buf;

    /* V19: Merged Steps 1+2 — per-channel pack+H2D pipeline (parity with CUDA V56).
     * Pack one colour plane (4 threads), then H2D that plane before packing the next.
     * H2D for ch0 starts at ~0.075ms; total pack+H2D: 0.075+1.26=1.335ms (was 1.56ms). */
    _impl->ensure_rgb12(width, height);
    {
        static constexpr int N_PACK = 4;
        const int    prs         = (width / 2) * 3;  /* packed_row_stride */
        const size_t plane_bytes = static_cast<size_t>(prs) * height;
        const int    chunk_rows  = (height + N_PACK - 1) / N_PACK;
        uint8_t*     dst         = _impl->h_rgb12_pinned[new_buf];

        for (int ch = 0; ch < 3; ++ch) {
            uint8_t* ch_plane = dst + ch * plane_bytes;
            std::future<void> futs[N_PACK - 1];
            for (int i = 1; i < N_PACK; ++i) {
                int y0 = i * chunk_rows, y1 = std::min(y0 + chunk_rows, height);
                futs[i - 1] = std::async(std::launch::async,
                    [=]{ pack_rgb12_plane(rgb16, rgb_stride_pixels, ch_plane,
                                          width, prs, y0, y1, ch); });
            }
            pack_rgb12_plane(rgb16, rgb_stride_pixels, ch_plane,
                             width, prs, 0, std::min(chunk_rows, height), ch);
            for (int i = 0; i < N_PACK - 1; ++i) futs[i].wait();

            /* H2D this channel immediately — overlaps with next channel's pack. */
            cudaMemcpyAsync(_impl->d_rgb12[new_buf] + ch * plane_bytes,
                            ch_plane, plane_bytes,
                            cudaMemcpyHostToDevice, _impl->st_h2d);
        }
        cudaEventRecord(_impl->h2d_done[new_buf], _impl->st_h2d);
    }

    /* Step 3: Rebuild comp graphs for new_buf if geometry/bitrate changed.
     * V18e: packed_row_stride = (width/2)*3, derived from width; use that for validity. */
    int v18e_packed_stride = (width / 2) * 3;
    bool v17s_valid = (_impl->cg_v17s[new_buf][0] != nullptr                       &&
                       _impl->cg_v17s_width[new_buf]      == width                 &&
                       _impl->cg_v17s_height[new_buf]     == height                &&
                       _impl->cg_v17s_rgb_stride[new_buf] == v18e_packed_stride    &&
                       _impl->cg_v17s_per_comp[new_buf]   == per_comp              &&
                       _impl->cg_v17s_is_4k[new_buf]      == is_4k                 &&
                       _impl->cg_v17s_is_3d[new_buf]      == is_3d);
    if (!v17s_valid)
        s17_rebuild_v17s_comp_graphs(_impl.get(), new_buf,
                                     width, height, v18e_packed_stride,
                                     per_comp, is_4k, is_3d);

    /* Step 4: Sync all 3 component streams for cur_buf (V17z: was only st[0]).
     * V17z race fix: all 3 streams write to h_enc[cur_buf]; syncing only st[0]
     * could race with st[1]/st[2]'s D2H still writing component 1/2 data. */
    std::vector<uint8_t> result;
    if (_impl->pipeline_active) {
        cudaStreamSynchronize(_impl->st[0]);
        cudaStreamSynchronize(_impl->st[1]);  /* V17z: race fix */
        cudaStreamSynchronize(_impl->st[2]);  /* V17z: race fix */
    }

    /* Step 5: Launch new_buf's graphs BEFORE codestream building (V17z early launch).
     * GPU starts new_buf compute as soon as h2d_done[new_buf] fires (~0.1ms sooner).
     * V17u: graph contains event-wait(h2d_done[new_buf]) + RGB+HDWT0 + DWT+Q+D2H. */
    for (int c = 0; c < 3; ++c)
        cudaGraphLaunch(_impl->cg_v17s[new_buf][c], _impl->st[c]);

    /* Step 6: Build cur_buf codestream on CPU while GPU computes new_buf. */
    if (_impl->pipeline_active) {
        result = s17_build_j2k_codestream(_impl.get(),
            _impl->p_width, _impl->p_height,
            _impl->p_per_comp, _impl->p_is_4k, _impl->p_is_3d,
            _impl->h_enc[_impl->cur_buf]);
    }

    _impl->cur_buf         = new_buf;
    _impl->pipeline_active = true;
    _impl->p_width         = width;
    _impl->p_height        = height;
    _impl->p_per_comp      = per_comp;
    _impl->p_is_4k         = is_4k;
    _impl->p_is_3d         = is_3d;

    return result;  /* empty on first call; caller must call flush() at end */
}


std::vector<uint8_t>
SlangJ2KEncoder::flush()
{
    /* V17s: Drain the pipeline — collect the last in-flight frame's codestream.
     * V17z: sync all 3 streams (was only st[0]) to fix race condition. */
    if (!_initialized || !_impl->pipeline_active) return {};
    cudaStreamSynchronize(_impl->st[0]);
    cudaStreamSynchronize(_impl->st[1]);  /* V17z: race fix */
    cudaStreamSynchronize(_impl->st[2]);  /* V17z: race fix */
    auto result = s17_build_j2k_codestream(_impl.get(),
        _impl->p_width, _impl->p_height,
        _impl->p_per_comp, _impl->p_is_4k, _impl->p_is_3d,
        _impl->h_enc[_impl->cur_buf]);
    _impl->pipeline_active = false;
    return result;
}


void
SlangJ2KEncoder::set_colour_params(GpuColourParams const& params)
{
    if (!_initialized || !params.valid) return;
    if (_impl->upload_colour(params))
        _colour_params_valid = true;
}


static std::shared_ptr<SlangJ2KEncoder> _s17inst;
static std::mutex _s17inst_mu;

std::shared_ptr<SlangJ2KEncoder>
slang_j2k_encoder_instance()
{
    std::lock_guard<std::mutex> l(_s17inst_mu);
    if (!_s17inst) _s17inst = std::make_shared<SlangJ2KEncoder>();
    return _s17inst;
}
