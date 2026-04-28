/*
    Copyright (C) 2024 DCP-o-matic contributors

    This file is part of DCP-o-matic.

    GPU-accelerated JPEG2000 encoder using CUDA — V23.

    V21 Optimizations over V20:
    1. Fused vertical DWT + deinterleave kernel (kernel_fused_vert_dwt_deinterleave)
       - Eliminates separate kernel_deinterleave_vert kernel
       - Saves 15 kernel launches per frame (5 levels × 3 components)
       - For DWT levels 2-4 (subband ≤ 440 KB ≤ GTX 1050 Ti L2 = 512 KB),
         the deinterleave reads d_work from L2 rather than DRAM
       - Thread-per-column design: each thread owns its column exclusively,
         so d_src == d_dst aliasing is safe (copy phase completes before write phase)

    V22 Optimizations over V21:
    1. Fixed per_comp bitrate formula (V20 bug: divided by 9 instead of 3)
       - V20 was encoding at 33% of the target bitrate (50 Mbps instead of 150 Mbps)
       - V22: per_comp = target_bytes / 3, so 3×per_comp ≈ target frame size
       - At 150 Mbps / 24 fps: per_comp ≈ 260 KB (was 87 KB with the bug)
       - Correct bitrate compliance; quality improved 3× at same bitrate setting
    2. All previous features retained: GPU colour conversion, pinned memory,
       3-stream parallel DWT, pointer-swap double-buffers, fused V DWT+deinterleave

    V23 Optimizations over V22:
    1. CDF 9/7 analysis normalization (CRITICAL quality fix):
       - The unnormalized DWT has DC gain K^(2×level) per 2D level:
         after 5 levels, LL5 coefficients are K^10 ≈ 7.4× the input range
       - For 12-bit input [0, 4095]: LL5 values up to ~30,000
       - With step=1.0 and magnitude cap=126: ALL LL5 coefficients saturate at 126!
         This means the most important (low-frequency) data is completely wrong.
       - Fix: apply CDF 9/7 analysis normalization at each H and V DWT output:
           L samples × NORM_L = 1/K = 0.812893  (shrink lowpass gain)
           H samples × NORM_H = K   = 1.230174  (amplify highpass)
       - After normalization, ALL subbands have coefficients in [-4095, 4095]
         for 12-bit input, regardless of decomposition level
       - The normalization also aligns with what J2K decoders expect when they
         apply the standard QCD step sizes to reconstruct the image
    2. Correct base step size: 32.5 (= 4095/126) instead of 1.0
       - With normalized coefficients in [-4095, 4095], step = 4095/126 ≈ 32.5
         ensures no saturation and uses the full 7-bit magnitude range
       - Previous step=1.0 gave q ≤ 126 for |c| ≤ 126, but LL5 |c| ≤ 30000
         so effectively LL5 = constant 126 (all information lost!)
       - V23 step=32.5 correctly quantizes: LL5 magnitudes spread 0..126
    3. Updated QCD quantization step markers to match actual step sizes
       - Consistent codestream: decoder's dequantization step matches encoder step
       - For R_b=13 (12-bit + 1 guard bit), step≈32.5 encodes as exp=8, man=32

    Performance history:
    V15: 16 fps  — baseline with many small kernel launches
    V16: 52 fps  — per-thread encoder instances, fused kernels
    V17: 56 fps  — CPU-bound: rgb_to_xyz on CPU was bottleneck
    V18: 113 fps — GPU colour conversion eliminates CPU rgb_to_xyz
    V19: 122 fps — pinned memory + magnitude cap (no byte stuffing)
    V20: 127 fps — per_comp early cut (but incorrect bitrate: 1/3 of target)
    V21: ~127 fps — fused V DWT+deinterleave (15 fewer kernel launches, L2 hits)
    V22: target 120+ fps — correct bitrate (3× data, 6× less work than V19)
    V23: target 120+ fps — correct coefficient normalization (quality correctness)
    V24: target 120+ fps — QCC per-component step consistency (decoder accuracy)
    V25: target 140-150 fps — fp16 vertical DWT workspace (halves V-DWT memory BW)
    V26: target 150-160 fps — fp16 H-DWT output (eliminates float→half copy in V-DWT)
    V27: target 155-165 fps — register-blocked V-DWT for small subbands (h ≤ 140)
    V28: target 165-175 fps — fused RGB→XYZ+H-DWT level 0 (eliminates 54MB d_in traffic)
    V29: target 175-185 fps — register-based tiled V-DWT (no global workspace, 2.5× less BW)
    V30: target 185-200 fps — 3-stream parallel single-channel fused colour+HDWT0 kernel
         (splits 24KB→3×8KB smem, raises occupancy 25%→100%, reuses d_rgb16 from L2 cache)
    V31: target 200-215 fps — pinned RGB staging buffer (truly-async H2D DMA) + reciprocal
         multiply quantization + codestream vector pre-reserve to eliminate reallocations
    V32: target 215-230 fps — CUDA Graphs for per-component DWT+quantize+D2H pipeline
         (cudaGraphLaunch replaces ~30 per-frame kernel launches; ~0.3ms saved per frame)
    V33: target 230-245 fps — H-DWT thread count 256→512 (occupancy 75%→100% on sm_61)
         (8KB smem/block: 256T → 6 blk/SM=75%; 512T → 4 blk/SM=100%; better latency hiding)
    V34: target 245-260 fps — V-DWT tile size 16→32 (reduce overlap overhead 38.9%→24.4%)
         (fewer tiles for same height: 68→34 for 2K level 0; ~19% fewer half loads per column)
    V35: target 260-275 fps — float4 vectorized quantize kernel (4× fewer load/store transactions)
         (16-byte float4 reads + uint32_t packed writes vs scalar float reads + byte writes)
    V36: target 275-295 fps — full half-precision DWT pipeline (d_a[c]: float→__half)
         V-DWT output half instead of float: ~16MB/frame saved; H-DWT levels 1+ half-io;
         quantize uses __half2 loads; total ~18MB/frame less global traffic → ~0.69ms
    V37: target 295-315 fps — unified super-graph (H2D + event + RGB+HDWT0 + DWT+Q+D2H)
         cudaStreamCaptureModeGlobal captures all 3 streams into one graph; encode_from_rgb48
         reduced from ~13 CUDA API calls per frame to 3 (graphLaunch + streamSync + CPU build)
         Expected savings: ~10 CUDA calls × 5µs = ~50µs/frame
    V38: target 335-360 fps — half-precision shared memory for all H-DWT kernels
         kernel_fused_i2f_horz_dwt_half_out, kernel_fused_horz_dwt_half_io,
         kernel_rgb48_xyz_hdwt0_1ch: extern __shared__ float smem[] → __half smem[]
         fp16 FMA: 2× throughput vs fp32 on sm_61+; smem 7.5KB→3.75KB per block;
         more L1 cache for LUT/matrix texture reads; parity with Slang V17o
    V39: target 360-385 fps — parity-split V-DWT lifting loops (branch-free unrolled)
         cdf97_lift_tiled<P0> template: 4×40 conditional iterations → 4×20 unrolled FMAs
         #pragma unroll + compile-time P0 eliminates all per-iteration branch overhead
         kernel_fused_vert_dwt_tiled_ho + kernel_fused_vert_dwt_tiled updated
         ~50% fewer loop iterations + full unrolling → ILP maximized; V-DWT ~40% of GPU time
    V40: target 385-410 fps — constant p0=1 hardcode + static_assert on tile params
         V_TILE=28 (even) and V_OVERLAP=5 (odd) → load_start always odd → p0 always 1
         Eliminates runtime p0 computation and if(p0) dispatch branch
         Output loop: `if ((p0+i) & 1)` → `if (!(i & 1))` (compile-time simplification)
         static_assert ensures correctness if V_TILE/V_OVERLAP ever change
    V41: target 410-440+ fps — 1-frame pipeline (double-buffer H2D+D2H)
         Double-buffered h_rgb16_pinned[2], h_packed_pinned[2], full_graph[2]
         CPU memcpy for frame N+1 overlaps with GPU executing frame N
         Steady-state time = max(GPU~2.16ms, memcpy~1.2ms) ≈ 2.16ms/frame → ~463fps
         encode_from_rgb48 returns PREVIOUS frame's codestream; flush() at end
    V42: target 550-600+ fps — H2D-compute overlap via dedicated st_h2d stream
         d_rgb16[2]: double-buffered GPU RGB; st_h2d: dedicated PCIe DMA stream
         H2D for frame N runs on st_h2d while SM computes frame N-1 on stream[0..2]
         cg_v42[buf][c]: per-buf comp graphs (DWT levels 1-4 + Q + D2H)
         CPU blocking: memcpy(1.2ms) + sync_wait(0.5ms) = 1.8ms > H2D(1.66ms) → 556fps
    V43: target 580-620 fps — parallel CPU memcpy via std::async (4 threads)
         Splits 12.4MB RGB copy into 4 concurrent chunks: 4× speedup → ~0.3ms
         CPU blocking drops to 0.3+0.5+0.1=0.9ms < H2D(1.66ms) → PCIe bottleneck
         T_frame ≈ H2D ≈ 1.66ms → ~602fps (PCIe-limited)
    V44: target 602+ fps — fuse event-wait + RGB+HDWT0 into per-buf CUDA Graphs
         rebuild_v42_comp_graphs now captures cudaStreamWaitEvent + RGB kernel
         Per-frame API calls: 9 (V43: 3 waitEvent + 3 RGB + 3 graphLaunch) → 3 (graphLaunch only)
         ~6 API calls × 5µs = ~30µs/frame saved; avoids per-frame driver sync points
    V45: target 610+ fps — 2-rows-per-block kernel_rgb48_xyz_hdwt0_1ch_2row
         Processes rows y0=blockIdx.x*2 and y1=y0+1 simultaneously; smem=2*w*sizeof(__half)
         Grid halved (540 vs 1080 blocks for 2K): 2× fewer block-scheduler entries per stream
         Matrix m0/m1/m2 amortized over 2× work; L2 reuse for adjacent RGB rows
         Same 4 syncthreads per 2 rows (vs 4+4); SM sees 8 rows/pass vs 4
    V46: target 615+ fps — 2-rows-per-block kernel_fused_horz_dwt_half_io_2row (levels 1-4)
         Halves grid for DWT levels 1-4: 540→270, 270→135, 135→68, 68→34 (2K)
         L2 reuse for adjacent d_a rows; smem=2*w*sizeof(__half); same 4 syncthreads
    V47: target 618+ fps — 4-rows-per-block kernel_fused_horz_dwt_half_io_4row (levels 1-4)
         Halves grid again vs V46: 270→135, 135→68, 68→34, 34→17 blocks (2K)
         smem=4*w*sizeof(__half); thread-limited at 4 blk/SM = 100% occ at all levels
         4 adjacent rows per block → improved L2 spatial locality for DWT input reads
         4 rows amortize 1 syncthreads set over 4× the work (vs 2× in V46)
    V48: target 620+ fps — uint16_t d_lut_out (was int32_t; halves LUT GPU texture cache)
         lut_out GPU allocation: 4096×4=16KB → 4096×2=8KB; fits in 32KB L1 alongside lut_in(16KB)
         Both LUTs (lut_in+lut_out = 24KB) now fit within sm_61 L1/texture cache (32KB)
         Fewer L1 evictions during RGB→XYZ conversion → better cache hit rate per SM
    V187: Fix tiled V-DWT out-of-bounds load at image bottom — 14-39 dB PSNR jump.
          kernel_fused_vert_dwt_tiled_ho_2col's `interior` flag only checked
          tile_start + V_TILE <= height, but the LOAD reads V_TILE_FL=32 rows
          starting at load_start=tile_start-V_OVERLAP.  For the LAST tile
          (tile_start + V_TILE = height exactly), interior=true, but
          load_start + V_TILE_FL = height + V_OVERLAP exceeded image height.
          The interior load path then read V_OVERLAP=4 rows past the image
          bottom as garbage memory, producing huge spurious H-band coefficients
          on the last 2-3 rows — visible as decoded values clipping to 0/4095
          on bars and ~80 LSB error on flat regions.
          Fix: interior = (load_start >= 0) && (load_start + V_TILE_FL <= height).
          Forces the last tile through the mirroring path, which gives correct
          symmetric extension.
          PSNR jumps:
            flat_50000: 48.8 → 63.4 dB
            h_bars_8:   47.3 → 60.3 dB
            v_bars_8:   48.6 → 61.0 dB
            two_value:  52.9 → 65.3 dB
    V186: Lower compute_base_step multiplier 0.25 → 0.06.  After V185 the T1
          step is internally scaled (×2 HL/LH, ×4 HH), so the same abstract
          base_step gives much coarser quantization.  Smaller multiplier
          recovers per-bitplane resolution; clamp lower bound 1.0 → 0.25.
    V185: Fix HL/LH/HH stepsize compensation for irreversible 9/7 inverse DWT.
          OpenJPEG inverse DWT uses two_invK = 2/K (not invK) per dwt.c "BUG_WEIRD_TWO_INVK"
          comment.  This doubles HL/LH coefficients and quadruples HH on decode.  OPJ's
          encoder compensates via log2_gain in tcd.c stepsize calculation.
          Our encoder didn't compensate → bars/checker reconstructed at 2x/4x magnitude
          near edges → saturation at 0/4095 → 15-25 dB PSNR.
          Fix in build_codeblock_table: cbi.quant_step = step * 2 for HL/LH, * 4 for HH.
          QCD continues to write the unscaled `step`. Encoder T1 quantizes coarser, so
          dequantized coefficients are 2x/4x smaller; inverse DWT amplification recovers
          the right magnitude.
          PSNR jumps: h_bars_8 23.9→47.3 dB, v_bars_8 22.5→48.6 dB, checker_64 15.5→29.8 dB.
    V177: Fix DC level shift + QCD guard bits — root cause of PSNR failures.
          Two-part fix for systematic flat-field anomaly (decoded≈2048 for all XYZ inputs):
          1. DC level shift: kernel_rgb48_xyz_hdwt0_1ch_2row now stores XYZ-2048 instead
             of XYZ. JPEG 2000 unsigned images require encoder to subtract 2^(prec-1)=2048
             before DWT; decoder adds 2048 after IDWT. Without this, decoded = XYZ+2048.
          2. Sqcd 0x22→0x42: numgbits 1→2 (SMPTE 422M DCP profile). OPJ tcd.c formula:
             band->numbps = expn+numgbits-1. With numgbits=1: band->numbps=pmax (not pmax+1),
             making z=pmax+1-nb give cblk->numbps=nb-1 (off-by-one: MSB lost in all CBs).
             With numgbits=2: band->numbps=pmax+1, z=pmax+1-nb → cblk->numbps=nb. Correct.
    V176: Unified pre-build for correct mode: FAST4=false now uses mag_bp_flat (single
          d_dwt scan, like FAST4=true) instead of per-bp col_mag_arr recompute. Template
          parameter MAX_BP=7 for correct mode (all blocks have ≤7 bp at 150Mbps). col_mag_arr
          eliminated from both paths. Correct: 47ms, 1568 LMEM; fast: 19ms, 1184 LMEM.
          Root cause finding: the 19ms (fast) vs 47ms (correct) timing difference comes from
          step_mult (3.0 vs 1.0) reducing bit-planes from ~4 to ~2, not kernel structure.
          Kernel PTX is identical for FAST4=false,MAX_BP=4 and FAST4=true,MAX_BP=4.
    V175: template<bool FAST4, int MAX_BP>: second template parameter, unified for both
          modes. No benchmark improvement — see V176 analysis.
    V174: Level-adaptive num_bp cap in FAST4=false kernel. LL5+L5-AC (level≥4) capped
          at 8 bp, L4 (level=3) at 9 bp. BENCHMARKED: no improvement — those blocks
          already have ≤8 bp naturally at 150Mbps. Timing 46.75ms (unchanged).
    V173: Two-pass correct mode: FAST4=true for all blocks (Pass 1, ~19ms), then
          FAST4=false for first num_hard blocks (LL5..L3, subbands[0..9], Pass 2).
          Hard blocks: 172 CBs/component (4+12+36+120). Full-quality overwrite for
          low-frequency subbands that need >4 bit-planes; easy high-freq blocks keep
          FAST4 approximation. BENCHMARKED: 66ms correct / 20ms fast — two sequential
          passes are SLOWER because hard blocks were already the V172 critical path.
          V173 reverted; V174 takes a different approach.
    V172: template<bool FAST4> kernel_ebcot_t1 — fast path clamps to 4 bit-planes.
          FAST4=true: pre-build mag_bp_flat[4*CB_DIM] in Pass 2 (1× d_dwt read vs num_bp×).
            Passes access mag_bp_flat directly; col_mag_arr eliminated (dead code).
            ptxas: 46 regs, 1184 bytes LMEM (mag_bp_flat 512 + others 672).
            Benchmark fast: 19.3ms/frame, 214969 bytes (59% faster than 47.5ms!).
          FAST4=false: V171 col_mag_arr per-iteration recompute, 800 bytes LMEM.
            Benchmark correct: 47.1ms/frame, 793109 bytes (unchanged).
          Dispatch: fast_mode flag → kernel_ebcot_t1<true>; else → kernel_ebcot_t1<false>.
          Quality trade-off: FAST4 clamps num_bp to 4 → 214KB vs 793KB at 150 Mbps.
            Acceptable for preview/draft; correct mode for final DCP export.
    V171: Eliminate mag_bp LMEM array — recompute per bit-plane from d_dwt.
          mag_bp[MAX_BPLANES][CB_DIM] (1280 bytes LMEM) removed; replaced with
          col_mag_arr[CB_DIM] computed per bit-plane by re-reading d_dwt coefficients.
          MAX_BPLANES raised to 16 (guard only — loop bound is num_bp from quantizer).
          Pass 2 (mag_bp pre-build loop) eliminated; bit-plane loop recomputes col_mag_arr.
          ptxas: 46 regs, 800 bytes LMEM (was 44 regs, 1952 bytes — 1152 bytes saved).
          Benchmark: 47.5ms (unchanged — d_dwt re-read DRAM cost ≈ LMEM savings).
          Phase tmarks split: "DWT" → "RGB+HDWT0" + "DWT_lv1+" for diagnostics.
          bench_phases.cc warmup increased 3→10 frames for thermal steady-state accuracy.
          Also added kernel_rgb48_xyz_hdwt0_1ch_4row (kept, not dispatched; 4-row hurts
          occupancy at 2K vs 2-row: 75% vs 100% due to 16KB vs 8KB smem).
    V128: Remove dead kernel_rgb48_xyz_hdwt0_1ch_4row_p12 — parity Slang V90.
          Never launched (annotated V124); runtime uses 2-row (2K) and 1-row (4K) p12 kernels.
          Deleted ~253 lines of function body + cudaFuncSetCacheConfig reference.
          Reduces compile time and binary size; no runtime change.
    V127: Template kernel_fused_horz_dwt_half_io_4row<DIV4> — parity Slang V89.
          Adds compile-time bool DIV4: when h%4==0, y3<height always → else block dead.
          DCI heights by level: 2K L1=540✓, L2=270✗, L3=135✗, L4=68✓; 4K L1=1080✓, L2=540✓, L3=270✗, L4=135✗.
          DIV4=true: if (DIV4 || y3 < height) → unconditional → else dead → compiler eliminates 69-line partial path.
          DIV4=false: same as if (y3 < height) → full if/else preserved for partial last block.
          Launch site: (height%4==0)?kernel<true><<<...>>>:kernel<false><<<...>>>; cudaFuncSetCacheConfig for both.
          Expected: ~69 dead lines removed for half the DWT level launches; slightly better reg/icache for DIV4=true.
    V126: DCI divisible-by-4 invariant in kernel_fused_i2f_horz_dwt_half_out_4row — parity Slang V88.
          Level-0 heights always 1080/2160 (both divisible by 4) → y3=y0+3 always < height → else block dead.
          Removes ~73 lines of dead scalar load + guarded DWT + scatter code.
          Compiler can now see interior path is unconditional → better register allocation.
          Same approach as V122 (removed dead else from 2-row RGB kernel).
    V125: Template kernel_fused_vert_dwt_fp16_hi_reg_ho<EVEN_HEIGHT> — parity Slang V87.
          Removes 4 runtime if(height&1)/if(!(height&1)) branches per column via compile-time template bool.
          EVEN_HEIGHT=true: Alpha/Gamma even-boundary executes; Beta/Delta odd-boundary eliminated.
          EVEN_HEIGHT=false: Beta/Delta odd-boundary executes; Alpha/Gamma even-boundary eliminated.
          Compiler also eliminates 2 dead __float2half constant computations per instantiation:
            EVEN: kB2/kD2 dead (odd paths unreachable); ODD: kA2/kG2 dead (even paths unreachable).
          DCI heights at reg-blocked levels: 68/34 (even) and 135/67 (odd) — both common paths.
          Launch site: (height%2==0)?kernel<true>:kernel<false>; cudaFuncSetCacheConfig updated for both.
          Expected: ~4-8 fewer PTX instructions per column thread; ~1-2% V-DWT reg-blocked speedup.
    V124: int2 __ldg load vectorization in kernel_fused_i2f_horz_dwt_half_out_4row (XYZ int32 path) — parity Slang V86.
          Load loop: for(x=t;x<w;x+=nt) 4×__ldg(int32) → for(x=t*2;x<w;x+=nt*2) 4×__ldg(int2).
          Each int2 load covers {d_input[row*stride+x], d_input[row*stride+x+1]}: 2 int32 in 64-bit load.
          8 × 32-bit loads → 4 × 64-bit loads: 2× fewer load instructions; same bytes transferred.
          Extract: sm01[x]={i2f(r0.x),i2f(r1.x)}, sm01[x+1]={i2f(r0.y),i2f(r1.y)}.
          DCI widths always even → x=t*2 always even → 8-byte aligned int2 load. Coalesced preserved.
          Annotation: kernel_rgb48_xyz_hdwt0_1ch_4row_p12 marked dead code (V124) — never launched.
    V123: __half2 __ldg load vectorization in kernel_fused_horz_dwt_half_io_4row interior load phase — parity Slang V85.
          Load loop: for(x=t;x<w;x+=nt) 4×__ldg(__half) → for(x=t*2;x<w;x+=nt*2) 4×__ldg(__half2).
          Each __half2 load covers {row[x], row[x+1]}: 2 __half in one 32-bit load instruction.
          8 scalar 16-bit loads → 4 vectorized 32-bit loads: 2× fewer load instructions per thread.
          Unpack: sm01[x]={low(r0),low(r1)}, sm01[x+1]={high(r0),high(r1)} via __low2half/__high2half.
          DCI widths always even → x=t*2 always even → 4-byte aligned (stride×y+x even guaranteed).
          Coalesced access preserved; 4K: 2 iters/thread, 2K: 1 iter → #pragma unroll 2 retained.
          Expected: 1-3% speedup in kernel_fused_horz_dwt_half_io_4row load phase (~15% of GPU time).
    V122: DCI even-height invariant in kernel_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT) — parity Slang V84.
          DCI heights always even (1080/2160) → y1=y0+1 always < height → else (partial scalar) branch is dead.
          Removed entire else block (35 lines: scalar load + scalar lifting + single-row scatter).
          Unconditional __half2 path: compiler eliminates register partitioning for dead branch.
          Better instruction cache coverage; smaller kernel binary; same correctness for all DCI content.
    V121: uint32_t __ldg replaces 3×byte __ldg per channel in hot p12 load loops — parity Slang V83
          kernel_rgb48_xyz_hdwt0_1ch_1row_p12 (4K HOT) + kernel_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT).
          9 byte __ldg → 3 uint32_t __ldg per pixel pair per row: 3× fewer load instructions.
          Pixel extraction: even=(byte0<<4)|(byte1>>4)=((rw&0xFF)<<4)|((rw>>12)&0xF); odd=((byte1&0xF)<<8)|byte2.
          Misaligned reads (off=p*3 not always 4B-aligned) correctly handled by CUDA __ldg texture cache.
          4K: 4 iters/thread × 3 uint32_t loads (was 9 bytes) = 4× better MLP per unrolled group.
          2K: 2 iters/thread × same benefit. Expected: 2-5% RGB+HDWT0 speedup → ~0.5-1% overall.
    V120: DCI even-width invariant in `width`-variable half/float 1-row kernels: kernel_fused_i2f_horz_dwt_half_out, kernel_fused_horz_dwt_half_io, kernel_fused_horz_dwt_half_out
          All three: Alpha/Gamma if(t==0&&width>1&&(width%2==0)) → if(t==0); Beta/Delta min(1,width-1)→1, min(x+1,width-1)→x+1.
          Fires on every row of levels 1-4 (2K: 540/270/135/68 rows per kernel); saves 2 ISETP+PREDAND+2 VMIN per block.
          All CUDA kernels now fully cleaned of DCI-redundant bounds guards; parity Slang V82.
    V119: DCI even-width invariant in remaining kernels: DO_HDWT_HALF macro, kernel_rgb48_xyz_hdwt0_1ch 1row/2row, and 2row_p12 partial — parity Slang V82
          DO_HDWT_HALF macro (float-path): Alpha/Gamma w>1&&!(w&1) → always, Beta/Delta min(1,w-1)→1, min(x+1,w-1)→x+1.
          kernel_rgb48_xyz_hdwt0_1ch (1-row): same simplification.
          kernel_rgb48_xyz_hdwt0_1ch_2row (2-row, y1<height guards): same simplification.
          kernel_rgb48_xyz_hdwt0_1ch_2row_p12 partial block: same simplification.
          All are non-p12 fallback or rarely-used paths; completes DCI invariant cleanup across all CUDA kernels.
    V118: DCI even-width invariant in kernel_rgb48_xyz_hdwt0_1ch_4row_p12 interior + partial, and kernel_fused_horz_dwt_half_io_4row partial — parity Slang V81
          kernel_rgb48_xyz_hdwt0_1ch_4row_p12 interior: Alpha/Gamma if(t==0&&w>1&&!(w&1)) → if(t==0);
          Beta/Delta boundary sm01[min(1,w-1)] → sm01[1] (sm23 same); saves 2 VMIN+2 ISETP per block.
          kernel_rgb48_xyz_hdwt0_1ch_4row_p12 partial block: same simplifications.
          kernel_fused_horz_dwt_half_io_4row partial (else branch): Alpha/Gamma/Beta/Delta simplified.
          4row_p12 fires on 134/135 blocks per 2K frame (interior); partial on last block only.
    V117: DCI even-width invariant in kernel_fused_horz_dwt_half_io_4row partial block + kernel_fused_horz_dwt_half_io_2row — parity Slang V80
          4-row kernel: interior (V110) already simplified; partial (else branch) still had old guards.
          2-row kernel: also still had old guards — both kernels fixed in this version.
          Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); saves 2 ISETP+PREDAND per block.
          Beta/Delta: smem[min(1,w-1)] → smem[1]; saves 1 VMIN+SEL per boundary thread.
          Beta/Delta loop: smem[min(x+1,w-1)] → smem[x+1] (x even, w even: x<w → x≤w-2 → x+1≤w-1).
          2-row kernel fires on all 540 rows of level-1 (2K); 4-row partial on last row-group only.
    V116: __launch_bounds__(512,4) → (512,3) on 1row/2row RGB p12 kernels — parity Slang V79
          V88 set (512,4) targeting ≤32 regs/T for 4 blk/SM, but V90 added 8KB sm_lut preload:
          smem = 8KB (sm_lut) + 7.68KB (DWT) = 15.68KB → PreferShared(48KB) → 3 blk/SM (smem-limited).
          Actual occupancy was already capped at 3 blk/SM by smem; (512,4) forced needlessly tight 32-reg limit.
          (512,3) → ≤floor(65536/512/3)=42 regs/T; still 3 blk/SM (smem-limited: 48KB/15.68KB=3).
          10 extra registers allow compiler to reduce/eliminate register spills in the complex color+DWT kernel.
          Applied to kernel_rgb48_xyz_hdwt0_1ch_2row_p12 and kernel_rgb48_xyz_hdwt0_1ch_1row_p12.
          Expected: 5-15% RGB+HDWT0 speedup if compiler was spilling under 32-reg limit → ~1-3% overall.
    V115: quantize L1-row early exit before inv_* computation — parity Slang V78
          L1 rows (row >= ll5_h*16) cover >50% of 2K/4K rows and need only inv_l1 = base_inv*0.833…
          Moving the early exit before the other 5 inv_* FMULs saves 5 FMUL per thread for majority rows.
          Register pressure for L1 path: 6 floats → 1 float (base_inv) + inline constant.
          row_lv ternary simplified: last branch now 2 (L1 never reaches it after early exit).
          Dispatch: if(row_lv==1) branch removed (dead code after early exit added above lambda).
    V114: Drop dead h>2 guard in reg-blocked V-DWT Beta/Delta boundary — parity Slang V77
          kernel_fused_vert_dwt_fp16_hi_reg_ho: DCI subband heights 34/68/135 always > 2.
          (height & 1) && (height > 2) → (height & 1): saves 1 ISETP+AND per boundary invocation.
          Applied to both Beta and Delta odd-height boundary cases. Minor but correct DCI invariant.
    V113: Simplify H-DWT i2f-4row Alpha/Gamma boundary + Beta/Delta min() — parity Slang V76
          DCI even-width invariant applied to kernel_fused_i2f_horz_dwt_half_out_4row (int32 XYZ path).
          Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); saves 2 ISETP+PREDAND per block.
          Beta/Delta: sm01[min(1,w-1)] → sm01[1]; saves 1 VMIN+SEL per boundary thread.
          This kernel handles level-0 H-DWT for the fallback encode() path (int32 XYZ input).
    V112: Simplify reg V-DWT boundary guards — parity Slang V75
          kernel_fused_vert_dwt_fp16_hi_reg_ho: all DCI heights > 1 → drop h>1 from Alpha/Gamma guards.
          col[min(1,height-1)] → col[1] in Beta/Delta (height always > 1 for DCI subbands: 34/68/135).
          Eliminates 2 VMIN + 2 ISETP+AND per column across 3 V-DWT levels.
    V111: Simplify boundary conditions in HOT kernels — parity Slang V74
          Apply DCI even-width invariant to kernel_rgb48_xyz_hdwt0_1ch_2row_p12 (2K HOT)
          and kernel_rgb48_xyz_hdwt0_1ch_1row_p12 (4K HOT) interior DWT lifting passes.
          Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); Beta/Delta: sm[min(1,w-1)] → sm[1].
          More impactful than V110 (4-row dead path): these are the actual hot kernels.
    V110: Simplify H-DWT 4-row Alpha/Gamma boundary + Beta/Delta min() — parity Slang V73
          DCI subbands: widths always even (1920/960/480/240/120/60) and always >1.
          Alpha/Gamma: if(t==0&&w>1&&!(w&1)) → if(t==0); saves 2 ISETP+PREDAND per block.
          Beta/Delta: sm01[min(1,w-1)] → sm01[1]; saves 1 VMIN+SEL per boundary thread.
          Applied to kernel_fused_horz_dwt_half_io_4row interior (both Alpha and Gamma passes).
    V109: PRMT sign extraction in vq4 tail — parity Slang V72 (parity with vq8 core V103/V66)
          Replace 4 SETP+SELP+IABS per element with 1 PRMT+AND (signs) + 2 AND (abs) per 4 elements.
          Applicable to 2K DC rows only (ll5_c=60, 60%8=4 → 4 tail elements); ~102 calls/frame/channel.
    V108: __launch_bounds__(512,4) on kernel_fused_horz_dwt_half_io_4row — parity Slang V71
          Comment said "Thread-limited at 4 blk/SM" but without annotation compiler may use 36+ regs/T.
          Forces ≤32 regs/T → 65536/(512×32)=4 blk/SM guaranteed. smem for L1 (7.68KB) allows 6 blk
          → register was the bottleneck. Hot kernel (~15% GPU time): +33% throughput if was 3 blk/SM.
    V107: __saturatef() replaces fmaxf(0,fminf(1,x)) in all RGB→XYZ clamp paths — parity Slang V70
          __saturatef(x) compiles to single PTX cvt.sat.f32.f32 instruction (vs 2 for fmaxf/fminf).
          Applied to: 3-ch legacy kernel (×3), 1ch_2row (×1), 4row_p12 load (×4), partial path (×3),
          xyz12 kernel (×3). Also removed min(…,4095) from xyz12 output indices (parity V105).
          Saves ~13 FMIN/FMAX instructions per pixel across all color-conversion kernels.
    V106: __half2 row-pair lifting in kernel_rgb48_xyz_hdwt0_1ch_4row_p12 interior — parity Slang V69
          Interior load now packs {row0[px],row1[px]} into sm01[px] and {row2[px],row3[px]} into sm23[px].
          4 lifting passes use __hfma2 (2 rows/instruction) → 2× FMA throughput vs 4 scalar HFMA.
          Same smem layout as kernel_fused_horz_dwt_half_io_4row (V82) and _2row_p12 (V83).
          + #pragma unroll 2 on all interior lifting loops (parity V96).
          + Drop MIN from Beta/Delta loops: x always even, w even (DCI) → x+1≤w-1 (parity V102).
          + __launch_bounds__(512,2): forces ≤64 regs/T; smem=23.5KB → 2 blk/SM smem-limited.
          Scatter: deinterleave via __low2half/__high2half + __hmul2 (parity V82/V83).
          4-row p12 is fallback path (3 launch selections; 4K uses 1-row, 2K uses 2-row for main path).
          Expected: ~2× lifting throughput for this kernel → ~0.5-1% overall for frames using 4-row path.
    V105: Remove redundant min(int(v*4095.5f),4095) from all d_lut_out index computations — parity Slang V68
          v = fmaxf(0,fminf(1,x)) ∈ [0,1] → v*4095.5f ≤ 4095.5 → int(truncate) ≤ 4095.
          min(...,4095) is dead code (never fires) — saves 1 IMIN per pixel per channel.
          Applied to all kernels: _4row_p12, _2row_p12, _1row_p12 interior/partial, legacy 3-ch.
          Parity: Slang V68
    V104: u16_to_f16() helper: PTX cvt.rn.f16.u16 replaces __float2half(float(u16)) — parity Slang V67
          All d_lut_out lookups: uint16→__half was CVT.F32.U16 + CVT.F16.F32 = 2 instructions.
          PTX cvt.rn.f16.u16 converts uint16→__half in 1 instruction directly.
          Exact for [0, 4095]: all DCP XYZ output values fit exactly in both float and half.
          Applied to all 11 __float2half(float(d_lut_out[...])) calls in hot RGB→XYZ kernels.
          Hot kernels: _1ch_2row_p12 (2K), _1ch_1row_p12 (4K), _1ch_4row_p12 (fallback).
          Saves ~6M CVT instructions per 2K frame, ~25M per 4K frame.
          Expected: ~5-10% RGB+HDWT0 compute speedup → ~1-2% overall (RGB~20% of GPU time).
          Parity: Slang V67
    V103: vq8 PRMT sign extraction + frcp_rn reduction in kernel_quantize_subband_ml — parity Slang V66
          Sign extraction: __byte_perm(r01.x, r01.y, 0x7531) & 0x80808080 replaces 8×SETP+8×SELP.
          IEEE half sign bit = bit 15 = byte index 1 (bits[15:8]); 0x7531 selects bytes 1,3,1,3
            from r01.x and r01.y, placing each sign byte into the correct byte position of lo/hi.
          Abs via AND 0x7FFF7FFF: clears sign bit of each half in packed uint32 — 4 AND vs 8 IABS.
          frcp_rn: 1 __frcp_rn(base_step) + 5 FMUL(compile-time 1/mult) vs 6 __frcp_rn.
          Total: 8 ops (2 PRMT+2 AND for signs, 4 AND for abs) vs 24 ops (8 SETP+8 SELP+8 IABS).
          Saves 16 instructions per 8-element vq8 iteration (~23% quantize compute reduction).
          Plus: 5 fewer SFU ops per thread from frcp_rn reduction.
          Note: -0.0 half (0x8000) → sign bit=1, mag=0 → byte=0x80 (vs 0x00 in prev). Negligible.
          Expected: ~20-25% quantize compute speedup → ~2-3% overall (quantize ~10% of GPU time).
          Parity: Slang V66
    V102: H-DWT Beta/Delta boundary hoist — remove min(x+1,w-1) from hot loops — parity Slang V65
          x in Beta/Delta loops is always even: x=2+t*2+k*(nt*2) — all terms even.
          For even w (all DCI: 1920,960,...,60 all even): x ≤ w-2 → x+1 ≤ w-1 → min(x+1,w-1)=x+1.
          The MIN never fires for even w — it computes an unnecessary conditional every iteration.
          Loop bound x<w-1 (≡ x<w for even-x, even-w) makes equivalence explicit to compiler.
          → Compiler can drop 1 VMIN/SETP/SEL per iteration; enables cleaner HFMA2 chain scheduling.
          With #pragma unroll 2: 2 fewer MIN per unrolled body × 2 rows × 2 passes (Beta+Delta) = 8/group.
          Updated kernels: kernel_fused_horz_dwt_half_io_4row (H-DWT lv1-5, HOT, ~15% GPU time),
            kernel_rgb48_xyz_hdwt0_1ch_1row_p12 (4K lv0 HOT, #pragma unroll 4 → 4 fewer MIN/iter),
            kernel_rgb48_xyz_hdwt0_1ch_2row_p12 (2K lv0 HOT), kernel_fused_i2f_horz_dwt_half_out_4row.
          Safety: x<w-1 with even x iterates same values as x<w for even w (no code change for DCI).
                  For odd w (never in DCI): x=w-1 possible → would need boundary case (omitted, DCI only).
          Expected: ~2-5% H-DWT speedup → ~0.3-0.8% overall (H-DWT ~15% GPU time). Parity Slang V65
    V101: reg-blocked V-DWT: hoist Beta/Delta boundary case → #pragma unroll 4 on main loops
          Old Beta loop: `for y=2..h: yp1=(y+1<h)?y+1:y-1` — runtime conditional every 2 iters.
          New: split into main loop (y<h-1, always y+1<h → no boundary) + boundary case (odd h only).
          Main loop: `for y=2; y<height-1; y+=2` + `#pragma unroll 4`:
            4 independent HFMA per unrolled batch → 4× FMA ILP; compiler issues 4 HFMA in 4 cycles.
            Removes per-iteration ternary check: saves 1 SETP+SEL instruction per 2 elements.
          Boundary case: fires only when height is odd (e.g., h=135 at DWT level 3):
            col[height-1] += kB2 * col[height-2]; (kB2=2×kB = kB*(col[y-1]+col[y-1]))
          Same transformation applied to Delta loop: same analysis, same benefit.
          Even heights (1080, 540, 270, all DCI level-0/1/2): boundary case never executed.
          Odd heights (135, 67, ..., at levels 3-5): one extra FMA per thread for boundary.
          Safety: verified for h=1,2,3,4,5,135,540 — all match original yp1 logic.
          Expected: ~15-25% Beta/Delta speedup → ~7-12% V-DWT lifting → ~3-5% overall (4K).
          Parity: Slang V64
    V100: vq8 uint2 stores + #pragma unroll 2 in kernel_quantize_subband_ml — parity Slang V63
          uint2 store (st.global.v2.b32): replaces 2×uint32_t (2×st.global.b32) for aligned zones.
          Alignment: row_dst = d_packed(256-align) + row*stride(mult-8) → 8-byte aligned.
          c = col_start + tx*8 → row_dst+c is 8-byte aligned iff col_start%8==0.
          All col_starts are 0 or multiples of ll5_c (2K:60,4K:120).
            col_start=0: 8-aligned ✓. col_start=60 (2K DC zone 2): 60%8=4 → 4-aligned only ✗.
            col_start=120,240,...: all 8-aligned ✓. 4K (ll5_c=120): ALL zones aligned ✓.
          Branch on (col_start&7)==0 is loop-invariant → compiler hoists outside loop.
          Unaligned fallback (2K DC zone 2, ~0.5% of work): keeps 2×uint32_t stores unchanged.
          #pragma unroll 2: 4K L1 rows (stride=3840, nt=256): 3840/(256×8)≈2 iters → full inline.
            Compiler issues 2×(2 int2 loads + 8 computations) simultaneously → better latency hiding.
            2K L1 rows (1 iter): pragma degenerates gracefully (no code bloat, single loop body).
          Expected: ~25-40% fewer store instructions in hot path → ~3-6% quantize speedup.
          quantize ~10% of GPU time → ~0.3-0.6% overall improvement; bigger win on 4K.
          Parity: Slang V63
    V99: __half2 row-pair packing + HFMA2 in kernel_fused_i2f_horz_dwt_half_out_4row — parity V45+V47
         Interior path: sm01[x]={row0,row1}, sm23[x]={row2,row3} as __half2 (matches fused_horz_dwt_half_io_4row)
         All 4 lifting passes (Alpha/Beta/Gamma/Delta): scalar 4×HMUL/HADD → HFMA2 (2× throughput per pass)
         #pragma unroll 2 on all interior loops — 4K level-0 (w=1920, nt=512): ~2 iters → full ILP
         Combined L+H scatter: 2 separate loops → 1 loop via __low2half/__high2half (fewer syncthreads)
         Partial path (else): unchanged (scalar, boundary-guarded, at most 1 block per frame component)
         Expected: ~2× lifting throughput for i2f H-DWT level-0 (fallback XYZ-plane encode path)
         Parity: Slang V62
    V98: vq8 int2 loads — replace 4×__half2 __ldg with 2×int2 __ldg in kernel_quantize_subband_ml
         4×ld.global.b32 → 2×ld.global.b64: halves load instruction count for vq8 inner loop
         int2 load at row_src+c (__half*): byte offset c*2; c=col_start+tx*8 always →
           c*2 = (col_start+tx*8)*2 — all col_start multiples of 8 → 16-byte aligned → int2 OK
           Exception: 2K ll5_c=60 zone (col_start=60): c=60+tx*8 → c*2=120+tx*16 → 8-byte aligned ✓
         Stores: keep 2×uint32_t — c is 4-byte aligned always; int2 needs 8B but col_start=60 gives c%8=4
           Skip int2 stores to avoid special-casing 2K DC zone 2
         Expected: ~2-4% quantize speedup → ~0.2-0.4% overall (fewer load transactions + PTX instr count)
         Parity: Slang V61
    V97: Reg-blocked V-DWT: hoist __half constants + #pragma unroll 4 on load/Alpha/Gamma/write loops
         kernel_fused_vert_dwt_fp16_hi_reg_ho: 6 __half consts hoisted (kA/kB/kG/kD/kNL/kNH)
         #pragma unroll 4 on load loop: 4 concurrent __ldg in flight → better MLP for column loads
           Each load at y*stride+x → different cache lines → 4× latency hiding over scalar loop
         #pragma unroll 4 on Alpha/Gamma loops: independent iterations (step=2, no inter-dep)
           4 independent HFMA per unrolled batch → better ILP on sm_61+ __half FMA units
         #pragma unroll 4 on both scatter write loops: 4 concurrent global stores per batch
         Beta/Delta loops have runtime conditional yp1 — skipped (bounds check complicates unroll)
         Expected: ~5-10% speedup for reg-blocked V-DWT path (small subbands h ≤ 140)
         Parity: Slang V60
    V96: #pragma unroll 2 for H-DWT 4-row lifting loops (Alpha/Beta/Gamma/Delta) — 4K focus
         Interior Alpha/Beta/Gamma/Delta: `for x=1+t*2; x<w-1; x+=nt*2` — 2 iters for 4K level 1
         4K L1: w=1920, nt=512 → (1920-2)/(512*2) ≈ 2 iters/thread → unroll 2 = fully inline
         2K L1: w=960, nt=512 → ~1 iter → pragma ignored; no code bloat on 2K path
         Each lifting loop: 2× HFMA2 pairs per unrolled body → 4 independent HFMA2/HADD2 in flight
         Compiler interleaves 2 iterations' smem reads (23-cycle latency) → better latency hiding
         Applies to all 4 DWT lifting stages (Alpha/Beta/Gamma/Delta) per block
         Expected: ~3-7% H-DWT speedup for 4K content → ~0.5-1% overall (H-DWT ~15% GPU time)
         Parity Slang V59. Safe: no data dependency between unrolled iterations; smem positions nt*2 apart
    V95: sm_lut preload for kernel_rgb48_xyz_hdwt0_1ch_4row_p12 — parity with V90/V91 (1-row/2-row)
         Add __shared__ __half sm_lut[4096] (8KB static smem) + int4 vectorized preload
         4096 __half = 512 int4; nt=512T → each thread loads exactly 1 int4 (same as V91 pattern)
         Total smem: 15.36KB (DWT) + 8KB (sm_lut) = 23.36KB
         PreferShared (48KB/SM): 48/23.36=2 blk/SM (same as PreferNone before) → no occupancy change
         Each thread handles ~2 pixels × 4 rows = 8 LUT reads: 8 × __ldg (~30-50cy) → smem (~3cy)
         Savings: 8 reads × 27cy = 216cy/thread vs 0 overhead (preload already done by V91 pattern)
         Actually preload cost: 1 int4 __ldg (~10cy) + syncthreads (~5cy) = ~15cy total overhead
         Net: ~200 cycles saved per thread in load section → ~5-10% 4-row HDWT0 speedup
         Parity Slang V58. Safe: sm_lut replaces d_lut_in __ldg; occupancy unchanged at 2 blk/SM
    V94: #pragma unroll 2 in H-DWT 4-row interior load+scatter loops (kernel_fused_horz_dwt_half_io_4row)
         Load loop: `for x=t; x<w; x+=nt` — 2 iters per thread (2K L1: w=960/nt=512=2; 4K L1: w=1920/512=4)
         Scatter loop: `for p=t; p<w/2; p+=nt` — up to 2 iters (4K L1: 960/512=2; 2K L1: 480/512≤1)
         Unroll 2 on load: compiler issues 2×4=8 concurrent __ldg per thread → better DRAM latency hiding
         Unroll 2 on scatter: compiler issues 2×8 concurrent smem reads + 2×8 global writes → BW pipelining
         Load loop dominates: __ldg latency ~100 cycles; 2× concurrent loads cuts effective latency by 2
         All lifting loops (Alpha/Beta/Gamma/Delta) have ~1 iter at level 1 → no benefit, left unchanged
         Expected: ~2-5% H-DWT speedup → ~0.3-0.7% overall (H-DWT levels 1-4 ~15% of GPU compute)
         Parity Slang V57. Safe: pure code-gen change; loop semantics unchanged
    V93: vq8 — 8-element vectorized quantize (upgrade vq4→vq8 in kernel_quantize_subband_ml)
         vq8: 4×__half2 loads (16B) + 2×uint32_t stores (8B) per thread per iteration
         vs vq4: 2×__half2 loads (8B) + 1×uint32_t store (4B) — halves loop iterations again
         Same total memory traffic as vq4; 50% fewer loop-counter updates, branch checks, c+=nt ops
         All zone widths ≥ 2*ll5_c are div-by-8 (2K: ≥120/8=15 ✓; 4K: ≥240/8=30 ✓; zero tail)
         DC row only: two zones of width ll5_c (2K: 60, 4K: 120); for 2K: 60%8=4 residual tail
         vq4 tail: [vq8_end, col_end) for residual < 8; DC rows are 3.1% of rows → negligible cost
         For 4K (ll5_c=120, div-by-8): zero tail in ALL zones → pure vq8 everywhere
         Expected: ~3-5% quantize speedup → ~0.3-0.5% overall (quantize ~10% of GPU compute)
         Parity Slang V56. Safe: tail handler covers all residuals; static loop covers main body
    V92: V_TILE 22→24 — reduce V-DWT tile count while staying within 42-reg budget
         V_TILE_FL: 30→32; col2[32] uses 32 __half2 regs; + ~10 other = ~42 total → launch_bounds OK
         Tiles for 2K (height=1080): ceil(1080/22)=49 → ceil(1080/24)=45 → 8% fewer tiles
         Tiles for 4K (height=2160): ceil(2160/22)=99 → ceil(2160/24)=90 → 9% fewer tiles
         Overlap ratio: V_TILE_FL/V_TILE = 30/22=1.364 → 32/24=1.333 → 3% less overlap overhead
         Total reads/col: 49×30=1470 (2K lvl0) → 45×32=1440 → 2% less DRAM bandwidth
         V_TILE=24 even, V_OVERLAP=4 even → load_start=even → P0=0 unchanged; static_assert passes ✓
         Expected: ~2-4% V-DWT speedup → ~0.8-1.6% overall (V-DWT is ~40% of GPU compute)
         Parity Slang V55. Safe: constexpr propagates to all loops, grid, static_assert auto-verifies
    V91: int4 vectorized LUT preload — 1 × 128-bit __ldg per thread vs 8 scalar loads (V90)
         sm_lut[4096 __half] = 512 × int4 (8KB); nt=512T → each thread loads exactly 1 int4
         Replaces V90's #pragma unroll 8 loop with a single reinterpret_cast<int4*> + __ldg
         1 ld.global.b128 per thread vs 8 ld.global.b16 → saves 7 instructions per thread
         512 threads × 7 instructions = 3584 fewer global load instructions per block
         cudaMalloc guarantees 256B alignment → int4 access of d_lut_in is 16B aligned ✓
         Same __syncthreads() barrier preserved; sm_lut read pattern unchanged
         Expected: ~5-10% preload speedup (minor) but reduces instruction pressure
         Parity Slang V54. Safe: pure load vectorization, no functional change
    V90: Shared-memory preload of d_lut_in — 0-cycle smem LUT access vs ~30-50 cycle L1/L2 read
         Add __shared__ __half sm_lut[4096] (8KB static smem) to 1-row and 2-row p12 kernels
         Preload loop: 4096/512=8 sequential coalesced reads per thread (cheap, one L1 fill)
         Main load loop then reads sm_lut[ri] at smem speed (~3 cycles) vs d_lut_in __ldg (~30-50 cycles)
         3 LUT reads saved per pixel pair × ~4 pairs per thread = ~12 reads × ~27 cycles = 324 cycles/thread
         Cost: 1 extra __syncthreads (~5-10 cycles) + 8 preload reads (coalesced, ~1 cycle each) → ~20 cycles
         Net: ~300 cycles saved per thread = ~15× reduction in LUT access time
         Smem: 8192 (lut) + 7680 (DWT) = 15872B → PreferShared (48KB/SM) → 3 blk/SM (was 4 with PreferNone)
         3→4 blk/SM is -25% warps but +300 cycles/thread >> -25% warp count overhead
         d_lut_out stays as __ldg (accessed once/pair, more predictable, better L1 hit rate)
         Parity Slang V53. Expected: ~20-40% load-section speedup → ~3-8% overall improvement
    V89: #pragma unroll in load/lifting/scatter loops — expose ILP to hide LUT miss latency
         1-row p12 (4K): w=3840, nt=512 → ~4 iters/thread → #pragma unroll 4
         2-row p12 (2K): w=1920, nt=512 → ~2 iters/thread → #pragma unroll 2
         Load loop: compiler issues all N iterations' __ldg requests simultaneously (memory ILP)
         Lifting loops: compiler interleaves N iterations' smem reads/FMAs → hides 23-cycle smem latency
         Scatter loop: compiler interleaves N smem reads + global writes for better throughput
         d_lut_in has 4096 __half entries (8KB): random-access, ~50-cycle L2 miss latency
         With #pragma unroll 4: 4× more in-flight LUT requests → 4× better miss latency hiding
         Expected: ~5-15% load-section speedup → ~1-3% overall improvement
         Parity Slang V52. Safe: pure code-gen change, no functional change
    V88: __launch_bounds__(512,4) for HDWT0 RGB kernels — guarantee 4 blk/SM occupancy (smem-limited)
         kernel_rgb48_xyz_hdwt0_1ch_1row_p12 and _2row_p12: 512T, smem=7680B → 4 blk/SM smem-limited
         Without bounds: compiler may use 36 regs/T → 36×512=18432 → 65536/18432=3.55 → 3 blk/SM
         With __launch_bounds__(512,4): forces ≤ floor(65536/512/4)=32 regs/T → 4 blk/SM guaranteed
         3→4 blk/SM = +33% concurrent warp slots = better LUT miss latency hiding (d_lut_in random access)
         1-row and 2-row HDWT0 kernels are LUT-lookup-heavy → +33% warps significantly improve issue rate
         Expected: ~5-10% HDWT0 speedup → ~1-2% overall (HDWT0 is ~15-20% of GPU compute)
         Parity Slang V51. Safe: compiler spills ≤4 regs to L1 if needed; L1 latency < LUT miss latency
    V87: __launch_bounds__(256,6) for register-limited kernels — guarantee 6 blk/SM occupancy
         kernel_quantize_subband_ml and kernel_fused_vert_dwt_tiled_ho_2col: no smem → register-limited
         Without bounds: compiler may use 44 regs/T → 44×256=11264 → 65536/11264=5.82→5 blk/SM
         With __launch_bounds__(256,6): forces ≤ floor(65536/256/6)=42 regs/T → 6 blk/SM guaranteed
         5→6 blk/SM = +20% concurrent warp slots = better latency hiding for __half2float + __ldg
         V-DWT 2-col is ~40% GPU compute → 20% V-DWT gain = ~8% overall; quantize ~10% → ~2%
         Total expected: ~5-10% GPU compute speedup → ~1-2% overall (H2D-limited pipeline)
         Parity Slang V50. Safe: worst case no change; compiler may spill ≤2 regs to L1 (negligible)
    V86: __hmul2 scatter in 1-row RGB+HDWT0 kernel — 2× FP16 multiply throughput in scatter
         sm[p*2] and sm[p*2+1] are adjacent __half → form natural __half2 at smem bank p (no conflict)
         Load as __half2 (one ld.shared.b32) and apply {NORM_L, NORM_H} via one HMUL2 instruction
         Saves 1 smem load instruction and 1 scalar HMUL per scatter iteration (vs 2 loads + 2 HMULs)
         __low2half/__high2half extract L/H for separate scatter stores (2 st.global.b16 unchanged)
         bank(sm[p*2]) = p%32 for p=t → all 32 warp threads access distinct banks → conflict-free ✓
         Parity Slang V49. Expected: ~3-7% scatter speedup → ~0.3-0.7% overall 4K improvement
    V85: PreferL1 cache config for reg-blocked V-DWT + quantize kernels — parity Slang V48
         kernel_fused_vert_dwt_fp16_hi_reg_ho: no smem → PreferL1 gives 48KB L1 (was default 16KB)
         kernel_quantize_subband_ml: no smem → PreferL1 gives 48KB L1
         Both kernels are register-limited not smem-limited → cache config change is free (no occ loss)
         Larger L1: better temporal locality for column data in reg-blocked V-DWT;
                    better instruction cache coverage for the quantize zone branches
         Expected: 1-3% speedup in reg-blocked V-DWT and quantize → ~0.1-0.2% overall
    V84: vq4 — 4-element vectorized quantize (upgrade vq2 → vq4 in kernel_quantize_subband_ml)
         vq4 loads 2×__half2 (8 bytes) per thread and stores 1×uint32_t (4 bytes) per iteration
         vs vq2: 1×__half2 load (4 bytes) + 1×uint16_t store (2 bytes) per iteration
         Halves loop overhead (iterations, branch, counter): same total memory traffic, 50% less control flow
         Zone boundaries are multiples of ll5_c=stride/32 (=60 for 2K, 120 for 4K), all div by 4 → no tail
         All zones have widths divisible by 4 → vq4 covers 100% of columns in every zone
         Expected: ~3-5% quantize speedup → ~0.2-0.4% overall improvement — parity Slang V47
    V83: __half2 row-pair packing in kernel_rgb48_xyz_hdwt0_1ch_2row_p12 — parity Slang V46
         Hoist y1<height to block level; interior path uses __half2 interleaved smem (w __half2 = 2w __half)
         Load: store {e0,e1} and {o0,o1} as __half2 pairs into sm2[] (sm2[x]={row0[x],row1[x]})
         Lifting: __hfma2 for Alpha/Beta/Gamma/Delta — 2× FMA throughput (2 ops vs 4 scalar per step)
         Scatter: combined L+H loop via __low2half/__high2half (same as V82 pattern)
         Partial path: y0-only scalar (height always even for 2K/4K → partial never taken in practice)
         Expected: 5-10% H-DWT speedup within RGB+HDWT0 kernel → ~1-2% overall improvement
    V81: Pair-wise 12-bit unpack in RGB+HDWT0 kernels — 25% fewer byte loads + combined scatter
         Load loop: px=t,t+nt per pixel → pair=t,t+nt per 2-pixel pair (shared middle byte)
         Per pair: 9 byte loads (3 bytes × 3 channels, shared byte[1]) vs 12 (4 × 3 with duplicate)
         Combined scatter: two separate even/odd loops → one paired L+H loop per pair index
         Expected: 5-10% RGB+HDWT0 speedup (25% kernel) → ~1-2% overall improvement
    V80: V_TILE=22, V_OVERLAP=4 for 2-col V-DWT — +20% throughput; write-combining H2D
         V_TILE: 28→22, V_OVERLAP: 5→4, V_TILE_FL: 38→30 → 2-col col2[] from 38→30 __half2 regs
         2-col kernel: ~50 regs/T (V_TILE_FL=38) → ~42 regs/T (V_TILE_FL=30) → 6 blk/SM (75% occ)
         Throughput: 5×28/38=3.68 → 6×22/30=4.40 → +19.6% V-DWT throughput; ~8% overall gain
         P0 changes to 0 (load_start always even: V_TILE=22 even, V_OVERLAP=4 even)
         Lifting loops: ALPHA/GAMMA start at i=1 (odd, globally-odd), BETA/DELTA start at i=2
         write-combining: h_rgb12_pinned uses cudaHostAlloc(..., cudaHostAllocWriteCombined)
         CPU only writes into h_rgb12_pinned (GPU reads via DMA) → safe + up to 40% H2D gain
    V79: 4K RGB+HDWT0: new 1-row p12 kernel — 50% → 100% occupancy for 4K
         kernel_rgb48_xyz_hdwt0_1ch_1row_p12: 1 row per block, smem=width×2B.
         For 4K (width=3840): smem=7,680B. PreferNone(32KB/SM)→4 blk/SM → 100% occ.
         Replaces 4K 2-row (smem=15.36KB→2 blk/SM=50% occ). Same wave count.
         Simpler code: no y1 guard, no second-row smem section.
         Expected: 30-50% RGB+HDWT0 speedup for 4K content → ~7-12% overall gain at 4K.
    V78: 2K RGB+HDWT0: switch from 4-row to 2-row kernel — 50% → 100% occupancy
         After V75: 4-row RGB+HDWT0 at 2K runs at 50% occ (PreferNone → 2 blk/SM × 512T = 1024T).
         2-row at 2K: smem=2×1920×2=7680B; PreferNone→32KB/SM→4 blk/SM → 2048T = 100% occ.
         Wave count identical: 540 blk/(20SM×4)=6.75 vs 270/(20SM×2)=6.75 (unchanged).
         But 100% occ → 2× warps/SM → 2× latency hiding slots for LUT random-access misses.
         4K path unchanged: 2-row at 4K with PreferNone → 2 blk/SM=50% occ (no regression).
         Change: `use_2row_rgb=(width>2048)` → `use_2row_rgb=true` (always use 2-row variant).
         Expected: 30-50% RGB+HDWT0 speedup (2K, 25% of GPU time) → ~7-12% overall gain.
    V77: Fix H-DWT 4-row PreferL1 occupancy bug at level 1 — 50% → 100% occupancy
         kernel_fused_horz_dwt_half_io_4row at level 1 (w=960): smem=4×960×2=7.68KB.
         PreferL1 → 16KB smem/SM → 16384/7680=2 blk/SM → 1024T = 50% occupancy — regression.
         PreferNone → 32KB smem/SM → 32768/7680=4 blk/SM → 2048T = 100% occupancy.
         V67 comment "smem fits in 16KB smem of PreferL1" was correct for per-block fit but
           ignored that 2 blocks × 7.68KB=15.36KB fills 16KB smem/SM → only 2 not 4 blk/SM.
         Level 2+ (w≤480, smem≤3.84KB): 4 blk/SM with both PreferL1 and PreferNone → no change.
         Trade-off: 48KB→32KB L1 at levels 2-5, but those are small levels with tiny data sets.
         Level 1 is the largest and slowest H-DWT level (270 blocks, 960×540 data = 1MB).
         With 100% occ: 4 blk/SM → 4× latency hiding → near-eliminate memory stall bubbles.
         Expected: 30-50% H-DWT level-1 speedup → H-DWT ~15% total → ~5-7% overall gain.
    V76: 2-column-per-thread __half2 tiled V-DWT — HFMA2 doubles lifting throughput
         kernel_fused_vert_dwt_tiled_ho_2col: each thread processes col x and x+1 simultaneously.
         cdf97_lift_tiled_h2: __half2 lifting using __hfma2(kA, __hadd2(prev,next), cur).
         HFMA2 on sm_61: processes 2 __half FMAs per instruction → 2× arithmetic throughput.
         Grid x = ceil(width/2/V_THREADS_TILED): half as many x-blocks; each block does 2× work.
         __half2 col2[38] ≈ 38-50 regs/T → 5-6 blk/SM (75-62.5% occ vs 8 blk = 100%).
         Effective throughput: 5-6 blk × 2 cols × 256T vs 8 × 1 × 256T → 25-50% gain.
         Load: `__ldg(int*)` at x (even) → reinterp as __half2; store: `*(int*)` at x.
         Req: width even (2K=1920, 4K=3840: yes); x always even (t×2 step) → aligned loads.
         Expected: 25-50% V-DWT speedup; V-DWT ~40% GPU time → 10-20% overall improvement.
    V75: Fix PreferL1 occupancy bug for 4-row/2-row RGB+HDWT0 kernels — 2× more blocks/SM
         kernel_rgb48_xyz_hdwt0_1ch_4row_p12 smem=4×1920×2=15.36KB; 2-row 4K=2×3840×2=15.36KB.
         PreferL1 → 16KB smem/SM → only 1 block/SM at 15.36KB (25% occupancy) — severe underuse.
         PreferNone (default, 32KB smem/SM) → 32/15.36=2 blocks/SM → 1024T = 50% occupancy.
         32KB L1 (PreferNone) still caches 16KB LUTs (lut_in+lut_out) with 16KB room for RGB.
         V72 comment "PreferL1 honored at 2K 4-row" was wrong: smem fits but 2nd block does not.
         Only kernel_rgb48_xyz_hdwt0_1ch (1-row, smem=3.84KB) retains PreferL1 (4 blk/SM, 100% occ).
         Expected: 5-15% RGB+HDWT0 throughput improvement from 2× more active warps/SM.
    V74: 4-rows-per-block H-DWT level 0 (i2f+DWT) — 75% fewer block dispatches for level-0
         kernel_fused_i2f_horz_dwt_half_out: 1 row/block (height=1080 blocks) → 4 rows/block (270 blocks).
         Same 4 __syncthreads amortized over 4× the work; 4 row load chains in-flight → better latency hiding.
         smem=4×1920×2=15.36KB (2K): set PreferShared (48KB smem → 3 blk/SM = 75% occ).
           PreferL1 would give 16KB smem limit → 1 blk/SM (25% occ) since 15.36≥16 is tight.
           No LUTs in this kernel → no benefit from L1 LUT caching; occupancy dominates.
         smem=4×3840×2=30.72KB (4K): PreferShared mandatory; 3 blk/SM if SMem is enough.
         V73 adaptive blk count applies to levels 1+ only; level 0 always uses full width → H_THREADS.
         Expected: 3-8% level-0 H-DWT speedup → ~0.5-1.5% overall improvement (level-0 ~15% GPU time).
    V73: Adaptive H-DWT thread count for levels 1+ — eliminates 50-77% wasted threads at small widths
         H-DWT levels 1+: fixed H_THREADS=512 wastes threads when w < 256 (level 2-4 for 2K).
         2K level-2 w=480: 512T → 480/512=93.75% util. level-3 w=240: 512T → 240/512=46.9%.
         level-4 w=120: 512T → 120/512=23.4%. Adaptive: (w>480)?512:(w>240)?256:(w>120)?128:64.
         level-3 2K: 512T→256T → 240/256=93.75% utilization; 256T occupancy ≥ 512T on sm_61.
         level-4 2K: 512T→128T → 120/128=93.75%; occupancy: 65536/(128×regs)>128/(512×regs)×4.
         4K levels 1-2 (w≥960): still 512T (no change); level-3 (w=480) 256T, level-4 128T.
         Kernel s17_dwt_h_half_io_4row uses blockDim.x dynamically — works with any thread count.
         Expected: 5-15% H-DWT speedup for levels 2-4 (~15% of GPU time → ~0.75-2% overall).
    V72: 2-row RGB+HDWT0 for 4K — enables PreferL1 at 4K, improves LUT cache hit rate
         4K 4-row smem=30.72KB > 16KB PreferL1 limit → runtime uses PreferShared (16KB L1).
         4K 2-row smem=15.36KB < 16KB → PreferL1 honored (48KB L1 for lut_in+lut_out=16KB).
         With PreferShared at 4K: 16KB L1 fits LUTs (16KB) exactly → prone to eviction with RGB.
         With PreferL1 at 4K: lut_in(8KB)+lut_out(8KB)=16KB in 48KB L1, ample room for RGB.
         2K path unchanged: 4-row smem=15.36KB < 16KB → PreferL1 already works.
         4K grid: ceil(2160/2)=1080 vs ceil(2160/4)=540 blocks (+100% dispatch), but LUT cached.
         Expected: 5-15% RGB+HDWT0 speedup at 4K (~25% GPU time → 1-4% overall 4K improvement).
    V71: V_TILE 24→28 — 4K DRAM savings, 100% occupancy retained (31 regs, within 32 limit)
         V_TILE=28: V_TILE_FL=38 → __half col[38] → ~31 regs/T → 65536/(256×31)=8.2 → 8 blk = 100%.
         4K level-0 overlap rows: 10 rows × 3840px × 2B = 75KB > 48KB L1 → DRAM-bound.
           V_TILE=24: 90×34-2160=900 overlap reads/col; V_TILE=28: 77×38-2160=766 → 15% fewer.
         2K level-0 (overlap in L1): 45→39 tiles, 450→402 overlap reads/col → 11% fewer.
         Grid tiles: 2K ceil(1080/28)=39 (vs 45), 4K ceil(2160/28)=77 (vs 90) → 13-14% fewer dispatches.
         Expected: 2-6% V-DWT speedup, more for 4K (DRAM-bound overlaps) than 2K (L1-cached overlaps).
    V70: V_TILE 16→24 for tiled V-DWT — reduce overlap overhead, 100% occupancy retained
         V69 __half col[V_TILE_FL]: ~23 regs/thread → thread-limited at 8 blk/SM (100% occ).
         V_TILE=24: V_TILE_FL=34 → __half col[34] → ~29 regs/thread → 65536/(256×29)=8.8 → 8 blk/SM.
         8 blk/SM = thread-limited = 100% occupancy (same as V69 V_TILE=16, no regression).
         Useful rows per tile: 16/26=61.5% → 24/34=70.6% → 14.8% fewer total DRAM reads/col.
         Grid tiles: ceil(1080/16)=68 → ceil(1080/24)=45 (2K level-0) → 34% fewer block dispatches.
         #pragma unroll covers V_TILE_FL=34 — no code changes needed (constexpr propagates).
         Expected: 3-8% V-DWT speedup from reduced memory traffic (~40% GPU time → ~1-3% overall).
    V69: __half column registers in V-DWT tiled + reg-blocked kernels — 2× HFMA throughput
         kernel_fused_vert_dwt_tiled_ho: float col[V_TILE_FL] → __half col[V_TILE_FL].
         kernel_fused_vert_dwt_fp16_hi_reg_ho: float col[MAX_REG_HEIGHT] → __half col[MAX_REG_HEIGHT].
         Eliminates 26 __half2float loads + 16 __float2half stores per tile (tiled kernel).
         New cdf97_lift_tiled_h: __half lifting with P0=1 hardcoded + #pragma unroll.
         __half HFMA has 2× throughput vs float FMA on sm_61+; V-DWT lifting is ~40 FMAs/tile-col.
         Reg-blocked: 140-float col[] → 70-half-register-equivalent; compiler may pack pairs.
         Tiled: interior path #pragma unroll over __half col[26] → HFMA2 sequences possible.
         Expected: 3-8% V-DWT speedup (V-DWT ~40% of GPU time → ~1-3% overall speedup).
    V68: Fuse 4-row RGB loads into interior if/else block — 12 __ldg in-flight per pixel
         kernel_rgb48_xyz_hdwt0_1ch_4row_p12: load section restructured with if(y3<height).
         Interior (100% of 2K blocks): single for loop issues 24 byte __ldg + 8 LUT __ldg
         simultaneously (vs 4 loops of 6+2 sequential) → full 4-row load ILP from GPU MLP.
         Else (partial last block): original guarded per-row loads (y3 not loaded — out of bounds).
         #undef UP12 moved to after the if/else block.
         Expected: 3-8% RGB+HDWT0 speedup from improved load-pipeline MLP utilization.
    V67: CachePreferL1 for H-DWT levels-1+ kernels (kernel_fused_horz_dwt_half_io_4row/2row)
         Adds 3 cudaFuncSetCacheConfig(PreferL1) calls at init time.
         H-DWT kernels use smem ≤ 4*960*2=7.5KB (level-1 2K) — fits in 16KB smem of PreferL1.
         PreferL1 expands L1 from 16KB→48KB for `__ldg` input caching (V-DWT output).
         Adjacent row-groups share ~7.5KB of input rows across sequential blocks → L1 hit rate.
         Expected: 2-5% H-DWT levels-1+ speedup from improved `__ldg` cache hit rate.
    V66: Hoist row-presence checks out of H-DWT levels-1+ 4-row lifting loops
         kernel_fused_horz_dwt_half_io_4row: single `if (y3 < height)` check at block level.
         Interior path: levels 1/4 (h=540/68 for 2K, 100% blocks), levels 2/3 mostly interior.
         Lifting loops (ALPHA/BETA/GAMMA/DELTA): 3 fewer `if(yN<height)` guards per iteration.
         Load loop: 4 rows loaded in single for-body (no yN<height) → coalesced 4-row read.
         Store loops: 4 unconditional writes each → 12 fewer branches/thread in store phase.
         Partial last block (height%4!=0): original checked path preserved unchanged.
         Expected: 3-7% H-DWT levels-1+ speedup (levels 1-4 together ~20% of GPU time).
    V65: Hoist row-presence checks out of RGB+HDWT0 4-row lifting loops
         kernel_rgb48_xyz_hdwt0_1ch_4row_p12: single `if (y3 < height)` check at block level.
         Interior path (100% of 2K blocks since 1080%4==0): no yN<height inside lifting loops.
         4 rows' ALPHA/BETA/GAMMA/DELTA loops fused per iteration → compiler can schedule ILP across
         rows (4 FP16 FMAs/iter instead of 1+3 conditionals); branch-predictor pressure eliminated.
         Store loops: 4 unconditional writes each (no yN<height checks) → 12 fewer branches/thread.
         Partial last block (height%4!=0, e.g., odd heights): original checked path preserved.
         Expected: 5-10% RGB+HDWT0 speedup (second-largest kernel, ~25% of GPU time).
    V64: Interior-tile unrolled load+store for tiled V-DWT — branch-free for 97% of blocks
         For tile_y ∈ [1, num_tiles-2] (interior): load_start ≥ 0 and load_start+V_TILE_FL ≤ height.
         Interior path: #pragma unroll loop with NO bounds check → 26 independent __ldg instructions.
         GPU memory pipeline issues all 26 loads simultaneously; warp scheduler hides full latency.
         Interior output loop: #pragma unroll over exactly V_TILE=28 iterations (compile-time bound).
         !(i&1) evaluated at compile time per unrolled step → 8 L stores + 8 H stores, straight-line.
         Boundary path (first + last tile, 2/68 = 3%): keep existing WS bounds-check code unchanged.
         Expected: 5-12% V-DWT speedup (eliminates loop-overhead + branch-predict pressure for 97% tiles).
    V63: V_THREADS 128→256 for V-DWT tiled path (large-h levels 0-2)
         8 warps/block vs 4 at 128T: warp scheduler has 2× more choices to hide memory latency.
         Fewer blocks: ceil(1920/256)×68=544 (level 0) vs ceil(1920/128)×68=1020 → 47% fewer.
         Same 87.5% SM occupancy (register-limited: 65536/(256×36)=7 blk/SM × 256=1792T).
         128T kept for reg-blocked path (h≤140): 140-float col → ~150 regs → 1 blk/SM at 256T.
         Expected: ~3-5% additional V-DWT speedup on dominant large-h levels.
    V62: V_TILE 32→16 for V-DWT tiled kernels — higher occupancy wins vs fewer loads
         V59 PreferL1 makes overlap rows (10 rows=37.5KB) always hit 48KB L1, regardless of V_TILE.
         V_TILE=32: V_TILE_FL=42 floats → ~50 regs/thread → 10 blk/SM → 40 warps (need ~116 for DRAM).
         V_TILE=16: V_TILE_FL=26 floats → ~36 regs/thread → 14 blk/SM → 56 warps (+37% latency cover).
         Extra 340 overlap loads/column all hit L1 → effectively free bandwidth.
         Est. 20-25% V-DWT speedup → ~8-10% overall GPU improvement
    V61: 4-rows-per-block kernel_rgb48_xyz_hdwt0_1ch_4row_p12 for level-0 RGB+HDWT0
         grid=(height+3)/4=270 blocks per component (vs 540 for 2K); smem=4*w*sizeof(__half)
         Halves block count vs V54 2-row; same 100% SM occupancy (thread-limited: 4×512T=2048T)
         Matrix m0/m1/m2 amortized over 4 rows; 4 syncthreads passes over 4 rows
         CachePreferL1 applied: 16KB LUTs cached alongside 15.4KB smem
         Expected: ~1-2% RGB+HDWT0 speedup from halved block dispatch overhead
    V60: full 100% vectorized quantize — extend vq2 to L4/L5/DC rows (was scalar fallback)
         L4/L5/DC rows (row_lv ≥ 4, ~12.5% of rows) now use zone-structured vq2 calls.
         DC rows: 6 zones [0,ll5_c)→inv_dc, [ll5_c,2c)→inv_l5, [2c,4c)→inv_l4, [4c,8c)→inv_l3,
                            [8c,16c)→inv_l2, [16c,stride)→inv_l1
         L5 rows: 5 zones starting at 0→inv_l5; L4 rows: 4 zones starting at 0→inv_l4.
         Zone boundaries are multiples of ll5_c (= stride/32): even for 2K (60) and 4K (120).
         Eliminates all scalar loads (uint8_t/half) and stores; 100% of rows now use vq2.
         Expected: ~3% additional quantize speedup (12.5% more rows vectorized).
    V59: cudaFuncSetCacheConfig(CachePreferL1) for V-DWT tiled + RGB+HDWT0 kernels
         V-DWT tiled has no smem — default split gives only 16KB L1; PreferL1 gives 48KB
         Adjacent tiles in Y share V_OVERLAP=5 rows (10 rows overlap) — 48KB L1 can hold
         10 rows × 1920 cols × 2B = 38.4KB (fits in 48KB L1, not in 16KB)
         RGB+HDWT0 has 7.5KB smem < 16KB limit — 48KB L1 holds full 16KB LUT set
         PreferL1 reduces DRAM traffic for V-DWT level 0 (dominant BW kernel): est. 5-15% speedup
    V58: vectorized quantize for L1/L2/L3 rows — __half2 loads + uint16_t stores (~87.5% of rows)
         L1 rows (row ≥ 544 for 2K, 50%): 1 uniform zone → inv_l1; full vectorized
         L2 rows (272..543, 25%): 2 zones at stride/2: [0..960)→inv_l2, [960..1920)→inv_l1
         L3 rows (136..271, 12.5%): 3 zones at 480/960: inv_l3/inv_l2/inv_l1
         vq2 lambda: __half2 load + 2-sample quantize + uint16_t store per iteration
         Zone boundaries (ll5_c × {8,16}) always even → uint16_t alignment guaranteed
         L4/L5/DC rows (~12.5%): scalar fallback unchanged
         Expected: ~40% quantize speedup → ~0.12ms/frame saved
    V57: per-subband QCD correctness fix — codestream step entries match V53 perceptual weights
         V53 applied level-dependent weights (LL5×0.65 … L1×1.20) but QCD/QCC wrote uniform steps
         Decoder dequantized LL5 at 1/0.65=1.54× amplitude → DC reconstructed wrong
         Fix: j2k_perceptual_sb_entry(base, i, is_4k) encodes actual step for each subband i
         j2k_qcd_step_entry converts float step → (eps<<11)|man using J2K scalar-expounded formula
         Both build_j2k_codestream (pipeline) and run_dwt_and_build_codestream (fallback) updated
         4K path unchanged (uniform — 4K doesn't apply perceptual weights)
    V56: per-channel pack+H2D pipeline — H2D starts 0.2ms earlier (pack_rgb12_plane)
         Previous: pack all 3 channels (0.3ms) → H2D all at once → H2D done at 1.56ms
         Now: pack ch0 (0.075ms) → H2D ch0; pack ch1 (0.075ms) → H2D ch1; pack ch2 → H2D ch2
         H2D starts at 0.075ms instead of 0.3ms; total pack+H2D: 0.075+1.26=1.335ms (vs 1.56ms)
         Each channel plane packed independently; 4 threads × (height/4) rows per channel
         pack_rgb12_plane replaces pack_rgb12_chunk inner ch loop; encode_from_rgb48 steps 1+2 merged
    V55: half-precision lut_in — L1 cache footprint 24KB → 16KB (d_lut_in float*→__half*)
         lut_in GPU allocation: 4096×4=16KB → 4096×2=8KB; lut_in+lut_out=16KB fits in 16KB L1
         Kernels: __ldg returns __half; __half2float() before matrix multiply (no precision loss: lut values ≤1.0)
         Host GpuColourParams::lut_in stays float[4096]; converted float→__half at upload time
         Reduces texture cache pressure per SM; more L1 available for DWT smem and d_a reads
    V54: 12-bit planar packed H2D — 25% smaller PCIe transfer (kernel_rgb48_xyz_hdwt0_1ch_2row_p12)
         CPU: pack_rgb12_chunk (4 threads) packs RGB48LE → 12-bit planar (3 channels × (w/2*3) × h)
         H2D: 9.5MB vs 12.6MB for 2K (1.26ms vs 1.68ms) → FPS ceiling raised ~600→~790fps
         GPU: better coalescing (pair-sharing threads, ~48-byte spans vs stride-6 uint16_t)
         New buffers: d_rgb12[2] + h_rgb12_pinned[2]; kernel reads from d_rgb12
         CPU pack time ~0.3ms (4 threads, 270 rows each) — hidden by 1.26ms H2D
    V53: 6-band perceptual quantization — multi-level step weighting (kernel_quantize_subband_ml)
         Distinguishes all 5 DWT levels: LL5×0.65, L5-AC×0.85, L4×0.95, L3×1.05, L2×1.12, L1×1.20
         V52 treated all non-LL5/non-L5 subbands as one band (×1.15); now each level is distinct
         6 __frcp_rn per block (amortized over stride cols); per-col: 2 chain selects + 1 min
         Better PSNR in level-2/3/4 content (textures, edges) without throughput change
    V52: 2D subband-aware quantization — correct LL5 column boundary (kernel_quantize_subband_2d)
         V50 used row < ll5_height to detect DC, but rows 0..ll5_h-1 also contain LH1 (cols stride/2..stride)
         Fix: LL5 = row < ll5_h AND col < ll5_cols (where ll5_cols = stride >> 5)
         Step weights: LL5×0.70 (DC, finest), LH5/HL5/HH5×0.90, all higher-freq×1.15
         Result: DC gets proper bit allocation; LH1 no longer incorrectly gets DC-quality precision
         kernel_quantize_subband_h replaced by kernel_quantize_subband_2d (same launch dims)
    V51: adaptive codestream trimming — trailing-zero removal per component
         find_actual_per_comp() scans backward (8-byte words) to find last non-zero byte
         TLM Ptlm[c] and SOT Psot updated to use actual[c] instead of max per_comp
         Simple/dark frames shrink significantly; complex frames unaffected; no quality loss
         Both build_j2k_codestream paths updated (pipeline V42+ and non-pipeline fallback)
    V50: target 620+ fps (quality) — subband-aware quantization (kernel_quantize_subband_h)
         Rows 0..ll5_h-1 (DC/LL5 region): step × 0.80 (20% finer — DC is most perceptually important)
         Rows ll5_h..2*ll5_h-1 (level-4 region): step × 0.95 (slight boost for mid-frequency)
         Rows ≥ 2*ll5_h (level 3 and above): step × 1.10 (coarser — higher-frequency detail)
         ll5_height = ceil(height/32); for 2K: 34; for 4K: 68
         Better PSNR/SSIM in low-frequency content at same bitrate; no throughput change
         launch_comp_pipeline now uses kernel_quantize_subband_h (row-grid, width×1 threads)
    V49: target 620+ fps (correctness) — fix 4K encoder + stream sync race
         4K correctness: launch_comp_pipeline now loops num_levels=is_4k?6:5 (was hardcoded 5)
           - 4K DCI requires 6-level DWT; codestream COD already declared 6 but GPU only did 5
         4K QCD/QCC: nsb now 3*(is_4k?6:5)+1=19 for 4K (was hardcoded 16 for both resolutions)
           - Incorrect QCD length caused invalid 4K codestreams (decoder step mismatch)
         Stream sync: sync stream[1] and stream[2] before build_j2k_codestream (was only stream[0])
           - Race condition: 3 component streams write h_packed_pinned independently; reading
             all 3 while only stream[0] was synchronized could yield stale component data
         Early graph launch: launch new_buf graphs BEFORE CPU codestream building (V49 timing)
           - Reduces sync_wait for next frame by ~0.1ms (GPU starts new_buf compute sooner)

    V24 Improvements over V23:
    1. Per-component QCC markers (codestream consistency fix):
       - V23 quantizes Y with step≈32.5 and X/Z with step≈35.75 (10% coarser)
       - But QCD written as 34.1 (geometric mean) — inconsistent for all 3 components
       - V24: QCD for Y (component 1, step=32.5), QCC for X (comp 0) and Z (comp 2)
         with their actual step=35.75. Decoder dequantization now matches encoder.
       - J2K decoders that use QCD/QCC to reconstruct the signal are now correct.
       - Note: DCP validator (dcpverify) typically ignores QCC for structural compliance,
         but this makes the decoder output numerically match the encoded coefficients.

    V25 Improvements over V24:
    1. fp16 (half-precision) vertical DWT workspace — halves V-DWT memory bandwidth:
       - The vertical DWT is memory-bandwidth-bound on GTX 1050 Ti (sm_61).
       - kernel_fused_vert_dwt_fp16 uses __half for the per-column lifting workspace.
       - Column copy: float→half (2 bytes vs 4 bytes per element saved on writes).
       - Lifting loops (4 passes, 2 reads+1 write each): half vs float → ~2× bandwidth.
       - Output: half→float with NORM_L/NORM_H.
       - Precision: fp16 has 10-bit mantissa; for DWT coefficients ±8192, ULP ≈ 8,
         << quantization step 32.5. Accumulated error across 4 lifting steps < 1 quant.
       - d_b[c] (float) still used for H-DWT temp. d_half[c] (half) added for V-DWT.
    2. Estimated gain: ~1-2ms per frame (V-DWT is ~20-30% of total GPU time).

    V26 Improvements over V25:
    1. fp16 H-DWT output — eliminates float→half copy in V-DWT:
       - H-DWT kernels now write __half to d_b[c] (was float)
       - Shared memory lifting stays float32 (full precision, only output is half)
       - V-DWT kernel reads __half d_b directly — no float2half conversion
       - d_b[c] allocation: pixels × 2 bytes (was × 4 bytes) — saves GPU memory too
       - No pointer-swapping needed: d_a=float (V-DWT output), d_b=half (H-DWT output)
    2. Saves ~24MB memory traffic per frame (H-DWT write + V-DWT input read saved)
       → ~0.21ms at 112 GB/s

    V27 Improvements over V26:
    1. Register-blocked V-DWT for small subbands (height ≤ MAX_REG_HEIGHT = 140):
       - New kernel: kernel_fused_vert_dwt_fp16_hi_reg
         * Reads __half d_src (H-DWT half output, same as V26)
         * Loads entire column into float registers (col[MAX_REG_HEIGHT])
         * CDF 9/7 lifting purely in registers — no global memory workspace at all
         * Writes float d_dst (same as V26)
       - For 2K: levels 3 (h=135) and 4 (h=68) use register-blocked path
         (levels 0-2 have h > 140 and use fp16-workspace path as in V26)
       - Eliminates d_half_work accesses for small subbands:
         * 4 lifting steps × (read + write) × 135 rows × 2048 cols × 2 bytes ≈ 4.4MB
         * Level 4: ~2.2MB; total ~6.6MB × 3 components ≈ 20MB/frame saved
         → ~0.18ms at 112 GB/s
       - Matches Slang V17's existing s17_dwt_v_reg optimization (now parity)
    2. Trade-off: higher register pressure (140 float regs/thread) → lower occupancy
       for these kernels, but small subbands are latency-bound not throughput-bound,
       so register blocking wins over extra memory traffic.

    V28 Improvements over V27:
    1. Fused RGB+H-DWT level-0 kernel (kernel_rgb48_xyz_hdwt0):
       - One block per row; shared memory 3×width floats (24KB for 2K)
       - Phase 1: RGB48→XYZ into smX/smY/smZ via LUT+matrix
       - Phase 2-4: sequential in-place H-DWT on each channel, write __half output
       - Eliminates d_in[0..2] (int32 XYZ planes): saves 27MB write + 27MB read = 54MB
       → ~0.48ms at 112 GB/s
    2. encode_from_rgb48 calls fused kernel, passes skip_level0_hdwt=true to DWT loop.

    V29 Improvements over V28:
    1. Register-based tiled vertical DWT (kernel_fused_vert_dwt_tiled):
       - Processes height in tiles of V_TILE=28 rows, V_OVERLAP=5 halo rows per side
       - V_TILE_FL=38 rows loaded from __half into float registers; NO d_half_work at all
       - CDF 9/7 lifting purely in float registers (no DRAM intermediate writes)
       - Whole-point symmetric extension handles borders without special-casing:
           y<0 → -y;  y>=h → 2(h-1)-y  → automatically matches standard boundary conds
       - Memory traffic: read(7.25MB) + write(8.85MB) = 16.1MB per component
         vs fp16-workspace: copy+4×lifting+deinterleave ≈ 39.8MB → 2.5× reduction
         Total savings: (39.8-16.1) × 3 = 71MB/frame → ~0.63ms at 112 GB/s
       - Grid: dim3((w+128-1)/128, (h+16-1)/16) — 2D launch, no shared memory
       - Replaces kernel_fused_vert_dwt_fp16_hi for h > MAX_REG_HEIGHT
*/

#include "cuda_j2k_encoder.h"
#include "gpu_ebcot.h"
#include "gpu_ebcot_t2.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <future>
#include <mutex>
#include <vector>


/* ===== J2K Codestream Constants ===== */
static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_QCC = 0xFF5D;  /* V24: per-component QCD override */
static constexpr uint16_t J2K_TLM = 0xFF55;
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;

static constexpr int NUM_DWT_LEVELS  = 5;
/* V27: column height threshold for register-blocked V-DWT.
 * For h ≤ MAX_REG_HEIGHT, the entire column fits in registers (float col[140]).
 * Eliminates fp16 workspace accesses for small subbands; reduces DRAM traffic. */
static constexpr int MAX_REG_HEIGHT  = 140;
/* V29/V34/V62/V70/V71/V80: tiled V-DWT parameters.
 * V80: V_TILE 28→22, V_OVERLAP 5→4 — reduces 2-col col2[] from 38→30 __half2 regs.
 *   ~50 regs/T (V_TILE_FL=38) → ~42 regs/T (V_TILE_FL=30) → 6 blk/SM (75% occ, up from 5).
 *   Throughput: 5×28/38=3.68 → 6×22/30=4.40 → +19.6% V-DWT; V-DWT~40% total → ~8% gain.
 *   P0 changes to 0: load_start always even (V_TILE=22 even, V_OVERLAP=4 even).
 *   Lifting loop indices updated: ALPHA/GAMMA start i=1, BETA/DELTA start i=2 (skip boundary i=0).
 * V71: V_TILE 24→28 — further reduces overlap overhead; 100% occupancy retained (31 regs).
 * V70: V_TILE 16→24 — V69 __half lowered regs to ~23/T → thread-limited (100% occ); 14.8% fewer loads.
 * V62: V_TILE 32→16 (reverts V34) to improve SM occupancy given PreferL1 (V59).
 * OVERLAP=4 halo rows each side (covers 4-step CDF 9/7 stencil; P0=0 boundary proof verified). */
static constexpr int V_TILE    = 24;  /* V92: was 22; V_TILE_FL→32; 8% fewer tiles, ~2% less DRAM BW */
static constexpr int V_OVERLAP = 4;
static constexpr int V_TILE_FL = V_TILE + 2 * V_OVERLAP;  /* 32 */
/* V44: threads for fused RGB+HDWT0 kernel — 512 for 100% SM occupancy on Pascal */
static constexpr int H_THREADS_FUSED = 512;

/* CDF 9/7 lifting coefficients */
static constexpr float ALPHA = -1.586134342f;
static constexpr float BETA  = -0.052980118f;
static constexpr float GAMMA =  0.882911075f;
static constexpr float DELTA =  0.443506852f;

/* V23: CDF 9/7 analysis normalization constants.
 *
 * After the 4 lifting steps, the L (lowpass) channel has DC gain K = 1.230174
 * and the H (highpass) channel has DC gain 1/K = 0.812893 per 1D DWT level.
 *
 * Applied at output of each 1D DWT (both H and V directions):
 *   L samples ×= NORM_L  (divides out the lowpass gain: K × (1/K) = 1)
 *   H samples ×= NORM_H  (amplifies highpass to match energy: (1/K) × K = 1)
 *
 * Net effect over 5 2D DWT levels:
 *   LL5 gain = (NORM_L × NORM_L)^5 × K^10 = (1/K^2)^5 × K^10 = 1.0 ✓
 *   H subbands gain → 1.0 ✓
 *
 * After normalization, all subbands are in [-4095, 4095] for 12-bit input,
 * consistent with the standard J2K QCD step-size derivation. */
static constexpr float NORM_L = 0.812893197535108f;  /* 1/K: shrinks lowpass */
static constexpr float NORM_H = 1.230174104914001f;  /* K:   amplifies highpass */


/* ===== Device helpers ===== */

/** V104: Direct uint16→__half conversion via PTX cvt.rn.f16.u16 (1 instruction).
 *  Replaces __float2half(float(v)) = CVT.F32.U16 + CVT.F16.F32 (2 instructions).
 *  Exact for v ∈ [0, 4095]: all DCP XYZ output values in d_lut_out fit in half.
 *  Saves 1 CVT per d_lut_out lookup — ~6M instructions/frame for 2K, ~25M for 4K. */
__device__ __forceinline__ __half u16_to_f16(uint16_t v)
{
    __half h;
    asm("cvt.rn.f16.u16 %0, %1;" : "=h"(*reinterpret_cast<unsigned short*>(&h)) : "h"(v));
    return h;
}


/**
 * V127: GPU RGB48LE → XYZ12 conversion kernel for hybrid encoding.
 * Produces 3 planar int32 outputs (compatible with OpenJPEGImage::data()).
 * Each thread processes one pixel: applies input LUT, 3×3 matrix, output LUT.
 * Output values are 12-bit (0-4095) stored as int32_t per DCI spec.
 */
__global__ void
kernel_rgb48_to_xyz12_planar(
    const uint16_t* __restrict__ d_rgb16,
    const float*    __restrict__ d_lut_in,    /* 4096 float: 12-bit → linear [0,1] */
    const uint16_t* __restrict__ d_lut_out,   /* 4096 uint16: linear → DCI 12-bit */
    const float*    __restrict__ d_matrix,     /* 9 floats: combined RGB→XYZ matrix */
    int32_t* __restrict__ d_xyz_x,
    int32_t* __restrict__ d_xyz_y,
    int32_t* __restrict__ d_xyz_z,
    int width, int height, int rgb_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    int y = idx / width;
    int x = idx % width;
    int base = y * rgb_stride + x * 3;

    /* Read RGB48LE and extract 12-bit indices */
    int ri = min(static_cast<int>(d_rgb16[base + 0] >> 4), 4095);
    int gi = min(static_cast<int>(d_rgb16[base + 1] >> 4), 4095);
    int bi = min(static_cast<int>(d_rgb16[base + 2] >> 4), 4095);

    /* Input LUT: 12-bit → linear float */
    float r = d_lut_in[ri];
    float g = d_lut_in[gi];
    float b = d_lut_in[bi];

    /* 3×3 matrix multiply: RGB linear → XYZ linear */
    float xv = d_matrix[0]*r + d_matrix[1]*g + d_matrix[2]*b;
    float yv = d_matrix[3]*r + d_matrix[4]*g + d_matrix[5]*b;
    float zv = d_matrix[6]*r + d_matrix[7]*g + d_matrix[8]*b;

    /* Clamp to [0,1] */
    xv = fminf(fmaxf(xv, 0.0f), 1.0f);
    yv = fminf(fmaxf(yv, 0.0f), 1.0f);
    zv = fminf(fmaxf(zv, 0.0f), 1.0f);

    /* Output LUT: linear [0,1] → 12-bit DCI value */
    int out_idx = y * width + x;
    d_xyz_x[out_idx] = static_cast<int32_t>(d_lut_out[static_cast<int>(xv * 4095.5f)]);
    d_xyz_y[out_idx] = static_cast<int32_t>(d_lut_out[static_cast<int>(yv * 4095.5f)]);
    d_xyz_z[out_idx] = static_cast<int32_t>(d_lut_out[static_cast<int>(zv * 4095.5f)]);
}

/* ===== CUDA Kernels ===== */

#if 0  /* Dead kernels — superseded by __half-smem variants; disabled to avoid extern __shared__ type conflict */
/**
 * Fused int32→float conversion + horizontal DWT (level 0).
 * One block per row. All threads cooperate in shared memory.
 * Writes deinterleaved (L|H) result to d_tmp.
 */
__global__ void
kernel_fused_i2f_horz_dwt(
    const int32_t* __restrict__ d_input,
    float* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ float smem[];
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x;
    int nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = __int2float_rn(d_input[y * stride + x]);
    __syncthreads();

    /* V120: DCI width always even and >1 → drop width>1&&(width%2==0); min(1,w-1)=1; min(x+1,w-1)=x+1. */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    /* V23: Apply CDF 9/7 analysis normalization at deinterleave output.
     * L×NORM_L and H×NORM_H ensure all subbands are in [-input_range, input_range]. */
    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x] * NORM_L;
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * NORM_H;
}


/**
 * Fused horizontal DWT for levels 1+.
 * Reads from d_data (current level), writes deinterleaved to d_tmp.
 */
__global__ void
kernel_fused_horz_dwt(
    const float* __restrict__ d_data,
    float* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ float smem[];
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x;
    int nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = d_data[y * stride + x];
    __syncthreads();

    /* V120: DCI width always even and >1 → simplify boundary guards. */
    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    /* V23: Apply CDF 9/7 analysis normalization at deinterleave output.
     * L×NORM_L and H×NORM_H ensure all subbands stay in [-input_range, input_range]. */
    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x] * NORM_L;
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * NORM_H;
}
#endif  /* dead float-smem kernels */


/**
 * V26/V38: Fused int32→half + horizontal DWT (level 0), writes __half output.
 * V38: __half shared memory — fp16 FMA 2× throughput on sm_61+; smem halved.
 */
__global__ void
kernel_fused_i2f_horz_dwt_half_out(
    const int32_t* __restrict__ d_input,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ __half smem[];  /* V38: was float; fp16 lifting 2× faster on Pascal+ */
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x, nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = __float2half(__int2float_rn(__ldg(&d_input[y * stride + x])));
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(ALPHA) * (smem[x - 1] + smem[x + 1]);
    /* V120: DCI width always even and >1 → drop width>1&&(width%2==0) guard. */
    if (t == 0) smem[width - 1] += __half(2.0f * ALPHA) * smem[width - 2];
    __syncthreads();

    /* V120: DCI width>1 → min(1,width-1)=1; even x<width → x+1≤width-1 → drop min. */
    if (t == 0) smem[0] += __half(2.0f * BETA) * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(BETA) * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(GAMMA) * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += __half(2.0f * GAMMA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * DELTA) * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(DELTA) * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2]      = smem[x] * __half(NORM_L);
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * __half(NORM_H);
}


/**
 * V99: __half2 row-pair packing + HFMA2 for i2f + H-DWT 4-row kernel — parity with V45 (levels 1+).
 * V74: 4-rows-per-block i2f + H-DWT level 0 (int32 input, __half output).
 * grid=(height+3)/4; smem=4*width*sizeof(__half).
 *
 * V99 interior path upgrade (mirrors kernel_fused_horz_dwt_half_io_4row V45):
 *   sm01[x] = {row0[x], row1[x]} as __half2; sm23[x] = {row2[x], row3[x]} as __half2.
 *   Lifting uses HFMA2: each instruction processes 2 rows simultaneously (2× throughput vs scalar).
 *   #pragma unroll 2 on all interior lifting loops (4K: ~2 iters, full ILP).
 *   Scatter uses __hmul2 + __low2half/__high2half (combined L/H scatter in one loop).
 * V126: DCI div-by-4 invariant (h∈{1080,2160}) → y3<height always → else block removed.
 */
__global__ void
kernel_fused_i2f_horz_dwt_half_out_4row(
    const int32_t* __restrict__ d_input,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ __half smem[];
    int y0 = blockIdx.x * 4;
    int y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    /* V126: y3 < height always for DCI (height∈{1080,2160} div by 4) — if wrapper removed. */
    /* V99: __half2 row-pair packing — sm01[x]={row0,row1}, sm23[x]={row2,row3}.
     * HFMA2 processes 2 rows per instruction (2× throughput vs V74 scalar lifting). */
        __half2* sm01 = reinterpret_cast<__half2*>(smem);
        __half2* sm23 = reinterpret_cast<__half2*>(smem + 2*w);
        /* V124: int2 __ldg loads — 4 × 64-bit loads replace 8 × 32-bit loads per 2-col pair.
         * DCI widths always even → x=t*2 even → 8-byte aligned int2 load.
         * x and x+1 loaded together; extract .x/.y for separate i2f→f16 conversion.
         * 2× fewer load instructions; same bytes; coalesced access preserved. Parity Slang V86. */
        #pragma unroll 2
        for (int x = t*2; x < w; x += nt*2) {
            int2 r0 = __ldg(reinterpret_cast<const int2*>(&d_input[y0*stride+x]));
            int2 r1 = __ldg(reinterpret_cast<const int2*>(&d_input[y1*stride+x]));
            int2 r2 = __ldg(reinterpret_cast<const int2*>(&d_input[y2*stride+x]));
            int2 r3 = __ldg(reinterpret_cast<const int2*>(&d_input[y3*stride+x]));
            sm01[x]   = __halves2half2(__float2half(__int2float_rn(r0.x)), __float2half(__int2float_rn(r1.x)));
            sm01[x+1] = __halves2half2(__float2half(__int2float_rn(r0.y)), __float2half(__int2float_rn(r1.y)));
            sm23[x]   = __halves2half2(__float2half(__int2float_rn(r2.x)), __float2half(__int2float_rn(r3.x)));
            sm23[x+1] = __halves2half2(__float2half(__int2float_rn(r2.y)), __float2half(__int2float_rn(r3.y)));
        }
        __syncthreads();
        /* V99: Alpha — HFMA2; #pragma unroll 2 for 4K MLP (2 iters at 4K level-0 w=1920). */
        {
            const __half2 kA = __half2half2(__float2half(ALPHA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kA, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kA, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V113: DCI w always even and >1 → simplify Alpha boundary; parity Slang V76. */
            if(t==0) {
                const __half2 kA2 = __half2half2(__float2half(2.f*ALPHA));
                sm01[w-1] = __hfma2(kA2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kA2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* V99: Beta — HFMA2; #pragma unroll 2. */
        {
            const __half2 kB = __half2half2(__float2half(BETA));
            if(t==0) {
                const __half2 kB2 = __half2half2(__float2half(2.f*BETA));
                /* V113: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kB2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kB2, sm23[1], sm23[0]);
            }
            /* V102: x always even → even w: x≤w-2 → x+1≤w-1 → min(x+1,w-1)=x+1; remove MIN. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kB, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kB, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* V99: Gamma — HFMA2; #pragma unroll 2. */
        {
            const __half2 kG = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x=1+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kG, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kG, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
            /* V113: DCI w always even and >1 → simplify Gamma boundary. */
            if(t==0) {
                const __half2 kG2 = __half2half2(__float2half(2.f*GAMMA));
                sm01[w-1] = __hfma2(kG2, sm01[w-2], sm01[w-1]);
                sm23[w-1] = __hfma2(kG2, sm23[w-2], sm23[w-1]);
            }
        }
        __syncthreads();
        /* V99: Delta — HFMA2; #pragma unroll 2. */
        {
            const __half2 kD = __half2half2(__float2half(DELTA));
            if(t==0) {
                const __half2 kD2 = __half2half2(__float2half(2.f*DELTA));
                /* V113: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
                sm01[0] = __hfma2(kD2, sm01[1], sm01[0]);
                sm23[0] = __hfma2(kD2, sm23[1], sm23[0]);
            }
            /* V102: same even-w invariant as Beta above — min(x+1,w-1)=x+1 for even w. */
            #pragma unroll 2
            for (int x=2+t*2; x<w-1; x+=nt*2) {
                sm01[x] = __hfma2(kD, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
                sm23[x] = __hfma2(kD, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
            }
        }
        __syncthreads();
        /* V99: Combined L+H scatter via __low2half/__high2half — single loop vs two. */
        {
            const __half2 nL = __half2half2(__float2half(NORM_L));
            const __half2 nH = __half2half2(__float2half(NORM_H));
            #pragma unroll 2
            for (int p=t; p<w/2; p+=nt) {
                __half2 v01L = __hmul2(sm01[p*2],   nL);
                __half2 v23L = __hmul2(sm23[p*2],   nL);
                d_tmp[y0*stride+p] = __low2half(v01L);
                d_tmp[y1*stride+p] = __high2half(v01L);
                d_tmp[y2*stride+p] = __low2half(v23L);
                d_tmp[y3*stride+p] = __high2half(v23L);
                __half2 v01H = __hmul2(sm01[p*2+1], nH);
                __half2 v23H = __hmul2(sm23[p*2+1], nH);
                d_tmp[y0*stride+hw+p] = __low2half(v01H);
                d_tmp[y1*stride+hw+p] = __high2half(v01H);
                d_tmp[y2*stride+hw+p] = __low2half(v23H);
                d_tmp[y3*stride+hw+p] = __high2half(v23H);
        }
    }

}


/**
 * V36/V38: Horizontal DWT for levels 1+, half-io.
 * V38: __half shared memory — direct half load (no float conv), fp16 FMA 2× throughput.
 */
__global__ void
kernel_fused_horz_dwt_half_io(
    const __half* __restrict__ d_data,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ __half smem[];  /* V38: was float; halves smem + fp16 FMA on sm_61+ */
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x, nt = blockDim.x;

    /* V38: direct half load — no float conversion. */
    for (int x = t; x < width; x += nt)
        smem[x] = __ldg(&d_data[y * stride + x]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(ALPHA) * (smem[x - 1] + smem[x + 1]);
    /* V120: DCI width always even and >1 → drop width>1&&(width%2==0) guard. */
    if (t == 0) smem[width - 1] += __half(2.0f * ALPHA) * smem[width - 2];
    __syncthreads();

    /* V120: DCI width>1 → min(1,width-1)=1; even x<width → x+1≤width-1 → drop min. */
    if (t == 0) smem[0] += __half(2.0f * BETA) * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(BETA) * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(GAMMA) * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += __half(2.0f * GAMMA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * DELTA) * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(DELTA) * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2]      = smem[x] * __half(NORM_L);
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * __half(NORM_H);
}


/**
 * V46: 2-rows-per-block variant of kernel_fused_horz_dwt_half_io (DWT levels 1-4).
 * smem[0..w-1]=row y0, smem[w..2w-1]=row y1; grid=(height+1)/2; smem=2*width*sizeof(__half).
 * Halves grid for levels 1-4: 540→270, 270→135, 135→68, 68→34 blocks (2K content).
 * Same 4 syncthreads per 2 rows; amortizes block overhead; L2 spatial reuse for adjacent rows.
 */
__global__ void
kernel_fused_horz_dwt_half_io_2row(
    const __half* __restrict__ d_data,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ __half smem[];
    int y0 = blockIdx.x * 2;
    int y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    for (int x = t; x < w; x += nt) smem[x]   = __ldg(&d_data[y0 * stride + x]);
    if (y1 < height)
        for (int x = t; x < w; x += nt) smem[w+x] = __ldg(&d_data[y1 * stride + x]);
    __syncthreads();

    for (int x=1+t*2; x<w-1; x+=nt*2) {
        smem[x]  +=__half(ALPHA)*(smem[x-1]+smem[x+1]);
        if(y1<height) smem[w+x]+=__half(ALPHA)*(smem[w+x-1]+smem[w+x+1]);
    }
    /* V117: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
    if(t==0) { smem[w-1]+=__half(2.f*ALPHA)*smem[w-2]; if(y1<height) smem[2*w-1]+=__half(2.f*ALPHA)*smem[2*w-2]; }
    __syncthreads();
    /* V117: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
    if(t==0) { smem[0]+=__half(2.f*BETA)*smem[1]; if(y1<height) smem[w]+=__half(2.f*BETA)*smem[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        smem[x]  +=__half(BETA)*(smem[x-1]+smem[x+1]);
        if(y1<height) smem[w+x]+=__half(BETA)*(smem[w+x-1]+smem[w+x+1]);
    }
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) {
        smem[x]  +=__half(GAMMA)*(smem[x-1]+smem[x+1]);
        if(y1<height) smem[w+x]+=__half(GAMMA)*(smem[w+x-1]+smem[w+x+1]);
    }
    /* V117: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
    if(t==0) { smem[w-1]+=__half(2.f*GAMMA)*smem[w-2]; if(y1<height) smem[2*w-1]+=__half(2.f*GAMMA)*smem[2*w-2]; }
    __syncthreads();
    /* V117: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
    if(t==0) { smem[0]+=__half(2.f*DELTA)*smem[1]; if(y1<height) smem[w]+=__half(2.f*DELTA)*smem[w+1]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        smem[x]  +=__half(DELTA)*(smem[x-1]+smem[x+1]);
        if(y1<height) smem[w+x]+=__half(DELTA)*(smem[w+x-1]+smem[w+x+1]);
    }
    __syncthreads();
    for (int x=t*2; x<w; x+=nt*2) {
        d_tmp[y0*stride+x/2]       = smem[x]   * __half(NORM_L);
        if(y1<height) d_tmp[y1*stride+x/2]   = smem[w+x] * __half(NORM_L);
    }
    for (int x=t*2+1; x<w; x+=nt*2) {
        d_tmp[y0*stride+hw+x/2]    = smem[x]   * __half(NORM_H);
        if(y1<height) d_tmp[y1*stride+hw+x/2] = smem[w+x] * __half(NORM_H);
    }
}


/**
 * V47: 4-rows-per-block variant of kernel_fused_horz_dwt_half_io (DWT levels 1-4).
 * smem[0..w-1]=y0, [w..2w-1]=y1, [2w..3w-1]=y2, [3w..4w-1]=y3.
 * grid=(height+3)/4; smem=4*width*sizeof(__half).
 * Halves grid vs V46: 270→135, 135→68, 68→34, 34→17 blocks (2K content).
 * Thread-limited at 4 blk/SM (512T×4=2048=max) = 100% occupancy at all DWT levels.
 * 4 adjacent rows read consecutively → 4× L2 spatial reuse for DWT input.
 * Each syncthreads amortized over 4 rows (vs 2 in V46, 1 in base kernel).
 * V108: __launch_bounds__(512,4) — guarantees 4 blk/SM (comment said "Thread-limited at 4 blk/SM"
 *        but without annotation compiler may use 36+ regs → 3 blk/SM; LB forces ≤32 regs/T).
 */
/* V127: DIV4=true (h%4==0) → y3<height always → else block dead → compiler eliminates ~69 lines. */
template<bool DIV4>
__global__ __launch_bounds__(512, 4)
void kernel_fused_horz_dwt_half_io_4row(
    const __half* __restrict__ d_data,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ __half smem[];
    int y0 = blockIdx.x * 4;
    int y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    /* V66: Hoist row-presence check to block level. y3, height uniform across block → no divergence.
     * V127: DIV4=true → if (true || ...) → unconditional; else block dead at compile time.
     * Else: original per-iteration yN<height guards for partial last block. */
    if (DIV4 || y3 < height) {
        /* V82: __half2 row-pair packing — sm01[x]={row0[x],row1[x]}, sm23[x]={row2[x],row3[x]}.
         * Lifting uses HFMA2 (processes 2 rows/instruction) → 2× FMA throughput vs 4 scalar HFMA.
         * Interleaved layout: sm01[k] at smem[2k..2k+1], sm23[k] at smem[2w+2k..2w+2k+1]. */
        __half2* sm01 = reinterpret_cast<__half2*>(smem);
        __half2* sm23 = reinterpret_cast<__half2*>(smem + 2*w);
        /* V123: __half2 __ldg loads — 4 × 32-bit loads replace 8 × 16-bit loads per 2-col pair.
         * DCI widths always even (960/480/240/120) → x=t*2 always even → 4-byte aligned __half2.
         * Each __half2 load covers {row[x], row[x+1]}; unpack into interleaved sm01/sm23 layout.
         * 2× fewer load instructions per thread; same bytes (coalesced access preserved).
         * V94: #pragma unroll 2 retained — 4K: 2 iters; 2K: 1 iter; better DRAM latency hiding. */
        /* V125: loop to x+1<w — for odd w, __half2 write at sm[w-1] is OOB in smem. */
        #pragma unroll 2
        for (int x = t*2; x+1 < w; x += nt*2) {
            __half2 r0 = __ldg(reinterpret_cast<const __half2*>(&d_data[y0*stride+x]));
            __half2 r1 = __ldg(reinterpret_cast<const __half2*>(&d_data[y1*stride+x]));
            __half2 r2 = __ldg(reinterpret_cast<const __half2*>(&d_data[y2*stride+x]));
            __half2 r3 = __ldg(reinterpret_cast<const __half2*>(&d_data[y3*stride+x]));
            sm01[x]   = __halves2half2(__low2half(r0), __low2half(r1));
            sm01[x+1] = __halves2half2(__high2half(r0), __high2half(r1));
            sm23[x]   = __halves2half2(__low2half(r2), __low2half(r3));
            sm23[x+1] = __halves2half2(__high2half(r2), __high2half(r3));
        }
        if ((w & 1) && t == 0) {
            int last = w - 1;
            sm01[last] = __halves2half2(__ldg(&d_data[y0*stride+last]), __ldg(&d_data[y1*stride+last]));
            sm23[last] = __halves2half2(__ldg(&d_data[y2*stride+last]), __ldg(&d_data[y3*stride+last]));
        }
        __syncthreads();
        /* V96: Alpha: odd positions — 2 HFMA2 per iter. #pragma unroll 2: 4K L1 has 2 iters
         * → compiler interleaves 2×4 HFMA2/HADD2 to hide 23-cycle smem read latency. */
        const __half2 kA2 = __half2half2(__float2half(ALPHA));
        #pragma unroll 2
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm01[x] = __hfma2(kA2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
            sm23[x] = __hfma2(kA2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
        }
        /* V110: DCI w always even and >1 — simplify Alpha boundary: drop w>1&&!(w&1) guard.
         * All DCI levels: 1920/960/480/240/120/60 all even. Saves 2 ISETP+PREDAND per block. */
        if(t==0) {
            const __half2 kA2bd = __half2half2(__float2half(2.f*ALPHA));
            sm01[w-1] = __hfma2(kA2bd, sm01[w-2], sm01[w-1]);
            sm23[w-1] = __hfma2(kA2bd, sm23[w-2], sm23[w-1]);
        }
        __syncthreads();
        /* V96: Beta: even positions. #pragma unroll 2 for 4K MLP. */
        const __half2 kB2 = __half2half2(__float2half(BETA));
        if(t==0) {
            const __half2 kB2bd = __half2half2(__float2half(2.f*BETA));
            /* V110: DCI w always >1 → min(1,w-1)=1 always. Drop runtime min. */
            sm01[0] = __hfma2(kB2bd, sm01[1], sm01[0]);
            sm23[0] = __hfma2(kB2bd, sm23[1], sm23[0]);
        }
        /* V102: x always even → even w (all DCI): x≤w-2 → x+1≤w-1 → min(x+1,w-1)=x+1.
         * Drop MIN: 1 fewer VMIN/SETP/SEL per iteration; cleaner HFMA2 scheduling with unroll 2. */
        #pragma unroll 2
        for (int x=2+t*2; x<w-1; x+=nt*2) {
            sm01[x] = __hfma2(kB2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
            sm23[x] = __hfma2(kB2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
        }
        __syncthreads();
        /* V96: Gamma: odd positions. #pragma unroll 2 for 4K MLP. */
        const __half2 kG2 = __half2half2(__float2half(GAMMA));
        #pragma unroll 2
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm01[x] = __hfma2(kG2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
            sm23[x] = __hfma2(kG2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
        }
        /* V110: same DCI even-width invariant — simplify Gamma boundary. */
        if(t==0) {
            const __half2 kG2bd = __half2half2(__float2half(2.f*GAMMA));
            sm01[w-1] = __hfma2(kG2bd, sm01[w-2], sm01[w-1]);
            sm23[w-1] = __hfma2(kG2bd, sm23[w-2], sm23[w-1]);
        }
        __syncthreads();
        /* V96: Delta: even positions. #pragma unroll 2 for 4K MLP. */
        const __half2 kD2 = __half2half2(__float2half(DELTA));
        if(t==0) {
            const __half2 kD2bd = __half2half2(__float2half(2.f*DELTA));
            /* V110: DCI w always >1 → min(1,w-1)=1 always. Drop runtime min. */
            sm01[0] = __hfma2(kD2bd, sm01[1], sm01[0]);
            sm23[0] = __hfma2(kD2bd, sm23[1], sm23[0]);
        }
        /* V102: same even-w invariant as Beta — min(x+1,w-1)=x+1; remove MIN instruction. */
        #pragma unroll 2
        for (int x=2+t*2; x<w-1; x+=nt*2) {
            sm01[x] = __hfma2(kD2, __hadd2(sm01[x-1], sm01[x+1]), sm01[x]);
            sm23[x] = __hfma2(kD2, __hadd2(sm23[x-1], sm23[x+1]), sm23[x]);
        }
        __syncthreads();
        /* V94+V82+V81: combined L+H scatter. #pragma unroll 2 issues 2×8 concurrent smem reads
         * + global writes (4K L1: 2 iters; 2K L1: ≤1 iter → compiler elides extra body). */
        const __half2 nL2 = __half2half2(__float2half(NORM_L));
        const __half2 nH2 = __half2half2(__float2half(NORM_H));
        #pragma unroll 2
        for (int p = t; p < w/2; p += nt) {
            __half2 v01L = __hmul2(sm01[p*2],   nL2);
            __half2 v23L = __hmul2(sm23[p*2],   nL2);
            d_tmp[y0*stride+p] = __low2half(v01L);
            d_tmp[y1*stride+p] = __high2half(v01L);
            d_tmp[y2*stride+p] = __low2half(v23L);
            d_tmp[y3*stride+p] = __high2half(v23L);
            __half2 v01H = __hmul2(sm01[p*2+1], nH2);
            __half2 v23H = __hmul2(sm23[p*2+1], nH2);
            d_tmp[y0*stride+hw+p] = __low2half(v01H);
            d_tmp[y1*stride+hw+p] = __high2half(v01H);
            d_tmp[y2*stride+hw+p] = __low2half(v23H);
            d_tmp[y3*stride+hw+p] = __high2half(v23H);
        }
    } else {
        /* Partial last block: original per-iteration yN<height guards. */
        for (int x = t; x < w; x += nt)           smem[x]     = __ldg(&d_data[y0*stride+x]);
        if (y1<height) for (int x=t; x<w; x+=nt)  smem[w+x]   = __ldg(&d_data[y1*stride+x]);
        if (y2<height) for (int x=t; x<w; x+=nt)  smem[2*w+x] = __ldg(&d_data[y2*stride+x]);
        __syncthreads();
        /* Alpha */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            smem[x]     +=__half(ALPHA)*(smem[x-1]     +smem[x+1]);
            if(y1<height) smem[w+x]   +=__half(ALPHA)*(smem[w+x-1]  +smem[w+x+1]);
            if(y2<height) smem[2*w+x] +=__half(ALPHA)*(smem[2*w+x-1]+smem[2*w+x+1]);
        }
        /* V118: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            smem[w-1]   +=__half(2.f*ALPHA)*smem[w-2];
            if(y1<height) smem[2*w-1] +=__half(2.f*ALPHA)*smem[2*w-2];
            if(y2<height) smem[3*w-1] +=__half(2.f*ALPHA)*smem[3*w-2];
        }
        __syncthreads();
        /* Beta */
        /* V118: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            smem[0]     +=__half(2.f*BETA)*smem[1];
            if(y1<height) smem[w]   +=__half(2.f*BETA)*smem[w+1];
            if(y2<height) smem[2*w] +=__half(2.f*BETA)*smem[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(BETA)*(smem[x-1]     +smem[x+1]);
            if(y1<height) smem[w+x]   +=__half(BETA)*(smem[w+x-1]  +smem[w+x+1]);
            if(y2<height) smem[2*w+x] +=__half(BETA)*(smem[2*w+x-1]+smem[2*w+x+1]);
        }
        __syncthreads();
        /* Gamma */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            smem[x]     +=__half(GAMMA)*(smem[x-1]     +smem[x+1]);
            if(y1<height) smem[w+x]   +=__half(GAMMA)*(smem[w+x-1]  +smem[w+x+1]);
            if(y2<height) smem[2*w+x] +=__half(GAMMA)*(smem[2*w+x-1]+smem[2*w+x+1]);
        }
        /* V118: DCI w always even and >1 → drop w>1&&!(w&1) guard. */
        if(t==0) {
            smem[w-1]   +=__half(2.f*GAMMA)*smem[w-2];
            if(y1<height) smem[2*w-1] +=__half(2.f*GAMMA)*smem[2*w-2];
            if(y2<height) smem[3*w-1] +=__half(2.f*GAMMA)*smem[3*w-2];
        }
        __syncthreads();
        /* Delta */
        /* V118: DCI w>1 → min(1,w-1)=1; even x<w → x+1≤w-1 → drop min(x+1,w-1). */
        if(t==0) {
            smem[0]     +=__half(2.f*DELTA)*smem[1];
            if(y1<height) smem[w]   +=__half(2.f*DELTA)*smem[w+1];
            if(y2<height) smem[2*w] +=__half(2.f*DELTA)*smem[2*w+1];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(DELTA)*(smem[x-1]     +smem[x+1]);
            if(y1<height) smem[w+x]   +=__half(DELTA)*(smem[w+x-1]  +smem[w+x+1]);
            if(y2<height) smem[2*w+x] +=__half(DELTA)*(smem[2*w+x-1]+smem[2*w+x+1]);
        }
        __syncthreads();
        /* Deinterleave and write. */
        for (int x=t*2; x<w; x+=nt*2) {
            d_tmp[y0*stride+x/2]             = smem[x]     * __half(NORM_L);
            if(y1<height) d_tmp[y1*stride+x/2] = smem[w+x]   * __half(NORM_L);
            if(y2<height) d_tmp[y2*stride+x/2] = smem[2*w+x] * __half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            d_tmp[y0*stride+hw+x/2]             = smem[x]     * __half(NORM_H);
            if(y1<height) d_tmp[y1*stride+hw+x/2] = smem[w+x]   * __half(NORM_H);
            if(y2<height) d_tmp[y2*stride+hw+x/2] = smem[2*w+x] * __half(NORM_H);
        }
    } /* end V66 if/else */
}


#if 0  /* Dead kernel — superseded by __half-smem H-DWT variants */
/**
 * V26: Horizontal DWT for levels 1+, writes __half output.
 * Shared memory lifting stays float32. Only DRAM write uses half.
 */
__global__ void
kernel_fused_horz_dwt_half_out(
    const float* __restrict__ d_data,
    __half* __restrict__ d_tmp,
    int width, int height, int stride)
{
    extern __shared__ float smem[];
    int y = blockIdx.x;
    if (y >= height) return;

    int t = threadIdx.x, nt = blockDim.x;

    for (int x = t; x < width; x += nt)
        smem[x] = __ldg(&d_data[y * stride + x]);  /* __ldg: V-DWT just wrote this */
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    /* V120: DCI width always even and >1 → drop width>1&&(width%2==0) guard. */
    if (t == 0) smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    /* V120: DCI width>1 → min(1,width-1)=1; even x<width → x+1≤width-1 → drop min. */
    if (t == 0) smem[0] += 2.0f * BETA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0) smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[1];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[x + 1]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2]      = __float2half(smem[x]     * NORM_L);
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = __float2half(smem[x]     * NORM_H);
}
#endif  /* dead float-smem kernel_fused_horz_dwt_half_out */


/**
 * V21: Fused vertical DWT + deinterleave.
 *
 * Replaces the former two-kernel sequence (kernel_fused_vert_dwt +
 * kernel_deinterleave_vert) with a single pass:
 *   1. Copy column from d_src into d_work (workspace)
 *   2. Apply all 4 CDF 9/7 lifting steps in-place on d_work
 *   3. Write deinterleaved (L|H) result from d_work to d_dst
 *
 * d_src and d_dst may alias (same pointer) because each thread processes
 * only its own column: the copy phase (step 1) completes before the write
 * phase (step 3), so there is no cross-thread aliasing hazard.
 *
 * Savings vs two-kernel approach:
 *   - 1 kernel launch per DWT level per component (15 fewer launches / frame)
 *   - For DWT levels 2-4 (subband ≤ 440 KB ≤ GTX 1050 Ti L2 = 512 KB):
 *     the deinterleave reads d_work from L2 rather than DRAM
 *
 * Parameters:
 *   d_src   — source column data (H-DWT result for this level)
 *   d_work  — workspace for in-place lifting (double-buffer d_aux)
 *   d_dst   — deinterleaved output; may equal d_src (same column, safe)
 */
__global__ void
kernel_fused_vert_dwt_deinterleave(
    const float* __restrict__ d_src,
    float* __restrict__ d_work,
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int h = height;

    /* Step 1: copy column into workspace */
    for (int y = 0; y < h; y++)
        d_work[y * stride + x] = __ldg(&d_src[y * stride + x]);

    /* Step 2: 4-step CDF 9/7 lifting in-place on d_work */

    /* Alpha — update odd samples */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += ALPHA * (d_work[(y - 1) * stride + x]
                                            + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * ALPHA * d_work[(h - 2) * stride + x];

    /* Beta — update even samples */
    d_work[x] += 2.0f * BETA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += BETA * (d_work[(y - 1) * stride + x]
                                           + d_work[yp1 * stride + x]);
    }

    /* Gamma — update odd samples */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] += GAMMA * (d_work[(y - 1) * stride + x]
                                             + d_work[(y + 1) * stride + x]);
    if (h > 1 && (h % 2 == 0))
        d_work[(h - 1) * stride + x] += 2.0f * GAMMA * d_work[(h - 2) * stride + x];

    /* Delta — update even samples */
    d_work[x] += 2.0f * DELTA * d_work[min(1, h - 1) * stride + x];
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y * stride + x] += DELTA * (d_work[(y - 1) * stride + x]
                                             + d_work[yp1 * stride + x]);
    }

    /* Step 3: write deinterleaved result to d_dst with CDF 9/7 normalization.
     * V23: L×NORM_L, H×NORM_H so that LL5 gain = (NORM_L²)^5 × K^10 = 1.0
     * and all subbands remain within [-4095, 4095] for 12-bit input.
     * Even rows → L (lowpass) subband in first half
     * Odd rows  → H (highpass) subband in second half */
    int hh = (h + 1) / 2;
    for (int y = 0; y < h; y += 2)
        d_dst[(y / 2) * stride + x] = d_work[y * stride + x] * NORM_L;
    for (int y = 1; y < h; y += 2)
        d_dst[(hh + y / 2) * stride + x] = d_work[y * stride + x] * NORM_H;
}


/**
 * V25: Vertical DWT + deinterleave using fp16 workspace.
 *
 * Same algorithm as kernel_fused_vert_dwt_deinterleave but stores the
 * per-column lifting workspace in __half (16-bit float) instead of float.
 * Halves memory bandwidth for the 4 lifting passes, which dominate V-DWT cost.
 *
 * Precision note: fp16 ULP for values ±8192 ≈ 8 (exponent=13, mantissa=10 bits).
 * Over 4 CDF 9/7 lifting steps, accumulated error ≈ 20 units << step=32.5.
 * Final output written as float (via __half2float) with NORM_L/NORM_H.
 */
__global__ void
kernel_fused_vert_dwt_fp16(
    const float* __restrict__ d_src,
    __half* __restrict__ d_work,  /* fp16 lifting workspace — same layout as d_src */
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int h = height;

    /* Precompute fp16 lifting coefficients (constant per-thread — in registers) */
    const __half hALPHA = __float2half(ALPHA);
    const __half hBETA  = __float2half(BETA);
    const __half hGAMMA = __float2half(GAMMA);
    const __half hDELTA = __float2half(DELTA);
    const __half h2     = __float2half(2.0f);

    /* Step 1: copy column from float d_src into fp16 d_work */
    for (int y = 0; y < h; y++)
        d_work[y * stride + x] = __float2half(__ldg(&d_src[y * stride + x]));

    /* Step 2: 4-step CDF 9/7 lifting on fp16 column */

    /* Alpha — update odd samples */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y * stride + x] = __hadd(d_work[y * stride + x],
            __hmul(hALPHA, __hadd(d_work[(y-1)*stride+x], d_work[(y+1)*stride+x])));
    if (h > 1 && (h % 2 == 0))
        d_work[(h-1)*stride+x] = __hadd(d_work[(h-1)*stride+x],
            __hmul(h2, __hmul(hALPHA, d_work[(h-2)*stride+x])));

    /* Beta — update even samples */
    d_work[x] = __hadd(d_work[x],
        __hmul(h2, __hmul(hBETA, d_work[min(1, h-1)*stride+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hBETA, __hadd(d_work[(y-1)*stride+x], d_work[yp1*stride+x])));
    }

    /* Gamma — update odd samples */
    for (int y = 1; y < h - 1; y += 2)
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hGAMMA, __hadd(d_work[(y-1)*stride+x], d_work[(y+1)*stride+x])));
    if (h > 1 && (h % 2 == 0))
        d_work[(h-1)*stride+x] = __hadd(d_work[(h-1)*stride+x],
            __hmul(h2, __hmul(hGAMMA, d_work[(h-2)*stride+x])));

    /* Delta — update even samples */
    d_work[x] = __hadd(d_work[x],
        __hmul(h2, __hmul(hDELTA, d_work[min(1, h-1)*stride+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hDELTA, __hadd(d_work[(y-1)*stride+x], d_work[yp1*stride+x])));
    }

    /* Step 3: write deinterleaved result to float d_dst with CDF 9/7 normalization */
    const __half hNORM_L = __float2half(NORM_L);
    const __half hNORM_H = __float2half(NORM_H);
    int hh = (h + 1) / 2;
    for (int y = 0; y < h; y += 2)
        d_dst[(y/2)*stride+x] = __half2float(__hmul(d_work[y*stride+x], hNORM_L));
    for (int y = 1; y < h; y += 2)
        d_dst[(hh+y/2)*stride+x] = __half2float(__hmul(d_work[y*stride+x], hNORM_H));
}


/**
 * V26: Vertical DWT + deinterleave with __half input and __half workspace.
 *
 * Identical to kernel_fused_vert_dwt_fp16 but d_src is __half (H-DWT output
 * stored as half in V26). The column copy is now half=half (no float2half).
 * This saves the DRAM read bandwidth for d_src: 4 bytes/elem → 2 bytes/elem.
 */
__global__ void
kernel_fused_vert_dwt_fp16_hi(
    const __half* __restrict__ d_src,  /* half input (V26: H-DWT output as half) */
    __half* __restrict__ d_work,
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int h = height;

    const __half hALPHA = __float2half(ALPHA);
    const __half hBETA  = __float2half(BETA);
    const __half hGAMMA = __float2half(GAMMA);
    const __half hDELTA = __float2half(DELTA);
    const __half h2     = __float2half(2.0f);

    /* Step 1: copy column from half d_src into half d_work (no conversion).
     * __ldg: read-only texture cache for coalesced half reads. */
    for (int y = 0; y < h; y++)
        d_work[y * stride + x] = __ldg(&d_src[y * stride + x]);

    /* Step 2: 4-step CDF 9/7 lifting (identical to kernel_fused_vert_dwt_fp16) */

    for (int y = 1; y < h - 1; y += 2)
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hALPHA, __hadd(d_work[(y-1)*stride+x], d_work[(y+1)*stride+x])));
    if (h > 1 && (h % 2 == 0))
        d_work[(h-1)*stride+x] = __hadd(d_work[(h-1)*stride+x],
            __hmul(h2, __hmul(hALPHA, d_work[(h-2)*stride+x])));

    d_work[x] = __hadd(d_work[x],
        __hmul(h2, __hmul(hBETA, d_work[min(1, h-1)*stride+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hBETA, __hadd(d_work[(y-1)*stride+x], d_work[yp1*stride+x])));
    }

    for (int y = 1; y < h - 1; y += 2)
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hGAMMA, __hadd(d_work[(y-1)*stride+x], d_work[(y+1)*stride+x])));
    if (h > 1 && (h % 2 == 0))
        d_work[(h-1)*stride+x] = __hadd(d_work[(h-1)*stride+x],
            __hmul(h2, __hmul(hGAMMA, d_work[(h-2)*stride+x])));

    d_work[x] = __hadd(d_work[x],
        __hmul(h2, __hmul(hDELTA, d_work[min(1, h-1)*stride+x])));
    for (int y = 2; y < h; y += 2) {
        int yp1 = (y + 1 < h) ? y + 1 : y - 1;
        d_work[y*stride+x] = __hadd(d_work[y*stride+x],
            __hmul(hDELTA, __hadd(d_work[(y-1)*stride+x], d_work[yp1*stride+x])));
    }

    /* Step 3: deinterleave + normalize, write float d_dst */
    const __half hNORM_L = __float2half(NORM_L);
    const __half hNORM_H = __float2half(NORM_H);
    int hh = (h + 1) / 2;
    for (int y = 0; y < h; y += 2)
        d_dst[(y/2)*stride+x] = __half2float(__hmul(d_work[y*stride+x], hNORM_L));
    for (int y = 1; y < h; y += 2)
        d_dst[(hh+y/2)*stride+x] = __half2float(__hmul(d_work[y*stride+x], hNORM_H));
}


/**
 * V27: Vertical DWT + deinterleave — __half input, register-blocked column (no global workspace).
 *
 * For small subbands (height ≤ MAX_REG_HEIGHT = 140) the entire column fits in
 * registers. Eliminates the fp16 d_work global memory accesses entirely:
 *   - 4 lifting passes × 2 reads+1 write × column_height elements removed from DRAM
 *   - For level 3 (h=135): saves 135×2×8 = 2.2KB/thread of global traffic
 *   - Trade-off: fewer resident blocks (higher register pressure), but memory-bound
 *     subbands benefit from the elimination of workspace traffic.
 *
 * Precision: all arithmetic in float32 (column loaded via __half2float).
 * Input: __half d_src (H-DWT output). Output: float d_dst (deinterleaved).
 */
__global__ void
kernel_fused_vert_dwt_fp16_hi_reg(
    const __half* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    /* Load column from __half into float registers; __ldg uses read-only texture cache */
    float col[MAX_REG_HEIGHT];
    for (int y = 0; y < height; y++)
        col[y] = __half2float(__ldg(&d_src[y * stride + x]));

    /* CDF 9/7 lifting in float registers */
    for (int y = 1; y < height - 1; y += 2)
        col[y] += ALPHA * (col[y-1] + col[y+1]);
    if (height > 1 && (height % 2 == 0))
        col[height-1] += 2.0f * ALPHA * col[height-2];

    col[0] += 2.0f * BETA * col[min(1, height-1)];
    for (int y = 2; y < height; y += 2) {
        int yp1 = (y+1 < height) ? y+1 : y-1;
        col[y] += BETA * (col[y-1] + col[yp1]);
    }

    for (int y = 1; y < height - 1; y += 2)
        col[y] += GAMMA * (col[y-1] + col[y+1]);
    if (height > 1 && (height % 2 == 0))
        col[height-1] += 2.0f * GAMMA * col[height-2];

    col[0] += 2.0f * DELTA * col[min(1, height-1)];
    for (int y = 2; y < height; y += 2) {
        int yp1 = (y+1 < height) ? y+1 : y-1;
        col[y] += DELTA * (col[y-1] + col[yp1]);
    }

    /* Deinterleave + normalize to d_dst */
    int hh = (height + 1) / 2;
    for (int y = 0; y < height; y += 2)
        d_dst[(y/2)*stride+x] = col[y] * NORM_L;
    for (int y = 1; y < height; y += 2)
        d_dst[(hh+y/2)*stride+x] = col[y] * NORM_H;
}


/**
 * V97: Reg-blocked V-DWT — hoisted __half constants + #pragma unroll 4 on load/Alpha/Gamma/write.
 * V36: Register-blocked V-DWT (small subbands), writes __half output to d_dst_h.
 * Mirrors kernel_fused_vert_dwt_fp16_hi_reg but output is half (d_a is now half in V36).
 *
 * V97 optimizations:
 *   1. Hoist kA/kB/kG/kD/kNL/kNH — one __float2half each, not per-loop-iteration.
 *   2. #pragma unroll 4 on load loop — 4 concurrent __ldg; hides ~800cy L2/DRAM latency.
 *   3. #pragma unroll 4 on Alpha/Gamma loops — step=2 → fully independent iterations;
 *      4 independent HFMA per batch → 4× ILP on sm_61+ __half FMA units.
 *   4. #pragma unroll 4 on both scatter write loops — 4 concurrent global stores per batch.
 * Beta/Delta: runtime conditional yp1 — unroll omitted (boundary check complicates it).
 *
 * V125: Template on EVEN_HEIGHT — eliminates 4 runtime branches + 2 dead constant computations.
 *   EVEN_HEIGHT=true  (height=68,34): Alpha/Gamma even-boundary always executes; Beta/Delta odd dead.
 *   EVEN_HEIGHT=false (height=135,67): Beta/Delta odd-boundary always executes; Alpha/Gamma even dead.
 *   Compiler removes unreachable if-body and eliminates dead kA2/kG2 (odd) or kB2/kD2 (even).
 *   Launch: (height%2==0)?kernel<true>:kernel<false>; both instantiations get PreferL1 cache config.
 */
template<bool EVEN_HEIGHT>
__global__ void
kernel_fused_vert_dwt_fp16_hi_reg_ho(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst_h,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    /* V97: hoist all __half constants — one __float2half each, not per-loop-iteration. */
    const __half kA  = __float2half(ALPHA),       kA2 = __float2half(2.f * ALPHA);
    const __half kB  = __float2half(BETA),        kB2 = __float2half(2.f * BETA);
    const __half kG  = __float2half(GAMMA),       kG2 = __float2half(2.f * GAMMA);
    const __half kD  = __float2half(DELTA),       kD2 = __float2half(2.f * DELTA);
    const __half kNL = __float2half(NORM_L);
    const __half kNH = __float2half(NORM_H);

    /* V69: __half col[] — direct __ldg→__half; HFMA 2× throughput; 70 regs (140 packed halves). */
    __half col[MAX_REG_HEIGHT];
    /* V97: #pragma unroll 4 — 4 concurrent __ldg in flight; hides column load latency.
     * Each load at y*stride+x → different L2 cache lines → MLP covers latency. */
    #pragma unroll 4
    for (int y = 0; y < height; y++)
        col[y] = __ldg(&d_src[y * stride + x]);

    /* Alpha: odd rows — V97: #pragma unroll 4 — step=2 → fully independent iterations;
     * 4 independent HFMA per unrolled batch → 4× ILP on sm_61+ __half FMA units. */
    #pragma unroll 4
    for (int y = 1; y < height - 1; y += 2)
        col[y] += kA * (col[y-1] + col[y+1]);
    /* V112: DCI heights always > 1 → h>1 guard removed. Check only even parity.
     * V125: EVEN_HEIGHT template → compiler eliminates dead branch + kA2 for odd heights. */
    if (EVEN_HEIGHT)
        col[height-1] += kA2 * col[height-2];

    /* V101: Beta — hoisted boundary case enables #pragma unroll 4 on main loop.
     * Main: y<height-1 guarantees y+1<height → no yp1 conditional ever fired.
     * Boundary: fires only for odd height (height-1 is even → col[height-1] is the last even y). */
    /* V112: DCI heights always > 1 → min(1,height-1)=1 always; drop runtime min. */
    col[0] += kB2 * col[1];
    #pragma unroll 4
    for (int y = 2; y < height - 1; y += 2)
        col[y] += kB * (col[y-1] + col[y+1]);
    /* V114: DCI heights always > 2 (34/68/135) → drop h>2 guard; parity Slang V77.
     * V125: !EVEN_HEIGHT template → compiler eliminates dead branch + kB2 for even heights. */
    if (!EVEN_HEIGHT)                          /* odd height: last even y = height-1 */
        col[height-1] += kB2 * col[height-2]; /* kB*(col[h-2]+col[h-2]) = 2kB*col[h-2] */

    /* Gamma: odd rows — same structure as Alpha; V97: #pragma unroll 4 → 4× ILP. */
    #pragma unroll 4
    for (int y = 1; y < height - 1; y += 2)
        col[y] += kG * (col[y-1] + col[y+1]);
    /* V112: DCI heights always > 1 → h>1 guard removed. Check only even parity.
     * V125: EVEN_HEIGHT template → compiler eliminates dead branch + kG2 for odd heights. */
    if (EVEN_HEIGHT)
        col[height-1] += kG2 * col[height-2];

    /* V101: Delta — same boundary-hoist transformation as Beta above. */
    /* V112: DCI heights always > 1 → min(1,height-1)=1 always; drop runtime min. */
    col[0] += kD2 * col[1];
    #pragma unroll 4
    for (int y = 2; y < height - 1; y += 2)
        col[y] += kD * (col[y-1] + col[y+1]);
    /* V114: DCI heights always > 2 → drop h>2 guard (same as Beta above).
     * V125: !EVEN_HEIGHT template → compiler eliminates dead branch + kD2 for even heights. */
    if (!EVEN_HEIGHT)                          /* odd height: last even y = height-1 */
        col[height-1] += kD2 * col[height-2]; /* kD*(col[h-2]+col[h-2]) = 2kD*col[h-2] */

    /* V36/V69: write __half output directly — no __float2half conversion needed.
     * V97: #pragma unroll 4 on both scatter loops — 4 concurrent global stores per batch. */
    int hh = (height + 1) / 2;
    #pragma unroll 4
    for (int y = 0; y < height; y += 2)
        d_dst_h[(y/2)*stride+x] = col[y] * kNL;
    #pragma unroll 4
    for (int y = 1; y < height; y += 2)
        d_dst_h[(hh+y/2)*stride+x] = col[y] * kNH;
}


/**
 * V39: Parity-split CDF 9/7 lifting helper for tiled V-DWT.
 * Template parameter P0: parity of local index 0 in global row space (0 or 1).
 *   P0=0 → local row 0 is globally even; odd rows start at i=1.
 *   P0=1 → local row 0 is globally odd;  even rows start at i=1.
 *
 * Replaces 4×(loop-with-branch) with 4×(branch-free stride-2 loop), halving
 * iteration count and enabling full #pragma unroll → straight-line FMA sequences.
 * Old: 4 × 40 iterations × 1 branch each  = 160 branches + 80 FMAs
 * New: 4 × 20 iterations × 0 branches     =   0 branches + 80 FMAs
 */
template<int P0>
__device__ __forceinline__ void
cdf97_lift_tiled(float col[V_TILE_FL])
{
    /* Alpha: update globally-odd rows */
    #pragma unroll
    for (int i = (P0 ? 2 : 1); i < V_TILE_FL - 1; i += 2)
        col[i] += ALPHA * (col[i-1] + col[i+1]);
    /* Beta: update globally-even rows */
    #pragma unroll
    for (int i = (P0 ? 1 : 2); i < V_TILE_FL - 1; i += 2)
        col[i] += BETA * (col[i-1] + col[i+1]);
    /* Gamma: update globally-odd rows */
    #pragma unroll
    for (int i = (P0 ? 2 : 1); i < V_TILE_FL - 1; i += 2)
        col[i] += GAMMA * (col[i-1] + col[i+1]);
    /* Delta: update globally-even rows */
    #pragma unroll
    for (int i = (P0 ? 1 : 2); i < V_TILE_FL - 1; i += 2)
        col[i] += DELTA * (col[i-1] + col[i+1]);
}


/**
 * V69: __half version of cdf97_lift_tiled — P0=1 hardcoded (V40 static_assert invariant).
 * Processes __half col[V_TILE_FL] using __half FMA arithmetic.
 * On sm_61+: HFMA has 2× throughput vs float FMA; replaces 4×12 float FMAs per tile-column
 * with 4×12 __half FMAs → compute section runs 2× faster.
 * Eliminates __half2float and __float2half conversions in the caller (direct __half I/O).
 */
__device__ __forceinline__ void
cdf97_lift_tiled_h(__half col[V_TILE_FL])
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


/**
 * V76: 2-column-per-thread __half2 lifting helper.
 * col2[i] = {col_x[i], col_{x+1}[i]} — two adjacent columns packed as __half2.
 * __hfma2(kA, __hadd2(prev, next), cur) processes both columns simultaneously:
 *   HFMA2 = 1 instruction per 2 FMAs → 2× arithmetic throughput on sm_61+.
 * P0=1 hardcoded (V40 invariant: V_TILE even + V_OVERLAP odd).
 */
__device__ __forceinline__ void
cdf97_lift_tiled_h2(__half2 col2[V_TILE_FL])
{
    const __half2 kA = __half2half2(__float2half(ALPHA));
    const __half2 kB = __half2half2(__float2half(BETA));
    const __half2 kG = __half2half2(__float2half(GAMMA));
    const __half2 kD = __half2half2(__float2half(DELTA));
    /* Alpha: globally-odd rows (P0=0 → local odd indices starting at i=1) */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2)
        col2[i] = __hfma2(kA, __hadd2(col2[i-1], col2[i+1]), col2[i]);
    /* Beta: globally-even rows (P0=0 → local even indices starting at i=2; skip i=0 boundary) */
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


/**
 * V36: Tiled V-DWT, writes __half output to d_dst_h.
 * Mirrors kernel_fused_vert_dwt_tiled but output is half (d_a is now half in V36).
 * V80: V_TILE=22, V_TILE_FL=30, P0=0. V71 used V_TILE=28, V_TILE_FL=38. No shared memory.
 *
 * V64: Interior-tile branch-free path for 97% of blocks.
 *   interior = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= height).
 *   Interior load: #pragma unroll V_TILE_FL iters, no bounds check → 38 independent __ldg.
 *   Interior store: #pragma unroll V_TILE iters, compile-time i → !(i&1) resolved at compile
 *     time, emitting 14 L stores + 14 H stores as straight-line code (no loop/branch overhead).
 *   Boundary tiles (first + last, 2/39=5% for 2K lvl-0): keep WS bounds-check path unchanged.
 */
__global__ void
kernel_fused_vert_dwt_tiled_ho(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst_h,
    int width, int height, int stride)
{
    int x          = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= width || tile_start >= height) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, height);

    /* V64: interior check — uniform across all threads in block (no divergence).
     * interior ↔ load_start ≥ 0 AND load_start + V_TILE_FL ≤ height. */
    bool interior = (tile_start >= V_OVERLAP) && (tile_start + V_TILE <= height);

    /* V69: __half col[] — direct __ldg→__half load (no __half2float); cdf97_lift_tiled_h
     * uses __half FMA (HFMA: 2× throughput vs float on sm_61+); output stored without
     * __float2half conversion. Tiled path eliminates 42 conversion instructions per tile. */
    __half col[V_TILE_FL];
    if (interior) {
        /* V64+V69+V71: no bounds check → #pragma unroll emits 38 independent __ldg (direct __half). */
        #pragma unroll
        for (int i = 0; i < V_TILE_FL; i++)
            col[i] = __ldg(&d_src[(load_start + i) * stride + x]);
    } else {
        for (int i = 0; i < V_TILE_FL; i++) {
            int gy = load_start + i;
            if (gy < 0) gy = -gy;
            else if (gy >= height) gy = 2*(height-1) - gy;
            col[i] = __ldg(&d_src[gy * stride + x]);
        }
    }

    /* V80: V_TILE (even) + V_OVERLAP (even) → load_start always even → p0 always 0.
     * V69: use cdf97_lift_tiled_h — __half lifting, P0=0 hardcoded. */
    static_assert(V_TILE % 2 == 0 && V_OVERLAP % 2 == 0,
                  "V80: requires V_TILE even + V_OVERLAP even for constant p0=0");
    cdf97_lift_tiled_h(col);

    /* V80: output parity p0=0 → odd i = H, even i = L; direct __half stores. */
    int hh = (height + 1) / 2;
    if (interior) {
        /* V64+V69: exactly V_TILE outputs, fully unrolled, no __float2half conversions. */
        #pragma unroll
        for (int i = V_OVERLAP; i < V_OVERLAP + V_TILE; i++) {
            int gy = load_start + i;
            if (i & 1)  /* V80: odd i (p0=0) → H subband */
                d_dst_h[(hh + gy/2) * stride + x] = col[i] * __half(NORM_H);
            else
                d_dst_h[(gy/2) * stride + x] = col[i] * __half(NORM_L);
        }
    } else {
        for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
            int gy = load_start + i;
            if (i & 1)  /* V80: odd i (p0=0) → H subband */
                d_dst_h[(hh + gy/2) * stride + x] = col[i] * __half(NORM_H);
            else
                d_dst_h[(gy/2) * stride + x] = col[i] * __half(NORM_L);
        }
    }
}


/**
 * V76: 2-column-per-thread tiled V-DWT using __half2 arithmetic (HFMA2 = 2× lifting throughput).
 * Each thread processes columns x and x+1 simultaneously.
 * Grid x = ceil(width/2/V_THREADS_TILED): half the x-blocks; each block does 2× more work.
 * V80: __half2 col2[V_TILE_FL=30]: ~42 regs/T → 6 blk/SM (75% occ). V76 had col2[38] ~50 regs → 5 blk.
 * Throughput: 6×22/30=4.40 effective col-sets/SM vs V76 5×28/38=3.68 → +19.6% V-DWT.
 * V80: P0=0 — odd i → H subband, even i → L subband.
 * Load: reinterp int* → __half2 (2×__half packed). Store: __half2 result → int* store.
 * Req: x always even (blockIdx.x*blockDim.x*2+t*2); width even (2K=1920, 4K=3840: yes).
 */
/* V87: __launch_bounds__(256,6) — 256T matches V_THREADS_TILED; 6 blk/SM target for register savings. */
__global__ __launch_bounds__(256, 6)
void kernel_fused_vert_dwt_tiled_ho_2col(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst_h,
    int width, int height, int stride)
{
    int x          = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= width || tile_start >= height) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, height);
    /* V187: `interior` must guarantee the LOAD reads are all in-bounds, i.e.
     * load_start + V_TILE_FL <= height (NOT just tile_start + V_TILE <= height).
     * The previous condition let the last tile (whose tile_end touches height)
     * fall into the interior path, which then read V_OVERLAP rows past the
     * image bottom as garbage memory and produced huge spurious H-band
     * coefficients on the last few rows. */
    bool interior  = (load_start >= 0) && (load_start + V_TILE_FL <= height);

    __half2 col2[V_TILE_FL];
    if (interior) {
        /* Interior path: all rows valid — #pragma unroll emits 38 paired __ldg. */
        #pragma unroll
        for (int i = 0; i < V_TILE_FL; i++) {
            int raw = __ldg(reinterpret_cast<const int*>(&d_src[(load_start + i) * stride + x]));
            col2[i] = *reinterpret_cast<const __half2*>(&raw);
        }
    } else {
        for (int i = 0; i < V_TILE_FL; i++) {
            int gy = load_start + i;
            if (gy < 0) gy = -gy;
            else if (gy >= height) gy = 2*(height-1) - gy;
            int raw = __ldg(reinterpret_cast<const int*>(&d_src[gy * stride + x]));
            col2[i] = *reinterpret_cast<const __half2*>(&raw);
        }
    }

    cdf97_lift_tiled_h2(col2);

    int hh = (height + 1) / 2;
    const __half2 nL = __half2half2(__float2half(NORM_L));
    const __half2 nH = __half2half2(__float2half(NORM_H));
    if (interior) {
        #pragma unroll
        for (int i = V_OVERLAP; i < V_OVERLAP + V_TILE; i++) {
            int gy = load_start + i;
            /* V80: P0=0 — odd i → H subband, even i → L subband */
            __half2 r = __hmul2(col2[i], (i & 1) ? nH : nL);
            int* p = reinterpret_cast<int*>(&d_dst_h[
                ((i & 1) ? (hh + gy/2) : (gy/2)) * stride + x]);
            *p = *reinterpret_cast<const int*>(&r);
        }
    } else {
        for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
            int gy = load_start + i;
            /* V80: P0=0 — odd i → H subband, even i → L subband */
            __half2 r = __hmul2(col2[i], (i & 1) ? nH : nL);
            int* p = reinterpret_cast<int*>(&d_dst_h[
                ((i & 1) ? (hh + gy/2) : (gy/2)) * stride + x]);
            *p = *reinterpret_cast<const int*>(&r);
        }
    }
}


/**
 * V35: Quantize + GPU sign-magnitude pack — float4 vectorized reads + uint32_t packed writes.
 *
 * Processes 4 elements per thread using float4 loads (16 bytes per read, 4× fewer transactions)
 * and packs 4 uint8 outputs into a single uint32_t store (4× fewer write transactions).
 *
 * n4 = floor(per_comp / 4) elements handled by this kernel (multiple-of-4 portion).
 * The last 0-3 elements are beyond n4*4 and skipped — negligible quality impact.
 * d_comp and d_packed must be 4/16-byte aligned (guaranteed by cudaMalloc).
 */
__global__ void
kernel_quantize_and_pack(
    const float* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int n4,   /* number of float4 groups = floor(per_comp / 4) */
    float step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;

    /* V35: float4 read — 16 bytes in one load, 4× coalescing improvement. */
    float4 v = __ldg(reinterpret_cast<const float4*>(d_comp) + i);
    float inv_step = __frcp_rn(step_size);  /* V31: reciprocal multiply */

    /* Pack 4 sign-magnitude bytes into uint32_t, one 4-byte store vs four 1-byte stores. */
    auto pack_byte = [inv_step](float fv) -> uint8_t {
        int q   = __float2int_rn(fv * inv_step);
        uint8_t sign = (q < 0) ? 0x80u : 0x00u;
        /* Cap magnitude at 126: prevents 0xFF (would require byte stuffing). */
        uint8_t mag  = static_cast<uint8_t>(min(126, abs(q)));
        return sign | mag;
    };

    uint32_t word = static_cast<uint32_t>(pack_byte(v.x))
                  | (static_cast<uint32_t>(pack_byte(v.y)) << 8)
                  | (static_cast<uint32_t>(pack_byte(v.z)) << 16)
                  | (static_cast<uint32_t>(pack_byte(v.w)) << 24);
    reinterpret_cast<uint32_t*>(d_packed)[i] = word;
}


/**
 * V36: Quantize + pack — __half2 vectorized input (reads 4 halves = 8 bytes per thread).
 * Used when d_a[c] is __half (V36 full-half pipeline). Output uint32_t same as V35.
 * n4 = floor(per_comp / 4); last 0-3 elements skipped — negligible.
 */
__global__ void
kernel_quantize_and_pack_h(
    const __half* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int n4,
    float step_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;

    /* Read 4 halves using two __half2 loads (8 bytes total per thread). */
    __half2 h12 = __ldg(reinterpret_cast<const __half2*>(d_comp) + i * 2);
    __half2 h34 = __ldg(reinterpret_cast<const __half2*>(d_comp) + i * 2 + 1);

    float inv_step = __frcp_rn(step_size);

    auto pack_byte = [inv_step](__half hv) -> uint8_t {
        int q   = __float2int_rn(__half2float(hv) * inv_step);
        uint8_t sign = (q < 0) ? 0x80u : 0x00u;
        uint8_t mag  = static_cast<uint8_t>(min(126, abs(q)));
        return sign | mag;
    };

    uint32_t word = static_cast<uint32_t>(pack_byte(h12.x))
                  | (static_cast<uint32_t>(pack_byte(h12.y)) << 8)
                  | (static_cast<uint32_t>(pack_byte(h34.x)) << 16)
                  | (static_cast<uint32_t>(pack_byte(h34.y)) << 24);
    reinterpret_cast<uint32_t*>(d_packed)[i] = word;
}


/**
 * V50: Subband-aware quantize kernel — per-row step weighting for visual quality.
 *
 * After the DWT, d_a[c] stores subbands in nested rectangles.
 * Rows 0..ll5_height-1 contain the LL5 (DC) region — the most perceptually important
 * subband. Applying a finer step there reduces quantization error in the DC term,
 * improving global brightness/contrast accuracy at no bitrate cost.
 *
 * Step weights vs base_step:
 *   rows [0..ll5_height):       STEP_LL = 0.80 (20% finer — DC subband, highest importance)
 *   rows [ll5_height..2*ll5_h): STEP_MID = 0.95 (slight boost for level-4 subbands)
 *   rows >= 2*ll5_height:       STEP_HI = 1.10 (coarser for higher-frequency content)
 *
 * Launch: kernel_quantize_subband_h<<<n_rows, 256, 0, st>>>(d_a[c], d_packed, stride, n_rows, step, ll5_h)
 * where n_rows = ceil(per_comp / stride), stride = original DWT width, ll5_h = ceil(height/32).
 *
 * Output: n_rows × stride uint8 bytes of sign-magnitude packed coefficients.
 * D2H copies only per_comp bytes (≤ n_rows × stride), so partial last row is safe.
 */
__global__ void
kernel_quantize_subband_h(
    const __half* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int stride,    /* original DWT width (row pitch of d_comp in d_a[c]) */
    int n_rows,    /* number of rows to quantize = ceil(per_comp / stride) */
    float base_step,
    int ll5_height)  /* V50: rows < ll5_height contain DC subbands (finer step) */
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    /* Row-based step weight: finer for DC (LL5) rows, coarser above. */
    float factor;
    if      (row < ll5_height)     factor = 0.80f;  /* DC subband — finest */
    else if (row < 2 * ll5_height) factor = 0.95f;  /* level-4 subbands */
    else                           factor = 1.10f;  /* level 3 and above */
    float inv_step = __frcp_rn(base_step * factor);

    for (int col = threadIdx.x; col < stride; col += blockDim.x) {
        float val = __half2float(__ldg(&d_comp[row * stride + col]));
        int q = __float2int_rn(val * inv_step);
        uint8_t sign = (q < 0) ? 0x80u : 0x00u;
        uint8_t mag  = static_cast<uint8_t>(min(126, abs(q)));
        d_packed[row * stride + col] = sign | mag;
    }
}


/**
 * V52: 2D subband-aware quantization — correct DC column range check.
 *
 * V50's kernel_quantize_subband_h used only row < ll5_height to detect the DC region,
 * but rows 0..ll5_height-1 also contain LH1 (finest horizontal detail, cols stride/2..stride-1).
 * Applying a finer step to LH1 wastes bits that should go to DC (LL5).
 *
 * The 5-level 2D DWT on a W×H image packs subbands as:
 *   LL5: rows [0..ll5_h), cols [0..ll5_cols)       where ll5_cols = stride >> 5
 *   LH5: rows [0..ll5_h), cols [ll5_cols..2*ll5_cols)
 *   HL5: rows [ll5_h..2*ll5_h), cols [0..ll5_cols)
 *   HH5: rows [ll5_h..2*ll5_h), cols [ll5_cols..2*ll5_cols)
 *   LH4..HH4: rows [0..2*ll5_h), cols [2*ll5_cols..4*ll5_cols)  — and so on
 *   LH1: rows [0..ll5_h), cols [stride/2..stride)   ← this was getting DC step in V50!
 *
 * Step weights:
 *   LL5 (DC):       base_step × 0.70  — finest, most perceptually important
 *   LH5/HL5/HH5:    base_step × 0.90  — level-5 AC subbands
 *   all other:      base_step × 1.15  — higher-frequency subbands
 *
 * ll5_cols = stride >> 5  (= stride/32, since 2^5=32 for 5-level DWT)
 * ll5_height = ceil(orig_height / 32)
 *
 * Launch: kernel_quantize_subband_2d<<<n_rows, 256, 0, st>>>(
 *             d_a[c], d_packed, stride, n_rows, step, ll5_height, ll5_cols)
 */
__global__ void
kernel_quantize_subband_2d(
    const __half* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int stride, int n_rows, float base_step,
    int ll5_height, int ll5_cols)  /* V52: both row AND column checks */
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    float inv_dc = __frcp_rn(base_step * 0.70f);  /* LL5: DC, finest */
    float inv_l5 = __frcp_rn(base_step * 0.90f);  /* LH5/HL5/HH5 */
    float inv_hf = __frcp_rn(base_step * 1.15f);  /* all higher-frequency */

    bool is_l5_row = (row < ll5_height * 2);
    bool is_dc_row = (row < ll5_height);

    for (int col = threadIdx.x; col < stride; col += blockDim.x) {
        float inv_step;
        if (is_dc_row && col < ll5_cols)
            inv_step = inv_dc;                          /* LL5: DC subband */
        else if (is_l5_row && col < ll5_cols * 2)
            inv_step = inv_l5;                          /* LH5/HL5/HH5 */
        else
            inv_step = inv_hf;                          /* higher-frequency subbands */

        float val = __half2float(__ldg(&d_comp[row * stride + col]));
        int q = __float2int_rn(val * inv_step);
        uint8_t sign = (q < 0) ? 0x80u : 0x00u;
        uint8_t mag  = static_cast<uint8_t>(min(126, abs(q)));
        d_packed[row * stride + col] = sign | mag;
    }
}


/**
 * V53: Multi-level perceptual quantization — 6-band step weighting.
 *
 * V52 used 3 bands (LL5, LH5/HL5/HH5, all-others). This version distinguishes all 5 DWT
 * levels plus the DC subband for accurate perceptual bit allocation:
 *
 *   Band              Step factor   Rationale
 *   LL5 (DC)            × 0.65     Lowest frequency, most perceptually important
 *   LH5/HL5/HH5         × 0.85     Level-5 AC: important diagonal/edge detail
 *   Level-4 subbands    × 0.95     Moderate frequency: visible textures
 *   Level-3 subbands    × 1.05     Higher frequency: fine edges
 *   Level-2 subbands    × 1.12     Very high frequency: noise-like
 *   Level-1 subbands    × 1.20     Finest detail: perceptually masked
 *
 * Subband level detection from (row, col):
 *   row_lv = 5 if row < 2*ll5_h,  4 if < 4*ll5_h,  3 if < 8*ll5_h,
 *            2 if < 16*ll5_h,  else 1
 *   col_lv = 5 if col < 2*ll5_c,  4 if < 4*ll5_c,  3 if < 8*ll5_c,
 *            2 if < 16*ll5_c,  else 1
 *   subband_level = min(row_lv, col_lv)
 *   DC condition: row < ll5_h AND col < ll5_c  (LL5 subband)
 *
 * ll5_h = ceil(orig_height/32),  ll5_c = stride >> 5
 *
 * Launch: kernel_quantize_subband_ml<<<n_rows, 256, 0, st>>>(
 *             d_a[c], d_packed, stride, n_rows, step, ll5_h, ll5_c)
 */
/**
 * V93: 8-element vectorized quantize — vq8 replaces vq4 in all zones.
 *
 * vq8 processes 8 elements per thread per loop iteration:
 *   4×__half2 loads (16 bytes) → 8 quantize → 2×uint32_t stores (8 bytes)
 * vs vq4: 2×__half2 loads (8B) + 1×uint32_t store (4B) per iteration.
 * Halves loop iterations vs vq4: 50% less loop overhead; same total memory traffic.
 * vq4 tail handles residual < 8 elements (only 2K DC zones: ll5_c=60, 60%8=4 tail).
 * 4K: ll5_c=120, all zones div-by-8 → zero tail everywhere in 4K mode.
 *
 * V60: Fully vectorized multi-level perceptual quantization — all rows use __half2 loads.
 *
 * V58 vectorized L1/L2/L3 rows (~87.5%). V60 extends to L4/L5/DC rows (~12.5%):
 *   DC rows: 6 zones [0,ll5_c)→inv_dc, [ll5_c,2c)→inv_l5, ..., [16c,stride)→inv_l1
 *   L5 rows: 5 zones starting [0,2c)→inv_l5; L4 rows: 4 zones starting [0,4c)→inv_l4
 *   All zone boundaries are multiples of ll5_c=stride/32: even for 2K(60) and 4K(120).
 * 100% of rows now use vectorized loads + stores.
 *
 * Requirement: stride div-by-4 (2K=1920, 4K=3840); ll5_c = stride/32 always div-by-4.
 */
/* V87: __launch_bounds__(256,6) — forces ≤42 regs/T → guaranteed 6 blk/SM (was potentially 5). */
__global__ __launch_bounds__(256, 6)
void kernel_quantize_subband_ml(
    const __half* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int stride, int n_rows, float base_step,
    int ll5_h, int ll5_c)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    /* V103: __frcp_rn computed first; V115: inv_* deferred until after L1 early exit. */
    float base_inv = __frcp_rn(base_step);

    const __half* row_src = d_comp  + static_cast<size_t>(row) * stride;
    uint8_t*      row_dst = d_packed + static_cast<size_t>(row) * stride;

    /* V93: All rows vectorized via vq8 — 8 elements per thread per iteration.
     * vq8: 4×__half2 loads (16B) + 2×uint32_t stores (8B) vs vq4's 2×__half2+uint32_t.
     * Halves loop iterations vs vq4: 50% less loop overhead; same total memory traffic.
     * Zone widths ≥ 2*ll5_c are always div-by-8 → zero tail for L1..L5 rows (100% of non-DC rows).
     * DC rows only: two zones of width ll5_c (2K: 60, 4K: 120).
     *   4K: ll5_c=120, 120%8=0 → zero tail. 2K: ll5_c=60, 60%8=4 → vq4 tail (4 elements).
     * L1 rows: 1 zone. L2: 2 zones. L3: 3 zones. L4: 4 zones. L5: 5 zones. DC: 6 zones.
     */
    auto vq8 = [row_src, row_dst](int col_start, int col_end, float inv, int nt) {
        /* V125: Use 4-byte (__half2) loads — int2 (8-byte) loads are misaligned on odd rows
         * because stride=1998 → row byte offset=row*3996 (3996%8=4). */
        #pragma unroll 2
        for (int c = col_start + threadIdx.x * 8; c + 7 < col_end; c += nt * 8) {
            const __half2* hp = reinterpret_cast<const __half2*>(row_src) + c / 2;
            __half2 hv0_raw = __ldg(hp);
            __half2 hv1_raw = __ldg(hp + 1);
            __half2 hv2_raw = __ldg(hp + 2);
            __half2 hv3_raw = __ldg(hp + 3);
            int2 r01, r23;
            __builtin_memcpy(&r01.x, &hv0_raw, 4);
            __builtin_memcpy(&r01.y, &hv1_raw, 4);
            __builtin_memcpy(&r23.x, &hv2_raw, 4);
            __builtin_memcpy(&r23.y, &hv3_raw, 4);
            /* V103: Sign extraction via __byte_perm on raw int2 loads.
             * IEEE 754 half-float: sign bit = bit 15 = byte-index 1 of the 16-bit value.
             * r01.x layout: hv0.x in bits[15:0], hv0.y in bits[31:16].
             *   byte 1 = bits[15:8] of r01.x → contains sign of hv0.x in its bit 7.
             *   byte 3 = bits[31:24] of r01.x → contains sign of hv0.y in its bit 7.
             * __byte_perm(r01.x, r01.y, 0x7531):
             *   result byte0 ← byte1(r01.x) → sign of hv0.x at bit 7 ✓
             *   result byte1 ← byte3(r01.x) → sign of hv0.y at bit 7 ✓
             *   result byte2 ← byte1(r01.y) → sign of hv1.x at bit 7 ✓
             *   result byte3 ← byte3(r01.y) → sign of hv1.y at bit 7 ✓
             * & 0x80808080: isolate sign bits — total 2 ops (PRMT+AND) vs 8 (4×SETP+4×SELP).
             * Abs via AND 0x7FFF7FFF: clears sign bit in each half packed in uint32 — 1 op vs 4 IABS.
             * Total savings: 16 fewer ops per 8 elements vs SETP+SELP+IABS approach. */
            const uint32_t lo_signs = __byte_perm(uint32_t(r01.x), uint32_t(r01.y), 0x7531u) & 0x80808080u;
            const uint32_t hi_signs = __byte_perm(uint32_t(r23.x), uint32_t(r23.y), 0x7531u) & 0x80808080u;
            /* Abs: clear sign bit of each half (bit 15) in the packed uint32 representation. */
            __half2 hv0, hv1, hv2, hv3;
            const uint32_t a01x = uint32_t(r01.x) & 0x7FFF7FFFu;
            const uint32_t a01y = uint32_t(r01.y) & 0x7FFF7FFFu;
            const uint32_t a23x = uint32_t(r23.x) & 0x7FFF7FFFu;
            const uint32_t a23y = uint32_t(r23.y) & 0x7FFF7FFFu;
            __builtin_memcpy(&hv0, &a01x, 4);  /* hv0 = {|hv0.y|, |hv0.x|} */
            __builtin_memcpy(&hv1, &a01y, 4);
            __builtin_memcpy(&hv2, &a23x, 4);
            __builtin_memcpy(&hv3, &a23y, 4);
            /* Quantize absolute values — all ≥ 0, no abs() needed in magnitude computation. */
            int aq0 = __float2int_rn(__half2float(hv0.x) * inv);
            int aq1 = __float2int_rn(__half2float(hv0.y) * inv);
            int aq2 = __float2int_rn(__half2float(hv1.x) * inv);
            int aq3 = __float2int_rn(__half2float(hv1.y) * inv);
            int aq4 = __float2int_rn(__half2float(hv2.x) * inv);
            int aq5 = __float2int_rn(__half2float(hv2.y) * inv);
            int aq6 = __float2int_rn(__half2float(hv3.x) * inv);
            int aq7 = __float2int_rn(__half2float(hv3.y) * inv);
            /* Pack magnitudes (≤126, safe in bits[6:0]) then OR in sign bits (bit 7 each byte). */
            const uint32_t lo = lo_signs | (uint32_t(min(aq0,126)) | (uint32_t(min(aq1,126))<<8) | (uint32_t(min(aq2,126))<<16) | (uint32_t(min(aq3,126))<<24));
            const uint32_t hi = hi_signs | (uint32_t(min(aq4,126)) | (uint32_t(min(aq5,126))<<8) | (uint32_t(min(aq6,126))<<16) | (uint32_t(min(aq7,126))<<24));
            /* V126: memcpy stores — stride=1998 makes odd-row addresses only 2-byte aligned. */
            __builtin_memcpy(row_dst + c,     &lo, 4);
            __builtin_memcpy(row_dst + c + 4, &hi, 4);
        }
        /* vq4 tail: handles residual elements when zone width % 8 != 0.
         * Only occurs for 2K DC rows (ll5_c=60, 60%8=4 → 4 tail elements per small zone).
         * V109: PRMT sign extraction — parity with vq8 core (replaces 4 SETP+SELP+IABS with PRMT+AND). */
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
            { uint32_t packed = signs |
                (uint32_t(min(126,aq0)) | (uint32_t(min(126,aq1))<<8)
                | (uint32_t(min(126,aq2))<<16) | (uint32_t(min(126,aq3))<<24));
              __builtin_memcpy(row_dst + c, &packed, 4); }
        }
    };

    /* V115: L1-row early exit — saves 5 FMUL per thread for >50% of 2K/4K rows.
     * L1 rows (row >= ll5_h*16): only inv_l1 = base_inv*0.833... needed.
     * Exiting here avoids the 5 other FMUL constants for the majority path.
     * Register pressure: 6 float constants → 1 (base_inv) + inline for L1 blocks.
     * Parity: Slang V78. */
    if (row >= ll5_h * 16) {
        vq8(0, stride, base_inv * 0.833333333f, blockDim.x);
        return;
    }

    /* V103: 5 FMUL instead of 5 __frcp_rn (only reached for non-L1 rows, <50% of rows).
     * 1/(step*mult) ≈ (1/step)*(1/mult); compile-time reciprocal constants for 1/mult. */
    float inv_dc = base_inv * 1.538461538f;  /* 1/0.65 */
    float inv_l5 = base_inv * 1.176470588f;  /* 1/0.85 */
    float inv_l4 = base_inv * 1.052631579f;  /* 1/0.95 */
    float inv_l3 = base_inv * 0.952380952f;  /* 1/1.05 */
    float inv_l2 = base_inv * 0.892857143f;  /* 1/1.12 */
    float inv_l1 = base_inv * 0.833333333f;  /* 1/1.20 */

    bool is_dc_row = (row < ll5_h);
    bool is_l5_row = (row < ll5_h * 2);
    /* V115: row < ll5_h*16 guaranteed by early exit above → row_lv in {2..5}. */
    int row_lv = is_l5_row ? 5 :
                 (row < ll5_h * 4  ? 4 :
                 (row < ll5_h * 8  ? 3 : 2));

    /* row_lv == 1 handled above (early exit); dispatch only L2..L5/DC rows here. */
    if (row_lv == 2) {
        /* L2: 2 zones split at stride/2 = ll5_c*16 */
        const int mid = ll5_c * 16;
        vq8(0,   mid,    inv_l2, blockDim.x);
        vq8(mid, stride, inv_l1, blockDim.x);
    } else if (row_lv == 3) {
        /* L3: 3 zones split at ll5_c*8 and ll5_c*16 */
        const int b1 = ll5_c * 8;
        const int b2 = ll5_c * 16;
        vq8(0,  b1,     inv_l3, blockDim.x);
        vq8(b1, b2,     inv_l2, blockDim.x);
        vq8(b2, stride, inv_l1, blockDim.x);
    } else {
        /* V93: L4/L5/DC rows — fully vectorized zone-based quantize via vq8.
         * Zone widths ≥ 2*ll5_c are div-by-8 → zero vq4 tail for L4/L5 rows.
         * DC rows: 6 zones; two smallest zones (width ll5_c) may need vq4 tail (2K only).
         * DC rows: 6 zones (LL5 uses inv_dc; LH5/HL5/HH5 uses inv_l5; then l4..l1).
         * L5 rows: 5 zones (first zone [0,2c)→inv_l5; no separate DC sub-zone).
         * L4 rows: 4 zones (first zone [0,4c)→inv_l4). */
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
            /* L4 rows (row_lv==4): [0,4c)→inv_l4, then l3, l2, l1 */
            vq8(0,          ll5_c*4,    inv_l4, blockDim.x);
            vq8(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq8(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq8(ll5_c*16,   stride,     inv_l1, blockDim.x);
        }
    }
}


/**
 * V29: Tiled vertical DWT — register-based column tiles, no global workspace.
 *
 * V34: Processes height in tiles of V_TILE=32 rows (V29 used 16) with V_OVERLAP=5 halo rows
 * on each side. All V_TILE_FL=42 rows loaded from __half d_src into float registers;
 * CDF 9/7 lifting done in float (no accumulation error); deinterleaved float written
 * to d_dst. No shared memory required — maximizes SM occupancy.
 *
 * V34 benefit: 32-row tiles reduce per-column load redundancy from 38.9%→24.4%.
 *   2K level 0 (h=1080): 68 tiles × 26 loads = 1768 → 34 tiles × 42 loads = 1428 (~19% fewer).
 *
 * Correctness: whole-point symmetric boundary extension (WS) handles image borders.
 * With WS, the standard CDF 9/7 lifting formula (with no boundary special-casing)
 * automatically matches the required 2×coeff×neighbor conditions at y=0 and y=h-1:
 *   y<0  → reflect to -y   (so y=-1 → y=1, giving 2×coeff×row[1] naturally)
 *   y≥h  → reflect to 2(h-1)-y (so y=h → y=h-2, giving 2×coeff×row[h-2] naturally)
 * Lifting range [1, V_TILE_FL-2] avoids out-of-bounds shared array access; the skipped
 * edge rows (i=0 and i=V_TILE_FL-1) are never part of the OVERLAP+TILE output range.
 *
 * Grid: (ceil(width/V_THREADS), ceil(height/V_TILE)) 2D blocks.
 * Block: V_THREADS=128 threads (1 per column). No shared memory.
 */
__global__ void
kernel_fused_vert_dwt_tiled(
    const __half* __restrict__ d_src,
    float* __restrict__ d_dst,
    int width, int height, int stride)
{
    int x          = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_start = blockIdx.y * V_TILE;
    if (x >= width || tile_start >= height) return;

    int load_start = tile_start - V_OVERLAP;
    int tile_end   = min(tile_start + V_TILE, height);  /* actual tile end (last tile is partial) */

    /* Load V_TILE_FL rows into float registers with whole-point symmetric extension.
     * Symmetric extension naturally handles boundary conditions (no special cases needed). */
    float col[V_TILE_FL];
    for (int i = 0; i < V_TILE_FL; i++) {
        int gy = load_start + i;
        if (gy < 0) gy = -gy;                        /* reflect: -1→1, -2→2, ... */
        else if (gy >= height) gy = 2*(height-1) - gy; /* reflect: h→h-2, h+1→h-3, ... */
        col[i] = __half2float(__ldg(&d_src[gy * stride + x]));
    }

    /* V80: p0 always 0 (V_TILE even, V_OVERLAP even → load_start always even).
     * static_assert in kernel_fused_vert_dwt_tiled_ho verifies this invariant. */
    cdf97_lift_tiled<0>(col);

    /* V80: output parity p0=0: odd i → H subband, even i → L subband. */
    int hh = (height + 1) / 2;
    for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
        int gy = load_start + i;
        if (i & 1)                 /* V80: odd i (p0=0) → globally odd → H subband */
            d_dst[(hh + gy/2) * stride + x] = col[i] * NORM_H;
        else                       /* even i → globally even → L subband */
            d_dst[(gy/2) * stride + x] = col[i] * NORM_L;
    }
}


#if 0  /* Dead kernel — superseded by kernel_rgb48_xyz_hdwt0_1ch variants */
/**
 * V28: Fused RGB48→XYZ colour conversion + horizontal DWT level 0 (all 3 components).
 *
 * Combines kernel_rgb48_to_xyz12 + 3× kernel_fused_i2f_horz_dwt_half_out into a single
 * kernel, eliminating the int32 XYZ intermediate planes (d_in[0..2]) for the RGB path.
 *
 * Savings vs separate kernels:
 *   - Eliminates write of d_in[0..2]: 3 × width × height × 4 bytes = 27MB/frame
 *   - Eliminates read of d_in[0..2] by H-DWT: 27MB/frame
 *   - Total: ~54MB/frame → ~0.48ms at 112 GB/s
 *
 * Structure: one block per row (blockIdx.x = row y).
 * Shared memory: 3 × width floats (XYZ channels, 24KB for 2K).
 * Phases per block:
 *   1. Load RGB48 → apply LUTs + matrix → store X, Y, Z into smX[], smY[], smZ[]
 *   2. In-place H-DWT (4 lifting steps) on smX → write __half to d_hx
 *   3. In-place H-DWT on smY → write __half to d_hy
 *   4. In-place H-DWT on smZ → write __half to d_hz
 * The H-DWT for each channel is sequential (smX is modified first, then smY, then smZ).
 *
 * Occupancy trade-off: 24KB shared memory → 2 blocks/SM → 25% occupancy.
 * This is acceptable because the kernel is compute-intensive (LUT + matrix + lifting)
 * and the bandwidth savings outweigh the occupancy reduction.
 */
__global__ void
kernel_rgb48_xyz_hdwt0(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V55: was float; 8KB vs 16KB GPU texture cache */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hx,   /* comp 0 H-DWT half output */
    __half* __restrict__ d_hy,   /* comp 1 H-DWT half output */
    __half* __restrict__ d_hz,   /* comp 2 H-DWT half output */
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];   /* reinterpreted as float below; 3 × width floats */
    float* smX = reinterpret_cast<float*>(sm);
    float* smY = smX + width;
    float* smZ = smX + 2 * width;

    int y = blockIdx.x;
    if (y >= height) return;
    int t = threadIdx.x, nt = blockDim.x;

    /* Phase 1: RGB48LE → XYZ float into shared memory.
     * Each thread handles ceil(width/nt) pixels. LUT indices cached via __ldg. */
    for (int px = t; px < width; px += nt) {
        int base = y * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[base + 0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[base + 1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[base + 2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55: __half→float */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float xv = __ldg(&d_matrix[0])*r + __ldg(&d_matrix[1])*g + __ldg(&d_matrix[2])*b;
        float yv = __ldg(&d_matrix[3])*r + __ldg(&d_matrix[4])*g + __ldg(&d_matrix[5])*b;
        float zv = __ldg(&d_matrix[6])*r + __ldg(&d_matrix[7])*g + __ldg(&d_matrix[8])*b;
        xv = __saturatef(xv);
        yv = __saturatef(yv);
        zv = __saturatef(zv);
        smX[px] = static_cast<float>(__ldg(&d_lut_out[static_cast<int>(xv * 4095.5f)]));
        smY[px] = static_cast<float>(__ldg(&d_lut_out[static_cast<int>(yv * 4095.5f)]));
        smZ[px] = static_cast<float>(__ldg(&d_lut_out[static_cast<int>(zv * 4095.5f)]));
    }
    __syncthreads();

    /* Phase 2-4: In-place H-DWT on each channel, then write __half output.
     * smX/smY/smZ are processed sequentially — each channel is fully lifted before
     * the next starts. smY and smZ are intact while smX is being processed, etc. */
    int w = width;
    int hw = (w + 1) / 2;

    /* Macro-style: apply CDF 9/7 lifting in-place on array 'smc', then write to 'dst' */
    /* V119: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
#define DO_HDWT_HALF(smc, dst)                                                              \
    for (int x = 1+t*2; x < w-1; x += nt*2) (smc)[x] += ALPHA*((smc)[x-1]+(smc)[x+1]);  \
    if (t==0) (smc)[w-1] += 2.f*ALPHA*(smc)[w-2];                                          \
    __syncthreads();                                                                         \
    if (t==0) (smc)[0] += 2.f*BETA*(smc)[1];                                               \
    for (int x = 2+t*2; x < w; x += nt*2) (smc)[x] += BETA*((smc)[x-1]+(smc)[x+1]);      \
    __syncthreads();                                                                         \
    for (int x = 1+t*2; x < w-1; x += nt*2) (smc)[x] += GAMMA*((smc)[x-1]+(smc)[x+1]);  \
    if (t==0) (smc)[w-1] += 2.f*GAMMA*(smc)[w-2];                                          \
    __syncthreads();                                                                         \
    if (t==0) (smc)[0] += 2.f*DELTA*(smc)[1];                                              \
    for (int x = 2+t*2; x < w; x += nt*2) (smc)[x] += DELTA*((smc)[x-1]+(smc)[x+1]);     \
    __syncthreads();                                                                         \
    for (int x = t*2;   x < w; x += nt*2) (dst)[y*stride + x/2]    = __float2half((smc)[x]*NORM_L); \
    for (int x = t*2+1; x < w; x += nt*2) (dst)[y*stride + hw+x/2] = __float2half((smc)[x]*NORM_H); \
    __syncthreads();

    DO_HDWT_HALF(smX, d_hx)
    DO_HDWT_HALF(smY, d_hy)
    DO_HDWT_HALF(smZ, d_hz)

#undef DO_HDWT_HALF
}
#endif  /* dead float-smem kernel_rgb48_xyz_hdwt0 */


/**
 * V30: Single-channel fused RGB48→XYZ colour conversion + horizontal DWT level 0.
 *
 * Split from kernel_rgb48_xyz_hdwt0 (3-channel, 24KB smem) into a 1-channel variant
 * using only 1×width floats of shared memory (8KB for 2K).
 *
 * Motivation:
 *   24KB smem per block → 2 blocks/SM → 25% thread occupancy on GTX 1050 Ti.
 *   8KB smem per block → 6 blocks/SM (smem limited, 48KB/8KB=6), but thread limit is
 *   8 blocks/SM (2048 threads/SM ÷ 256 threads/block). So we reach:
 *     3 parallel streams × 6 blocks/SM = 18 desired, capped at 8 → 100% thread occupancy.
 *   All 3 streams read the same d_rgb16 → L2 cache reuse across streams eliminates 2×
 *   the extra DRAM reads. Effective DRAM: 13.3MB RGB + 13.2MB d_hXYZ = 26.5MB (same as
 *   3-ch kernel), but at 100% vs 25% occupancy → ~4× better latency hiding.
 *
 * Launch pattern (encode_from_rgb48):
 *   kernel_rgb48_xyz_hdwt0_1ch<<<height, 512, width*sizeof(__half), stream[0]>>>(d_rgb16, ..., d_hx, 0);
 *   kernel_rgb48_xyz_hdwt0_1ch<<<height, 512, width*sizeof(__half), stream[1]>>>(d_rgb16, ..., d_hy, 1);
 *   kernel_rgb48_xyz_hdwt0_1ch<<<height, 512, width*sizeof(__half), stream[2]>>>(d_rgb16, ..., d_hz, 2);
 * Streams 1 & 2 must wait on h2d_done event before launch (same as before).
 *
 * Expected speedup: 0.95ms → ~0.25ms for this stage → ~0.7ms/frame saved.
 */
/* V38: __half shared memory — colour convert to float (needs precision), write half to smem, lift fp16. */
__global__ void
kernel_rgb48_xyz_hdwt0_1ch(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V55: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,   /* single component H-DWT half output */
    int comp,                       /* 0=X, 1=Y, 2=Z */
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];  /* V38: was float; halves smem (3.75KB/block for 2K) */
    int y = blockIdx.x;
    if (y >= height) return;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr + 1]);
    float m2 = __ldg(&d_matrix[mr + 2]);

    /* Phase 1: RGB48LE → single XYZ channel → __half shared memory. */
    for (int px = t; px < w; px += nt) {
        int base = y * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[base + 0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[base + 1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[base + 2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55 */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = m0 * r + m1 * g + m2 * b;
        v = __saturatef(v);
        sm[px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v * 4095.5f)]));
    }
    __syncthreads();

    /* Phase 2: In-place H-DWT with fp16 arithmetic; write __half to d_hout. */
    /* V119: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(ALPHA)*(sm[x-1]+sm[x+1]);
    if (t==0) sm[w-1] += __half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*BETA)*sm[1];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(BETA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(GAMMA)*(sm[x-1]+sm[x+1]);
    if (t==0) sm[w-1] += __half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*DELTA)*sm[1];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(DELTA)*(sm[x-1]+sm[x+1]);
    __syncthreads();
    for (int x = t*2;   x < w; x += nt*2) d_hout[y*stride + x/2]      = sm[x] * __half(NORM_L);
    for (int x = t*2+1; x < w; x += nt*2) d_hout[y*stride + hw + x/2] = sm[x] * __half(NORM_H);
}


/**
 * V45: 2-rows-per-block variant of kernel_rgb48_xyz_hdwt0_1ch.
 * Each block processes rows y0=blockIdx.x*2 and y1=y0+1 simultaneously.
 * smem[0..w-1] = row y0; smem[w..2w-1] = row y1.
 *
 * Benefits vs 1-row:
 *   - Grid halved (540 vs 1080 blocks for 2K) → half the block-scheduler overhead
 *   - Matrix m0/m1/m2 amortized over 2× the work (same 3 __ldg reads per 2 rows)
 *   - 2 adjacent RGB rows → L2 cache spatial locality (consecutive DRAM lines)
 *   - 4 __syncthreads for 2 rows (vs 4 per row × 2 invocations) → same sync count
 *   - SM occupancy unchanged (thread-limited at 4 blocks/SM): 8 rows/SM/pass vs 4
 *
 * Launch: kernel_rgb48_xyz_hdwt0_1ch_2row<<<(height+1)/2, 512, 2*w*sizeof(__half), st>>>
 */
__global__ void
kernel_rgb48_xyz_hdwt0_1ch_2row(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V55: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];  /* [0..w-1]=row y0, [w..2w-1]=row y1 */
    int y0 = blockIdx.x * 2;
    int y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr + 1]);
    float m2 = __ldg(&d_matrix[mr + 2]);

    /* Phase 1: load row y0 → sm[0..w-1] and row y1 → sm[w..2w-1]. */
    for (int px = t; px < w; px += nt) {
        int b0 = y0 * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[b0 + 0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[b0 + 1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[b0 + 2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55 */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = __saturatef(m0*r + m1*g + m2*b);
        /* DC level shift: encoder stores XYZ-2048 so decoder's +2048 reconstructs XYZ. */
        sm[px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)])) - __half(2048.0f);
    }
    if (y1 < height) {
        for (int px = t; px < w; px += nt) {
            int b1 = y1 * rgb_stride + px * 3;
            int ri = min((__ldg(&d_rgb16[b1 + 0]) >> 4), 4095);
            int gi = min((__ldg(&d_rgb16[b1 + 1]) >> 4), 4095);
            int bi = min((__ldg(&d_rgb16[b1 + 2]) >> 4), 4095);
            float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55 */
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = __saturatef(m0*r + m1*g + m2*b);
            sm[w + px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)])) - __half(2048.0f);
        }
    }
    __syncthreads();

    /* Phase 2: in-place CDF 9/7 H-DWT on both rows, 4 lifting passes. */
    /* V119: DCI w always even and >1 → drop w>1&&!(w&1); min(1,w-1)=1; min(x+1,w-1)=x+1. */
    /* Alpha: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(ALPHA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(ALPHA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0) {
        sm[w-1]   += __half(2.f*ALPHA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*ALPHA) * sm[2*w-2];
    }
    __syncthreads();
    /* Beta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*BETA) * sm[1];
        if (y1 < height) sm[w] += __half(2.f*BETA) * sm[w+1];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(BETA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(BETA) * (sm[w+x-1] + sm[w+x+1]);
    }
    __syncthreads();
    /* Gamma: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(GAMMA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(GAMMA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0) {
        sm[w-1]   += __half(2.f*GAMMA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*GAMMA) * sm[2*w-2];
    }
    __syncthreads();
    /* Delta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*DELTA) * sm[1];
        if (y1 < height) sm[w] += __half(2.f*DELTA) * sm[w+1];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(DELTA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(DELTA) * (sm[w+x-1] + sm[w+x+1]);
    }
    __syncthreads();

    /* Phase 3: deinterleave and write half output. */
    for (int x = t*2;   x < w; x += nt*2) {
        d_hout[y0*stride + x/2]        = sm[x]   * __half(NORM_L);
        if (y1 < height) d_hout[y1*stride + x/2]   = sm[w+x] * __half(NORM_L);
    }
    for (int x = t*2+1; x < w; x += nt*2) {
        d_hout[y0*stride + hw + x/2]      = sm[x]   * __half(NORM_H);
        if (y1 < height) d_hout[y1*stride + hw + x/2] = sm[w+x] * __half(NORM_H);
    }

}


/**
 * V171: 4-rows-per-block RGB→XYZ+HDWT0 kernel for RGB48 interleaved input.
 * Halves block count vs 2-row: (height+1)/2 → (height+3)/4 blocks.
 * Shared memory: 4*w*sizeof(__half).
 * Launch: <<<(height+3)/4, H_THREADS_FUSED, 4*w*sizeof(__half), st>>>
 */
__global__ void
kernel_rgb48_xyz_hdwt0_1ch_4row(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int rgb_stride, int stride)
{
    extern __shared__ __half sm[];  /* [0..w-1]=y0, [w..2w-1]=y1, [2w..3w-1]=y2, [3w..4w-1]=y3 */
    int y0 = blockIdx.x * 4;
    int y1 = y0 + 1, y2 = y0 + 2, y3 = y0 + 3;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr + 1]);
    float m2 = __ldg(&d_matrix[mr + 2]);

    /* Phase 1: colour-convert all 4 rows → sm[0..4w-1]. */
    for (int px = t; px < w; px += nt) {
        int b0 = y0 * rgb_stride + px * 3;
        int ri = min((__ldg(&d_rgb16[b0 + 0]) >> 4), 4095);
        int gi = min((__ldg(&d_rgb16[b0 + 1]) >> 4), 4095);
        int bi = min((__ldg(&d_rgb16[b0 + 2]) >> 4), 4095);
        float r = __half2float(__ldg(&d_lut_in[ri]));
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = __saturatef(m0*r + m1*g + m2*b);
        sm[px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)]));
    }
    if (y1 < height) {
        for (int px = t; px < w; px += nt) {
            int b1 = y1 * rgb_stride + px * 3;
            int ri = min((__ldg(&d_rgb16[b1 + 0]) >> 4), 4095);
            int gi = min((__ldg(&d_rgb16[b1 + 1]) >> 4), 4095);
            int bi = min((__ldg(&d_rgb16[b1 + 2]) >> 4), 4095);
            float r = __half2float(__ldg(&d_lut_in[ri]));
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = __saturatef(m0*r + m1*g + m2*b);
            sm[w + px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)]));
        }
    }
    if (y2 < height) {
        for (int px = t; px < w; px += nt) {
            int b2 = y2 * rgb_stride + px * 3;
            int ri = min((__ldg(&d_rgb16[b2 + 0]) >> 4), 4095);
            int gi = min((__ldg(&d_rgb16[b2 + 1]) >> 4), 4095);
            int bi = min((__ldg(&d_rgb16[b2 + 2]) >> 4), 4095);
            float r = __half2float(__ldg(&d_lut_in[ri]));
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = __saturatef(m0*r + m1*g + m2*b);
            sm[2*w + px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)]));
        }
    }
    if (y3 < height) {
        for (int px = t; px < w; px += nt) {
            int b3 = y3 * rgb_stride + px * 3;
            int ri = min((__ldg(&d_rgb16[b3 + 0]) >> 4), 4095);
            int gi = min((__ldg(&d_rgb16[b3 + 1]) >> 4), 4095);
            int bi = min((__ldg(&d_rgb16[b3 + 2]) >> 4), 4095);
            float r = __half2float(__ldg(&d_lut_in[ri]));
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = __saturatef(m0*r + m1*g + m2*b);
            sm[3*w + px] = u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)]));
        }
    }
    __syncthreads();

    /* Phase 2: in-place CDF 9/7 H-DWT on all 4 rows, 4 lifting passes. */
    /* Alpha: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]     += __half(ALPHA) * (sm[x-1]       + sm[x+1]);
        if (y1 < height) sm[w+x]   += __half(ALPHA) * (sm[w+x-1]   + sm[w+x+1]);
        if (y2 < height) sm[2*w+x] += __half(ALPHA) * (sm[2*w+x-1] + sm[2*w+x+1]);
        if (y3 < height) sm[3*w+x] += __half(ALPHA) * (sm[3*w+x-1] + sm[3*w+x+1]);
    }
    if (t==0) {
        sm[w-1]   += __half(2.f*ALPHA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*ALPHA) * sm[2*w-2];
        if (y2 < height) sm[3*w-1] += __half(2.f*ALPHA) * sm[3*w-2];
        if (y3 < height) sm[4*w-1] += __half(2.f*ALPHA) * sm[4*w-2];
    }
    __syncthreads();
    /* Beta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*BETA) * sm[1];
        if (y1 < height) sm[w]   += __half(2.f*BETA) * sm[w+1];
        if (y2 < height) sm[2*w] += __half(2.f*BETA) * sm[2*w+1];
        if (y3 < height) sm[3*w] += __half(2.f*BETA) * sm[3*w+1];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]     += __half(BETA) * (sm[x-1]       + sm[x+1]);
        if (y1 < height) sm[w+x]   += __half(BETA) * (sm[w+x-1]   + sm[w+x+1]);
        if (y2 < height) sm[2*w+x] += __half(BETA) * (sm[2*w+x-1] + sm[2*w+x+1]);
        if (y3 < height) sm[3*w+x] += __half(BETA) * (sm[3*w+x-1] + sm[3*w+x+1]);
    }
    __syncthreads();
    /* Gamma: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]     += __half(GAMMA) * (sm[x-1]       + sm[x+1]);
        if (y1 < height) sm[w+x]   += __half(GAMMA) * (sm[w+x-1]   + sm[w+x+1]);
        if (y2 < height) sm[2*w+x] += __half(GAMMA) * (sm[2*w+x-1] + sm[2*w+x+1]);
        if (y3 < height) sm[3*w+x] += __half(GAMMA) * (sm[3*w+x-1] + sm[3*w+x+1]);
    }
    if (t==0) {
        sm[w-1]   += __half(2.f*GAMMA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*GAMMA) * sm[2*w-2];
        if (y2 < height) sm[3*w-1] += __half(2.f*GAMMA) * sm[3*w-2];
        if (y3 < height) sm[4*w-1] += __half(2.f*GAMMA) * sm[4*w-2];
    }
    __syncthreads();
    /* Delta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*DELTA) * sm[1];
        if (y1 < height) sm[w]   += __half(2.f*DELTA) * sm[w+1];
        if (y2 < height) sm[2*w] += __half(2.f*DELTA) * sm[2*w+1];
        if (y3 < height) sm[3*w] += __half(2.f*DELTA) * sm[3*w+1];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]     += __half(DELTA) * (sm[x-1]       + sm[x+1]);
        if (y1 < height) sm[w+x]   += __half(DELTA) * (sm[w+x-1]   + sm[w+x+1]);
        if (y2 < height) sm[2*w+x] += __half(DELTA) * (sm[2*w+x-1] + sm[2*w+x+1]);
        if (y3 < height) sm[3*w+x] += __half(DELTA) * (sm[3*w+x-1] + sm[3*w+x+1]);
    }
    __syncthreads();

    /* Phase 3: deinterleave and write half output for all 4 rows. */
    for (int x = t*2;   x < w; x += nt*2) {
        d_hout[y0*stride + x/2]                         = sm[x]     * __half(NORM_L);
        if (y1 < height) d_hout[y1*stride + x/2]        = sm[w+x]   * __half(NORM_L);
        if (y2 < height) d_hout[y2*stride + x/2]        = sm[2*w+x] * __half(NORM_L);
        if (y3 < height) d_hout[y3*stride + x/2]        = sm[3*w+x] * __half(NORM_L);
    }
    for (int x = t*2+1; x < w; x += nt*2) {
        d_hout[y0*stride + hw + x/2]                    = sm[x]     * __half(NORM_H);
        if (y1 < height) d_hout[y1*stride + hw + x/2]  = sm[w+x]   * __half(NORM_H);
        if (y2 < height) d_hout[y2*stride + hw + x/2]  = sm[2*w+x] * __half(NORM_H);
        if (y3 < height) d_hout[y3*stride + hw + x/2]  = sm[3*w+x] * __half(NORM_H);
    }
}


/**
 * V54: 2-rows-per-block RGB→XYZ+HDWT0 kernel for packed 12-bit planar input.
 *
 * Input format (d_rgb12): 3 contiguous planes, each holding all pixels of one channel
 * in packed 12-bit pairs.  For plane 'ch' at row 'y', pixel pair 'px/2':
 *   byte offset = ch * height * packed_row_stride + y * packed_row_stride + (px/2)*3
 *   packed_row_stride = (width/2) * 3  (bytes per channel per row)
 *   Pair unpack:
 *     even px: val12 = (b0 << 4) | (b1 >> 4)
 *     odd  px: val12 = ((b1 & 0xF) << 8) | b2
 *
 * Benefits over uint16_t interleaved (kernel_rgb48_xyz_hdwt0_1ch_2row):
 *   - 25% smaller H2D transfer (9.5MB vs 12.6MB for 2K) → H2D 1.26ms vs 1.68ms
 *   - FPS ceiling raised from ~600fps to ~800fps
 *   - Better GPU cache coalescing: adjacent thread pairs share cache lines (bytes 0-47
 *     for 32 threads vs stride-6 uint16_t interleaved)
 *   - No change to DWT or downstream kernels
 *
 * Launch: same as V45 2-row kernel: <<<(height+1)/2, H_THREADS_FUSED, 2*w*sizeof(__half), st>>>
 */
/* V116: __launch_bounds__(512,3) — matches actual 3 blk/SM smem limit (was 512,4 from V88).
 * V90 added 8KB sm_lut; smem=15.68KB → PreferShared(48KB)=3 blk/SM. (512,4)→32 regs was wrong.
 * (512,3) → ≤42 regs/T → 42×512=21504 regs/block → 65536/21504=3 blk/SM (same occupancy).
 * 10 extra registers per thread allow compiler to reduce spills in the complex color+DWT kernel. */
__global__ __launch_bounds__(512, 3)
void kernel_rgb48_xyz_hdwt0_1ch_2row_p12(
    const uint8_t*  __restrict__ d_rgb12,   /* V54: packed 12-bit planar (was uint16_t*) */
    const __half*   __restrict__ d_lut_in,  /* V55: was float */
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    int y0 = blockIdx.x * 2;
    int y1 = y0 + 1;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr + 1]);
    float m2 = __ldg(&d_matrix[mr + 2]);

    /* V90: 8KB static smem for d_lut_in preload — 0-cycle smem access vs ~30-50 cycle L1/L2. */
    __shared__ __half sm_lut[4096];

    /* Plane bases: each plane = height * packed_row_stride bytes */
    size_t plane_stride = (size_t)height * packed_row_stride;
    const uint8_t* r_plane = d_rgb12 + 0 * plane_stride;
    const uint8_t* g_plane = d_rgb12 + 1 * plane_stride;
    const uint8_t* b_plane = d_rgb12 + 2 * plane_stride;

    /* V91: int4 vectorized preload — 1 × 128-bit __ldg per thread (8 × __half) vs 8 scalar loads.
     * nt=512T and 4096 __half / 8 per int4 = 512 int4 entries → each thread loads exactly 1 int4. */
    reinterpret_cast<int4*>(sm_lut)[t] = __ldg(reinterpret_cast<const int4*>(d_lut_in) + t);
    __syncthreads();

    /* V83: hoist y1<height to block level — interior uses __half2 interleaved smem (same 2w __half size).
     * sm2[x]={row0[x],row1[x]}; all lifting steps use HFMA2 for 2× FMA throughput. */
    /* V90: lambda reads sm_lut (smem) instead of __ldg d_lut_in (L1/L2). */
    auto lut_xyz_2row = [&](int ri, int gi, int bi) -> __half {
        float r = __half2float(sm_lut[ri]);
        float g = __half2float(sm_lut[gi]);
        float b = __half2float(sm_lut[bi]);
        float v = __saturatef(m0*r + m1*g + m2*b);
        return u16_to_f16(__ldg(&d_lut_out[static_cast<int>(v*4095.5f)]));
    };
    /* V122: DCI height always even (1080/2160) → y1=y0+1 always < height; partial else branch removed.
     * Compiler can now optimize unconditionally — no register partitioning for dead branch. */
    {
        /* Interior path: both rows valid — __half2 lifting. */
        __half2* sm2 = reinterpret_cast<__half2*>(sm);
        /* V125: Byte-level loads — V121's uint32_t __ldg causes misaligned address on sm_61
         * because packed_row_stride (e.g. 2997) is not 4-byte aligned.
         * Each pixel pair occupies 3 bytes: [b0: lo8_of_pix0, b1: hi4_pix0|lo4_pix1, b2: hi8_pix1].
         * pix0 = (b0 << 4) | (b1 >> 4);  pix1 = ((b1 & 0xF) << 8) | b2. */
        #pragma unroll 2
        for (int p = t; p * 2 < w; p += nt) {
            int off0 = y0 * packed_row_stride + p * 3;
            uint8_t rb0=__ldg(r_plane+off0), rb1=__ldg(r_plane+off0+1), rb2=__ldg(r_plane+off0+2);
            uint8_t gb0=__ldg(g_plane+off0), gb1=__ldg(g_plane+off0+1), gb2=__ldg(g_plane+off0+2);
            uint8_t bb0=__ldg(b_plane+off0), bb1=__ldg(b_plane+off0+1), bb2=__ldg(b_plane+off0+2);
            __half e0 = lut_xyz_2row((rb0<<4)|(rb1>>4), (gb0<<4)|(gb1>>4), (bb0<<4)|(bb1>>4));
            __half o0 = lut_xyz_2row(((rb1&0xF)<<8)|rb2, ((gb1&0xF)<<8)|gb2, ((bb1&0xF)<<8)|bb2);
            int off1 = y1 * packed_row_stride + p * 3;
            uint8_t rb3=__ldg(r_plane+off1), rb4=__ldg(r_plane+off1+1), rb5=__ldg(r_plane+off1+2);
            uint8_t gb3=__ldg(g_plane+off1), gb4=__ldg(g_plane+off1+1), gb5=__ldg(g_plane+off1+2);
            uint8_t bb3=__ldg(b_plane+off1), bb4=__ldg(b_plane+off1+1), bb5=__ldg(b_plane+off1+2);
            __half e1 = lut_xyz_2row((rb3<<4)|(rb4>>4), (gb3<<4)|(gb4>>4), (bb3<<4)|(bb4>>4));
            __half o1 = lut_xyz_2row(((rb4&0xF)<<8)|rb5, ((gb4&0xF)<<8)|gb5, ((bb4&0xF)<<8)|bb5);
            sm2[p*2]   = __halves2half2(e0, e1);
            sm2[p*2+1] = __halves2half2(o0, o1);
        }
        __syncthreads();
        {
            const __half2 kA = __half2half2(__float2half(ALPHA));
            /* V89: #pragma unroll 2 — 2K ~2 iters; interleaves smem reads + HFMA2s. */
            #pragma unroll 2
            for (int x = 1+t*2; x < w-1; x += nt*2)
                sm2[x] = __hfma2(kA, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
            /* V111: DCI w always even and >1 → simplify Alpha boundary (parity Slang V74). */
            if (t==0)
                sm2[w-1] = __hfma2(__half2half2(__float2half(2.f*ALPHA)), sm2[w-2], sm2[w-1]);
        }
        __syncthreads();
        {
            const __half2 kB = __half2half2(__float2half(BETA));
            /* V111: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
            if (t==0) sm2[0] = __hfma2(__half2half2(__float2half(2.f*BETA)), sm2[1], sm2[0]);
            /* V102: x always even → even w: min(x+1,w-1)=x+1; drop MIN per iter. */
            #pragma unroll 2
            for (int x = 2+t*2; x < w-1; x += nt*2)
                sm2[x] = __hfma2(kB, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
        }
        __syncthreads();
        {
            const __half2 kG = __half2half2(__float2half(GAMMA));
            #pragma unroll 2
            for (int x = 1+t*2; x < w-1; x += nt*2)
                sm2[x] = __hfma2(kG, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
            /* V111: DCI w always even and >1 → simplify Gamma boundary. */
            if (t==0)
                sm2[w-1] = __hfma2(__half2half2(__float2half(2.f*GAMMA)), sm2[w-2], sm2[w-1]);
        }
        __syncthreads();
        {
            const __half2 kD = __half2half2(__float2half(DELTA));
            /* V111: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
            if (t==0) sm2[0] = __hfma2(__half2half2(__float2half(2.f*DELTA)), sm2[1], sm2[0]);
            /* V102: same even-w invariant as Beta — drop MIN. */
            #pragma unroll 2
            for (int x = 2+t*2; x < w-1; x += nt*2)
                sm2[x] = __hfma2(kD, __hadd2(sm2[x-1], sm2[x+1]), sm2[x]);
        }
        __syncthreads();
        /* V83+V82: combined L+H scatter via __low2half/__high2half. */
        {
            const __half2 nL = __half2half2(__float2half(NORM_L));
            const __half2 nH = __half2half2(__float2half(NORM_H));
            /* V89: #pragma unroll 2 — interleaves smem reads + global writes. */
            #pragma unroll 2
            for (int p = t; p < w/2; p += nt) {
                __half2 vL = __hmul2(sm2[p*2],   nL);
                __half2 vH = __hmul2(sm2[p*2+1], nH);
                d_hout[y0*stride + p]       = __low2half(vL);
                d_hout[y1*stride + p]       = __high2half(vL);
                d_hout[y0*stride + hw + p]  = __low2half(vH);
                d_hout[y1*stride + hw + p]  = __high2half(vH);
            }
        }
    }
}


/**
 * V79: 1-row-per-block RGB+HDWT0 p12 for 4K — 100% occupancy (parity with Slang V42).
 * grid=height; smem=width*sizeof(__half) (3.84KB@2K, 7.68KB@4K).
 * For 4K: smem=7.68KB, PreferNone(32KB/SM)→4 blk/SM=100% occ vs 2-row's 2 blk/SM=50%.
 * Simpler than 2-row: no y1 guard, single-row smem. Wave count same as 2-row (grid/SM×blk).
 */
/* V116: __launch_bounds__(512,3) — matches actual 3 blk/SM smem limit (same as 2row_p12 fix above).
 * smem=15.68KB (sm_lut 8KB + DWT 7.68KB for 4K) → PreferShared(48KB)=3 blk/SM actual limit.
 * (512,3) → ≤42 regs/T; gives compiler 10 extra registers vs old (512,4)→32 regs. */
__global__ __launch_bounds__(512, 3)
void kernel_rgb48_xyz_hdwt0_1ch_1row_p12(
    const uint8_t*  __restrict__ d_rgb12,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    /* V90: 8KB static smem for d_lut_in preload — 0-cycle access vs ~30-50 cycle L1/L2 read. */
    __shared__ __half sm_lut[4096];
    int y0 = blockIdx.x;
    int t = threadIdx.x, nt = blockDim.x;
    int w = width, hw = (w + 1) / 2;

    /* V91: int4 vectorized preload — all block threads participate; early exit after sync (deadlock safe).
     * 512T and 512 int4 entries → each thread loads exactly 1 int4 (8 × __half) in one ld.global.b128. */
    reinterpret_cast<int4*>(sm_lut)[t] = __ldg(reinterpret_cast<const int4*>(d_lut_in) + t);
    __syncthreads();

    /* Guard after sync — y0 is always < height (grid=height) but safe for any future grid change. */
    if (y0 >= height) return;

    int mr = comp * 3;
    float m0 = __ldg(&d_matrix[mr]);
    float m1 = __ldg(&d_matrix[mr + 1]);
    float m2 = __ldg(&d_matrix[mr + 2]);

    size_t plane_stride = (size_t)height * packed_row_stride;
    const uint8_t* r_p = d_rgb12 + 0 * plane_stride;
    const uint8_t* g_p = d_rgb12 + 1 * plane_stride;
    const uint8_t* b_p = d_rgb12 + 2 * plane_stride;

    /* V125: Byte-level loads — V121's uint32_t __ldg causes misaligned address on sm_61
     * because packed_row_stride (e.g. 2997) is not 4-byte aligned.
     * Each pixel pair = 3 bytes: pix0=(b0<<4)|(b1>>4), pix1=((b1&0xF)<<8)|b2. */
    #pragma unroll 4
    for (int p = t; p * 2 < w; p += nt) {
        int off = y0 * packed_row_stride + p * 3;
        uint8_t rb0=__ldg(r_p+off), rb1=__ldg(r_p+off+1), rb2=__ldg(r_p+off+2);
        uint8_t gb0=__ldg(g_p+off), gb1=__ldg(g_p+off+1), gb2=__ldg(g_p+off+2);
        uint8_t bb0=__ldg(b_p+off), bb1=__ldg(b_p+off+1), bb2=__ldg(b_p+off+2);
        int ri0 = (rb0<<4)|(rb1>>4), gi0 = (gb0<<4)|(gb1>>4), bi0 = (bb0<<4)|(bb1>>4);
        int ri1 = ((rb1&0xF)<<8)|rb2, gi1 = ((gb1&0xF)<<8)|gb2, bi1 = ((bb1&0xF)<<8)|bb2;
        /* V90: use sm_lut (smem) instead of __ldg (L1/L2) for zero-latency LUT reads. */
        auto lut_xyz = [&](int ri, int gi, int bi) -> __half {
            float r = __half2float(sm_lut[ri]);
            float g = __half2float(sm_lut[gi]);
            float b = __half2float(sm_lut[bi]);
            float v = __saturatef(m0*r + m1*g + m2*b);
            return u16_to_f16(__ldg(&d_lut_out[(int)(v*4095.5f)]));
        };
        sm[p*2]   = lut_xyz(ri0, gi0, bi0);
        sm[p*2+1] = lut_xyz(ri1, gi1, bi1);
    }
    __syncthreads();

    /* V89: #pragma unroll 4 — ~4 iters for 4K; compiler interleaves smem reads + FMAs across iters. */
    #pragma unroll 4
    for (int x = 1+t*2; x < w-1; x += nt*2)
        sm[x] += __half(ALPHA) * (sm[x-1] + sm[x+1]);
    /* V111: DCI w always even and >1 → simplify Alpha boundary; parity Slang V74. */
    if (t==0) sm[w-1] += __half(2.f*ALPHA) * sm[w-2];
    __syncthreads();
    /* V111: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
    if (t==0) sm[0] += __half(2.f*BETA) * sm[1];
    /* V102: x always even → even w (4K=3840 even): x≤w-2 → x+1≤w-1 → min()=x+1; drop MIN. */
    #pragma unroll 4
    for (int x = 2+t*2; x < w-1; x += nt*2)
        sm[x] += __half(BETA) * (sm[x-1] + sm[x+1]);
    __syncthreads();
    #pragma unroll 4
    for (int x = 1+t*2; x < w-1; x += nt*2)
        sm[x] += __half(GAMMA) * (sm[x-1] + sm[x+1]);
    /* V111: DCI w always even and >1 → simplify Gamma boundary. */
    if (t==0) sm[w-1] += __half(2.f*GAMMA) * sm[w-2];
    __syncthreads();
    /* V111: DCI w always >1 → min(1,w-1)=1; drop runtime min. */
    if (t==0) sm[0] += __half(2.f*DELTA) * sm[1];
    /* V102: same even-w invariant — drop MIN from Delta loop. */
    #pragma unroll 4
    for (int x = 2+t*2; x < w-1; x += nt*2)
        sm[x] += __half(DELTA) * (sm[x-1] + sm[x+1]);
    __syncthreads();

    /* V86: __hmul2 scatter — load sm[p*2]||sm[p*2+1] as __half2 (one ld.shared.b32) and
     * apply NORM_L/NORM_H simultaneously via one HMUL2 instruction (2× FP16 throughput).
     * sm[p*2]={L-coeff, H-coeff} are adjacent __half → natural __half2 pair at bank p (no conflict).
     * Saves 1 smem load instruction and 1 multiply instruction per scatter iteration. */
    {
        const __half2 norms = __halves2half2(__float2half(NORM_L), __float2half(NORM_H));
        /* V89: #pragma unroll 4 — ~4 iters for 4K; interleaves smem reads + global writes. */
        #pragma unroll 4
        for (int p = t; p < w / 2; p += nt) {
            __half2 lh = __hmul2(*reinterpret_cast<const __half2*>(&sm[p*2]), norms);
            d_hout[y0*stride + p]      = __low2half(lh);
            d_hout[y0*stride + hw + p] = __high2half(lh);
        }
    }
}


/* V128: kernel_rgb48_xyz_hdwt0_1ch_4row_p12 removed — dead code (never launched). */


/**
 * V18: GPU colour conversion kernel.
 * Converts RGB48LE → XYZ12 using precomputed LUTs and Bradford matrix.
 *
 * Each thread handles one pixel:
 *   1. Shift RGB16 right by 4 → 12-bit LUT index
 *   2. Apply input LUT (linearizes gamma): lut_in[idx] → linear float
 *   3. Apply 3x3 Bradford+RGB→XYZ matrix
 *   4. Clamp to [0, 1]
 *   5. Apply output LUT (DCP companding): → int32 XYZ value
 *
 * Uses __ldg for all read-only accesses (texture cache).
 */
__global__ void
kernel_rgb48_to_xyz12(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,   /* V55: was float */
    const uint16_t* __restrict__ d_lut_out,  /* V48: was int32_t; 8KB vs 16KB GPU texture */
    const float*    __restrict__ d_matrix,
    int32_t* __restrict__ d_out_x,
    int32_t* __restrict__ d_out_y,
    int32_t* __restrict__ d_out_z,
    int width, int height, int rgb_stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;

    int px = i % width;
    int py = i / width;
    int base = py * rgb_stride + px * 3;

    /* Load RGB48LE → 12-bit index (shift right 4) */
    int ri = min((__ldg(&d_rgb16[base + 0]) >> 4), 4095);
    int gi = min((__ldg(&d_rgb16[base + 1]) >> 4), 4095);
    int bi = min((__ldg(&d_rgb16[base + 2]) >> 4), 4095);

    /* Input LUT: linearize gamma (V55: __half→float) */
    float r = __half2float(__ldg(&d_lut_in[ri]));
    float g = __half2float(__ldg(&d_lut_in[gi]));
    float b = __half2float(__ldg(&d_lut_in[bi]));

    /* Bradford + RGB→XYZ matrix multiply (row-major) */
    float xv = __ldg(&d_matrix[0]) * r + __ldg(&d_matrix[1]) * g + __ldg(&d_matrix[2]) * b;
    float yv = __ldg(&d_matrix[3]) * r + __ldg(&d_matrix[4]) * g + __ldg(&d_matrix[5]) * b;
    float zv = __ldg(&d_matrix[6]) * r + __ldg(&d_matrix[7]) * g + __ldg(&d_matrix[8]) * b;

    /* Clamp to [0, 1] */
    xv = __saturatef(xv);
    yv = __saturatef(yv);
    zv = __saturatef(zv);

    /* Output LUT: DCP gamma companding → int32 */
    int xi = (int)(xv * 4095.5f);
    int yi = (int)(yv * 4095.5f);
    int zi = (int)(zv * 4095.5f);

    d_out_x[i] = __ldg(&d_lut_out[xi]);
    d_out_y[i] = __ldg(&d_lut_out[yi]);
    d_out_z[i] = __ldg(&d_lut_out[zi]);
}


/* ===== Encoder Implementation ===== */

struct CudaJ2KEncoderImpl
{
    /* DWT buffers */
    __half*  d_a[3]  = {nullptr};  /* V36: half: V-DWT output (was float; saves 2B/pixel) */
    __half*  d_b[3]  = {nullptr};  /* V26: half: H-DWT output (saves 50% DRAM vs float) */
    /* d_half[3] removed in V30: V29 tiled kernel uses registers, no fp16 workspace */
    /* Integer input per component (encode() path only; not used in encode_from_rgb48) */
    int32_t* d_in[3] = {nullptr};
    /* GPU-packed tier-1 output */
    uint8_t* d_packed = nullptr;
    /* One CUDA stream per component for parallel DWT */
    cudaStream_t stream[3] = {nullptr};

    /* V32: per-component CUDA Graphs for XYZ fallback path (DWT+quantize+D2H).
     * These graphs write to h_packed_pinned[0] (comp graph / non-pipeline path). */
    cudaGraphExec_t cg_exec[3]  = {nullptr, nullptr, nullptr};
    int    cg_width   = 0;
    int    cg_height  = 0;
    size_t cg_per_comp = 0;
    bool   cg_is_4k   = false;
    bool   cg_is_3d   = false;

    size_t buf_pixels = 0;

    /* V42: Dedicated H2D stream + per-buf H2D completion events.
     * H2D runs on st_h2d (PCIe DMA engine); compute runs on stream[0..2] (SM engine).
     * These hardware engines are independent → H2D for frame N+1 overlaps with
     * DWT compute for frame N, hiding DWT (0.5ms) under H2D (1.66ms).
     * Steady-state bottleneck = H2D alone = 1.66ms/frame → ~602fps. */
    cudaStream_t st_h2d = nullptr;
    cudaEvent_t h2d_done[2] = {nullptr, nullptr};

    /* V42: Per-buf per-channel comp graphs for RGB pipeline.
     * cg_v42[buf][c] captures skip_l0_hdwt=true path; D2H writes to h_packed_pinned[buf]. */
    cudaGraphExec_t cg_v42[2][3] = {{nullptr,nullptr,nullptr},{nullptr,nullptr,nullptr}};
    int    cg_v42_width[2]      = {0, 0};
    int    cg_v42_height[2]     = {0, 0};
    int    cg_v42_rgb_stride[2] = {0, 0};  /* V44: rgb_stride_pixels baked into graph */
    size_t cg_v42_per_comp[2]   = {0, 0};
    bool   cg_v42_is_4k[2]      = {false, false};
    bool   cg_v42_is_3d[2]      = {false, false};

    /* V42: 1-frame pipeline state */
    int    cur_buf        = 0;
    bool   pipeline_active = false;
    bool   graphs_failed   = false;  /* V125: true after graph capture failure → direct launches */
    int    p_width        = 0;
    int    p_height       = 0;
    size_t p_per_comp     = 0;
    bool   p_is_4k        = false;
    bool   p_is_3d        = false;

    void destroy_v42_graphs() {
        for (int i = 0; i < 2; ++i)
            for (int c = 0; c < 3; ++c)
                if (cg_v42[i][c]) { cudaGraphExecDestroy(cg_v42[i][c]); cg_v42[i][c] = nullptr; }
        pipeline_active = false;
    }

    /* V18: colour conversion device buffers */
    uint16_t* d_rgb16[2]   = {nullptr, nullptr};  /* V42: double-buffered GPU RGB48LE input */
    __half*   d_lut_in     = nullptr;  /* V55: 4096-entry input gamma LUT (was float; halves L1 cache 16KB→8KB) */
    float*    d_lut_in_f32 = nullptr;  /* V127: full-precision input LUT for XYZ conversion */
    uint16_t* d_lut_out    = nullptr;  /* V48: 4096-entry output gamma LUT (was int32_t; saves 8KB GPU texture cache) */
    float*    d_matrix     = nullptr;  /* 9-float Bradford+RGB→XYZ matrix */

    /* V127: GPU RGB→XYZ conversion buffers */
    int32_t*  d_xyz[3]        = {nullptr, nullptr, nullptr};  /* device XYZ planar output */
    int32_t*  h_xyz_pinned    = nullptr;  /* pinned host buffer for D2H (3 * pixels int32_t) */
    uint16_t* d_rgb16_xyz     = nullptr;  /* device RGB input for XYZ conversion */
    size_t    xyz_buf_pixels  = 0;
    size_t    rgb_buf_pixels = 0;
    bool      colour_loaded  = false;

    /* V41: double-buffered pinned D2H download buffers */
    uint8_t*  h_packed_pinned[2]  = {nullptr, nullptr};
    size_t    pinned_buf_pixels = 0;

    /* V41: double-buffered pinned H2D upload staging buffers */
    uint16_t* h_rgb16_pinned[2]   = {nullptr, nullptr};
    size_t    pinned_rgb_pixels = 0;

    /* V54: packed 12-bit planar RGB buffers (3 channels × (width/2*3) bytes/row × height).
     * Replaces h_rgb16_pinned/d_rgb16 for the V42 pipeline — 25% less PCIe H2D traffic. */
    uint8_t*  d_rgb12[2]          = {nullptr, nullptr};
    uint8_t*  h_rgb12_pinned[2]   = {nullptr, nullptr};
    size_t    rgb12_buf_pixels     = 0;
    size_t    pinned_rgb12_pixels  = 0;

    /* V127: EBCOT T1 buffers */
    CodeBlockInfo* d_cb_info     = nullptr;
    uint8_t*  d_ebcot_data[3]    = {nullptr, nullptr, nullptr};
    uint16_t* d_ebcot_len[3]     = {nullptr, nullptr, nullptr};
    uint8_t*  d_ebcot_npasses[3] = {nullptr, nullptr, nullptr};
    uint16_t* d_ebcot_passlens[3]= {nullptr, nullptr, nullptr};
    uint8_t*  d_ebcot_numbp[3]   = {nullptr, nullptr, nullptr};
    uint8_t*  h_ebcot_data[3]    = {nullptr, nullptr, nullptr};
    uint16_t* h_ebcot_len[3]     = {nullptr, nullptr, nullptr};
    uint8_t*  h_ebcot_npasses[3] = {nullptr, nullptr, nullptr};
    uint16_t* h_ebcot_passlens[3]= {nullptr, nullptr, nullptr};
    uint8_t*  h_ebcot_numbp[3]   = {nullptr, nullptr, nullptr};
    int       ebcot_num_cbs      = 0;
    std::vector<SubbandGeom> ebcot_subbands;
    std::vector<CodeBlockInfo> ebcot_cb_table;

    bool init() {
        for (int c = 0; c < 3; ++c) {
            if (cudaStreamCreate(&stream[c]) != cudaSuccess) return false;
        }
        if (cudaStreamCreate(&st_h2d) != cudaSuccess) return false;
        for (int i = 0; i < 2; ++i)
            if (cudaEventCreateWithFlags(&h2d_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        /* V59: prefer L1 cache over shared memory for memory-BW-bound kernels.
         * V-DWT tiled: no smem — larger L1 lets adjacent tiles share loaded rows.
         * RGB+HDWT0 1-row: smem=3.84KB → PreferL1 → 16KB smem/SM → 4 blk/SM = 100% occ. */
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_tiled,            cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_tiled_ho,         cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_tiled_ho_2col,    cudaFuncCachePreferL1);
        /* V85: reg-blocked V-DWT + quantize have no smem → PreferL1 is free (no occupancy cost).
         * Larger L1 improves column-data locality in reg-blocked kernel; quantize benefits from
         * larger instruction cache for multi-zone branch chains. */
        /* V125: template<bool EVEN_HEIGHT> — both instantiations need PreferL1. */
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_fp16_hi_reg_ho<true>,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_fp16_hi_reg_ho<false>, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_quantize_subband_ml,              cudaFuncCachePreferL1);
        /* V90: 2-row p12: smem = 7.68KB (DWT) + 8KB (sm_lut static) = 15.87KB.
         * PreferNone (32KB/SM): 2 blk/SM. PreferShared (48KB/SM): 3 blk/SM.
         * Use PreferShared → 3 blk/SM; sm_lut enables 0-cycle LUT reads → big speedup vs -1 block. */
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch_2row_p12, cudaFuncCachePreferShared);
        /* V90: 1-row p12: smem = 7.68KB (DWT) + 8KB (sm_lut static) = 15.87KB → same as 2-row.
         * PreferShared (48KB/SM) → 3 blk/SM; 0-cycle sm_lut access outweighs -1 block vs PreferNone. */
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch_1row_p12, cudaFuncCachePreferShared);
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch,          cudaFuncCachePreferL1);
        /* V74: level-0 4-row kernel: smem=4×w×2B. 2K=15.36KB, 4K=30.72KB.
         * No LUT access → L1 caching LUTs not needed. Use PreferShared: 48KB smem → 3 blk/SM.
         * PreferL1 would limit to 16KB smem → only 1 blk/SM at 2K (25% occ) — avoid. */
        cudaFuncSetCacheConfig(kernel_fused_i2f_horz_dwt_half_out_4row, cudaFuncCachePreferShared);
        /* V77: 4-row H-DWT: level-1 smem=7.68KB; PreferL1→16KB smem/SM→2 blk/SM (50% occ).
         * PreferNone→32KB smem/SM→4 blk/SM=100% occ at level 1 (largest, bottleneck level).
         * Levels 2-5 (smem≤3.84KB): 4 blk/SM with both → PreferNone has no occupancy cost.
         * 2-row/1-row kernels: smem≤3.84KB at all levels → 4 blk/SM with PreferL1 → keep. */
        /* V127: template<bool DIV4> — both instantiations need PreferNone. */
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io_4row<true>,  cudaFuncCachePreferNone);
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io_4row<false>, cudaFuncCachePreferNone);
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io_2row,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io,       cudaFuncCachePreferL1);
        return true;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;

        cleanup_dwt_buffers();

        /* V126: generous padding — V-DWT tiled kernel reads up to V_TILE_FL rows past
         * the last valid tile, and int __ldg reads 2 extra bytes at row ends.
         * Padding = stride * V_OVERLAP * sizeof(__half) + 64 to cover worst-case overshoot. */
        size_t pad = static_cast<size_t>(width) * 8 * sizeof(__half) + 64;
        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a[c],  pixels * sizeof(__half) + pad);
            cudaMalloc(&d_b[c],  pixels * sizeof(__half) + pad);
            cudaMalloc(&d_in[c], pixels * sizeof(int32_t)); /* used by non-RGB encode() path */
            /* V30: d_half[c] workspace removed — V29 tiled kernel no longer uses it */
        }
        cudaMalloc(&d_packed, pixels * 3 * sizeof(uint8_t));
        buf_pixels = pixels;
        ensure_pinned_buffer(width, height);
    }

    void ensure_rgb_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= rgb_buf_pixels) return;
        for (int i = 0; i < 2; ++i) {
            if (d_rgb16[i]) { cudaFree(d_rgb16[i]); d_rgb16[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i)
            cudaMalloc(&d_rgb16[i], pixels * 3 * sizeof(uint16_t));
        rgb_buf_pixels = pixels;
        ensure_pinned_rgb_buffer(width, height);
        /* V54: allocate packed 12-bit GPU buffers */
        ensure_rgb12_buffer(width, height);
    }

    /* V54: allocate/reallocate packed 12-bit planar GPU buffers.
     * Size = 3 channels × (width/2 * 3 bytes) × height = width * height * 9/2 bytes. */
    void ensure_rgb12_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= rgb12_buf_pixels) return;
        /* packed_row_stride = (width/2) * 3; total = 3 channels × height × packed_row_stride */
        size_t packed_size = static_cast<size_t>((width / 2) * 3) * height * 3;
        for (int i = 0; i < 2; ++i) {
            if (d_rgb12[i]) { cudaFree(d_rgb12[i]); d_rgb12[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i)
            cudaMalloc(&d_rgb12[i], packed_size);
        rgb12_buf_pixels = pixels;
        ensure_pinned_rgb12_buffer(width, height);
    }

    void ensure_pinned_rgb12_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= pinned_rgb12_pixels) return;
        size_t packed_size = static_cast<size_t>((width / 2) * 3) * height * 3;
        for (int i = 0; i < 2; ++i) {
            if (h_rgb12_pinned[i]) { cudaFreeHost(h_rgb12_pinned[i]); h_rgb12_pinned[i] = nullptr; }
        }
        destroy_v42_graphs();  /* graphs bake old pointer; force rebuild */
        /* V80: write-combining — CPU only writes (packs) into this buffer, GPU reads via DMA.
         * Eliminates PCIe snooping overhead → up to 40% H2D bandwidth gain. */
        for (int i = 0; i < 2; ++i)
            cudaHostAlloc(&h_rgb12_pinned[i], packed_size, cudaHostAllocWriteCombined);
        pinned_rgb12_pixels = pixels;
    }

    void ensure_pinned_rgb_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= pinned_rgb_pixels) return;
        for (int i = 0; i < 2; ++i) {
            if (h_rgb16_pinned[i]) { cudaFreeHost(h_rgb16_pinned[i]); h_rgb16_pinned[i] = nullptr; }
        }
        destroy_v42_graphs();  /* graphs bake old pointer; force rebuild */
        for (int i = 0; i < 2; ++i)
            cudaMallocHost(&h_rgb16_pinned[i], pixels * 3 * sizeof(uint16_t));
        pinned_rgb_pixels = pixels;
    }

    void ensure_pinned_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= pinned_buf_pixels) return;
        for (int i = 0; i < 2; ++i) {
            if (h_packed_pinned[i]) { cudaFreeHost(h_packed_pinned[i]); h_packed_pinned[i] = nullptr; }
        }
        for (int i = 0; i < 2; ++i)
            cudaMallocHost(&h_packed_pinned[i], pixels * 3 * sizeof(uint8_t));
        pinned_buf_pixels = pixels;
    }

    void upload_colour_params(GpuColourParams const& p) {
        if (!d_lut_in)     cudaMalloc(&d_lut_in,     4096 * sizeof(__half));
        if (!d_lut_in_f32) cudaMalloc(&d_lut_in_f32, 4096 * sizeof(float));  /* V127: full-precision for XYZ */
        if (!d_lut_out)    cudaMalloc(&d_lut_out,    4096 * sizeof(uint16_t));
        if (!d_matrix)     cudaMalloc(&d_matrix,     9    * sizeof(float));

        /* V55: convert float→__half before upload; host array stays float for precision */
        __half h_lut_in_tmp[4096];
        for (int i = 0; i < 4096; ++i) h_lut_in_tmp[i] = __float2half(p.lut_in[i]);
        cudaMemcpy(d_lut_in,     h_lut_in_tmp, 4096 * sizeof(__half),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_in_f32, p.lut_in,     4096 * sizeof(float),     cudaMemcpyHostToDevice);  /* V127 */
        cudaMemcpy(d_lut_out,    p.lut_out,    4096 * sizeof(uint16_t),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix,     p.matrix,     9    * sizeof(float),     cudaMemcpyHostToDevice);
        colour_loaded = true;
    }

    void cleanup_dwt_buffers() {
        for (int c = 0; c < 3; ++c) {
            if (d_a[c])  { cudaFree(d_a[c]);  d_a[c]  = nullptr; }
            if (d_b[c])  { cudaFree(d_b[c]);  d_b[c]  = nullptr; }
            if (d_in[c]) { cudaFree(d_in[c]); d_in[c] = nullptr; }
            /* d_half[c] was removed in V30 — was fp16 V-DWT workspace, now unused */
        }
        if (d_packed) { cudaFree(d_packed); d_packed = nullptr; }
        buf_pixels = 0;
    }

    void destroy_comp_graphs() {
        for (int c = 0; c < 3; ++c) {
            if (cg_exec[c]) { cudaGraphExecDestroy(cg_exec[c]); cg_exec[c] = nullptr; }
        }
        cg_width = cg_height = 0; cg_per_comp = 0;
    }

    ~CudaJ2KEncoderImpl() {
        destroy_v42_graphs();
        destroy_comp_graphs();
        cleanup_dwt_buffers();
        for (int i = 0; i < 2; ++i) {
            if (d_rgb16[i])         { cudaFree(d_rgb16[i]);             d_rgb16[i]         = nullptr; }
            if (d_rgb12[i])         { cudaFree(d_rgb12[i]);             d_rgb12[i]         = nullptr; }
            if (h_packed_pinned[i]) { cudaFreeHost(h_packed_pinned[i]); h_packed_pinned[i] = nullptr; }
            if (h_rgb16_pinned[i])  { cudaFreeHost(h_rgb16_pinned[i]);  h_rgb16_pinned[i]  = nullptr; }
            if (h_rgb12_pinned[i])  { cudaFreeHost(h_rgb12_pinned[i]);  h_rgb12_pinned[i]  = nullptr; }
            if (h2d_done[i])        { cudaEventDestroy(h2d_done[i]);    h2d_done[i]        = nullptr; }
        }
        if (d_lut_in)  { cudaFree(d_lut_in);  d_lut_in  = nullptr; }
        if (d_lut_out) { cudaFree(d_lut_out); d_lut_out = nullptr; }
        if (d_matrix)  { cudaFree(d_matrix);  d_matrix  = nullptr; }
        for (int c = 0; c < 3; ++c)
            if (stream[c]) cudaStreamDestroy(stream[c]);
        if (st_h2d) cudaStreamDestroy(st_h2d);
    }
};


/* ===== J2K Codestream Writer ===== */
class J2KCodestreamWriter
{
public:
    void reserve(size_t n)   { _data.reserve(n); }
    void write_u8(uint8_t v)  { _data.push_back(v); }
    void write_u16(uint16_t v) {
        _data.push_back(static_cast<uint8_t>(v >> 8));
        _data.push_back(static_cast<uint8_t>(v & 0xFF));
    }
    void write_u32(uint32_t v) {
        write_u16(static_cast<uint16_t>(v >> 16));
        write_u16(static_cast<uint16_t>(v & 0xFFFF));
    }
    void write_marker(uint16_t m) { write_u16(m); }
    void write_bytes(const uint8_t* d, size_t n) { _data.insert(_data.end(), d, d + n); }

    /** Write tier-1 (entropy-coded) data with J2K byte stuffing.
     *  Inserts 0x00 after every 0xFF to prevent false marker detection. */
    void write_bytes_stuffed(const uint8_t* d, size_t n) {
        _data.reserve(_data.size() + n + 32);
        for (size_t i = 0; i < n; ++i) {
            _data.push_back(d[i]);
            if (d[i] == 0xFF) _data.push_back(0x00);
        }
    }
    size_t position() const { return _data.size(); }
    void patch_u32(size_t offset, uint32_t v) {
        _data[offset+0] = static_cast<uint8_t>(v >> 24);
        _data[offset+1] = static_cast<uint8_t>((v >> 16) & 0xFF);
        _data[offset+2] = static_cast<uint8_t>((v >> 8)  & 0xFF);
        _data[offset+3] = static_cast<uint8_t>(v         & 0xFF);
    }
    std::vector<uint8_t>& data() { return _data; }
private:
    std::vector<uint8_t> _data;
};


/* ===== Public API ===== */

CudaJ2KEncoder::CudaJ2KEncoder()
    : _impl(std::make_unique<CudaJ2KEncoderImpl>())
{
    _initialized = _impl->init();
}

CudaJ2KEncoder::~CudaJ2KEncoder() = default;


/**
 * V27: Perform one DWT level on one component.
 * Fixed buffer roles (no pointer-swapping):
 *   d_float     — float buffer: V-DWT output (always result of current level)
 *   d_half_h    — half buffer: H-DWT output (saves 50% H-DWT write BW vs float)
 *   d_half_work — half workspace for V-DWT lifting (large subbands only)
 * After the call, the level result is in d_float.
 *
 * V27 vs V26: for height ≤ MAX_REG_HEIGHT (140), V-DWT uses register-blocking
 * (kernel_fused_vert_dwt_fp16_hi_reg) which eliminates all d_half_work global
 * memory accesses for small subbands (DWT levels 3-4 for 2K content).
 */
static void
gpu_dwt97_level(
    __half* d_half_a,       /* V36: d_a[c]: half (V-DWT output, was float) */
    __half* d_half_h,       /* d_b[c]: half (H-DWT output) */
    __half* d_half_work,    /* legacy: unused since V30 */
    const int32_t* d_input, /* int32 input (level 0 only, ignored if skip_hdwt) */
    int width, int height, int stride,
    int level, cudaStream_t st,
    bool skip_hdwt = false) /* V28: true when H-DWT already done (fused kernel path) */
{
    /* V33: 512 threads for H-DWT gives 100% SM occupancy (vs 75% at 256T for 8KB smem/block). */
    constexpr int H_THREADS = 512;
    /* V63: 256T for tiled V-DWT (h>MAX_REG_HEIGHT): 8 warps/block vs 4 at 128T; better latency
     *       hiding; fewer blocks (544 vs 1020 for 2K level 0); same 87.5% occupancy.
     *       128T kept for reg-blocked (h≤140): 140-float col[] → ~150 regs/T → 256T would give
     *       256×150=38400 per block → 1 blk/SM = 12.5% occ → regression for levels 3-4. */
    constexpr int V_THREADS_TILED = 256;
    constexpr int V_THREADS_REG   = 128;

    size_t smem = static_cast<size_t>(width) * sizeof(__half);  /* V38: half smem */
    int grid_v  = (width + V_THREADS_REG - 1) / V_THREADS_REG;

    /* Step 1: Horizontal DWT — writes __half to d_half_h.
     * Level 0 non-skip: reads int32 d_input. Levels 1+: reads half d_half_a (V36). */
    if (!skip_hdwt) {
        if (level == 0) {
            /* V74: 4-row level-0 kernel — 75% fewer block dispatches (1080→270 for 2K).
             * 4 rows' i2f chains in-flight; __syncthreads amortized over 4× more work. */
            kernel_fused_i2f_horz_dwt_half_out_4row<<<(height+3)/4, H_THREADS, 4*smem, st>>>(
                d_input, d_half_h, width, height, stride);
        } else {
            /* V47: 4-rows-per-block for levels 1+; grid quartered vs V45, halved vs V46.
             * V73: adaptive thread count — fixed 512T wastes 50-77% threads at small widths.
             *   level-3 2K (w=240): 512T→256T (93.75%); level-4 2K (w=120): 512T→128T (93.75%).
             *   level-3 4K (w=480): 512T→256T; level-4 4K (w=240): 256T; level-5 4K (w=120): 128T. */
            int h_blk = (width > 480) ? H_THREADS :
                        (width > 240) ? 256 :
                        (width > 120) ? 128 : 64;
            /* V127: dispatch DIV4 template at graph capture time — h%4==0 → else block dead. */
            if (height % 4 == 0)
                kernel_fused_horz_dwt_half_io_4row<true><<<(height+3)/4, h_blk, 4*smem, st>>>(
                    d_half_a, d_half_h, width, height, stride);
            else
                kernel_fused_horz_dwt_half_io_4row<false><<<(height+3)/4, h_blk, 4*smem, st>>>(
                    d_half_a, d_half_h, width, height, stride);
        }
    }

    /* Step 2+3: V-DWT reads __half d_half_h, writes __half d_half_a (V36).
     * V27: h ≤ MAX_REG_HEIGHT → register-blocked (128T); V62: h > → tiled (256T, V63).
     * V76: tiled path uses 2-col kernel — HFMA2 doubles lifting throughput. */
    if (height <= MAX_REG_HEIGHT) {
        /* V125: dispatch even/odd height template instantiation at CUDA graph capture time. */
        if (height % 2 == 0)
            kernel_fused_vert_dwt_fp16_hi_reg_ho<true><<<grid_v, V_THREADS_REG, 0, st>>>(
                d_half_h, d_half_a, width, height, stride);
        else
            kernel_fused_vert_dwt_fp16_hi_reg_ho<false><<<grid_v, V_THREADS_REG, 0, st>>>(
                d_half_h, d_half_a, width, height, stride);
    } else {
        /* V76: grid x = ceil(width/2/V_THREADS_TILED): half x-blocks, each thread does 2 cols. */
        dim3 v_grid2d((width/2 + V_THREADS_TILED - 1) / V_THREADS_TILED,
                      (height + V_TILE - 1) / V_TILE);
        kernel_fused_vert_dwt_tiled_ho_2col<<<v_grid2d, V_THREADS_TILED, 0, st>>>(
            d_half_h, d_half_a, width, height, stride);
    }
    (void)d_half_work;  /* V30+: no longer used */
}


/**
 * V32: Launches all GPU kernels for one component: DWT0-4 + quantize + D2H.
 * V41: h_dest selects which double-buffer slot receives the D2H output.
 */
static void
launch_comp_pipeline(
    CudaJ2KEncoderImpl* impl, int c, int width, int height,
    size_t per_comp, float step, bool skip_level0_hdwt, cudaStream_t st,
    uint8_t* h_dest,  /* V41: D2H destination (h_packed_pinned[0] or [1]) */
    int num_levels = NUM_DWT_LEVELS)  /* V49: 5 for 2K, 6 for 4K */
{
    int stride = width;    /* original row pitch — unchanged throughout DWT */
    int orig_height = height;
    int w = width, h = height;
    for (int level = 0; level < num_levels; ++level) {
        bool skip_hdwt = skip_level0_hdwt && (level == 0);
        gpu_dwt97_level(impl->d_a[c], impl->d_b[c], nullptr,
                        impl->d_in[c], w, h, stride, level, st, skip_hdwt);
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    /* V53: Multi-level perceptual quantization — 6-band step weighting (DC, L5, L4, L3, L2, L1).
     * Distinguishes all 5 DWT levels: finer step for low-freq, coarser for high-freq subbands.
     * ll5_h = ceil(orig_height/32),  ll5_c = stride >> 5 (stride/32). */
    int n_rows   = std::min(static_cast<int>((per_comp + stride - 1) / stride), orig_height);
    int ll5_h    = (orig_height + 31) / 32;
    int ll5_c    = stride >> 5;
    kernel_quantize_subband_ml<<<n_rows, 256, 0, st>>>(
        impl->d_a[c], impl->d_packed + c * per_comp,
        stride, n_rows, step, ll5_h, ll5_c);
    cudaMemcpyAsync(h_dest + c * per_comp,
                    impl->d_packed + c * per_comp,
                    per_comp, cudaMemcpyDeviceToHost, st);  /* V41: dest buf */
}


/**
 * V32: Capture per-component CUDA Graphs for the DWT+quantize+D2H pipeline.
 * Each of the 3 component graphs captures: DWT levels 0-4 + quantize + D2H.
 * Graphs are run in parallel on separate streams (no cross-component dependency).
 * Graphs are valid as long as width/height/per_comp/is_4k don't change.
 */
/**
 * V125: Adaptive base quantization step — scales with target compression ratio.
 *
 * With 1-byte sign-magnitude packing (max magnitude 126), the step must balance:
 *   - Low step → more non-zero coefficients → larger output → higher quality
 *   - High step → more zeros → smaller output → lower quality (and DC clipping)
 *
 * Formula: base_step = clamp(compression_ratio × 0.25, 1.0, 32.5)
 *   150 Mbps / 24 fps (2K): ratio=8.3 → step=2.1
 *   250 Mbps / 24 fps (2K): ratio=5.0 → step=1.25
 *   100 Mbps / 24 fps (2K): ratio=12.4 → step=3.1
 *
 * Replaces the fixed step of 32.5 (2K) / 16.25 (4K) which over-quantized,
 * producing ~5KB/comp instead of the target ~260KB/comp at 150 Mbps.
 */
static float
compute_base_step(int width, int height, size_t per_comp)
{
    size_t pixels = static_cast<size_t>(width) * height;
    float ratio = static_cast<float>(pixels) / std::max(per_comp, static_cast<size_t>(1));
    /* V186: drop step by 4× — was clamp(ratio*0.25, 1.0, 32.5) → 2.13 at 150Mbps/2K.
     * After V185 fix our T1 step is internally ×2 (HL/LH) or ×4 (HH), so the same
     * "abstract" base_step gives much coarser HH quantization than before.  Lower
     * base_step means finer quantization and more bit-planes; PSNR rises substantially
     * on hard patterns (bars, checker) until num_bp brushes against MAX_BP. */
    /* V186: 0.06 multiplier (was 0.25 pre-V185) gives ~4× finer base quantization.
     * Combined with V185 step×2/×4 compensation, final T1 step matches OPJ encoder
     * roughly.  Tried 0.04 and 0.025 — 0.04 cost 27 dB on fast flat_30000 (lossless
     * lost when step gets near DWT FP16 noise); 0.025 hurts checker_64 worse.
     * V191: tried 0.04 with MAX_BP=16 (correct) and MAX_BP=14 (fast) — correct mode
     * gains ~1 dB but fast mode loses lossless on flat_30000. */
    return std::clamp(ratio * 0.06f, 0.25f, 32.5f);
}

static void
rebuild_comp_graphs(
    CudaJ2KEncoderImpl* impl,
    int width, int height, size_t per_comp, bool is_4k, bool is_3d, bool skip_level0_hdwt)
{
    impl->destroy_comp_graphs();
    float base_step = compute_base_step(width, height, per_comp);
    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;  /* V49: 6-level DWT for 4K */
    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.1f);
        cudaGraph_t g;
        cudaStreamBeginCapture(impl->stream[c], cudaStreamCaptureModeThreadLocal);
        launch_comp_pipeline(impl, c, width, height, per_comp, step, skip_level0_hdwt,
                             impl->stream[c], impl->h_packed_pinned[0],  /* V41: comp graphs use buf 0 */
                             num_levels);  /* V49 */
        cudaStreamEndCapture(impl->stream[c], &g);
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
 * V44: Build per-buf per-channel full-pipeline graphs.
 * cg_v42[buf][c] captures:
 *   1. cudaStreamWaitEvent(h2d_done[buf])   — wait for H2D of d_rgb16[buf]
 *   2. kernel_rgb48_xyz_hdwt0_1ch           — RGB→XYZ + H-DWT level 0 → d_b[c]
 *   3. DWT levels 1-4 + quantize + D2H     — → h_packed_pinned[buf]
 * Per-frame launch: 3× cudaGraphLaunch (was 3× waitEvent + 3× RGB + 3× graph = 9 calls).
 * d_a[c]/d_b[c] intermediates shared between buf=0 and buf=1 (safe: sequential).
 * Graphs valid as long as width/height/rgb_stride/per_comp/is_4k/is_3d unchanged.
 */
static void
rebuild_v42_comp_graphs(
    CudaJ2KEncoderImpl* impl,
    int buf,
    int width, int height, int rgb_stride_pixels,
    size_t per_comp, bool is_4k, bool is_3d)
{
    for (int c = 0; c < 3; ++c) {
        if (impl->cg_v42[buf][c]) {
            cudaGraphExecDestroy(impl->cg_v42[buf][c]);
            impl->cg_v42[buf][c] = nullptr;
        }
    }
    float base_step = compute_base_step(width, height, per_comp);  /* V125 */
    fprintf(stderr, "[V125] rebuild_v42: w=%d h=%d per_comp=%zu base_step=%.3f\n",
            width, height, per_comp, base_step);
    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;  /* V49: 6-level DWT for 4K */
    /* V61: 4-row kernel: smem=4×width halves; grid=(height+3)/4 blocks.
     * V72: Conditionally use 2-row at 4K (width>2048) to enable PreferL1 (48KB L1).
     *   4K 4-row smem=30.72KB > 16KB PreferL1 → runtime PreferShared (16KB L1, LUT evictions).
     *   4K 2-row smem=15.36KB < 16KB → PreferL1 honored (48KB L1, all LUTs cached). */
    size_t ch_smem_4row = static_cast<size_t>(4 * width) * sizeof(__half);
    size_t ch_smem_2row = static_cast<size_t>(2 * width) * sizeof(__half);
    size_t ch_smem_1row = static_cast<size_t>(1 * width) * sizeof(__half);
    int rgb_grid = (height + 3) / 4;
    int rgb_grid_2row = (height + 1) / 2;
    int rgb_grid_1row = height;
    /* V78: Use 2-row for 2K (100% occ: smem=7.68KB, PreferNone→4 blk/SM).
     * V79: Use 1-row for 4K (100% occ: smem=7.68KB@4K, PreferNone→4 blk/SM).
     * 4K 2-row smem=15.36KB → 2 blk/SM=50% occ; 1-row smem=7.68KB → 4 blk/SM=100% occ. */
    bool use_1row_4k = (width > 2048);  /* V79: 4K uses 1-row for 100% occ */
    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.1f);
        cudaGraph_t g;
        cudaStreamBeginCapture(impl->stream[c], cudaStreamCaptureModeThreadLocal);
        /* V44: event wait baked into graph — waits for h2d_done[buf] before RGB kernel */
        cudaStreamWaitEvent(impl->stream[c], impl->h2d_done[buf], 0);
        int packed_row_stride = (width / 2) * 3;  /* bytes per channel per row */
        if (use_1row_4k) {
            /* V79: 4K — 1-row: smem=7.68KB, PreferNone→4 blk/SM=100% occ (vs 2-row's 50%). */
            kernel_rgb48_xyz_hdwt0_1ch_1row_p12<<<rgb_grid_1row, H_THREADS_FUSED, ch_smem_1row, impl->stream[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_b[c], c,
                width, height, packed_row_stride, width);
        } else {
            /* V78: 2K — 2-row: smem=7.68KB, PreferNone→4 blk/SM=100% occ. */
            kernel_rgb48_xyz_hdwt0_1ch_2row_p12<<<rgb_grid_2row, H_THREADS_FUSED, ch_smem_2row, impl->stream[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_b[c], c,
                width, height, packed_row_stride, width);
        }
        /* DWT levels 1+ + quantize + D2H: reads d_b[c], writes h_packed_pinned[buf] */
        launch_comp_pipeline(impl, c, width, height, per_comp, step, true,
                             impl->stream[c], impl->h_packed_pinned[buf],
                             num_levels);  /* V49 */
        cudaError_t cap_err = cudaStreamEndCapture(impl->stream[c], &g);
        if (cap_err != cudaSuccess) {
            impl->cg_v42[buf][c] = nullptr;
            continue;
        }
        cudaError_t inst_err = cudaGraphInstantiate(&impl->cg_v42[buf][c], g, nullptr, nullptr, 0);
        if (inst_err != cudaSuccess)
            impl->cg_v42[buf][c] = nullptr;
        cudaGraphDestroy(g);
    }
    impl->cg_v42_width[buf]      = width;
    impl->cg_v42_height[buf]     = height;
    impl->cg_v42_rgb_stride[buf] = rgb_stride_pixels;
    impl->cg_v42_per_comp[buf]   = per_comp;
    impl->cg_v42_is_4k[buf]      = is_4k;
    impl->cg_v42_is_3d[buf]      = is_3d;
}


/**
 * V57: Encode a scalar J2K quantization step into a QCD/QCC 16-bit entry.
 * Format (ISO 15444-1 §A.6.4): bits [15:11] = eps, bits [10:0] = man.
 * Decoder reconstructs: step = 2^(R_b - eps) × (1 + man/2048), R_b = 13 (12-bit XYZ + 1 guard bit).
 * eps  = R_b - floor(log2(step)) ensures 1 ≤ step/2^(R_b-eps) < 2.
 */
static uint16_t
j2k_qcd_step_entry(float step)
{
    int log2s = static_cast<int>(std::floor(std::log2(step)));
    int eps   = 13 - log2s;  /* R_b=13; ensures denominator = 2^floor(log2(step)) */
    float denom = std::ldexp(1.0f, 13 - eps);
    int man = static_cast<int>((step / denom - 1.0f) * 2048.0f);
    man = std::max(0, std::min(2047, man));
    return static_cast<uint16_t>((eps << 11) | man);
}

/**
 * V57: Return V53-weight-adjusted QCD/QCC entry for a given subband index and base step.
 * For 2K (5-level DWT): subband order is LL5 (0), L5-AC (1-3), L4 (4-6),
 *   L3 (7-9), L2 (10-12), L1 (13-15).
 * For 4K (6-level DWT): uses uniform base_step (4K ll5_h = height/32, not height/64;
 *   the 4K perceptual mapping is left as-is until a full 4K ll6_h fix is implemented).
 */
static uint16_t
j2k_perceptual_sb_entry(float base_step, int sb_idx, bool is_4k)
{
    if (is_4k) return j2k_qcd_step_entry(base_step);  /* 4K: uniform */
    /* 2K per-subband V53 perceptual weights */
    static const float kWeights[16] = {
        0.65f,                      /* sb 0:   LL5 (DC) */
        0.85f, 0.85f, 0.85f,        /* sb 1-3: L5-AC */
        0.95f, 0.95f, 0.95f,        /* sb 4-6: L4 */
        1.05f, 1.05f, 1.05f,        /* sb 7-9: L3 */
        1.12f, 1.12f, 1.12f,        /* sb 10-12: L2 */
        1.20f, 1.20f, 1.20f         /* sb 13-15: L1 (finest detail) */
    };
    float w = (sb_idx < 16) ? kWeights[sb_idx] : 1.0f;
    return j2k_qcd_step_entry(base_step * w);
}

/**
 * V51: Find the actual number of meaningful bytes in a packed component buffer.
 * Scans backward in 8-byte words to skip trailing zeros quickly.
 * Returns at least 1 (even for an all-zero component) to keep valid tile parts.
 */
static size_t
find_actual_per_comp(const uint8_t* data, size_t n)
{
    /* Scan backward in 8-byte chunks while all-zero, then byte-by-byte. */
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


/**
 * V37: Build J2K codestream from quantized tier-1 data already downloaded.
 * V41: h_src selects which double-buffer slot holds the completed D2H data.
 */
static std::vector<uint8_t>
build_j2k_codestream(
    CudaJ2KEncoderImpl* impl,
    int width, int height,
    size_t per_comp, bool is_4k, bool is_3d,
    uint8_t* h_src)  /* V41: source buffer (h_packed_pinned[0] or [1]) */
{
    (void)is_3d;  /* Currently unused in codestream format; kept for future use */
    J2KCodestreamWriter cs;
    cs.reserve(std::max(static_cast<size_t>(16384),
                        static_cast<size_t>(300) + 3 * (14 + per_comp)));

    cs.write_marker(J2K_SOC);

    /* SIZ */
    {
        cs.write_marker(J2K_SIZ);
        cs.write_u16(2 + 2 + 32 + 2 + 3 * 3);
        cs.write_u16(is_4k ? 0x0004 : 0x0003);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u16(3);
        for (int c = 0; c < 3; ++c) { cs.write_u8(11); cs.write_u8(1); cs.write_u8(1); }
    }

    /* COD */
    {
        int num_precincts = is_4k ? 7 : (NUM_DWT_LEVELS + 1);
        cs.write_marker(J2K_COD);
        cs.write_u16(2 + 1 + 4 + 5 + num_precincts);
        cs.write_u8(0x01); cs.write_u8(0x04); cs.write_u16(1); cs.write_u8(1);
        cs.write_u8(is_4k ? 6 : NUM_DWT_LEVELS);
        cs.write_u8(3); cs.write_u8(3); cs.write_u8(0x01); cs.write_u8(0x00); /* SPcod: BYPASS bit=1 */
        cs.write_u8(0x77);
        for (int i = 1; i < num_precincts; ++i) cs.write_u8(0x88);
    }

    /* QCD — V57/V125: per-subband step matching V53 perceptual weights.
     * V125: base_step is now adaptive (compute_base_step), not hardcoded.
     * V49: nsb = 19 for 4K, 16 for 2K. */
    float base_y = compute_base_step(width, height, per_comp);
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;
        cs.write_marker(J2K_QCD);
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        cs.write_u8(0x22);
        for (int i = 0; i < nsb; ++i)
            cs.write_u16(j2k_perceptual_sb_entry(base_y, i, is_4k));
    }

    /* QCC for X (comp 0) and Z (comp 2) — V57: per-subband step (X/Z base = base_y×1.1). */
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;
        uint16_t lqcc = static_cast<uint16_t>(4 + 2 * nsb);
        float base_xz = base_y * 1.1f;
        for (int c : {0, 2}) {
            cs.write_marker(J2K_QCC);
            cs.write_u16(lqcc);
            cs.write_u8(static_cast<uint8_t>(c));
            cs.write_u8(0x22);
            for (int i = 0; i < nsb; ++i)
                cs.write_u16(j2k_perceptual_sb_entry(base_xz, i, is_4k));
        }
    }

    /* V51: Compute actual per-component sizes (trim trailing zeros). */
    size_t actual[3];
    for (int c = 0; c < 3; ++c)
        actual[c] = find_actual_per_comp(h_src + c * per_comp, per_comp);
    /* TLM */
    {
        cs.write_marker(J2K_TLM);
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 1 + 3 * 4));
        cs.write_u8(0); cs.write_u8(0x40);
        for (int c = 0; c < 3; ++c)
            cs.write_u32(static_cast<uint32_t>(14 + actual[c]));  /* V51: actual trimmed size */
    }

    /* SOT + SOD × 3 */
    for (int c = 0; c < 3; ++c) {
        cs.write_marker(J2K_SOT);
        cs.write_u16(10);
        cs.write_u16(0);
        size_t psot_pos = cs.position();
        cs.write_u32(0);
        cs.write_u8(static_cast<uint8_t>(c));
        cs.write_u8(3);
        cs.write_marker(J2K_SOD);
        cs.write_bytes(h_src + c * per_comp, actual[c]);  /* V51: write only non-zero bytes */
        cs.patch_u32(psot_pos, static_cast<uint32_t>(cs.position() - psot_pos + 4));
    }
    while (cs.data().size() < 16384) cs.write_u8(0);
    cs.write_marker(J2K_EOC);
    return std::move(cs.data());
}


/**
 * Internal: run DWT on d_in[0..2] and build J2K codestream.
 * Called by both encode() and encode_from_rgb48() after d_in is populated.
 */
static std::vector<uint8_t>
run_dwt_and_build_codestream(
    CudaJ2KEncoderImpl* impl,
    int width, int height,
    int64_t bit_rate, int fps,
    bool is_3d, bool is_4k,
    bool skip_level0_hdwt = false)  /* V28: true when fused RGB→H-DWT kernel was used */
{
    int stride = width;
    size_t pixels = static_cast<size_t>(width) * height;

    /* V22: Correct per_comp formula — divide target_bytes equally across 3 components.
     *
     * V20 bug: used (pixels * ratio / 3) which double-counted the /3 factor,
     * yielding per_comp = target_bytes/9.  Result: frames encoded at only 33%
     * of the target bitrate (e.g. 50 Mbps instead of 150 Mbps).
     *
     * V22 fix: per_comp = target_bytes / 3, so 3×per_comp ≈ target_bytes.
     * At 150 Mbps / 24 fps: per_comp ≈ 260 KB (was 87 KB with the bug).
     * Frame size increases 3×, but still <<< 1.76 MB per component (full DWT),
     * so the savings over V19 are still ~6.8×.
     *
     * The minimum frame size of 16384 bytes from DCI Bv2.1 is enforced
     * by the padding loop before EOC. */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t target_bytes = static_cast<size_t>(frame_bits / 8);
    size_t per_comp = std::min(
        std::max(target_bytes / 3,
                 static_cast<size_t>(1)),
        pixels);
    size_t actual_sz[3] = {per_comp, per_comp, per_comp};  /* V51: trimmed per-component sizes */

    /* V32: CUDA Graph dispatch for DWT+quantize+D2H pipeline.
     *
     * All 3 per-component pipelines (DWT → quantize → D2H) are captured as
     * independent CUDA Graphs on first call or whenever frame geometry changes.
     * cudaGraphLaunch replaces ~30 individual kernel launches per frame, saving
     * ~0.3ms of SM scheduling overhead at 24 fps.
     *
     * Graphs are invalidated and rebuilt if width/height/per_comp/is_4k changes
     * (e.g., first 4K frame after 2K). The captured graphs include the D2H
     * cudaMemcpyAsync, so no separate D2H loop is needed below. */
    impl->ensure_pinned_buffer(width, height);
    {
        bool graphs_valid = (impl->cg_exec[0] && impl->cg_exec[1] && impl->cg_exec[2] &&
                             impl->cg_width  == width  && impl->cg_height  == height  &&
                             impl->cg_per_comp == per_comp && impl->cg_is_4k == is_4k &&
                             impl->cg_is_3d == is_3d);
        if (!graphs_valid)
            rebuild_comp_graphs(impl, width, height, per_comp, is_4k, is_3d, skip_level0_hdwt);
        for (int c = 0; c < 3; ++c)
            cudaGraphLaunch(impl->cg_exec[c], impl->stream[c]);
    }
    for (int c = 0; c < 3; ++c)
        cudaStreamSynchronize(impl->stream[c]);

    /* V19: magnitude capped at 126, so 0xFF never appears in packed data.
     * No byte stuffing needed; tile part size is exact. */
    /* Build J2K codestream.
     * V31: pre-reserve to avoid vector reallocations (~800KB for 2K at 150Mbps). */
    J2KCodestreamWriter cs;
    cs.reserve(std::max(static_cast<size_t>(16384),
                        static_cast<size_t>(300) + 3 * (14 + per_comp)));

    cs.write_marker(J2K_SOC);

    /* SIZ */
    {
        cs.write_marker(J2K_SIZ);
        cs.write_u16(2 + 2 + 32 + 2 + 3 * 3);
        cs.write_u16(is_4k ? 0x0004 : 0x0003);  /* Rsiz: OPJ_PROFILE_CINEMA_4K / 2K */
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u32(width); cs.write_u32(height);
        cs.write_u32(0);     cs.write_u32(0);
        cs.write_u16(3);
        for (int c = 0; c < 3; ++c) { cs.write_u8(11); cs.write_u8(1); cs.write_u8(1); }
    }

    /* COD — SMPTE 429-4 / dcpverify-required fields:
       Scod=1 (precinct partition), CPRL, 1 layer, MCT=1,
       5 levels, 32x32 blocks, filter=0 (9/7 irreversible per DCP convention),
       precinct sizes: 0x77 (LL), 0x88×(levels) (other subbands) */
    {
        int num_precincts = NUM_DWT_LEVELS + 1;   /* 6 for 2K, 7 for 4K */
        if (is_4k) num_precincts = 7;
        cs.write_marker(J2K_COD);
        cs.write_u16(2 + 1 + 4 + 5 + num_precincts); /* Length includes precinct bytes */
        cs.write_u8(0x01);                       /* Scod=1: precinct partition enabled */
        cs.write_u8(0x04);                       /* SGcod: CPRL progression order */
        cs.write_u16(1);                         /* SGcod: 1 quality layer */
        cs.write_u8(1);                          /* SGcod: MCT=1 (required by DCI/dcpverify) */
        cs.write_u8(is_4k ? 6 : NUM_DWT_LEVELS); /* SPcod: decomposition levels */
        cs.write_u8(3);                          /* SPcod: xcb'=3 → 32-sample code blocks */
        cs.write_u8(3);                          /* SPcod: ycb'=3 → 32-sample code blocks */
        cs.write_u8(0x01);                       /* SPcod: BYPASS bit=1 (V165 bypass mode) */
        cs.write_u8(0x00);                       /* SPcod: filter=0 (9/7 irreversible, DCI) */
        cs.write_u8(0x77);                       /* Precinct: LL band = 128×128 */
        for (int i = 1; i < num_precincts; ++i)
            cs.write_u8(0x88);                   /* Precinct: other bands = 256×256 */
    }

    /* QCD — V57: per-subband step matching V53 perceptual weights.
     *
     * V53 applied level-dependent perceptual weights (0.65×LL5 … 1.20×L1).
     * The codestream QCD must report those actual steps so the decoder
     * dequantizes at the correct amplitude (not 1/0.65 = 1.54× too large).
     *
     * j2k_perceptual_sb_entry(base, i, is_4k) encodes the exact step used
     * for subband i: base_step × kWeights[i] → (eps<<11)|man.
     *
     * V125: base_step is now adaptive (compute_base_step). */
    float base_y2 = compute_base_step(width, height, per_comp);
    {
        cs.write_marker(J2K_QCD);
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;  /* V49: 19 for 4K, 16 for 2K */
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        cs.write_u8(0x22);  /* Sqcd: scalar expounded, 1 guard bit */
        for (int i = 0; i < nsb; ++i)
            cs.write_u16(j2k_perceptual_sb_entry(base_y2, i, is_4k));
    }

    /* V24/V57: QCC markers — per-component quantization step overrides for X and Z.
     * V125: uses adaptive base step. */
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;  /* V49: 19 for 4K, 16 for 2K */
        uint16_t lqcc = static_cast<uint16_t>(4 + 2 * nsb);
        float base_xz = base_y2 * 1.1f;
        for (int c : {0, 2}) {
            cs.write_marker(J2K_QCC);
            cs.write_u16(lqcc);
            cs.write_u8(static_cast<uint8_t>(c));  /* Cqcc: component index */
            cs.write_u8(0x22);                     /* Sqcc: scalar expounded, 1 guard bit */
            for (int i = 0; i < nsb; ++i)
                cs.write_u16(j2k_perceptual_sb_entry(base_xz, i, is_4k));
        }
    }

    /* V51: Compute actual per-component sizes (trim trailing zeros). */
    {
        uint8_t* src = impl->h_packed_pinned[0];
        for (int c = 0; c < 3; ++c)
            actual_sz[c] = find_actual_per_comp(src + c * per_comp, per_comp);
    }

    /* TLM (Tile-part Length Marker) — required by DCI Bv2.1.
       Must appear in the main header before the first SOT.
       Stlm = 0x40: ST=00 (no tile-part index), SP=1 (4-byte Ptlm).
       Ptlm[c] = SOT(2+10) + SOD(2) + stuffed_data = 14 + actual_sz[c]. */
    {
        /* J2K_TLM now defined at file scope */
        cs.write_marker(J2K_TLM);
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 1 + 3 * 4)); /* Ltlm = 4 + 3*4 = 16 */
        cs.write_u8(0);       /* Ztlm: 0 = first TLM segment */
        cs.write_u8(0x40);    /* Stlm: ST=00 (no tile index), SP=1 (4-byte Ptlm) */
        for (int c = 0; c < 3; ++c)
            cs.write_u32(static_cast<uint32_t>(14 + actual_sz[c]));  /* V51: actual sizes */
    }

    /* SOT + SOD: 3 tile parts (one per component) — DCI Bv2.1 requires 3 for 2K.
       Each SOT/SOD covers one component in CPRL order. */
    for (int c = 0; c < 3; ++c) {
        cs.write_marker(J2K_SOT);
        cs.write_u16(10);
        cs.write_u16(0);                              /* Isot: tile 0 */
        size_t psot_pos = cs.position();
        cs.write_u32(0);                              /* Psot: patched after data */
        cs.write_u8(static_cast<uint8_t>(c));         /* TPsot: tile part index */
        cs.write_u8(3);                               /* TNsot: 3 tile parts */
        cs.write_marker(J2K_SOD);
        cs.write_bytes(impl->h_packed_pinned[0] + c * per_comp, actual_sz[c]);  /* V51: actual bytes */
        cs.patch_u32(psot_pos, static_cast<uint32_t>(cs.position() - psot_pos + 4));
    }
    /* Pad final codestream to DCP minimum frame size (16384 bytes) */
    while (cs.data().size() < 16384) cs.write_u8(0);

    cs.write_marker(J2K_EOC);
    return std::move(cs.data());
}


/**
 * V17 path: encode from pre-converted XYZ int32 planes.
 * Used as fallback when colour params are not available.
 */
std::vector<uint8_t>
CudaJ2KEncoder::encode(
    const int32_t* const xyz_planes[3],
    int width,
    int height,
    int64_t bit_rate,
    int fps,
    bool is_3d,
    bool is_4k
)
{
    if (!_initialized) return {};

    size_t pixels = static_cast<size_t>(width) * height;
    _impl->ensure_buffers(width, height);

    /* Upload XYZ planes on component streams */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(_impl->d_in[c], xyz_planes[c],
                        pixels * sizeof(int32_t), cudaMemcpyHostToDevice,
                        _impl->stream[c]);
    }

    return run_dwt_and_build_codestream(_impl.get(), width, height,
                                        bit_rate, fps, is_3d, is_4k);
}


/**
 * V54: Pack RGB48LE interleaved uint16_t into 12-bit planar format.
 *
 * Output: 3 contiguous planes (R, G, B), each holding height rows of (width/2)*3 bytes.
 * Pair packing: for two adjacent 12-bit values A and B:
 *   byte 0 = A[11:4],  byte 1 = A[3:0]<<4 | B[11:8],  byte 2 = B[7:0]
 * Where A = src_val >> 4 (top 12 bits of 16-bit value).
 *
 * CPU time ~0.3ms (4 threads × 270 rows); fits within H2D time of 1.26ms.
 */
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

/**
 * V56: Single-channel variant of pack_rgb12_chunk.
 * Packs one colour plane (ch=0..2) into a pre-allocated contiguous plane buffer.
 * Allows per-channel H2D to start as soon as each plane is ready (0.075ms per plane).
 *
 * @param dst_plane  Start of the channel's plane in the packed buffer
 * @param ch         Source channel index (0=R, 1=G, 2=B) within interleaved RGB48LE
 */
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


/**
 * V18 path: encode from RGB48LE input with GPU colour conversion.
 * Eliminates CPU rgb_to_xyz bottleneck by running LUT+matrix on GPU.
 *
 * @param rgb16              Interleaved RGB48LE, row-major
 * @param rgb_stride_pixels  Row stride in uint16_t values (= width*3 typically)
 */
std::vector<uint8_t>
CudaJ2KEncoder::encode_from_rgb48(
    const uint16_t* rgb16,
    int width,
    int height,
    int rgb_stride_pixels,
    int64_t bit_rate,
    int fps,
    bool is_3d,
    bool is_4k
)
{
    if (!_initialized || !_colour_params_valid) return {};

    _impl->ensure_buffers(width, height);
    _impl->ensure_rgb_buffer(width, height);
    _impl->ensure_pinned_buffer(width, height);

    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    size_t pixels = static_cast<size_t>(width) * height;
    size_t per_comp = std::min(
        std::max(static_cast<size_t>(frame_bits / 8) / 3, static_cast<size_t>(1)), pixels);

    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);

    /* V42: 1-frame pipeline with H2D-compute overlap.
     * new_buf: write slot for this frame (RGB staging + D2H destination).
     * cur_buf: previous frame's slot (GPU currently executing or finished compute).
     *
     * Timeline (steady state):
     *   CPU memcpy(new_buf, 1.2ms) concurrent with GPU compute(cur_buf, 0.5ms on SM).
     *   H2D(new_buf, 1.66ms on PCIe st_h2d) starts after memcpy, overlaps with anything.
     *   sync(stream[0]) collects cur_buf result; compute(new_buf) kicks off after h2d_done.
     *   Steady-state bottleneck = H2D(1.66ms) + sync_tail(0.5ms) ≈ 1.73ms → ~578fps. */
    int new_buf = 1 - _impl->cur_buf;

    /* V56: Merged Steps 1+2 — per-channel pack+H2D pipeline.
     * Pack one colour plane (4 threads), then immediately submit H2D for that plane before
     * packing the next channel.  H2D for ch0 starts at ~0.075ms instead of ~0.3ms;
     * pack+H2D total latency: 0.075ms + 1.26ms = 1.335ms (was 1.56ms). */
    _impl->ensure_rgb12_buffer(width, height);
    {
        static constexpr int N_PACK = 4;
        const int packed_row_stride = (width / 2) * 3;
        const size_t plane_bytes    = static_cast<size_t>(packed_row_stride) * height;
        const int    chunk_rows     = (height + N_PACK - 1) / N_PACK;
        uint8_t*     dst            = _impl->h_rgb12_pinned[new_buf];

        for (int ch = 0; ch < 3; ++ch) {
            uint8_t* ch_plane = dst + ch * plane_bytes;
            std::future<void> futs[N_PACK - 1];
            for (int i = 1; i < N_PACK; ++i) {
                int y0 = i * chunk_rows, y1 = std::min(y0 + chunk_rows, height);
                futs[i - 1] = std::async(std::launch::async,
                    [=]{ pack_rgb12_plane(rgb16, rgb_stride_pixels, ch_plane,
                                          width, packed_row_stride, y0, y1, ch); });
            }
            pack_rgb12_plane(rgb16, rgb_stride_pixels, ch_plane,
                             width, packed_row_stride, 0, std::min(chunk_rows, height), ch);
            for (int i = 0; i < N_PACK - 1; ++i) futs[i].wait();

            /* H2D this channel immediately — runs on st_h2d while next channel is packed. */
            cudaMemcpyAsync(_impl->d_rgb12[new_buf] + ch * plane_bytes,
                            ch_plane, plane_bytes,
                            cudaMemcpyHostToDevice, _impl->st_h2d);
        }
        cudaEventRecord(_impl->h2d_done[new_buf], _impl->st_h2d);
    }

    /* Step 3: Rebuild comp graphs for new_buf if geometry/bitrate changed. */
    /* V54: packed_row_stride depends only on width (already tracked by cg_v42_width).
     * cg_v42_rgb_stride still stored for API compat but checked against packed_row_stride. */
    int v54_packed_stride = (width / 2) * 3;
    bool v42_valid = (_impl->cg_v42[new_buf][0] != nullptr                     &&
                      _impl->cg_v42_width[new_buf]      == width               &&
                      _impl->cg_v42_height[new_buf]     == height              &&
                      _impl->cg_v42_rgb_stride[new_buf] == v54_packed_stride   &&
                      _impl->cg_v42_per_comp[new_buf]   == per_comp            &&
                      _impl->cg_v42_is_4k[new_buf]      == is_4k               &&
                      _impl->cg_v42_is_3d[new_buf]      == is_3d);
    /* V125: Skip graph capture entirely — direct kernel launches are more robust
     * across GPU architectures and avoid misaligned-address errors during capture. */
    (void)v42_valid;  /* suppress unused warning */

    /* Step 4: Sync all 3 component streams for cur_buf (V49: was only stream[0]).
     * V49 race fix: all 3 streams write to h_packed_pinned[cur_buf]; syncing only
     * stream[0] could race with stream[1] and stream[2]'s D2H still in flight. */
    std::vector<uint8_t> result;
    if (_impl->pipeline_active) {
        cudaStreamSynchronize(_impl->stream[0]);
        cudaStreamSynchronize(_impl->stream[1]);  /* V49: race fix */
        cudaStreamSynchronize(_impl->stream[2]);  /* V49: race fix */
    }

    /* Step 5: Launch pipeline for new_buf.
     * V125: If graphs are available, use cudaGraphLaunch. Otherwise, fall back to
     * direct kernel launches (handles graph capture failures on older GPUs). */
    {
        /* V125: Direct kernel launch — no CUDA Graphs (robust across all GPU architectures). */
        float base_step_fb = compute_base_step(width, height, per_comp);
        int packed_row_stride_fb = (width / 2) * 3;
        const int num_levels_fb = is_4k ? 6 : NUM_DWT_LEVELS;
        size_t ch_smem_2row = static_cast<size_t>(2 * width) * sizeof(__half);
        size_t ch_smem_1row = static_cast<size_t>(1 * width) * sizeof(__half);
        int rgb_grid_2row = (height + 1) / 2;
        int rgb_grid_1row = height;
        bool use_1row_4k = (width > 2048);
        for (int c = 0; c < 3; ++c) {
            float step_c = base_step_fb * (c == 1 ? 1.0f : 1.1f);
            cudaStreamWaitEvent(_impl->stream[c], _impl->h2d_done[new_buf], 0);
            if (use_1row_4k) {
                kernel_rgb48_xyz_hdwt0_1ch_1row_p12<<<rgb_grid_1row, H_THREADS_FUSED, ch_smem_1row, _impl->stream[c]>>>(
                    _impl->d_rgb12[new_buf],
                    _impl->d_lut_in, _impl->d_lut_out, _impl->d_matrix,
                    _impl->d_b[c], c,
                    width, height, packed_row_stride_fb, width);
            } else {
                kernel_rgb48_xyz_hdwt0_1ch_2row_p12<<<rgb_grid_2row, H_THREADS_FUSED, ch_smem_2row, _impl->stream[c]>>>(
                    _impl->d_rgb12[new_buf],
                    _impl->d_lut_in, _impl->d_lut_out, _impl->d_matrix,
                    _impl->d_b[c], c,
                    width, height, packed_row_stride_fb, width);
            }
            launch_comp_pipeline(_impl.get(), c, width, height, per_comp, step_c, true,
                                 _impl->stream[c], _impl->h_packed_pinned[new_buf],
                                 num_levels_fb);
        }
    }

    /* Step 6: Sync cur_buf, then build codestream while GPU computes new_buf. */
    if (_impl->pipeline_active) {
        for (int c2 = 0; c2 < 3; ++c2)
            cudaStreamSynchronize(_impl->stream[c2]);
        result = build_j2k_codestream(_impl.get(),
            _impl->p_width, _impl->p_height,
            _impl->p_per_comp, _impl->p_is_4k, _impl->p_is_3d,
            _impl->h_packed_pinned[_impl->cur_buf]);
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
CudaJ2KEncoder::flush()
{
    /* V42: Drain the pipeline — collect the last in-flight frame's codestream.
     * V49: sync all 3 streams (was only stream[0]) to fix race condition. */
    if (!_initialized || !_impl->pipeline_active) return {};
    cudaStreamSynchronize(_impl->stream[0]);
    cudaStreamSynchronize(_impl->stream[1]);  /* V49: race fix */
    cudaStreamSynchronize(_impl->stream[2]);  /* V49: race fix */
    auto result = build_j2k_codestream(_impl.get(),
        _impl->p_width, _impl->p_height,
        _impl->p_per_comp, _impl->p_is_4k, _impl->p_is_3d,
        _impl->h_packed_pinned[_impl->cur_buf]);
    _impl->pipeline_active = false;
    return result;
}


/**
 * V127: GPU-accelerated RGB48→XYZ12 conversion.
 * Returns 3 planar int32 arrays (12-bit XYZ, 0-4095) compatible with OpenJPEGImage.
 * xyz_out must be pre-allocated: 3 * width * height * sizeof(int32_t).
 * Layout: xyz_out[0..pixels-1] = X, xyz_out[pixels..2*pixels-1] = Y, xyz_out[2*pixels..3*pixels-1] = Z.
 */
bool
CudaJ2KEncoder::gpu_rgb_to_xyz(
    const uint16_t* rgb16,
    int width,
    int height,
    int rgb_stride_pixels,
    int32_t* xyz_out)
{
    if (!_initialized || !_colour_params_valid) return false;

    size_t pixels = static_cast<size_t>(width) * height;
    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);

    /* Ensure device buffers */
    if (pixels > _impl->xyz_buf_pixels) {
        if (_impl->d_rgb16_xyz) cudaFree(_impl->d_rgb16_xyz);
        for (int c = 0; c < 3; ++c)
            if (_impl->d_xyz[c]) { cudaFree(_impl->d_xyz[c]); _impl->d_xyz[c] = nullptr; }
        if (_impl->h_xyz_pinned) cudaFreeHost(_impl->h_xyz_pinned);

        cudaMalloc(&_impl->d_rgb16_xyz, rgb_bytes);
        for (int c = 0; c < 3; ++c)
            cudaMalloc(&_impl->d_xyz[c], pixels * sizeof(int32_t));
        cudaHostAlloc(&_impl->h_xyz_pinned, 3 * pixels * sizeof(int32_t), cudaHostAllocDefault);
        _impl->xyz_buf_pixels = pixels;
    }

    /* H2D: upload RGB48LE to GPU */
    cudaMemcpy(_impl->d_rgb16_xyz, rgb16, rgb_bytes, cudaMemcpyHostToDevice);

    /* Launch conversion kernel */
    int threads = 256;
    int blocks = (static_cast<int>(pixels) + threads - 1) / threads;
    kernel_rgb48_to_xyz12_planar<<<blocks, threads, 0, _impl->stream[0]>>>(
        _impl->d_rgb16_xyz,
        _impl->d_lut_in_f32,
        _impl->d_lut_out,
        _impl->d_matrix,
        _impl->d_xyz[0], _impl->d_xyz[1], _impl->d_xyz[2],
        width, height, rgb_stride_pixels);

    /* D2H: download XYZ planes */
    for (int c = 0; c < 3; ++c) {
        cudaMemcpyAsync(_impl->h_xyz_pinned + c * pixels,
                        _impl->d_xyz[c],
                        pixels * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, _impl->stream[0]);
    }
    cudaStreamSynchronize(_impl->stream[0]);

    /* Copy to output */
    memcpy(xyz_out, _impl->h_xyz_pinned, 3 * pixels * sizeof(int32_t));
    return true;
}


/**
 * V127: Full GPU EBCOT encoding pipeline.
 * GPU: RGB48 → XYZ (colour conversion) → DWT → EBCOT T1 (per code-block)
 * CPU: T2 packet assembly → J2K codestream
 *
 * This is synchronous (no pipelining) — returns a complete, decodable J2K codestream.
 */
std::vector<uint8_t>
CudaJ2KEncoder::encode_ebcot(
    const uint16_t* rgb16,
    int width, int height, int rgb_stride_pixels,
    int64_t bit_rate, int fps, bool is_3d, bool is_4k,
    bool fast_mode)
{
    if (!_initialized || !_colour_params_valid) return {};

    /* V134: fast_mode quality knobs.
     * - fast_step_mult: multiplies base_step → coarser quantization → more
     *   zero coefficients → fewer significant bit-planes → fewer T1 passes
     *   per code-block → faster EBCOT T1. Output remains standard J2K
     *   because QCD markers are derived from the same step.
     * - fast_bitrate_mult: target_bytes scaled down so T2 truncates earlier.
     */
    /* V135: balanced fast_mode — 3x step is ~1.6 fewer bit-planes (~5 fewer
     * T1 passes per CB) and 0.5x target bytes. GTX 1050 Ti @ 2048×1080:
     * ~2.2x faster, ~25-30% of the correct-mode output size. Output is
     * still a standard J2K codestream (QCD markers carry the coarser step),
     * so it decodes in any J2K decoder — just with visible quality loss. */
    /* V188: fast mode revival.
     * After V186 lowered base_step by ~4× and V185 made HH bands use a finer T1 step
     * (compared to QCD value), fast_mode with MAX_BP=4 became severely under-coded —
     * LL band coefficients have num_bp ~ 10, dropping 6 MSBs to fit MAX_BP=4 → 7 dB PSNR.
     *
     * Compromise: keep fast_step_mult = 3.0 (modest coarsening), bump fast template
     * MAX_BP to 8 below.  Output will be slightly larger than the pre-V188 fast path
     * but quality recovers to ~50+ dB. */
    const float fast_step_mult    = fast_mode ? 3.0f : 1.0f;
    const float fast_bitrate_mult = fast_mode ? 0.5f : 1.0f;

    static const bool s_bench = (getenv("DCP_GPU_BENCH") != nullptr);
    using Clk = std::chrono::high_resolution_clock;
    auto t_prev = Clk::now();
    auto tmark = [&](const char* label) {
        if (!s_bench) return;
        cudaDeviceSynchronize();
        auto now = Clk::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(now - t_prev).count() / 1000.0;
        fprintf(stderr, "[bench] %-10s %.3f ms\n", label, ms);
        t_prev = now;
    };

    _impl->ensure_buffers(width, height);
    _impl->ensure_rgb_buffer(width, height);

    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;
    int stride = width;

    tmark("setup");

    /* Step 1: H2D — upload RGB48 to GPU.
     * V147 (reverted): staging into pinned + async cudaMemcpy on a dedicated
     * DMA stream made bench_phases 0.6 ms slower.  The caller buffer is
     * already fast-path for sync cudaMemcpy, and the extra CPU→pinned memcpy
     * dominated any async win in the single-thread synchronous path. */
    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);
    cudaMemcpy(_impl->d_rgb16[0], rgb16, rgb_bytes, cudaMemcpyHostToDevice);
    tmark("H2D");

    /* Step 2: GPU colour conversion + H-DWT level 0 (fused kernel).
     * V146: switch from 1-row to 2-row kernel when height is even (always true
     * for DCI 1080/2160).  Grid halves: 1080→540 blocks for 2K, 2160→1080 for
     * 4K.  smem doubles (2·w·sizeof(__half)) but still fits PreferL1 at 2K and
     * PreferNone at 4K.  Matrix/lut register state amortised over 2× the work;
     * adjacent-row L2 locality improves.  encode_from_rgb48's p12 path used
     * this pattern for years; encode_ebcot simply never caught up. */
    size_t ch_smem = static_cast<size_t>(2 * width) * sizeof(__half);
    int rgb_grid_2row = (height + 1) / 2;

    for (int c = 0; c < 3; ++c) {
        kernel_rgb48_xyz_hdwt0_1ch_2row<<<rgb_grid_2row, H_THREADS_FUSED, ch_smem, _impl->stream[c]>>>(
            _impl->d_rgb16[0],
            _impl->d_lut_in, _impl->d_lut_out, _impl->d_matrix,
            _impl->d_b[c], c,
            width, height, rgb_stride_pixels, stride);
    }

    tmark("RGB+HDWT0");

    /* Step 3: DWT levels 1+ (H-DWT + V-DWT per level per component) */
    for (int c = 0; c < 3; ++c) {
        int w = width, h = height;
        for (int level = 0; level < num_levels; ++level) {
            gpu_dwt97_level(_impl->d_a[c], _impl->d_b[c], nullptr,
                            _impl->d_in[c], w, h, stride, level, _impl->stream[c],
                            level == 0 /* skip H-DWT for level 0 */);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
    }
    /* V148: removed intermediate cudaDeviceSynchronize() after color and DWT
     * kernels — intra-stream ordering on stream[c] makes them redundant.
     * Error check is deferred to the single stream sync before T2. */

    /* V136: drop the unconditional DWT→T1 sync.
     * T1 is launched on the same stream[c] as DWT, so the intra-stream
     * ordering is automatic. We only need to sync if we're about to do
     * CPU-side work that modifies the CB table (step 4). */
    tmark("DWT_lv1+");

    /* Step 4: Build code-block table if needed */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    int64_t target_bytes = static_cast<int64_t>(
        static_cast<double>(frame_bits / 8) * fast_bitrate_mult);

    float base_step = compute_base_step(width, height,
        static_cast<size_t>(target_bytes / 3)) * fast_step_mult;

    if (_impl->ebcot_cb_table.empty() || _impl->ebcot_cb_table[0].quant_step != base_step) {
        build_codeblock_table(width, height, stride, num_levels, base_step, is_4k,
                              _impl->ebcot_cb_table, _impl->ebcot_subbands);
        int num_cbs = static_cast<int>(_impl->ebcot_cb_table.size());

        /* Reallocate EBCOT buffers if CB count changed */
        if (num_cbs != _impl->ebcot_num_cbs) {
            /* Free old */
            if (_impl->d_cb_info) cudaFree(_impl->d_cb_info);
            for (int c = 0; c < 3; ++c) {
                if (_impl->d_ebcot_data[c])     cudaFree(_impl->d_ebcot_data[c]);
                if (_impl->d_ebcot_len[c])      cudaFree(_impl->d_ebcot_len[c]);
                if (_impl->d_ebcot_npasses[c])   cudaFree(_impl->d_ebcot_npasses[c]);
                if (_impl->d_ebcot_passlens[c])  cudaFree(_impl->d_ebcot_passlens[c]);
                if (_impl->d_ebcot_numbp[c])    cudaFree(_impl->d_ebcot_numbp[c]);
                if (_impl->h_ebcot_data[c])     cudaFreeHost(_impl->h_ebcot_data[c]);
                if (_impl->h_ebcot_len[c])      cudaFreeHost(_impl->h_ebcot_len[c]);
                if (_impl->h_ebcot_npasses[c])   cudaFreeHost(_impl->h_ebcot_npasses[c]);
                if (_impl->h_ebcot_passlens[c])  cudaFreeHost(_impl->h_ebcot_passlens[c]);
                if (_impl->h_ebcot_numbp[c])    cudaFreeHost(_impl->h_ebcot_numbp[c]);
            }

            /* Allocate new */
            cudaMalloc(&_impl->d_cb_info, num_cbs * sizeof(CodeBlockInfo));
            for (int c = 0; c < 3; ++c) {
                cudaMalloc(&_impl->d_ebcot_data[c],    (size_t)num_cbs * CB_BUF_SIZE);
                cudaMalloc(&_impl->d_ebcot_len[c],     num_cbs * sizeof(uint16_t));
                cudaMalloc(&_impl->d_ebcot_npasses[c],  num_cbs * sizeof(uint8_t));
                cudaMalloc(&_impl->d_ebcot_passlens[c], (size_t)num_cbs * MAX_PASSES * sizeof(uint16_t));
                cudaMalloc(&_impl->d_ebcot_numbp[c],   num_cbs * sizeof(uint8_t));
                cudaHostAlloc(&_impl->h_ebcot_data[c],    (size_t)num_cbs * CB_BUF_SIZE, cudaHostAllocDefault);
                cudaHostAlloc(&_impl->h_ebcot_len[c],     num_cbs * sizeof(uint16_t), cudaHostAllocDefault);
                cudaHostAlloc(&_impl->h_ebcot_npasses[c],  num_cbs * sizeof(uint8_t), cudaHostAllocDefault);
                cudaHostAlloc(&_impl->h_ebcot_passlens[c], (size_t)num_cbs * MAX_PASSES * sizeof(uint16_t), cudaHostAllocDefault);
                cudaHostAlloc(&_impl->h_ebcot_numbp[c],   num_cbs * sizeof(uint8_t), cudaHostAllocDefault);
            }
            _impl->ebcot_num_cbs = num_cbs;
        }

        /* Upload CB info table */
        cudaMemcpy(_impl->d_cb_info, _impl->ebcot_cb_table.data(),
                   num_cbs * sizeof(CodeBlockInfo), cudaMemcpyHostToDevice);
    }

    /* Step 5: Launch EBCOT T1 kernel per component */
    int num_cbs = _impl->ebcot_num_cbs;
    /* V141: 64 threads/block matches kernel's __launch_bounds__(64, 16).
     * Each thread uses ~2KB local memory (mag[1024]), so smaller blocks
     * reduce L1 pressure and increase occupancy.
     * V145: re-tested (128, 8) and (32, 32) — both ~1% slower than (64, 16). */
    constexpr int EBCOT_THREADS = 64;
    int ebcot_grid = (num_cbs + EBCOT_THREADS - 1) / EBCOT_THREADS;

    /* V143: fast-mode also skips the 1 LSB bit-plane inside T1, dropping
     * up to 3 coding passes per CB (SPP+MRP+CUP of the final bit-plane).
     * Effective quantization on those coefficients ≈ step × 2; combined
     * with step_mult=3.0 → effective step × 6 on the LSB. */
    int  bp_skip    = fast_mode ? 1 : 0;
    /* V165: BYPASS mode disabled — re-enabling produced "segment too long" decode
     * failures even after V187, indicating residual bugs in the bypass coding path.
     * Leave off until those are fixed; the speed gain isn't worth correctness. */
    bool use_bypass = false;
    /* V177: MAX_BP increased to 14 for correct mode.
     * base_step=2.12 at 150Mbps/2K → max q ≈ 4095/1.378 ≈ 2972 → 12 bit-planes.
     * p_max = 13 - floor(log2(step_LL5)) + 1 = 14 for LL5. MAX_BP=14 covers all.
     * Fast mode: MAX_BP=4 (step_mult=3.0 → coarse quant → ≤4 bit-planes). */
    for (int c = 0; c < 3; ++c) {
        if (fast_mode) {
            /* V188: fast path MAX_BP 4 → 12.  With V186 base_step now ~4× smaller and
             * V185 effective T1 step varying per band, max num_bp on real frames at
             * 150 Mbps lands around 10-11.  Tried MAX_BP=10 — h_gradient regressed
             * 52.9→26.8 dB (some CB hit 11 bit-planes, MSB truncated). MAX_BP=12 stays. */
            kernel_ebcot_t1<true, 12><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                _impl->d_a[c], stride,
                _impl->d_cb_info, num_cbs,
                _impl->d_ebcot_data[c],
                _impl->d_ebcot_len[c],
                _impl->d_ebcot_npasses[c],
                _impl->d_ebcot_passlens[c],
                _impl->d_ebcot_numbp[c],
                bp_skip, use_bypass);
        } else {
            kernel_ebcot_t1<false, 14><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                _impl->d_a[c], stride,
                _impl->d_cb_info, num_cbs,
                _impl->d_ebcot_data[c],
                _impl->d_ebcot_len[c],
                _impl->d_ebcot_npasses[c],
                _impl->d_ebcot_passlens[c],
                _impl->d_ebcot_numbp[c],
                bp_skip, use_bypass);
        }
    }

    /* V148: T1 error is checked after the stream syncs at the end of D2H. */
    tmark("EBCOT_T1");

    /* Step 6: D2H transfer of coded data.
     *
     * V137: strided D2H. Source device layout is [CB_BUF_SIZE] per CB but
     * actual coded bytes typically occupy 5-20% of that. cudaMemcpy2DAsync
     * with a smaller dst-pitch copies only the first max_cb_d2h bytes per
     * CB, dropping D2H traffic 2-4×. T2 then reads strided at the new
     * smaller pitch and caps per-CB len to max_cb_d2h-1 for safety.
     *
     * Budget:
     *   correct mode — max_cb_d2h = 1024 (halves 2048-byte D2H).
     *   fast mode    — max_cb_d2h = 640  (~3.2× smaller than full 2048).
     * Any CB whose coded output exceeds the budget is truncated, which in
     * practice only affects the lowest few code-blocks at very high rates.
     */
    const int max_cb_d2h = fast_mode ? 640 : 1024;
    for (int c = 0; c < 3; ++c) {
        cudaMemcpy2DAsync(
            _impl->h_ebcot_data[c], max_cb_d2h,
            _impl->d_ebcot_data[c], CB_BUF_SIZE,
            max_cb_d2h, num_cbs,
            cudaMemcpyDeviceToHost, _impl->stream[c]);
        cudaMemcpyAsync(_impl->h_ebcot_len[c], _impl->d_ebcot_len[c],
                        num_cbs * sizeof(uint16_t), cudaMemcpyDeviceToHost, _impl->stream[c]);
        cudaMemcpyAsync(_impl->h_ebcot_npasses[c], _impl->d_ebcot_npasses[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, _impl->stream[c]);
        cudaMemcpyAsync(_impl->h_ebcot_numbp[c], _impl->d_ebcot_numbp[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, _impl->stream[c]);
        /* V132: pass_lengths not used by T2 (single-layer CPRL) — skip D2H */
    }

    for (int c = 0; c < 3; ++c)
        cudaStreamSynchronize(_impl->stream[c]);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            fprintf(stderr, "GPU pipeline error: %s\n", cudaGetErrorString(err));
    }
    tmark("D2H");

    /* Step 7: CPU T2 assembly + codestream construction */
#ifdef GPU_J2K_DEBUG_T1
    for (int c = 0; c < 3; ++c) {
        long sum_bp = 0, sum_len = 0, nz_bp = 0;
        for (int i = 0; i < num_cbs; ++i) {
            sum_bp  += _impl->h_ebcot_numbp[c][i];
            sum_len += _impl->h_ebcot_len[c][i];
            if (_impl->h_ebcot_numbp[c][i] > 0) ++nz_bp;
        }
        fprintf(stderr, "DEBUG T1 comp=%d  avg_bp=%.2f nz=%ld/%d avg_len=%.1f\n",
                c, sum_bp / (double)num_cbs, nz_bp, num_cbs, sum_len / (double)num_cbs);
    }
#endif
    const uint8_t*  cd[3] = { _impl->h_ebcot_data[0], _impl->h_ebcot_data[1], _impl->h_ebcot_data[2] };
    const uint16_t* cl[3] = { _impl->h_ebcot_len[0],  _impl->h_ebcot_len[1],  _impl->h_ebcot_len[2] };
    const uint8_t*  np[3] = { _impl->h_ebcot_npasses[0], _impl->h_ebcot_npasses[1], _impl->h_ebcot_npasses[2] };
    const uint16_t* pl[3] = { _impl->h_ebcot_passlens[0], _impl->h_ebcot_passlens[1], _impl->h_ebcot_passlens[2] };
    const uint8_t*  nb[3] = { _impl->h_ebcot_numbp[0], _impl->h_ebcot_numbp[1], _impl->h_ebcot_numbp[2] };

    auto result = build_ebcot_codestream(
        width, height, is_4k, is_3d,
        num_levels, base_step,
        _impl->ebcot_subbands,
        cd, cl, np, pl, nb,
        target_bytes,
        max_cb_d2h);
    tmark("T2+CS");
    return result;
}


/**
 * Upload colour conversion LUT+matrix to GPU device memory.
 * Call once per film (or whenever colour conversion changes).
 */
void
CudaJ2KEncoder::set_colour_params(GpuColourParams const& params)
{
    if (!_initialized || !params.valid) return;
    _impl->upload_colour_params(params);
    _colour_params_valid = true;
}


std::vector<float>
CudaJ2KEncoder::debug_get_dwt_output(int c, int width, int height)
{
    if (!_initialized || c < 0 || c >= 3) return {};
    std::vector<__half> h_tmp(size_t(width) * height);
    cudaMemcpy(h_tmp.data(), _impl->d_a[c],
               size_t(width) * height * sizeof(__half), cudaMemcpyDeviceToHost);
    std::vector<float> out(h_tmp.size());
    for (size_t i = 0; i < h_tmp.size(); ++i)
        out[i] = __half2float(h_tmp[i]);
    return out;
}


/* Singleton for backward compatibility */
static std::shared_ptr<CudaJ2KEncoder> _cuda_j2k_instance;
static std::mutex _cuda_j2k_instance_mutex;

std::shared_ptr<CudaJ2KEncoder>
cuda_j2k_encoder_instance()
{
    std::lock_guard<std::mutex> lock(_cuda_j2k_instance_mutex);
    if (!_cuda_j2k_instance)
        _cuda_j2k_instance = std::make_shared<CudaJ2KEncoder>();
    return _cuda_j2k_instance;
}
