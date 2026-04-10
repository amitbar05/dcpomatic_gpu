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
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <future>
#include <mutex>
#include <vector>


/* ===== J2K Codestream Constants ===== */
static constexpr uint16_t J2K_SOC = 0xFF4F;
static constexpr uint16_t J2K_SIZ = 0xFF51;
static constexpr uint16_t J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C;
static constexpr uint16_t J2K_QCC = 0xFF5D;  /* V24: per-component QCD override */
static constexpr uint16_t J2K_SOT = 0xFF90;
static constexpr uint16_t J2K_SOD = 0xFF93;
static constexpr uint16_t J2K_EOC = 0xFFD9;

static constexpr int NUM_DWT_LEVELS  = 5;
/* V27: column height threshold for register-blocked V-DWT.
 * For h ≤ MAX_REG_HEIGHT, the entire column fits in registers (float col[140]).
 * Eliminates fp16 workspace accesses for small subbands; reduces DRAM traffic. */
static constexpr int MAX_REG_HEIGHT  = 140;
/* V29/V34/V62/V70/V71: tiled V-DWT parameters.
 * V71: V_TILE 24→28 — further reduces overlap overhead; 100% occupancy retained (31 regs).
 *   4K level-0: overlap rows (75KB) exceed 48KB L1 → DRAM-bound; 15% fewer vs V_TILE=24.
 *   V_TILE=28: V_TILE_FL=38 → __half col[38] → ~31 regs/T → 65536/(256×31)=8.2 → 8 blk = 100%.
 *   Grid: 2K level-0 45→39 tiles; 4K level-0 90→77 tiles (13-14% fewer dispatches).
 * V70: V_TILE 16→24 — V69 __half lowered regs to ~23/T → thread-limited (100% occ); 14.8% fewer loads.
 * V62: V_TILE 32→16 (reverts V34) to improve SM occupancy given PreferL1 (V59).
 *   V_TILE=32 float: ~50 regs → 10 blk/SM → 40 warps (< DRAM latency threshold).
 *   V_TILE=16 float: ~36 regs → 14 blk/SM → 56 warps → 37% better DRAM latency hiding.
 * OVERLAP=5 halo rows each side (covers 4-step CDF 9/7 stencil + 1 safety margin). */
static constexpr int V_TILE    = 28;
static constexpr int V_OVERLAP = 5;
static constexpr int V_TILE_FL = V_TILE + 2 * V_OVERLAP;  /* 38 */
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


/* ===== CUDA Kernels ===== */

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

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
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

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += ALPHA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    /* V23: Apply CDF 9/7 analysis normalization at deinterleave output.
     * L×NORM_L and H×NORM_H ensure all subbands stay in [-input_range, input_range]. */
    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2] = smem[x] * NORM_L;
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * NORM_H;
}


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
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += __half(2.0f * ALPHA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * BETA) * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(BETA) * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(GAMMA) * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += __half(2.0f * GAMMA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * DELTA) * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(DELTA) * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2]      = smem[x] * __half(NORM_L);
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = smem[x] * __half(NORM_H);
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
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += __half(2.0f * ALPHA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * BETA) * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(BETA) * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += __half(GAMMA) * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += __half(2.0f * GAMMA) * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += __half(2.0f * DELTA) * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += __half(DELTA) * (smem[x - 1] + smem[min(x + 1, width - 1)]);
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
    if(t==0&&w>1&&!(w&1)) { smem[w-1]+=__half(2.f*ALPHA)*smem[w-2]; if(y1<height) smem[2*w-1]+=__half(2.f*ALPHA)*smem[2*w-2]; }
    __syncthreads();
    if(t==0) { smem[0]+=__half(2.f*BETA)*smem[min(1,w-1)]; if(y1<height) smem[w]+=__half(2.f*BETA)*smem[w+min(1,w-1)]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        smem[x]  +=__half(BETA)*(smem[x-1]+smem[min(x+1,w-1)]);
        if(y1<height) smem[w+x]+=__half(BETA)*(smem[w+x-1]+smem[w+min(x+1,w-1)]);
    }
    __syncthreads();
    for (int x=1+t*2; x<w-1; x+=nt*2) {
        smem[x]  +=__half(GAMMA)*(smem[x-1]+smem[x+1]);
        if(y1<height) smem[w+x]+=__half(GAMMA)*(smem[w+x-1]+smem[w+x+1]);
    }
    if(t==0&&w>1&&!(w&1)) { smem[w-1]+=__half(2.f*GAMMA)*smem[w-2]; if(y1<height) smem[2*w-1]+=__half(2.f*GAMMA)*smem[2*w-2]; }
    __syncthreads();
    if(t==0) { smem[0]+=__half(2.f*DELTA)*smem[min(1,w-1)]; if(y1<height) smem[w]+=__half(2.f*DELTA)*smem[w+min(1,w-1)]; }
    for (int x=2+t*2; x<w; x+=nt*2) {
        smem[x]  +=__half(DELTA)*(smem[x-1]+smem[min(x+1,w-1)]);
        if(y1<height) smem[w+x]+=__half(DELTA)*(smem[w+x-1]+smem[w+min(x+1,w-1)]);
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
 */
__global__ void
kernel_fused_horz_dwt_half_io_4row(
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
     * Interior (all 4 rows present): branch-free lifting and stores for full 4-row ILP.
     * Else: original per-iteration yN<height guards for partial last block. */
    if (y3 < height) {
        /* Interior path: all 4 rows valid — no yN<height inside any loop. */
        for (int x = t; x < w; x += nt) {
            smem[x]     = __ldg(&d_data[y0*stride+x]);
            smem[w+x]   = __ldg(&d_data[y1*stride+x]);
            smem[2*w+x] = __ldg(&d_data[y2*stride+x]);
            smem[3*w+x] = __ldg(&d_data[y3*stride+x]);
        }
        __syncthreads();
        /* Alpha: odd positions */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            smem[x]     +=__half(ALPHA)*(smem[x-1]     +smem[x+1]);
            smem[w+x]   +=__half(ALPHA)*(smem[w+x-1]  +smem[w+x+1]);
            smem[2*w+x] +=__half(ALPHA)*(smem[2*w+x-1]+smem[2*w+x+1]);
            smem[3*w+x] +=__half(ALPHA)*(smem[3*w+x-1]+smem[3*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            smem[w-1]   +=__half(2.f*ALPHA)*smem[w-2];
            smem[2*w-1] +=__half(2.f*ALPHA)*smem[2*w-2];
            smem[3*w-1] +=__half(2.f*ALPHA)*smem[3*w-2];
            smem[4*w-1] +=__half(2.f*ALPHA)*smem[4*w-2];
        }
        __syncthreads();
        /* Beta: even positions */
        if(t==0) {
            smem[0]   +=__half(2.f*BETA)*smem[min(1,w-1)];
            smem[w]   +=__half(2.f*BETA)*smem[w+min(1,w-1)];
            smem[2*w] +=__half(2.f*BETA)*smem[2*w+min(1,w-1)];
            smem[3*w] +=__half(2.f*BETA)*smem[3*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(BETA)*(smem[x-1]     +smem[min(x+1,w-1)]);
            smem[w+x]   +=__half(BETA)*(smem[w+x-1]  +smem[w+min(x+1,w-1)]);
            smem[2*w+x] +=__half(BETA)*(smem[2*w+x-1]+smem[2*w+min(x+1,w-1)]);
            smem[3*w+x] +=__half(BETA)*(smem[3*w+x-1]+smem[3*w+min(x+1,w-1)]);
        }
        __syncthreads();
        /* Gamma: odd positions */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            smem[x]     +=__half(GAMMA)*(smem[x-1]     +smem[x+1]);
            smem[w+x]   +=__half(GAMMA)*(smem[w+x-1]  +smem[w+x+1]);
            smem[2*w+x] +=__half(GAMMA)*(smem[2*w+x-1]+smem[2*w+x+1]);
            smem[3*w+x] +=__half(GAMMA)*(smem[3*w+x-1]+smem[3*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            smem[w-1]   +=__half(2.f*GAMMA)*smem[w-2];
            smem[2*w-1] +=__half(2.f*GAMMA)*smem[2*w-2];
            smem[3*w-1] +=__half(2.f*GAMMA)*smem[3*w-2];
            smem[4*w-1] +=__half(2.f*GAMMA)*smem[4*w-2];
        }
        __syncthreads();
        /* Delta: even positions */
        if(t==0) {
            smem[0]   +=__half(2.f*DELTA)*smem[min(1,w-1)];
            smem[w]   +=__half(2.f*DELTA)*smem[w+min(1,w-1)];
            smem[2*w] +=__half(2.f*DELTA)*smem[2*w+min(1,w-1)];
            smem[3*w] +=__half(2.f*DELTA)*smem[3*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(DELTA)*(smem[x-1]     +smem[min(x+1,w-1)]);
            smem[w+x]   +=__half(DELTA)*(smem[w+x-1]  +smem[w+min(x+1,w-1)]);
            smem[2*w+x] +=__half(DELTA)*(smem[2*w+x-1]+smem[2*w+min(x+1,w-1)]);
            smem[3*w+x] +=__half(DELTA)*(smem[3*w+x-1]+smem[3*w+min(x+1,w-1)]);
        }
        __syncthreads();
        /* Deinterleave and write — 4 unconditional stores per loop. */
        for (int x=t*2; x<w; x+=nt*2) {
            d_tmp[y0*stride+x/2] = smem[x]     * __half(NORM_L);
            d_tmp[y1*stride+x/2] = smem[w+x]   * __half(NORM_L);
            d_tmp[y2*stride+x/2] = smem[2*w+x] * __half(NORM_L);
            d_tmp[y3*stride+x/2] = smem[3*w+x] * __half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            d_tmp[y0*stride+hw+x/2] = smem[x]     * __half(NORM_H);
            d_tmp[y1*stride+hw+x/2] = smem[w+x]   * __half(NORM_H);
            d_tmp[y2*stride+hw+x/2] = smem[2*w+x] * __half(NORM_H);
            d_tmp[y3*stride+hw+x/2] = smem[3*w+x] * __half(NORM_H);
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
        if(t==0&&w>1&&!(w&1)) {
            smem[w-1]   +=__half(2.f*ALPHA)*smem[w-2];
            if(y1<height) smem[2*w-1] +=__half(2.f*ALPHA)*smem[2*w-2];
            if(y2<height) smem[3*w-1] +=__half(2.f*ALPHA)*smem[3*w-2];
        }
        __syncthreads();
        /* Beta */
        if(t==0) {
            smem[0]     +=__half(2.f*BETA)*smem[min(1,w-1)];
            if(y1<height) smem[w]   +=__half(2.f*BETA)*smem[w+min(1,w-1)];
            if(y2<height) smem[2*w] +=__half(2.f*BETA)*smem[2*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(BETA)*(smem[x-1]     +smem[min(x+1,w-1)]);
            if(y1<height) smem[w+x]   +=__half(BETA)*(smem[w+x-1]  +smem[w+min(x+1,w-1)]);
            if(y2<height) smem[2*w+x] +=__half(BETA)*(smem[2*w+x-1]+smem[2*w+min(x+1,w-1)]);
        }
        __syncthreads();
        /* Gamma */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            smem[x]     +=__half(GAMMA)*(smem[x-1]     +smem[x+1]);
            if(y1<height) smem[w+x]   +=__half(GAMMA)*(smem[w+x-1]  +smem[w+x+1]);
            if(y2<height) smem[2*w+x] +=__half(GAMMA)*(smem[2*w+x-1]+smem[2*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            smem[w-1]   +=__half(2.f*GAMMA)*smem[w-2];
            if(y1<height) smem[2*w-1] +=__half(2.f*GAMMA)*smem[2*w-2];
            if(y2<height) smem[3*w-1] +=__half(2.f*GAMMA)*smem[3*w-2];
        }
        __syncthreads();
        /* Delta */
        if(t==0) {
            smem[0]     +=__half(2.f*DELTA)*smem[min(1,w-1)];
            if(y1<height) smem[w]   +=__half(2.f*DELTA)*smem[w+min(1,w-1)];
            if(y2<height) smem[2*w] +=__half(2.f*DELTA)*smem[2*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            smem[x]     +=__half(DELTA)*(smem[x-1]     +smem[min(x+1,w-1)]);
            if(y1<height) smem[w+x]   +=__half(DELTA)*(smem[w+x-1]  +smem[w+min(x+1,w-1)]);
            if(y2<height) smem[2*w+x] +=__half(DELTA)*(smem[2*w+x-1]+smem[2*w+min(x+1,w-1)]);
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
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * ALPHA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * BETA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += BETA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    for (int x = 1 + t * 2; x < width - 1; x += nt * 2)
        smem[x] += GAMMA * (smem[x - 1] + smem[x + 1]);
    if (t == 0 && width > 1 && (width % 2 == 0))
        smem[width - 1] += 2.0f * GAMMA * smem[width - 2];
    __syncthreads();

    if (t == 0) smem[0] += 2.0f * DELTA * smem[min(1, width - 1)];
    for (int x = 2 + t * 2; x < width; x += nt * 2)
        smem[x] += DELTA * (smem[x - 1] + smem[min(x + 1, width - 1)]);
    __syncthreads();

    int hw = (width + 1) / 2;
    for (int x = t * 2; x < width; x += nt * 2)
        d_tmp[y * stride + x / 2]      = __float2half(smem[x]     * NORM_L);
    for (int x = t * 2 + 1; x < width; x += nt * 2)
        d_tmp[y * stride + hw + x / 2] = __float2half(smem[x]     * NORM_H);
}


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
 * V36: Register-blocked V-DWT (small subbands), writes __half output to d_dst_h.
 * Mirrors kernel_fused_vert_dwt_fp16_hi_reg but output is half (d_a is now half in V36).
 */
__global__ void
kernel_fused_vert_dwt_fp16_hi_reg_ho(
    const __half* __restrict__ d_src,
    __half* __restrict__ d_dst_h,
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    /* V69: __half col[] — direct __ldg→__half; eliminates __half2float/__float2half conversions.
     * __half HFMA has 2× throughput vs float FMA on sm_61+. Compiler may pack adjacent halves
     * into 32-bit register pairs, halving register pressure (140 float → 70 effective regs). */
    __half col[MAX_REG_HEIGHT];
    for (int y = 0; y < height; y++)
        col[y] = __ldg(&d_src[y * stride + x]);

    for (int y = 1; y < height - 1; y += 2)
        col[y] += __half(ALPHA) * (col[y-1] + col[y+1]);
    if (height > 1 && (height % 2 == 0))
        col[height-1] += __half(2.f*ALPHA) * col[height-2];

    col[0] += __half(2.f*BETA) * col[min(1, height-1)];
    for (int y = 2; y < height; y += 2) {
        int yp1 = (y+1 < height) ? y+1 : y-1;
        col[y] += __half(BETA) * (col[y-1] + col[yp1]);
    }

    for (int y = 1; y < height - 1; y += 2)
        col[y] += __half(GAMMA) * (col[y-1] + col[y+1]);
    if (height > 1 && (height % 2 == 0))
        col[height-1] += __half(2.f*GAMMA) * col[height-2];

    col[0] += __half(2.f*DELTA) * col[min(1, height-1)];
    for (int y = 2; y < height; y += 2) {
        int yp1 = (y+1 < height) ? y+1 : y-1;
        col[y] += __half(DELTA) * (col[y-1] + col[yp1]);
    }

    /* V36/V69: write __half output directly — no __float2half conversion needed */
    int hh = (height + 1) / 2;
    for (int y = 0; y < height; y += 2)
        d_dst_h[(y/2)*stride+x] = col[y] * __half(NORM_L);
    for (int y = 1; y < height; y += 2)
        d_dst_h[(hh+y/2)*stride+x] = col[y] * __half(NORM_H);
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
    /* Alpha: globally-odd rows (P0=1 → local even indices starting at i=2) */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2) col[i] += kA*(col[i-1]+col[i+1]);
    /* Beta: globally-even rows (P0=1 → local odd indices starting at i=1) */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2) col[i] += kB*(col[i-1]+col[i+1]);
    /* Gamma: globally-odd rows */
    #pragma unroll
    for (int i = 2; i < V_TILE_FL - 1; i += 2) col[i] += kG*(col[i-1]+col[i+1]);
    /* Delta: globally-even rows */
    #pragma unroll
    for (int i = 1; i < V_TILE_FL - 1; i += 2) col[i] += kD*(col[i-1]+col[i+1]);
}


/**
 * V36: Tiled V-DWT, writes __half output to d_dst_h.
 * Mirrors kernel_fused_vert_dwt_tiled but output is half (d_a is now half in V36).
 * V71: V_TILE=28, V_TILE_FL=38. No shared memory.
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

    /* V40: V_TILE (even) + V_OVERLAP (odd) → load_start always odd → p0 always 1.
     * V69: use cdf97_lift_tiled_h — __half lifting, P0=1 hardcoded. */
    static_assert(V_TILE % 2 == 0 && V_OVERLAP % 2 == 1,
                  "V40: requires V_TILE even + V_OVERLAP odd for constant p0=1");
    cdf97_lift_tiled_h(col);

    /* V40/V69: output parity p0=1 → even i = H, odd i = L; direct __half stores. */
    int hh = (height + 1) / 2;
    if (interior) {
        /* V64+V69: exactly V_TILE outputs, fully unrolled, no __float2half conversions. */
        #pragma unroll
        for (int i = V_OVERLAP; i < V_OVERLAP + V_TILE; i++) {
            int gy = load_start + i;
            if (!(i & 1))
                d_dst_h[(hh + gy/2) * stride + x] = col[i] * __half(NORM_H);
            else
                d_dst_h[(gy/2) * stride + x] = col[i] * __half(NORM_L);
        }
    } else {
        for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
            int gy = load_start + i;
            if (!(i & 1))
                d_dst_h[(hh + gy/2) * stride + x] = col[i] * __half(NORM_H);
            else
                d_dst_h[(gy/2) * stride + x] = col[i] * __half(NORM_L);
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
 * V60: Fully vectorized multi-level perceptual quantization — all rows use __half2 loads.
 *
 * V58 vectorized L1/L2/L3 rows (~87.5%). V60 extends to L4/L5/DC rows (~12.5%):
 *   DC rows: 6 vq2 zones [0,ll5_c)→inv_dc, [ll5_c,2c)→inv_l5, ..., [16c,stride)→inv_l1
 *   L5 rows: 5 zones starting [0,2c)→inv_l5; L4 rows: 4 zones starting [0,4c)→inv_l4
 *   All zone boundaries are multiples of ll5_c=stride/32: even for 2K(60) and 4K(120).
 * 100% of rows now use vq2: __half2 loads (4B) + uint16_t stores (2B) vs scalar (2B+1B).
 *
 * Requirement: stride even (2K=1920, 4K=3840); ll5_c = stride/32 always even.
 */
__global__ void
kernel_quantize_subband_ml(
    const __half* __restrict__ d_comp,
    uint8_t* __restrict__ d_packed,
    int stride, int n_rows, float base_step,
    int ll5_h, int ll5_c)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    float inv_dc = __frcp_rn(base_step * 0.65f);
    float inv_l5 = __frcp_rn(base_step * 0.85f);
    float inv_l4 = __frcp_rn(base_step * 0.95f);
    float inv_l3 = __frcp_rn(base_step * 1.05f);
    float inv_l2 = __frcp_rn(base_step * 1.12f);
    float inv_l1 = __frcp_rn(base_step * 1.20f);

    bool is_dc_row = (row < ll5_h);
    bool is_l5_row = (row < ll5_h * 2);
    int row_lv = is_l5_row ? 5 :
                 (row < ll5_h * 4  ? 4 :
                 (row < ll5_h * 8  ? 3 :
                 (row < ll5_h * 16 ? 2 : 1)));

    const __half* row_src = d_comp  + static_cast<size_t>(row) * stride;
    uint8_t*      row_dst = d_packed + static_cast<size_t>(row) * stride;

    /* V60: All rows vectorized via vq2 (was V58: L1/L2/L3 only, L4/L5/DC scalar).
     * Each zone spans columns with uniform inv_step → __half2 loads + uint16_t stores.
     * All zone boundaries are multiples of ll5_c (even) → uint16_t alignment guaranteed.
     * L1 rows: 1 zone. L2 rows: 2 zones. L3 rows: 3 zones.
     * L4 rows: 4 zones. L5 rows: 5 zones. DC rows: 6 zones.
     */
    auto vq2 = [row_src, row_dst](int col_start, int col_end, float inv, int nt) {
        /* Vectorized quantize: [col_start, col_end) with uniform inv, 2 samples/thread. */
        for (int c = col_start + threadIdx.x * 2; c + 1 < col_end; c += nt * 2) {
            __half2 hv = __ldg(reinterpret_cast<const __half2*>(row_src) + c / 2);
            int q0 = __float2int_rn(__half2float(hv.x) * inv);
            int q1 = __float2int_rn(__half2float(hv.y) * inv);
            uint8_t b0 = (q0 < 0 ? 0x80u : 0u) | uint8_t(min(126, abs(q0)));
            uint8_t b1 = (q1 < 0 ? 0x80u : 0u) | uint8_t(min(126, abs(q1)));
            *reinterpret_cast<uint16_t*>(row_dst + c) = uint16_t(b0) | (uint16_t(b1) << 8);
        }
    };

    if (row_lv == 1) {
        /* L1: all cols uniform inv_l1 */
        vq2(0, stride, inv_l1, blockDim.x);
    } else if (row_lv == 2) {
        /* L2: 2 zones split at stride/2 = ll5_c*16 */
        const int mid = ll5_c * 16;
        vq2(0,   mid,    inv_l2, blockDim.x);
        vq2(mid, stride, inv_l1, blockDim.x);
    } else if (row_lv == 3) {
        /* L3: 3 zones split at ll5_c*8 and ll5_c*16 */
        const int b1 = ll5_c * 8;
        const int b2 = ll5_c * 16;
        vq2(0,  b1,     inv_l3, blockDim.x);
        vq2(b1, b2,     inv_l2, blockDim.x);
        vq2(b2, stride, inv_l1, blockDim.x);
    } else {
        /* V60: L4/L5/DC rows — fully vectorized zone-based quantize.
         * Zone boundaries are multiples of ll5_c (stride/32), always even → vq2-safe.
         * DC rows: 6 zones (LL5 uses inv_dc; LH5/HL5/HH5 uses inv_l5; then l4..l1).
         * L5 rows: 5 zones (first zone [0,2c)→inv_l5; no separate DC sub-zone).
         * L4 rows: 4 zones (first zone [0,4c)→inv_l4). */
        if (is_dc_row) {
            vq2(0,          ll5_c,      inv_dc, blockDim.x);
            vq2(ll5_c,      ll5_c*2,    inv_l5, blockDim.x);
            vq2(ll5_c*2,    ll5_c*4,    inv_l4, blockDim.x);
            vq2(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq2(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq2(ll5_c*16,   stride,     inv_l1, blockDim.x);
        } else if (is_l5_row) {
            vq2(0,          ll5_c*2,    inv_l5, blockDim.x);
            vq2(ll5_c*2,    ll5_c*4,    inv_l4, blockDim.x);
            vq2(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq2(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq2(ll5_c*16,   stride,     inv_l1, blockDim.x);
        } else {
            /* L4 rows (row_lv==4): [0,4c)→inv_l4, then l3, l2, l1 */
            vq2(0,          ll5_c*4,    inv_l4, blockDim.x);
            vq2(ll5_c*4,    ll5_c*8,    inv_l3, blockDim.x);
            vq2(ll5_c*8,    ll5_c*16,   inv_l2, blockDim.x);
            vq2(ll5_c*16,   stride,     inv_l1, blockDim.x);
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

    /* V40: p0 always 1 (V_TILE even, V_OVERLAP odd → load_start always odd).
     * static_assert in kernel_fused_vert_dwt_tiled_ho verifies this invariant. */
    cdf97_lift_tiled<1>(col);

    /* V40: output parity: even i → H subband, odd i → L subband (with p0=1 hardcoded). */
    int hh = (height + 1) / 2;
    for (int i = V_OVERLAP; i < V_OVERLAP + (tile_end - tile_start); i++) {
        int gy = load_start + i;
        if (!(i & 1))              /* even i (with p0=1) → globally odd → H subband */
            d_dst[(hh + gy/2) * stride + x] = col[i] * NORM_H;
        else                       /* odd i → globally even → L subband */
            d_dst[(gy/2) * stride + x] = col[i] * NORM_L;
    }
}


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
    extern __shared__ float sm[];   /* 3 × width floats: smX, smY, smZ */
    float* smX = sm;
    float* smY = sm + width;
    float* smZ = sm + 2 * width;

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
        xv = fmaxf(0.f, fminf(1.f, xv));
        yv = fmaxf(0.f, fminf(1.f, yv));
        zv = fmaxf(0.f, fminf(1.f, zv));
        smX[px] = static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(xv * 4095.5f), 4095)]));
        smY[px] = static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(yv * 4095.5f), 4095)]));
        smZ[px] = static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(zv * 4095.5f), 4095)]));
    }
    __syncthreads();

    /* Phase 2-4: In-place H-DWT on each channel, then write __half output.
     * smX/smY/smZ are processed sequentially — each channel is fully lifted before
     * the next starts. smY and smZ are intact while smX is being processed, etc. */
    int w = width;
    int hw = (w + 1) / 2;

    /* Macro-style: apply CDF 9/7 lifting in-place on array 'smc', then write to 'dst' */
#define DO_HDWT_HALF(smc, dst)                                                              \
    for (int x = 1+t*2; x < w-1; x += nt*2) (smc)[x] += ALPHA*((smc)[x-1]+(smc)[x+1]);  \
    if (t==0 && w>1 && !(w&1)) (smc)[w-1] += 2.f*ALPHA*(smc)[w-2];                        \
    __syncthreads();                                                                         \
    if (t==0) (smc)[0] += 2.f*BETA*(smc)[min(1,w-1)];                                     \
    for (int x = 2+t*2; x < w; x += nt*2) (smc)[x] += BETA*((smc)[x-1]+(smc)[min(x+1,w-1)]); \
    __syncthreads();                                                                         \
    for (int x = 1+t*2; x < w-1; x += nt*2) (smc)[x] += GAMMA*((smc)[x-1]+(smc)[x+1]);  \
    if (t==0 && w>1 && !(w&1)) (smc)[w-1] += 2.f*GAMMA*(smc)[w-2];                        \
    __syncthreads();                                                                         \
    if (t==0) (smc)[0] += 2.f*DELTA*(smc)[min(1,w-1)];                                    \
    for (int x = 2+t*2; x < w; x += nt*2) (smc)[x] += DELTA*((smc)[x-1]+(smc)[min(x+1,w-1)]); \
    __syncthreads();                                                                         \
    for (int x = t*2;   x < w; x += nt*2) (dst)[y*stride + x/2]    = __float2half((smc)[x]*NORM_L); \
    for (int x = t*2+1; x < w; x += nt*2) (dst)[y*stride + hw+x/2] = __float2half((smc)[x]*NORM_H); \
    __syncthreads();

    DO_HDWT_HALF(smX, d_hx)
    DO_HDWT_HALF(smY, d_hy)
    DO_HDWT_HALF(smZ, d_hz)

#undef DO_HDWT_HALF
}


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
        v = fmaxf(0.f, fminf(1.f, v));
        sm[px] = __float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(v * 4095.5f), 4095)])));
    }
    __syncthreads();

    /* Phase 2: In-place H-DWT with fp16 arithmetic; write __half to d_hout. */
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(ALPHA)*(sm[x-1]+sm[x+1]);
    if (t==0 && w>1 && !(w&1)) sm[w-1] += __half(2.f*ALPHA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*BETA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(BETA)*(sm[x-1]+sm[min(x+1,w-1)]);
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) sm[x] += __half(GAMMA)*(sm[x-1]+sm[x+1]);
    if (t==0 && w>1 && !(w&1)) sm[w-1] += __half(2.f*GAMMA)*sm[w-2];
    __syncthreads();
    if (t==0) sm[0] += __half(2.f*DELTA)*sm[min(1,w-1)];
    for (int x = 2+t*2; x < w; x += nt*2) sm[x] += __half(DELTA)*(sm[x-1]+sm[min(x+1,w-1)]);
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
        float v = fmaxf(0.f, fminf(1.f, m0*r + m1*g + m2*b));
        sm[px] = __float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(v*4095.5f), 4095)])));
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
            float v = fmaxf(0.f, fminf(1.f, m0*r + m1*g + m2*b));
            sm[w + px] = __float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(v*4095.5f), 4095)])));
        }
    }
    __syncthreads();

    /* Phase 2: in-place CDF 9/7 H-DWT on both rows, 4 lifting passes. */
    /* Alpha: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(ALPHA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(ALPHA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0 && w>1 && !(w&1)) {
        sm[w-1]   += __half(2.f*ALPHA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*ALPHA) * sm[2*w-2];
    }
    __syncthreads();
    /* Beta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*BETA) * sm[min(1, w-1)];
        if (y1 < height) sm[w] += __half(2.f*BETA) * sm[w + min(1, w-1)];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(BETA) * (sm[x-1]     + sm[min(x+1, w-1)]);
        if (y1 < height) sm[w+x] += __half(BETA) * (sm[w+x-1] + sm[w + min(x+1, w-1)]);
    }
    __syncthreads();
    /* Gamma: odd positions */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(GAMMA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(GAMMA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0 && w>1 && !(w&1)) {
        sm[w-1]   += __half(2.f*GAMMA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*GAMMA) * sm[2*w-2];
    }
    __syncthreads();
    /* Delta: even positions */
    if (t==0) {
        sm[0] += __half(2.f*DELTA) * sm[min(1, w-1)];
        if (y1 < height) sm[w] += __half(2.f*DELTA) * sm[w + min(1, w-1)];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(DELTA) * (sm[x-1]     + sm[min(x+1, w-1)]);
        if (y1 < height) sm[w+x] += __half(DELTA) * (sm[w+x-1] + sm[w + min(x+1, w-1)]);
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
__global__ void
kernel_rgb48_xyz_hdwt0_1ch_2row_p12(
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

    /* Plane bases: each plane = height * packed_row_stride bytes */
    size_t plane_stride = (size_t)height * packed_row_stride;
    const uint8_t* r_plane = d_rgb12 + 0 * plane_stride;
    const uint8_t* g_plane = d_rgb12 + 1 * plane_stride;
    const uint8_t* b_plane = d_rgb12 + 2 * plane_stride;

    /* Helper macro: unpack one 12-bit value from a 3-byte packed pair at plane+row_off+0 */
#define UNPACK12(plane, row_off, px_odd) \
    ((px_odd) ? ((int(__ldg((plane)+(row_off)+1) & 0xF) << 8) | int(__ldg((plane)+(row_off)+2))) \
              : ((int(__ldg((plane)+(row_off)+0)) << 4) | (int(__ldg((plane)+(row_off)+1)) >> 4)))

    /* Phase 1: load row y0 into sm[0..w-1] */
    for (int px = t; px < w; px += nt) {
        int off = y0 * packed_row_stride + (px / 2) * 3;
        int o   = px & 1;
        int ri  = UNPACK12(r_plane, off, o);
        int gi  = UNPACK12(g_plane, off, o);
        int bi  = UNPACK12(b_plane, off, o);
        float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55 */
        float g = __half2float(__ldg(&d_lut_in[gi]));
        float b = __half2float(__ldg(&d_lut_in[bi]));
        float v = fmaxf(0.f, fminf(1.f, m0*r + m1*g + m2*b));
        sm[px] = __float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(v*4095.5f), 4095)])));
    }
    if (y1 < height) {
        for (int px = t; px < w; px += nt) {
            int off = y1 * packed_row_stride + (px / 2) * 3;
            int o   = px & 1;
            int ri  = UNPACK12(r_plane, off, o);
            int gi  = UNPACK12(g_plane, off, o);
            int bi  = UNPACK12(b_plane, off, o);
            float r = __half2float(__ldg(&d_lut_in[ri]));  /* V55 */
            float g = __half2float(__ldg(&d_lut_in[gi]));
            float b = __half2float(__ldg(&d_lut_in[bi]));
            float v = fmaxf(0.f, fminf(1.f, m0*r + m1*g + m2*b));
            sm[w + px] = __float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(v*4095.5f), 4095)])));
        }
    }
#undef UNPACK12
    __syncthreads();

    /* Phases 2-3: identical to kernel_rgb48_xyz_hdwt0_1ch_2row — in-place CDF 9/7 + deinterleave */
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(ALPHA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(ALPHA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0 && w>1 && !(w&1)) {
        sm[w-1]   += __half(2.f*ALPHA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*ALPHA) * sm[2*w-2];
    }
    __syncthreads();
    if (t==0) {
        sm[0] += __half(2.f*BETA) * sm[min(1, w-1)];
        if (y1 < height) sm[w] += __half(2.f*BETA) * sm[w + min(1, w-1)];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(BETA) * (sm[x-1]     + sm[min(x+1, w-1)]);
        if (y1 < height) sm[w+x] += __half(BETA) * (sm[w+x-1] + sm[w + min(x+1, w-1)]);
    }
    __syncthreads();
    for (int x = 1+t*2; x < w-1; x += nt*2) {
        sm[x]   += __half(GAMMA) * (sm[x-1]     + sm[x+1]);
        if (y1 < height) sm[w+x] += __half(GAMMA) * (sm[w+x-1] + sm[w+x+1]);
    }
    if (t==0 && w>1 && !(w&1)) {
        sm[w-1]   += __half(2.f*GAMMA) * sm[w-2];
        if (y1 < height) sm[2*w-1] += __half(2.f*GAMMA) * sm[2*w-2];
    }
    __syncthreads();
    if (t==0) {
        sm[0] += __half(2.f*DELTA) * sm[min(1, w-1)];
        if (y1 < height) sm[w] += __half(2.f*DELTA) * sm[w + min(1, w-1)];
    }
    for (int x = 2+t*2; x < w; x += nt*2) {
        sm[x]   += __half(DELTA) * (sm[x-1]     + sm[min(x+1, w-1)]);
        if (y1 < height) sm[w+x] += __half(DELTA) * (sm[w+x-1] + sm[w + min(x+1, w-1)]);
    }
    __syncthreads();
    for (int x = t*2;   x < w; x += nt*2) {
        d_hout[y0*stride + x/2]                          = sm[x]   * __half(NORM_L);
        if (y1 < height) d_hout[y1*stride + x/2]         = sm[w+x] * __half(NORM_L);
    }
    for (int x = t*2+1; x < w; x += nt*2) {
        d_hout[y0*stride + hw + x/2]                     = sm[x]   * __half(NORM_H);
        if (y1 < height) d_hout[y1*stride + hw + x/2]   = sm[w+x] * __half(NORM_H);
    }
}


/**
 * V61: 4-rows-per-block packed-12-bit RGB+HDWT0 (parity with Slang V24).
 * grid=(height+3)/4; smem=4*width*sizeof(__half).
 * Halves block count vs V54 2-row (270 vs 540 for 2K); same 100% SM occupancy.
 */
__global__ void
kernel_rgb48_xyz_hdwt0_1ch_4row_p12(
    const uint8_t*  __restrict__ d_rgb12,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_hout,
    int comp,
    int width, int height, int packed_row_stride, int stride)
{
    extern __shared__ __half sm[];
    int y0=blockIdx.x*4, y1=y0+1, y2=y0+2, y3=y0+3;
    int t=threadIdx.x, nt=blockDim.x;
    int w=width, hw=(w+1)/2;

    int mr=comp*3;
    float m0=__ldg(&d_matrix[mr]), m1=__ldg(&d_matrix[mr+1]), m2=__ldg(&d_matrix[mr+2]);

    size_t plane_stride=(size_t)height*packed_row_stride;
    const uint8_t* r_plane=d_rgb12+0*plane_stride;
    const uint8_t* g_plane=d_rgb12+1*plane_stride;
    const uint8_t* b_plane=d_rgb12+2*plane_stride;

#define UP12(pl, roff, odd) \
    ((odd)?((int(__ldg((pl)+(roff)+1)&0xF)<<8)|int(__ldg((pl)+(roff)+2)))\
          :((int(__ldg((pl)+(roff)+0))<<4)|(int(__ldg((pl)+(roff)+1))>>4)))

    /* V68: Fuse 4-row loads into if/else block for maximum load MLP.
     * Interior (y3<height, 100% of 2K blocks): single for loop issues all 4 rows'
     *   12 texture loads per pixel simultaneously — GPU MLP hides all latencies at once.
     * Else (partial last block): original per-row guarded loads (y3 not loaded). */
    if (y3 < height) {
        /* Interior: fused 4-row load — 24 byte __ldg + 8 LUT __ldg per pixel in-flight. */
        for (int px=t; px<w; px+=nt) {
            int o=px&1, phalf=(px/2)*3;
            int off0=y0*packed_row_stride+phalf;
            float r0=__half2float(__ldg(&d_lut_in[UP12(r_plane,off0,o)]));
            float g0=__half2float(__ldg(&d_lut_in[UP12(g_plane,off0,o)]));
            float b0=__half2float(__ldg(&d_lut_in[UP12(b_plane,off0,o)]));
            sm[px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r0+m1*g0+m2*b0))*4095.5f),4095)])));
            int off1=y1*packed_row_stride+phalf;
            float r1=__half2float(__ldg(&d_lut_in[UP12(r_plane,off1,o)]));
            float g1=__half2float(__ldg(&d_lut_in[UP12(g_plane,off1,o)]));
            float b1=__half2float(__ldg(&d_lut_in[UP12(b_plane,off1,o)]));
            sm[w+px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r1+m1*g1+m2*b1))*4095.5f),4095)])));
            int off2=y2*packed_row_stride+phalf;
            float r2=__half2float(__ldg(&d_lut_in[UP12(r_plane,off2,o)]));
            float g2=__half2float(__ldg(&d_lut_in[UP12(g_plane,off2,o)]));
            float b2=__half2float(__ldg(&d_lut_in[UP12(b_plane,off2,o)]));
            sm[2*w+px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r2+m1*g2+m2*b2))*4095.5f),4095)])));
            int off3=y3*packed_row_stride+phalf;
            float r3=__half2float(__ldg(&d_lut_in[UP12(r_plane,off3,o)]));
            float g3=__half2float(__ldg(&d_lut_in[UP12(g_plane,off3,o)]));
            float b3=__half2float(__ldg(&d_lut_in[UP12(b_plane,off3,o)]));
            sm[3*w+px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r3+m1*g3+m2*b3))*4095.5f),4095)])));
        }
        __syncthreads();
        /* V65+V68 interior: branch-free 4-row lifting and 4× unconditional stores. */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(ALPHA)*(sm[x-1]    +sm[x+1]);
            sm[w+x]  +=__half(ALPHA)*(sm[w+x-1]  +sm[w+x+1]);
            sm[2*w+x]+=__half(ALPHA)*(sm[2*w+x-1]+sm[2*w+x+1]);
            sm[3*w+x]+=__half(ALPHA)*(sm[3*w+x-1]+sm[3*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            sm[w-1]  +=__half(2.f*ALPHA)*sm[w-2];
            sm[2*w-1]+=__half(2.f*ALPHA)*sm[2*w-2];
            sm[3*w-1]+=__half(2.f*ALPHA)*sm[3*w-2];
            sm[4*w-1]+=__half(2.f*ALPHA)*sm[4*w-2];
        }
        __syncthreads();
        if(t==0) {
            sm[0]  +=__half(2.f*BETA)*sm[min(1,w-1)];
            sm[w]  +=__half(2.f*BETA)*sm[w+min(1,w-1)];
            sm[2*w]+=__half(2.f*BETA)*sm[2*w+min(1,w-1)];
            sm[3*w]+=__half(2.f*BETA)*sm[3*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(BETA)*(sm[x-1]    +sm[min(x+1,w-1)]);
            sm[w+x]  +=__half(BETA)*(sm[w+x-1]  +sm[w+min(x+1,w-1)]);
            sm[2*w+x]+=__half(BETA)*(sm[2*w+x-1]+sm[2*w+min(x+1,w-1)]);
            sm[3*w+x]+=__half(BETA)*(sm[3*w+x-1]+sm[3*w+min(x+1,w-1)]);
        }
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(GAMMA)*(sm[x-1]    +sm[x+1]);
            sm[w+x]  +=__half(GAMMA)*(sm[w+x-1]  +sm[w+x+1]);
            sm[2*w+x]+=__half(GAMMA)*(sm[2*w+x-1]+sm[2*w+x+1]);
            sm[3*w+x]+=__half(GAMMA)*(sm[3*w+x-1]+sm[3*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            sm[w-1]  +=__half(2.f*GAMMA)*sm[w-2];
            sm[2*w-1]+=__half(2.f*GAMMA)*sm[2*w-2];
            sm[3*w-1]+=__half(2.f*GAMMA)*sm[3*w-2];
            sm[4*w-1]+=__half(2.f*GAMMA)*sm[4*w-2];
        }
        __syncthreads();
        if(t==0) {
            sm[0]  +=__half(2.f*DELTA)*sm[min(1,w-1)];
            sm[w]  +=__half(2.f*DELTA)*sm[w+min(1,w-1)];
            sm[2*w]+=__half(2.f*DELTA)*sm[2*w+min(1,w-1)];
            sm[3*w]+=__half(2.f*DELTA)*sm[3*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(DELTA)*(sm[x-1]    +sm[min(x+1,w-1)]);
            sm[w+x]  +=__half(DELTA)*(sm[w+x-1]  +sm[w+min(x+1,w-1)]);
            sm[2*w+x]+=__half(DELTA)*(sm[2*w+x-1]+sm[2*w+min(x+1,w-1)]);
            sm[3*w+x]+=__half(DELTA)*(sm[3*w+x-1]+sm[3*w+min(x+1,w-1)]);
        }
        __syncthreads();
        for (int x=t*2; x<w; x+=nt*2) {
            d_hout[y0*stride+x/2] = sm[x]     *__half(NORM_L);
            d_hout[y1*stride+x/2] = sm[w+x]   *__half(NORM_L);
            d_hout[y2*stride+x/2] = sm[2*w+x] *__half(NORM_L);
            d_hout[y3*stride+x/2] = sm[3*w+x] *__half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            d_hout[y0*stride+hw+x/2] = sm[x]     *__half(NORM_H);
            d_hout[y1*stride+hw+x/2] = sm[w+x]   *__half(NORM_H);
            d_hout[y2*stride+hw+x/2] = sm[2*w+x] *__half(NORM_H);
            d_hout[y3*stride+hw+x/2] = sm[3*w+x] *__half(NORM_H);
        }
    } else {
        /* Partial last block: guarded per-row loads (y3 not loaded), conditional lifting. */
        for (int px=t; px<w; px+=nt) {
            int off=y0*packed_row_stride+(px/2)*3, o=px&1;
            float r=__half2float(__ldg(&d_lut_in[UP12(r_plane,off,o)]));
            float g=__half2float(__ldg(&d_lut_in[UP12(g_plane,off,o)]));
            float b=__half2float(__ldg(&d_lut_in[UP12(b_plane,off,o)]));
            sm[px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r+m1*g+m2*b))*4095.5f),4095)])));
        }
        if(y1<height) for(int px=t; px<w; px+=nt) {
            int off=y1*packed_row_stride+(px/2)*3, o=px&1;
            float r=__half2float(__ldg(&d_lut_in[UP12(r_plane,off,o)]));
            float g=__half2float(__ldg(&d_lut_in[UP12(g_plane,off,o)]));
            float b=__half2float(__ldg(&d_lut_in[UP12(b_plane,off,o)]));
            sm[w+px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r+m1*g+m2*b))*4095.5f),4095)])));
        }
        if(y2<height) for(int px=t; px<w; px+=nt) {
            int off=y2*packed_row_stride+(px/2)*3, o=px&1;
            float r=__half2float(__ldg(&d_lut_in[UP12(r_plane,off,o)]));
            float g=__half2float(__ldg(&d_lut_in[UP12(g_plane,off,o)]));
            float b=__half2float(__ldg(&d_lut_in[UP12(b_plane,off,o)]));
            sm[2*w+px]=__float2half(static_cast<float>(__ldg(&d_lut_out[min(static_cast<int>(fmaxf(0.f,fminf(1.f,m0*r+m1*g+m2*b))*4095.5f),4095)])));
        }
        __syncthreads();
        /* Partial: conditional lifting — y3 >= height, so y3 guards are always false. */
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(ALPHA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(ALPHA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(ALPHA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            sm[w-1]  +=__half(2.f*ALPHA)*sm[w-2];
            if(y1<height) sm[2*w-1]+=__half(2.f*ALPHA)*sm[2*w-2];
            if(y2<height) sm[3*w-1]+=__half(2.f*ALPHA)*sm[3*w-2];
        }
        __syncthreads();
        if(t==0) {
            sm[0]    +=__half(2.f*BETA)*sm[min(1,w-1)];
            if(y1<height) sm[w]  +=__half(2.f*BETA)*sm[w+min(1,w-1)];
            if(y2<height) sm[2*w]+=__half(2.f*BETA)*sm[2*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(BETA)*(sm[x-1]    +sm[min(x+1,w-1)]);
            if(y1<height) sm[w+x]  +=__half(BETA)*(sm[w+x-1]  +sm[w+min(x+1,w-1)]);
            if(y2<height) sm[2*w+x]+=__half(BETA)*(sm[2*w+x-1]+sm[2*w+min(x+1,w-1)]);
        }
        __syncthreads();
        for (int x=1+t*2; x<w-1; x+=nt*2) {
            sm[x]    +=__half(GAMMA)*(sm[x-1]    +sm[x+1]);
            if(y1<height) sm[w+x]  +=__half(GAMMA)*(sm[w+x-1]  +sm[w+x+1]);
            if(y2<height) sm[2*w+x]+=__half(GAMMA)*(sm[2*w+x-1]+sm[2*w+x+1]);
        }
        if(t==0&&w>1&&!(w&1)) {
            sm[w-1]  +=__half(2.f*GAMMA)*sm[w-2];
            if(y1<height) sm[2*w-1]+=__half(2.f*GAMMA)*sm[2*w-2];
            if(y2<height) sm[3*w-1]+=__half(2.f*GAMMA)*sm[3*w-2];
        }
        __syncthreads();
        if(t==0) {
            sm[0]    +=__half(2.f*DELTA)*sm[min(1,w-1)];
            if(y1<height) sm[w]  +=__half(2.f*DELTA)*sm[w+min(1,w-1)];
            if(y2<height) sm[2*w]+=__half(2.f*DELTA)*sm[2*w+min(1,w-1)];
        }
        for (int x=2+t*2; x<w; x+=nt*2) {
            sm[x]    +=__half(DELTA)*(sm[x-1]    +sm[min(x+1,w-1)]);
            if(y1<height) sm[w+x]  +=__half(DELTA)*(sm[w+x-1]  +sm[w+min(x+1,w-1)]);
            if(y2<height) sm[2*w+x]+=__half(DELTA)*(sm[2*w+x-1]+sm[2*w+min(x+1,w-1)]);
        }
        __syncthreads();
        for (int x=t*2; x<w; x+=nt*2) {
            d_hout[y0*stride+x/2]               = sm[x]     *__half(NORM_L);
            if(y1<height) d_hout[y1*stride+x/2] = sm[w+x]   *__half(NORM_L);
            if(y2<height) d_hout[y2*stride+x/2] = sm[2*w+x] *__half(NORM_L);
        }
        for (int x=t*2+1; x<w; x+=nt*2) {
            d_hout[y0*stride+hw+x/2]               = sm[x]     *__half(NORM_H);
            if(y1<height) d_hout[y1*stride+hw+x/2] = sm[w+x]   *__half(NORM_H);
            if(y2<height) d_hout[y2*stride+hw+x/2] = sm[2*w+x] *__half(NORM_H);
        }
    } /* end V68 if/else */
#undef UP12
}


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
    xv = fmaxf(0.0f, fminf(1.0f, xv));
    yv = fmaxf(0.0f, fminf(1.0f, yv));
    zv = fmaxf(0.0f, fminf(1.0f, zv));

    /* Output LUT: DCP gamma companding → int32 */
    int xi = min((int)(xv * 4095.5f), 4095);
    int yi = min((int)(yv * 4095.5f), 4095);
    int zi = min((int)(zv * 4095.5f), 4095);

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
    uint16_t* d_lut_out    = nullptr;  /* V48: 4096-entry output gamma LUT (was int32_t; saves 8KB GPU texture cache) */
    float*    d_matrix     = nullptr;  /* 9-float Bradford+RGB→XYZ matrix */
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

    bool init() {
        for (int c = 0; c < 3; ++c) {
            if (cudaStreamCreate(&stream[c]) != cudaSuccess) return false;
        }
        if (cudaStreamCreate(&st_h2d) != cudaSuccess) return false;
        for (int i = 0; i < 2; ++i)
            if (cudaEventCreateWithFlags(&h2d_done[i], cudaEventDisableTiming) != cudaSuccess) return false;
        /* V59: prefer L1 cache over shared memory for memory-BW-bound kernels.
         * V-DWT tiled: no smem — larger L1 lets adjacent tiles share loaded rows.
         * RGB+HDWT0: smem=7.5KB (< 16KB L1 limit) — 48KB L1 fits both 8KB LUTs. */
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_tiled,     cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_vert_dwt_tiled_ho,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch_4row_p12, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch_2row_p12, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_rgb48_xyz_hdwt0_1ch,          cudaFuncCachePreferL1);
        /* V67: PreferL1 for H-DWT levels-1+ kernels. smem ≤ 7.5KB < 16KB limit for all levels.
         * 48KB L1 caches __ldg-loaded V-DWT output rows → better hit rate for 4-row stride. */
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io_4row,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io_2row,  cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(kernel_fused_horz_dwt_half_io,       cudaFuncCachePreferL1);
        return true;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;

        cleanup_dwt_buffers();

        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a[c],  pixels * sizeof(__half));  /* V36: half V-DWT output */
            cudaMalloc(&d_b[c],  pixels * sizeof(__half));  /* V26: half H-DWT output */
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
        for (int i = 0; i < 2; ++i)
            cudaMallocHost(&h_rgb12_pinned[i], packed_size);
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
        if (!d_lut_in)  cudaMalloc(&d_lut_in,  4096 * sizeof(__half));  /* V55: was float (16KB→8KB) */
        if (!d_lut_out) cudaMalloc(&d_lut_out, 4096 * sizeof(uint16_t));  /* V48: was int32_t */
        if (!d_matrix)  cudaMalloc(&d_matrix,  9    * sizeof(float));

        /* V55: convert float→__half before upload; host array stays float for precision */
        __half h_lut_in_tmp[4096];
        for (int i = 0; i < 4096; ++i) h_lut_in_tmp[i] = __float2half(p.lut_in[i]);
        cudaMemcpy(d_lut_in,  h_lut_in_tmp, 4096 * sizeof(__half),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_out, p.lut_out, 4096 * sizeof(uint16_t), cudaMemcpyHostToDevice);  /* V48 */
        cudaMemcpy(d_matrix,  p.matrix,  9    * sizeof(float),   cudaMemcpyHostToDevice);
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
            kernel_fused_i2f_horz_dwt_half_out<<<height, H_THREADS, smem, st>>>(
                d_input, d_half_h, width, height, stride);
        } else {
            /* V47: 4-rows-per-block for levels 1+; grid quartered vs V45, halved vs V46.
             * V73: adaptive thread count — fixed 512T wastes 50-77% threads at small widths.
             *   level-3 2K (w=240): 512T→256T (93.75%); level-4 2K (w=120): 512T→128T (93.75%).
             *   level-3 4K (w=480): 512T→256T; level-4 4K (w=240): 256T; level-5 4K (w=120): 128T. */
            int h_blk = (width > 480) ? H_THREADS :
                        (width > 240) ? 256 :
                        (width > 120) ? 128 : 64;
            kernel_fused_horz_dwt_half_io_4row<<<(height+3)/4, h_blk, 4*smem, st>>>(
                d_half_a, d_half_h, width, height, stride);
        }
    }

    /* Step 2+3: V-DWT reads __half d_half_h, writes __half d_half_a (V36).
     * V27: h ≤ MAX_REG_HEIGHT → register-blocked (128T); V62: h > → tiled (256T, V63). */
    if (height <= MAX_REG_HEIGHT) {
        kernel_fused_vert_dwt_fp16_hi_reg_ho<<<grid_v, V_THREADS_REG, 0, st>>>(
            d_half_h, d_half_a, width, height, stride);
    } else {
        dim3 v_grid2d((width + V_THREADS_TILED - 1) / V_THREADS_TILED,
                      (height + V_TILE - 1) / V_TILE);
        kernel_fused_vert_dwt_tiled_ho<<<v_grid2d, V_THREADS_TILED, 0, st>>>(
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
static void
rebuild_comp_graphs(
    CudaJ2KEncoderImpl* impl,
    int width, int height, size_t per_comp, bool is_4k, bool is_3d, bool skip_level0_hdwt)
{
    impl->destroy_comp_graphs();
    float base_step = is_4k ? 16.25f : 32.5f;
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
    float base_step = is_4k ? 16.25f : 32.5f;
    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;  /* V49: 6-level DWT for 4K */
    /* V61: 4-row kernel: smem=4×width halves; grid=(height+3)/4 blocks.
     * V72: Conditionally use 2-row at 4K (width>2048) to enable PreferL1 (48KB L1).
     *   4K 4-row smem=30.72KB > 16KB PreferL1 → runtime PreferShared (16KB L1, LUT evictions).
     *   4K 2-row smem=15.36KB < 16KB → PreferL1 honored (48KB L1, all LUTs cached). */
    size_t ch_smem_4row = static_cast<size_t>(4 * width) * sizeof(__half);
    size_t ch_smem_2row = static_cast<size_t>(2 * width) * sizeof(__half);
    int rgb_grid = (height + 3) / 4;
    int rgb_grid_2row = (height + 1) / 2;
    bool use_2row_rgb = (width > 2048);  /* V72: 4K uses 2-row for PreferL1 */
    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.1f);
        cudaGraph_t g;
        cudaStreamBeginCapture(impl->stream[c], cudaStreamCaptureModeThreadLocal);
        /* V44: event wait baked into graph — waits for h2d_done[buf] before RGB kernel */
        cudaStreamWaitEvent(impl->stream[c], impl->h2d_done[buf], 0);
        int packed_row_stride = (width / 2) * 3;  /* bytes per channel per row */
        if (use_2row_rgb) {
            /* V72: 4K — 2-row kernel enables PreferL1 (smem=15.36KB < 16KB limit). */
            kernel_rgb48_xyz_hdwt0_1ch_2row_p12<<<rgb_grid_2row, H_THREADS_FUSED, ch_smem_2row, impl->stream[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_b[c], c,
                width, height, packed_row_stride, width);
        } else {
            /* V61: 2K — 4-row kernel (smem=15.36KB < 16KB, PreferL1 already works). */
            kernel_rgb48_xyz_hdwt0_1ch_4row_p12<<<rgb_grid, H_THREADS_FUSED, ch_smem_4row, impl->stream[c]>>>(
                impl->d_rgb12[buf],
                impl->d_lut_in, impl->d_lut_out, impl->d_matrix,
                impl->d_b[c], c,
                width, height, packed_row_stride, width);
        }
        /* DWT levels 1+ + quantize + D2H: reads d_b[c], writes h_packed_pinned[buf] */
        launch_comp_pipeline(impl, c, width, height, per_comp, step, true,
                             impl->stream[c], impl->h_packed_pinned[buf],
                             num_levels);  /* V49 */
        cudaStreamEndCapture(impl->stream[c], &g);
        cudaGraphInstantiate(&impl->cg_v42[buf][c], g, nullptr, nullptr, 0);
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
        cs.write_u8(3); cs.write_u8(3); cs.write_u8(0x00); cs.write_u8(0x00);
        cs.write_u8(0x77);
        for (int i = 1; i < num_precincts; ++i) cs.write_u8(0x88);
    }

    /* QCD — V57: per-subband step matching V53 perceptual weights (Y component, base=32.5).
     * V49: nsb = 19 for 4K, 16 for 2K. */
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;
        cs.write_marker(J2K_QCD);
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        cs.write_u8(0x22);
        float base_y = is_4k ? 16.25f : 32.5f;
        for (int i = 0; i < nsb; ++i)
            cs.write_u16(j2k_perceptual_sb_entry(base_y, i, is_4k));
    }

    /* QCC for X (comp 0) and Z (comp 2) — V57: per-subband step (X/Z base = base_y×1.1). */
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;
        uint16_t lqcc = static_cast<uint16_t>(4 + 2 * nsb);
        float base_xz = (is_4k ? 16.25f : 32.5f) * 1.1f;
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
        cs.write_u8(0x00);                       /* SPcod: no bypass/reset/terminate */
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
     * Base step = 32.5 (2K) or 16.25 (4K). 4K uses uniform (no perceptual
     * weights applied in 4K path). */
    {
        cs.write_marker(J2K_QCD);
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;  /* V49: 19 for 4K, 16 for 2K */
        cs.write_u16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        cs.write_u8(0x22);  /* Sqcd: scalar expounded, 1 guard bit */
        float base_y = is_4k ? 16.25f : 32.5f;
        for (int i = 0; i < nsb; ++i)
            cs.write_u16(j2k_perceptual_sb_entry(base_y, i, is_4k));
    }

    /* V24/V57: QCC markers — per-component quantization step overrides for X and Z.
     *
     * X and Z components use 1.1× coarser quantization than Y.
     * V57: each subband entry uses the perceptual weight for that subband level. */
    {
        int nsb = 3 * (is_4k ? 6 : NUM_DWT_LEVELS) + 1;  /* V49: 19 for 4K, 16 for 2K */
        uint16_t lqcc = static_cast<uint16_t>(4 + 2 * nsb);
        float base_xz = (is_4k ? 16.25f : 32.5f) * 1.1f;
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
        static constexpr uint16_t J2K_TLM = 0xFF55;
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
    if (!v42_valid)
        rebuild_v42_comp_graphs(_impl.get(), new_buf,
                                width, height, v54_packed_stride,
                                per_comp, is_4k, is_3d);

    /* Step 4: Sync all 3 component streams for cur_buf (V49: was only stream[0]).
     * V49 race fix: all 3 streams write to h_packed_pinned[cur_buf]; syncing only
     * stream[0] could race with stream[1] and stream[2]'s D2H still in flight. */
    std::vector<uint8_t> result;
    if (_impl->pipeline_active) {
        cudaStreamSynchronize(_impl->stream[0]);
        cudaStreamSynchronize(_impl->stream[1]);  /* V49: race fix */
        cudaStreamSynchronize(_impl->stream[2]);  /* V49: race fix */
    }

    /* Step 5: Launch new_buf's graphs BEFORE codestream building (V49 early launch).
     * Graphs wait on h2d_done[new_buf] event — GPU starts as soon as H2D fires.
     * Starting GPU 0.1ms earlier reduces sync_wait for the next frame. */
    for (int c = 0; c < 3; ++c)
        cudaGraphLaunch(_impl->cg_v42[new_buf][c], _impl->stream[c]);

    /* Step 6: Build codestream for cur_buf on CPU while GPU computes new_buf. */
    if (_impl->pipeline_active) {
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
