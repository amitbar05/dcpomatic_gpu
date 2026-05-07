# DCP GPU JPEG2000 Encoder — Correctness Roadmap

## Current State (V211)

All changes target the **correct mode** (`fast_mode=false`) using FP32 DWT + full EBCOT T1/T2.
Fast mode remains with FP16 DWT + approximations (not part of this roadmap).

### Already Verified Correct
- MQ coder: 47-entry probability table matches T.800 Table C.2 exactly
- MQ init/flush/renorme/byteout/restart: matches OpenJPEG reference
- Context model: ZC (9 contexts), SC (5), MR (3), AGG, UNI — all correct
- Bit-plane coding: SPP → MRP → CUP per T.800, run-length coding correct
- Bypass mode: MQ termination with SETBITS, raw-bit coding with 0xFF stuffing
- DWT: CDF 9/7 coefficients, normalization constants, symmetric extension
- ICT: applied in pixel domain BEFORE DWT, coefficients match T.800 Annex G
- Dead-zone quantization: `__float2int_rd` (floor) per T.800 Annex E
- COD: CPRL progression, MCT=1, correct levels/CB-size/bypass/filter
- QCD: Rb=12 eps/man encoding, correct subband order
- TLM/SOT/SOD/EOC: correct multi-tile-part structure
- Tag trees: full hierarchical encoding matching `opj_tgt_encode`
- Packet headers: empty-flag, inclusion, ZBP, np (Table B.4), CB-length (Lblock+log2)
- BitWriter: correct 0xFF byte stuffing
- Rate control: sequential per-component truncation with pass-length data

---

## Remaining Issues (Priority Order)

### CRITICAL — DCP Compliance

#### #1: Scod=0x00 → Scod=0x01 with precinct partition
**File:** `src/lib/gpu_ebcot_t2.h` (COD marker, packet assembly)  
**Standard:** SMPTE 429-4 S6.3 requires `Scod=0x01` (precinct partition enabled)  
**Current:** `Scod=0x00` (default 2^15x2^15 precincts, one precinct per resolution)  
**Fix plan:**
1. Set `Scod=0x01` in COD, add 1 precinct-size byte per resolution to COD length
2. Compute precinct dimensions: LL = 2^(PPx-15)x2^(PPy-15) of image, others = 2^PPx x 2^PPy
3. DCP convention: PPx=PPy=7 for LL (128x128 image-unit precincts), 8 for other subbands (256x256)
4. Update `build_tp` to split each resolution into multiple packets by precinct grid
5. Update TLM/SOT/SOD to account for more packets per tile-part
6. Test: verify OpenJPEG decodes with Scod=0x01 correctly

---

### HIGH — Missing Tests

#### #2: ICT Correctness Test
**New file:** `test/ict_correctness.cu`  
**Purpose:** Verify ICT forward+inverse produces identity (within FP32 tolerance)  
**Test plan:**
- Generate known XYZ test vectors (0, 2048, 4095, gradients, random)
- Apply GPU ICT forward kernel
- Apply CPU inverse ICT (matching T.800 Annex G.2.2)
- Verify roundtrip error < 0.5 LSB per component
- Test edge cases: all-zero, all-max, single-impulse, checkerboard

#### #3: Bypass Mode Test
**New file:** `test/bypass_correctness.cu`  
**Purpose:** Verify T1 produces bit-identical output with bypass ON vs OFF for synthetic coefficients  
**Test plan:**
- Create synthetic DWT coefficient arrays with known structure
- Run `kernel_ebcot_t1` with `use_bypass=false` (MQ-only)
- Run `kernel_ebcot_t1` with `use_bypass=true` (MQ+bypass)
- Verify: decoded bits are identical (bypass is lossless — same data, different packing)
- Verify: bypass output is NOT larger than MQ-only (bypass is more efficient for lower bit-planes)

#### #4: Tag Tree Unit Test
**New file:** `test/tagtree_correctness.cc`  
**Purpose:** Verify `TagTree::encode()` matches reference implementation  
**Test plan:**
- Build tag trees with known leaf values
- Encode each leaf against various thresholds
- Compare output bit sequence against CPU reference (or OpenJPEG `opj_tgt_encode`)
- Test edge cases: single leaf, 1xN, Nx1, full 32x32
- Test propagation: verify min-value bubbles to root correctly
- Test re-entry: verify known nodes are skipped on subsequent encodes

#### #5: Packet Header 0xFF Stuffing Test
**New file:** `test/bitwriter_correctness.cc`  
**Purpose:** Verify `BitWriter` produces correct byte-stuffed output  
**Test plan:**
- Write sequences known to produce 0xFF bytes
- Verify stuffing bytes (0x00) are inserted after 0xFF
- Verify `prev_ff` tracking across byte boundaries
- Test flush() behavior with partial bytes and prev_ff
- Compare against reference byte-stuffing implementation

#### #6: QCD Subband Order Test
**New file:** `test/qcd_order_test.cc`  
**Purpose:** Verify QCD subband steps are in correct standard order  
**Test plan:**
- Encode a frame with known per-subband step values
- Parse the J2K codestream, extract QCD step entries
- Verify order: LL, then for each level (coarsest to finest): HL, LH, HH
- Verify step values match those from `build_codeblock_table`

#### #7: Full J2K Roundtrip Test (encode+decode validation)
**New file:** `test/roundtrip_correctness.cu`  
**Purpose:** End-to-end test: encode synthetic patterns, decode with OpenJPEG, compare pixels  
**Test plan:**
- Encode 2K (1998x1080) and 4K (3996x2160) frames via `encode_ebcot` (correct mode)
- Decode with OpenJPEG `opj_decompress`
- Compare decoded XYZ values against original (after ICT inverse)
- Measure PSNR per component — target > 50 dB for gradient, > 60 dB for flat
- Test patterns: flat (mid-gray, black, white), sine gradient, checkerboard (8px, 64px), noise
- Verify: no OpenJPEG warnings/errors during decode
- Verify: XML assetmap and CPL can be generated from output

---

### MEDIUM — Quality & Completeness

#### #8: Global Rate Control (vs per-component)
**File:** `src/lib/gpu_ebcot_t2.h`  
**Current:** Each component gets `target_bytes/3`  
**Issue:** If component 0 uses only 50% budget, remaining 50% is wasted  
**Fix plan:**
1. Remove `per_comp_target`, use global `remaining_bytes`
2. Process all 3 components in interleaved order
3. Allocate bytes to whichever component's next most-important code-block pass has the best rate-distortion slope
4. Or simpler: proportional allocation — compute total coded bytes for all 3 components, then scale each to target_bytes proportionally

#### #9: Adaptive Base-Step Tuning
**File:** `src/lib/cuda_j2k_encoder.cu` (adaptive retry in encode_ebcot)  
**Current:** Retry with 0.5x step when byte_ratio < 0.55 and > 0.005  
**Issue:** Thresholds are hardcoded; may not be optimal for all content  
**Fix plan:**
1. Make adaptive_thresh_low/high configurable
2. Add more retry attempts (up to 3) with finer step adjustments (0.75x, 0.5x, 0.25x)
3. Add guard: never go below minimum step (avoids num_bp overflow)

#### #10: DWT Level 5 in Fast Mode
**File:** `src/lib/cuda_j2k_encoder.cu`  
**Current:** Fast mode 2K uses 4 DWT levels instead of 5  
**Issue:** SMPTE 429-4 requires 5 levels for 2K  
**Fix plan:**
1. Change `num_levels = is_4k ? 6 : (fast_mode ? 4 : NUM_DWT_LEVELS)` to always use 5 for 2K
2. Benchmark: measure perf impact (~5% slower expected)
3. If too slow, make fast-mode level count configurable

#### #11: Edge-Case Code-Block Tests
**New test:** Add to `test/edge_case_tests.cu`  
**Purpose:** Test code-blocks with partial dimensions (width < 32 or height < 32)  
**Test plan:**
- Images with odd dimensions producing boundary CBs of size 1x32, 32x1, 1x1
- Verify T1 kernel handles these without out-of-bounds access
- Verify tag trees work with non-square leaf grids (ncbx != ncby)
- Verify packet headers correctly code partial CBs

---

### LOW — Optional Improvements

#### #12: ICT FP32 LUT Path
**File:** `src/lib/cuda_j2k_encoder.cu`  
**Current:** `kernel_rgb48_to_xyz12` uses `__half` LUT even in correct mode  
**Issue:** ~0.05% precision loss from FP16 LUT quantization  
**Fix:** Use `kernel_rgb48_to_xyz12` with `d_lut_in_f32` for correct mode (or create FP32 variant)

#### #13: 4K EBCOT Test
**New test:** Add 4K test patterns to verify 6 DWT levels, different TLM/SOT structure  

#### #14: 3D Stereo Mode Test
**New test:** Verify half-bitrate encoding works correctly with `is_3d=true`

#### #15: CB Overflow Stress Test
**New test:** Generate content that forces CB_BUF_SIZE overflow, verify graceful handling

#### #16: MQ Reset Between Code-Blocks
**File:** `src/lib/gpu_ebcot.h`  
**Current:** MQ contexts are re-initialized per code-block (via new MQCoder per CB)  
**Standard:** T.800 says contexts are reset between code-blocks (already correct)  
**Verify:** Add assertion in T1 kernel that context array is properly initialized

---

## Implementation Order

```
Phase 1 — Critical Fixes
+-- #1  Scod=0x01 + precinct partition

Phase 2 — Correctness Tests
+-- #4  Tag tree unit test
+-- #5  BitWriter 0xFF stuffing test
+-- #6  QCD subband order test
+-- #2  ICT correctness test
+-- #3  Bypass mode test
+-- #7  Full roundtrip test (encode+decode)

Phase 3 — Quality
+-- #8  Global rate control
+-- #9  Adaptive base-step tuning
+-- #10 DWT level 5 in fast mode
+-- #11 Edge-case CB tests

Phase 4 — Polish
+-- #12 ICT FP32 LUT
+-- #13 4K test
+-- #14 3D test
+-- #15 CB overflow test
+-- #16 MQ reset verification
```

---

## Test Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `test/verify_correctness.cu` | Full encode + J2K validation | Done (960x540) |
| `test/mq_correctness.cu` | MQ encode/decode roundtrip | Done |
| `test/dwt_correctness.cu` | DWT GPU vs CPU reference | Done |
| `test/verify_ebcot_correctness.cu` | EBCOT T1+T2 + OpenJPEG decode | Done |
| `test/edge_case_tests.cu` | Color conversion, odd dims, extremes | Done |
| `test/ict_correctness.cu` | ICT forward+inverse roundtrip | DONE |
| `test/bypass_correctness.cu` | Bypass mode vs MQ-only | DONE |
| `test/tagtree_correctness.cc` | Tag tree vs reference | DONE |
| `test/bitwriter_correctness.cc` | BitWriter 0xFF stuffing | DONE |
| `test/qcd_order_test.cc` | QCD subband order | #6 - TODO |
| `test/roundtrip_correctness.cu` | Full encode+decode pixel compare | #7 - TODO |
