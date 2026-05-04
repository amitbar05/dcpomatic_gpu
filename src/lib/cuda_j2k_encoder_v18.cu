/*
    Copyright (C) 2024-2025 DCP-o-matic contributors

    GPU-accelerated JPEG2000 encoder using CUDA — V18 (CUDA Graphs + All Optimizations).

    V18 combines all improvements from V16 plus the main encoder optimizations:
    1.  V16's LRU-cached code-block table (lookup_cb_table/store_cb_table)
    2.  V16's consolidated D2H (ebcot_d2h_consolidated) — 2 memcpys per component
    3.  V16's pre-allocated EBCOT pool (ensure_ebcot_pool)
    4.  V16's parallel CPU T2 assembly (std::async per component)
    5.  V196: FP32 DWT support for correct mode (gpu_dwt97_level_fp32)
    6.  V199: Adaptive retry (halve step, re-encode if under target)
    7.  V200: Pre-allocated pool with 25% headroom
    8.  V198: PCRD-OPT slope-sort rate allocation in T2
    9.  V42+: CUDA Graph capture for the DWT+quantize+EBCOT_T1 pipeline
        (rebuild_v42_comp_graphs + cudaGraphLaunch per frame)
    10. V47: 4-row kernel dispatch for H-DWT levels 1+
    11. V165: BYPASS mode support in EBCOT T1
    12. graphs_failed fallback for when graph capture fails

    This file is self-contained: all MQ coder, bypass coder, EBCOT T1 kernel,
    and T2 assembly are included inline.  DWT kernel helpers (gpu_dwt97_level,
    gpu_dwt97_level_fp32) are forward-declared and provided by the main
    cuda_j2k_encoder.cu during compilation.
*/

#include "cuda_j2k_encoder.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <mutex>
#include <future>
#include <chrono>
#include <array>
#include <cstdint>

/* ============================================================================
 * SECTION 1: Constants
 * ============================================================================ */

static constexpr int NUM_DWT_LEVELS = 5;
static constexpr float ALPHA = -1.586134342f, BETA = -0.052980118f;
static constexpr float GAMMA = 0.882911075f, DELTA = 0.443506852f;
static constexpr uint16_t J2K_SOC = 0xFF4F, J2K_SIZ = 0xFF51, J2K_COD = 0xFF52;
static constexpr uint16_t J2K_QCD = 0xFF5C, J2K_SOT = 0xFF90, J2K_SOD = 0xFF93, J2K_EOC = 0xFFD9;
static constexpr int MAX_REG_HEIGHT = 140;

static constexpr float NORM_L = 0.812893197535108f;
static constexpr float NORM_H = 1.230174104914001f;

static constexpr int V_TILE    = 28;
static constexpr int V_OVERLAP  = 5;
static constexpr int V_TILE_FL  = V_TILE + 2 * V_OVERLAP;
static constexpr int H_THREADS_FUSED = 512;

/* EBCOT constants (from gpu_ebcot.h) */
static constexpr int CB_DIM      = 32;
static constexpr int CB_PIXELS   = CB_DIM * CB_DIM;
static constexpr int STRIPE_H    = 4;
static constexpr int MAX_BPLANES = 16;
static constexpr int MAX_PASSES  = MAX_BPLANES * 3;
static constexpr int CB_BUF_SIZE = 2048;

static constexpr int SUBBAND_LL = 0;
static constexpr int SUBBAND_HL = 1;
static constexpr int SUBBAND_LH = 2;
static constexpr int SUBBAND_HH = 3;

static constexpr int T1_CTXNO_ZC  = 0;
static constexpr int T1_CTXNO_SC  = 9;
static constexpr int T1_CTXNO_MR  = 14;
static constexpr int T1_CTXNO_AGG = 17;
static constexpr int T1_CTXNO_UNI = 18;
static constexpr int T1_NUM_CTXS  = 19;

/* J2K T2 marker constants */
static constexpr uint16_t J2K_SOC_M = 0xFF4F;
static constexpr uint16_t J2K_SIZ_M = 0xFF51;
static constexpr uint16_t J2K_COD_M = 0xFF52;
static constexpr uint16_t J2K_QCD_M = 0xFF5C;
static constexpr uint16_t J2K_QCC_M = 0xFF5D;
static constexpr uint16_t J2K_TLM_M = 0xFF55;
static constexpr uint16_t J2K_SOT_M = 0xFF90;
static constexpr uint16_t J2K_SOD_M = 0xFF93;
static constexpr uint16_t J2K_EOC_M = 0xFFD9;


/* ============================================================================
 * SECTION 2: MQ Coder (inlined from gpu_ebcot.h)
 * ============================================================================ */

/* Packed MQ table entry */
#define MQ_PACK(qe, nmps, nlps, sw) \
    (uint32_t)((qe) | ((uint32_t)(nmps) << 16) | ((uint32_t)(nlps) << 22) | ((uint32_t)(sw) << 28))

__device__ __constant__ static const uint32_t MQ_TABLE[47] = {
    MQ_PACK(0x5601, 1, 1,1), MQ_PACK(0x3401, 2, 6,0), MQ_PACK(0x1801, 3, 9,0), MQ_PACK(0x0AC1, 4,12,0),
    MQ_PACK(0x0521, 5,29,0), MQ_PACK(0x0221,38,33,0), MQ_PACK(0x5601, 7, 6,1), MQ_PACK(0x5401, 8,14,0),
    MQ_PACK(0x4801, 9,14,0), MQ_PACK(0x3801,10,14,0), MQ_PACK(0x3001,11,17,0), MQ_PACK(0x2401,12,18,0),
    MQ_PACK(0x1C01,13,20,0), MQ_PACK(0x1601,29,21,0), MQ_PACK(0x5601,15,14,1), MQ_PACK(0x5401,16,14,0),
    MQ_PACK(0x5101,17,15,0), MQ_PACK(0x4801,18,16,0), MQ_PACK(0x3801,19,17,0), MQ_PACK(0x3401,20,18,0),
    MQ_PACK(0x3001,21,19,0), MQ_PACK(0x2801,22,19,0), MQ_PACK(0x2401,23,20,0), MQ_PACK(0x2201,24,21,0),
    MQ_PACK(0x1C01,25,22,0), MQ_PACK(0x1801,26,23,0), MQ_PACK(0x1601,27,24,0), MQ_PACK(0x1401,28,25,0),
    MQ_PACK(0x1201,29,26,0), MQ_PACK(0x1101,30,27,0), MQ_PACK(0x0AC1,31,28,0), MQ_PACK(0x09C1,32,29,0),
    MQ_PACK(0x08A1,33,30,0), MQ_PACK(0x0521,34,31,0), MQ_PACK(0x0441,35,32,0), MQ_PACK(0x02A1,36,33,0),
    MQ_PACK(0x0221,37,34,0), MQ_PACK(0x0141,38,35,0), MQ_PACK(0x0111,39,36,0), MQ_PACK(0x0085,40,37,0),
    MQ_PACK(0x0049,41,38,0), MQ_PACK(0x0025,42,39,0), MQ_PACK(0x0015,43,40,0), MQ_PACK(0x0009,44,41,0),
    MQ_PACK(0x0005,45,42,0), MQ_PACK(0x0001,45,43,0), MQ_PACK(0x5601,46,46,0)
};

struct MQCoder {
    uint32_t A;
    uint32_t C;
    int      CT;
    uint8_t* bp;
    uint8_t* start;
    uint8_t  ctx_packed[T1_NUM_CTXS];
};

__device__ static void mq_init(MQCoder* mq, uint8_t* buf) {
    mq->A  = 0x8000;
    mq->C  = 0;
    mq->CT = 12;
    buf[0] = 0;
    mq->bp    = buf;
    mq->start = buf;
    for (int i = 0; i < T1_NUM_CTXS; i++) mq->ctx_packed[i] = 0;
    mq->ctx_packed[0]            = 4 << 1;
    mq->ctx_packed[T1_CTXNO_AGG] = 3 << 1;
    mq->ctx_packed[T1_CTXNO_UNI] = 46 << 1;
}

__device__ static void mq_byteout(MQCoder* mq) {
    if (*mq->bp == 0xFF) {
        mq->bp++;
        *mq->bp = static_cast<uint8_t>(mq->C >> 20);
        mq->C &= 0xFFFFF;
        mq->CT = 7;
    } else {
        if (mq->C & 0x8000000) {
            (*mq->bp)++;
            mq->C &= 0x7FFFFFF;
            if (*mq->bp == 0xFF) {
                mq->bp++;
                *mq->bp = static_cast<uint8_t>(mq->C >> 20);
                mq->C &= 0xFFFFF;
                mq->CT = 7;
            } else {
                mq->bp++;
                *mq->bp = static_cast<uint8_t>(mq->C >> 19);
                mq->C &= 0x7FFFF;
                mq->CT = 8;
            }
        } else {
            mq->bp++;
            *mq->bp = static_cast<uint8_t>(mq->C >> 19);
            mq->C &= 0x7FFFF;
            mq->CT = 8;
        }
    }
}

__device__ static void mq_renorme(MQCoder* mq) {
    do {
        mq->A <<= 1;
        mq->C <<= 1;
        mq->CT--;
        if (mq->CT == 0) mq_byteout(mq);
    } while (mq->A < 0x8000);
}

__device__ static void mq_encode(MQCoder* mq, int ctx, int d) {
    uint8_t  packed = mq->ctx_packed[ctx];
    uint32_t entry  = MQ_TABLE[packed >> 1];
    uint16_t qe     = static_cast<uint16_t>(entry);
    int      mpsv   = packed & 1;
    mq->A -= qe;
    if (d != mpsv) {
        if (mq->A < qe) { mq->C += qe; }
        else { mq->A = qe; }
        uint8_t nlps    = static_cast<uint8_t>((entry >> 22) & 0x3F);
        uint8_t switchf = static_cast<uint8_t>((entry >> 28) & 1);
        mq->ctx_packed[ctx] = static_cast<uint8_t>((nlps << 1) | (mpsv ^ switchf));
        mq_renorme(mq);
    } else {
        if (mq->A < 0x8000) {
            if (mq->A < qe) { mq->A = qe; }
            else { mq->C += qe; }
            uint8_t nmps = static_cast<uint8_t>((entry >> 16) & 0x3F);
            mq->ctx_packed[ctx] = static_cast<uint8_t>((nmps << 1) | mpsv);
            mq_renorme(mq);
        } else { mq->C += qe; }
    }
}

__device__ static void mq_flush(MQCoder* mq) {
    uint32_t tempc = mq->C + mq->A;
    mq->C |= 0xFFFF;
    if (mq->C >= tempc) mq->C -= 0x8000;
    mq->C <<= mq->CT; mq_byteout(mq);
    mq->C <<= mq->CT; mq_byteout(mq);
    if (*mq->bp == 0xFF) mq->bp--;
}

__device__ static void mq_align(MQCoder* mq) {
    mq->C <<= mq->CT; mq_byteout(mq);
    mq->C <<= mq->CT; mq_byteout(mq);
    if (*mq->bp == 0xFF) mq->bp--;
}

__device__ static void mq_restart(MQCoder* mq, uint8_t* new_bp) {
    mq->bp = new_bp - 1;
    mq->A  = 0x8000;
    mq->C  = 0;
    mq->CT = 12;
}


/* ============================================================================
 * SECTION 3: Bypass Coder (inlined from gpu_ebcot.h)
 * ============================================================================ */

struct BypassCoder {
    uint8_t* bp;
    uint8_t  accum;
    int      cnt;
};

__device__ static void bypass_write(BypassCoder* bc, int bit) {
    bc->accum = static_cast<uint8_t>((bc->accum << 1) | bit);
    if (++bc->cnt == 8) {
        *bc->bp = bc->accum;
        if (bc->accum == 0xFF) { bc->bp++; *bc->bp = 0x00; }
        bc->bp++;
        bc->accum = 0;
        bc->cnt   = 0;
    }
}

__device__ static void bypass_flush(BypassCoder* bc) {
    if (bc->cnt > 0) {
        *bc->bp++ = static_cast<uint8_t>(bc->accum << (8 - bc->cnt));
        bc->cnt   = 0;
    }
}

__device__ static void bypass_write_bits(BypassCoder* bc, uint8_t bits, int n) {
    int space = 8 - bc->cnt;
    if (n <= space) {
        bc->accum = static_cast<uint8_t>((bc->accum << n) | bits);
        bc->cnt += n;
        if (bc->cnt == 8) {
            *bc->bp = bc->accum;
            if (bc->accum == 0xFF) { bc->bp++; *bc->bp = 0x00; }
            bc->bp++;
            bc->accum = 0;
            bc->cnt   = 0;
        }
    } else {
        int rem = n - space;
        bc->accum = static_cast<uint8_t>((bc->accum << space) | (bits >> rem));
        *bc->bp = bc->accum;
        if (bc->accum == 0xFF) { bc->bp++; *bc->bp = 0x00; }
        bc->bp++;
        bc->accum = static_cast<uint8_t>(bits & ((1u << rem) - 1u));
        bc->cnt   = rem;
    }
}


/* ============================================================================
 * SECTION 4: EBCOT T1 Context and Neighbor Functions (inlined from gpu_ebcot.h)
 * ============================================================================ */

#define SIGP(sigma_pad, r, c)   (((sigma_pad)[(r)+1] >> ((c)+1)) & 1)

__device__ static uint64_t has_sig_neighbor_mask(const uint64_t* sigma_pad, int r) {
    uint64_t north = sigma_pad[r];
    uint64_t cur   = sigma_pad[r + 1];
    uint64_t south = sigma_pad[r + 2];
    uint64_t nbr = north | south | (cur << 1) | (cur >> 1)
                 | (north << 1) | (north >> 1)
                 | (south << 1) | (south >> 1);
    return nbr;
}

__device__ static void neighbor_counts(const uint64_t* sigma_pad, int r, int c,
                                        int* sh_out, int* sv_out, int* sd_out) {
    int pc = c + 1, pr = r + 1;
    uint64_t north = sigma_pad[pr - 1], south = sigma_pad[pr + 1], cur = sigma_pad[pr];
    *sh_out = ((north >> pc) & 1) + ((south >> pc) & 1);
    *sv_out = ((cur >> (pc - 1)) & 1) + ((cur >> (pc + 1)) & 1);
    *sd_out = ((north >> (pc - 1)) & 1) + ((north >> (pc + 1)) & 1)
            + ((south >> (pc - 1)) & 1) + ((south >> (pc + 1)) & 1);
}

__device__ static int has_sig_neighbor(const uint64_t* sigma_pad, int r, int c) {
    int pc = c + 1, pr = r + 1;
    uint64_t north = sigma_pad[pr - 1], south = sigma_pad[pr + 1], cur = sigma_pad[pr];
    return (((north >> (pc - 1)) | (north >> pc) | (north >> (pc + 1)) |
             (cur >> (pc - 1)) | (cur >> (pc + 1)) |
             (south >> (pc - 1)) | (south >> pc) | (south >> (pc + 1))) & 1);
}

__device__ __constant__ static const uint8_t ZC_LUT[180] = {
    /* LL subband */
    0, 1, 2, 2, 2,  5, 6, 6, 6, 6,  8, 8, 8, 8, 8,
    3, 3, 3, 3, 3,  7, 7, 7, 7, 7,  8, 8, 8, 8, 8,
    4, 4, 4, 4, 4,  7, 7, 7, 7, 7,  8, 8, 8, 8, 8,
    /* HL subband */
    0, 1, 2, 2, 2,  3, 3, 3, 3, 3,  4, 4, 4, 4, 4,
    5, 6, 6, 6, 6,  7, 7, 7, 7, 7,  7, 7, 7, 7, 7,
    8, 8, 8, 8, 8,  8, 8, 8, 8, 8,  8, 8, 8, 8, 8,
    /* LH subband */
    0, 1, 2, 2, 2,  5, 6, 6, 6, 6,  8, 8, 8, 8, 8,
    3, 3, 3, 3, 3,  7, 7, 7, 7, 7,  8, 8, 8, 8, 8,
    4, 4, 4, 4, 4,  7, 7, 7, 7, 7,  8, 8, 8, 8, 8,
    /* HH subband */
    0, 3, 6, 6, 6,  1, 4, 7, 7, 7,  2, 5, 8, 8, 8,
    1, 4, 7, 7, 7,  2, 5, 8, 8, 8,  2, 5, 8, 8, 8,
    2, 5, 8, 8, 8,  2, 5, 8, 8, 8,  2, 5, 8, 8, 8
};

__device__ static int t1_zero_context_fast(const uint64_t* sigma_pad, int r, int c, int subband) {
    int sh, sv, sd;
    neighbor_counts(sigma_pad, r, c, &sh, &sv, &sd);
    int sd_clamp = (sd < 4) ? sd : 4;
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + sd_clamp];
}

__device__ static int t1_zc_if_neighbor(const uint64_t* sigma_pad, int r, int c, int subband) {
    int pc = c + 1, pr = r + 1;
    uint64_t north = sigma_pad[pr - 1], cur = sigma_pad[pr], south = sigma_pad[pr + 1];
    if (!(((north >> (pc-1)) | (north >> pc) | (north >> (pc+1)) |
           (cur >> (pc-1)) | (cur >> (pc+1)) |
           (south >> (pc-1)) | (south >> pc) | (south >> (pc+1))) & 1))
        return -1;
    int sh = ((north >> pc) & 1) + ((south >> pc) & 1);
    int sv = ((cur >> (pc-1)) & 1) + ((cur >> (pc+1)) & 1);
    int sd = ((north >> (pc-1)) & 1) + ((north >> (pc+1)) & 1)
           + ((south >> (pc-1)) & 1) + ((south >> (pc+1)) & 1);
    int sd_clamp = (sd < 4) ? sd : 4;
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + sd_clamp];
}

__device__ static void t1_sign_context_fast(const uint64_t* sigma_pad, const uint32_t* sign_bits,
                                             int r, int c, int /*cbw*/, int /*cbh*/,
                                             int* ctx_out, int* xor_bit_out) {
    int pc = c + 1, pr = r + 1;
    uint64_t sig_w = (sigma_pad[pr] >> (pc - 1)) & 1;
    uint64_t sig_e = (sigma_pad[pr] >> (pc + 1)) & 1;
    int hpos = 0, hneg = 0;
    if (sig_w) { if ((sign_bits[r] >> (c - 1)) & 1) hneg = 1; else hpos = 1; }
    if (sig_e) { if ((sign_bits[r] >> (c + 1)) & 1) hneg = 1; else hpos = 1; }
    int hc = hpos - hneg;
    uint64_t sig_n = (sigma_pad[pr - 1] >> pc) & 1;
    uint64_t sig_s = (sigma_pad[pr + 1] >> pc) & 1;
    int vpos = 0, vneg = 0;
    if (sig_n) { if ((sign_bits[r - 1] >> c) & 1) vneg = 1; else vpos = 1; }
    if (sig_s) { if ((sign_bits[r + 1] >> c) & 1) vneg = 1; else vpos = 1; }
    int vc = vpos - vneg;
    int flipped = 0;
    if (hc < 0) { hc = -hc; vc = -vc; flipped = 1; }
    if (hc == 0) { *xor_bit_out = (vc < 0) ? 1 : 0; *ctx_out = (vc == 0) ? 0 : 1; }
    else         { *xor_bit_out = flipped;          *ctx_out = (vc < 0) ? 2 : (vc == 0) ? 3 : 4; }
}

__device__ static int t1_mr_context_fast(const uint64_t* sigma_pad, const uint32_t* firstref_bits,
                                          int r, int c) {
    if (!((firstref_bits[r] >> c) & 1)) return 2;
    return has_sig_neighbor(sigma_pad, r, c) ? 1 : 0;
}

__device__ static int sig_bit(const uint32_t* sigma_bits, int r, int c, int h) {
    if (r < 0 || r >= h || c < 0 || c >= CB_DIM) return 0;
    return (sigma_bits[r] >> c) & 1;
}


/* ============================================================================
 * SECTION 5: CodeBlockInfo and kernel_ebcot_t1 (inlined from gpu_ebcot.h)
 * ============================================================================ */

struct CodeBlockInfo {
    int16_t  x0, y0;
    int16_t  width, height;
    uint8_t  subband_type;
    uint8_t  level;
    float    quant_step;
};

/* V196: DWT coefficient loader — dispatches __half vs float at compile time */
template<typename DWT_T>
__device__ __forceinline__ float dwt_load(const DWT_T* p) { return __half2float(__ldg(p)); }
template<>
__device__ __forceinline__ float dwt_load<float>(const float* p) { return __ldg(p); }

/* Main EBCOT T1 kernel — one code-block per thread.
 * V196: templated on DWT_T for FP16/FP32 support.
 * V172: FAST4=true clamps to 4 bit-planes with pre-built mag_bp_flat.
 * V175: MAX_BP compile-time bound for unrolling.
 * V165: use_bypass enables JPEG2000 BYPASS mode. */
template<bool FAST4, int MAX_BP = (FAST4 ? 4 : 10), typename DWT_T = __half>
__global__ __launch_bounds__(64, 16) void kernel_ebcot_t1(
    const DWT_T* __restrict__ d_dwt,
    int dwt_stride,
    const CodeBlockInfo* __restrict__ d_cb_info,
    int num_cbs,
    uint8_t*  __restrict__ d_coded_data,
    uint16_t* __restrict__ d_coded_len,
    uint8_t*  __restrict__ d_num_passes,
    uint16_t* __restrict__ d_pass_lengths,
    uint8_t*  __restrict__ d_num_bp,
    int bp_skip = 0,
    bool use_bypass = false)
{
    int cb_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cb_idx >= num_cbs) return;

    CodeBlockInfo cbi = d_cb_info[cb_idx];
    int cbw = cbi.width, cbh = cbi.height;
    if (cbw <= 0 || cbh <= 0) {
        d_coded_len[cb_idx] = 0;
        d_num_passes[cb_idx] = 0;
        return;
    }

    uint64_t sigma_pad[CB_DIM + 2];
    uint32_t sign_bits[CB_DIM];
    uint32_t firstref_bits[CB_DIM];
    uint32_t coded_bits[CB_DIM];

    for (int i = 0; i < CB_DIM + 2; i++) sigma_pad[i] = 0;
    for (int r = 0; r < cbh; r++) firstref_bits[r] = 0xFFFFFFFF;

    float inv_step = __frcp_rn(cbi.quant_step);

    /* Pass 1: compute sign_bits[], max_mag */
    int max_mag = 0;
    for (int r = 0; r < cbh; r++) {
        uint32_t sb = 0;
        const DWT_T* row_ptr = d_dwt + (cbi.y0 + r) * dwt_stride + cbi.x0;
        for (int c = 0; c < cbw; c++) {
            float val = dwt_load(row_ptr + c);
            int q = __float2int_rn(fabsf(val) * inv_step);
            if (val < 0.0f) sb |= (1u << c);
            max_mag |= q;
        }
        sign_bits[r] = sb;
    }

    int num_bp = (max_mag > 0) ? (32 - __clz(max_mag)) : 0;
    if (num_bp > MAX_BP) num_bp = MAX_BP;
    if (num_bp == 0) {
        d_coded_len[cb_idx] = 0;
        d_num_passes[cb_idx] = 0;
        d_num_bp[cb_idx] = 0;
        return;
    }

    /* V176: Unified pre-build of all bit-planes in one d_dwt scan */
    uint32_t mag_bp_flat[MAX_BP * CB_DIM];
    for (int bp_idx = 0; bp_idx < num_bp; bp_idx++)
        for (int c = 0; c < cbw; c++)
            mag_bp_flat[bp_idx * CB_DIM + c] = 0;
    for (int r = 0; r < cbh; r++) {
        uint32_t rmask = 1u << r;
        const DWT_T* row_ptr = d_dwt + (cbi.y0 + r) * dwt_stride + cbi.x0;
        for (int c = 0; c < cbw; c++) {
            int q = __float2int_rn(fabsf(dwt_load(row_ptr + c)) * inv_step);
            if (q == 0) continue;
            for (int bp_idx = 0; bp_idx < num_bp; bp_idx++) {
                if ((q >> (num_bp - 1 - bp_idx)) & 1)
                    mag_bp_flat[bp_idx * CB_DIM + c] |= rmask;
            }
        }
    }

    /* Initialize MQ coder */
    uint8_t* out_buf = d_coded_data + (size_t)cb_idx * CB_BUF_SIZE;
    out_buf[0] = 0;
    MQCoder mq;
    mq_init(&mq, out_buf);

    int total_passes = 0;
    uint16_t* pass_lens = d_pass_lengths + (size_t)cb_idx * MAX_PASSES;
    int any_significant = 0;
    int actual_bp_skip = (bp_skip > 0) ? (bp_skip < num_bp ? bp_skip : num_bp - 1) : 0;

    for (int bp = num_bp - 1; bp >= 0; bp--) {
        bool first_bp = (bp == num_bp - 1);
        const int bp_idx = num_bp - 1 - bp;
        bool bp_bypass = (use_bypass && bp < num_bp - 2);

        if (bp_bypass && bp < num_bp - 3)
            mq_restart(&mq, mq.bp + 1);

        for (int r2 = 0; r2 < cbh; r2++) coded_bits[r2] = 0;

        /* --- SPP --- */
        if (!first_bp) {
        if (any_significant) {
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            int stripe_len = stripe_end - stripe_y;
            uint64_t spp_all_sig = sigma_pad[stripe_y + 1] & sigma_pad[stripe_y + 2]
                                 & sigma_pad[stripe_y + 3] & sigma_pad[stripe_y + 4];
            uint32_t spp_sig[STRIPE_H] = {0, 0, 0, 0};
            for (int ri = 0; ri < stripe_len; ri++)
                spp_sig[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1);
            for (int c = 0; c < cbw; c++) {
                uint64_t cmask_pad = 1ull << (c + 1);
                uint32_t cmask = 1u << c;
                if (spp_all_sig & cmask_pad) continue;
                uint32_t col_mag = mag_bp_flat[bp_idx * CB_DIM + c];
                for (int r = stripe_y; r < stripe_end; r++) {
                    if (spp_sig[r - stripe_y] & cmask) continue;
                    int zc = t1_zc_if_neighbor(sigma_pad, r, c, cbi.subband_type);
                    if (zc < 0) continue;
                    int bit = (col_mag >> r) & 1;
                    mq_encode(&mq, T1_CTXNO_ZC + zc, bit);
                    if (bit) {
                        sigma_pad[r + 1] |= cmask_pad;
                        int sc_ctx, xor_bit;
                        t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                        mq_encode(&mq, T1_CTXNO_SC + sc_ctx, ((sign_bits[r] >> c) & 1) ^ xor_bit);
                    }
                    coded_bits[r] |= cmask;
                }
            }
        }
        }
        if (bp_bypass) mq_align(&mq);
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end SPP */

        /* --- MRP --- */
        if (!first_bp) {
        if (any_significant) {
        if (bp_bypass) {
            BypassCoder bc;
            bc.bp = mq.bp + 1;  bc.accum = 0;  bc.cnt = 0;
            for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                int stripe_end = min(stripe_y + STRIPE_H, cbh);
                int stripe_len_mrp = stripe_end - stripe_y;
                uint32_t mrp_proc[STRIPE_H] = {0, 0, 0, 0};
                for (int ri = 0; ri < stripe_len_mrp; ri++)
                    mrp_proc[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1)
                                 & ~coded_bits[stripe_y + ri];
                uint32_t mrp_active = mrp_proc[0] | mrp_proc[1] | mrp_proc[2] | mrp_proc[3];
                while (mrp_active) {
                    int c = __ffs(mrp_active) - 1;
                    mrp_active &= mrp_active - 1u;
                    uint32_t cmask = 1u << c;
                    uint32_t col_mag = mag_bp_flat[bp_idx * CB_DIM + c];
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if (!(mrp_proc[r - stripe_y] & cmask)) continue;
                        bypass_write(&bc, (col_mag >> r) & 1);
                        firstref_bits[r] &= ~(1u << c);
                    }
                }
            }
            bypass_flush(&bc);
            mq.bp = bc.bp - 1;
        } else {
            for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                int stripe_end = min(stripe_y + STRIPE_H, cbh);
                int stripe_len_mrp = stripe_end - stripe_y;
                uint32_t mrp_proc[STRIPE_H] = {0, 0, 0, 0};
                for (int ri = 0; ri < stripe_len_mrp; ri++)
                    mrp_proc[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1)
                                 & ~coded_bits[stripe_y + ri];
                uint32_t mrp_colmask = mrp_proc[0] | mrp_proc[1] | mrp_proc[2] | mrp_proc[3];
                for (int c = 0; c < cbw; c++) {
                    uint32_t cmask = 1u << c;
                    if (!(mrp_colmask & cmask)) continue;
                    uint32_t col_mag = mag_bp_flat[bp_idx * CB_DIM + c];
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if (!(mrp_proc[r - stripe_y] & cmask)) continue;
                        int bit = (col_mag >> r) & 1;
                        int mr = t1_mr_context_fast(sigma_pad, firstref_bits, r, c);
                        mq_encode(&mq, T1_CTXNO_MR + mr, bit);
                        firstref_bits[r] &= ~(1u << c);
                    }
                }
            }
        }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end MRP */

        /* --- CUP --- */
        if (bp_bypass) {
            BypassCoder bc;
            bc.bp = mq.bp + 1;  bc.accum = 0;  bc.cnt = 0;
            for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                int stripe_end = min(stripe_y + STRIPE_H, cbh);
                int stripe_len_cup = stripe_end - stripe_y;
                uint32_t cup_skip[STRIPE_H] = {0, 0, 0, 0};
                for (int ri = 0; ri < stripe_len_cup; ri++)
                    cup_skip[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1)
                                 | coded_bits[stripe_y + ri];
                uint32_t cup_all_skip = cup_skip[0] & cup_skip[1] & cup_skip[2] & cup_skip[3];
                uint32_t cup_active   = ~cup_all_skip & ((1u << cbw) - 1u);
                while (cup_active) {
                    int c = __ffs(cup_active) - 1;
                    cup_active &= cup_active - 1u;
                    uint64_t cmask_pad = 1ull << (c + 1);
                    uint32_t cmask = 1u << c;
                    uint32_t col_mag = mag_bp_flat[bp_idx * CB_DIM + c];
                    for (int r = stripe_y; r < stripe_end; r++) {
                        int ri = r - stripe_y;
                        if (cup_skip[ri] & cmask) continue;
                        int bit = (col_mag >> r) & 1;
                        bypass_write(&bc, bit);
                        if (bit) {
                            sigma_pad[r + 1] |= cmask_pad;
                            any_significant = 1;
                            bypass_write(&bc, (sign_bits[r] >> c) & 1);
                        }
                        coded_bits[r] |= cmask;
                    }
                }
            }
            bypass_flush(&bc);
            mq.bp = bc.bp - 1;
        } else {
            for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                int stripe_end = min(stripe_y + STRIPE_H, cbh);
                int stripe_len_cup = stripe_end - stripe_y;
                uint64_t s0 = sigma_pad[stripe_y];
                uint64_t s1 = sigma_pad[stripe_y + 1];
                uint64_t s2 = sigma_pad[stripe_y + 2];
                uint64_t s3 = sigma_pad[stripe_y + 3];
                uint64_t s4 = sigma_pad[stripe_y + 4];
                uint64_t sig_any = s0 | s1 | s2 | s3 | s4;
                uint64_t hn_pre  = sig_any | (sig_any << 1) | (sig_any >> 1);
                uint32_t coded_or = coded_bits[stripe_y] | coded_bits[stripe_y + 1]
                                  | coded_bits[stripe_y + 2] | coded_bits[stripe_y + 3];
                uint64_t fast_not_zero = hn_pre | ((uint64_t)coded_or << 1);
                uint32_t cup_skip[STRIPE_H] = {0, 0, 0, 0};
                for (int ri = 0; ri < stripe_len_cup; ri++)
                    cup_skip[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1)
                                 | coded_bits[stripe_y + ri];
                for (int c = 0; c < cbw; c++) {
                    uint64_t cmask_pad = 1ull << (c + 1);
                    uint32_t cmask = 1u << c;
                    int all_zero;
                    if (fast_not_zero & cmask_pad) {
                        all_zero = 0;
                    } else {
                        all_zero = (stripe_len_cup == 4) ? 1 : 0;
                        if (all_zero && any_significant) {
                            for (int r = stripe_y; r < stripe_end; r++) {
                                if (cup_skip[r - stripe_y] & cmask) { all_zero = 0; break; }
                                if (has_sig_neighbor(sigma_pad, r, c)) { all_zero = 0; break; }
                            }
                        }
                    }
                    uint32_t col_mag = mag_bp_flat[bp_idx * CB_DIM + c];
                    if (all_zero) {
                        uint32_t stripe_bits = (col_mag >> stripe_y) & 0xFu;
                        int any_sig = (stripe_bits != 0);
                        mq_encode(&mq, T1_CTXNO_AGG, any_sig);
                        if (!any_sig) continue;
                        int run_len = __ffs(stripe_bits) - 1;
                        mq_encode(&mq, T1_CTXNO_UNI, (run_len >> 1) & 1);
                        mq_encode(&mq, T1_CTXNO_UNI, run_len & 1);
                        int r = stripe_y + run_len;
                        sigma_pad[r + 1] |= cmask_pad;
                        any_significant = 1;
                        int sc_ctx, xor_bit;
                        t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                        mq_encode(&mq, T1_CTXNO_SC + sc_ctx, ((sign_bits[r] >> c) & 1) ^ xor_bit);
                        coded_bits[r] |= cmask;
                        for (r = stripe_y + run_len + 1; r < stripe_end; r++) {
                            int ri = r - stripe_y;
                            if (cup_skip[ri] & cmask) continue;
                            int bit = (col_mag >> r) & 1;
                            int zc2 = t1_zero_context_fast(sigma_pad, r, c, cbi.subband_type);
                            mq_encode(&mq, T1_CTXNO_ZC + zc2, bit);
                            if (bit) {
                                sigma_pad[r + 1] |= cmask_pad;
                                any_significant = 1;
                                t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                                mq_encode(&mq, T1_CTXNO_SC + sc_ctx, ((sign_bits[r] >> c) & 1) ^ xor_bit);
                            }
                            coded_bits[r] |= cmask;
                        }
                    } else {
#pragma unroll
                        for (int r = stripe_y; r < stripe_end; r++) {
                            int ri = r - stripe_y;
                            if (cup_skip[ri] & cmask) continue;
                            int bit = (col_mag >> r) & 1;
                            int zc2 = t1_zero_context_fast(sigma_pad, r, c, cbi.subband_type);
                            mq_encode(&mq, T1_CTXNO_ZC + zc2, bit);
                            if (bit) {
                                sigma_pad[r + 1] |= cmask_pad;
                                any_significant = 1;
                                int sc_ctx2, xor_bit2;
                                t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx2, &xor_bit2);
                                mq_encode(&mq, T1_CTXNO_SC + sc_ctx2, ((sign_bits[r] >> c) & 1) ^ xor_bit2);
                            }
                            coded_bits[r] |= cmask;
                        }
                    }
                }
            }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);

        if (actual_bp_skip > 0 && bp == actual_bp_skip) break;
    }

    /* Final flush */
    {
        int last_coded_bp = (actual_bp_skip > 0) ? actual_bp_skip : 0;
        bool last_is_bypass = use_bypass && (last_coded_bp < num_bp - 2);
        if (!last_is_bypass)
            mq_flush(&mq);
        else
            while (mq.bp > mq.start && *mq.bp == 0xFF) mq.bp--;
    }

    int coded_len = static_cast<int>(mq.bp - mq.start);
    if (coded_len < 0) coded_len = 0;
    if (coded_len > CB_BUF_SIZE - 1) coded_len = CB_BUF_SIZE - 1;

    while (coded_len > 0 && out_buf[coded_len] == 0xFF) coded_len--;
    d_coded_len[cb_idx] = static_cast<uint16_t>(coded_len);
    d_num_passes[cb_idx] = static_cast<uint8_t>(total_passes);
    d_num_bp[cb_idx]    = static_cast<uint8_t>(num_bp);
    for (int p = 0; p < total_passes; p++) {
        int pl = (pass_lens[p] > 0) ? (pass_lens[p] - 1) : 0;
        pass_lens[p] = static_cast<uint16_t>(min(pl, coded_len));
    }
}


/* ============================================================================
 * SECTION 6: T2 Data Structures and Functions (inlined from gpu_ebcot_t2.h)
 * ============================================================================ */

struct SubbandGeom {
    int x0, y0;
    int width, height;
    int type;
    int level;
    int res;
    float step;
    int cb_x0, cb_y0;
    int ncbx, ncby;
    int cb_start_idx;
};

inline void build_codeblock_table(
    int width, int height, int stride, int num_levels, float base_step, bool is_4k,
    std::vector<CodeBlockInfo>& cb_infos,
    std::vector<SubbandGeom>& subbands)
{
    cb_infos.clear();
    subbands.clear();

    int w[8], h[8];
    w[0] = width; h[0] = height;
    for (int l = 1; l <= num_levels; l++) {
        w[l] = (w[l-1] + 1) / 2;
        h[l] = (h[l-1] + 1) / 2;
    }

    float level_weight[8] = {1.20f, 1.12f, 1.05f, 0.95f, 0.85f, 0.65f, 0.65f, 0.65f};

    /* LL subband */
    {
        int ll_w = w[num_levels], ll_h = h[num_levels];
        float step = base_step * level_weight[num_levels];
        SubbandGeom sg;
        sg.x0 = 0; sg.y0 = 0;
        sg.width = ll_w; sg.height = ll_h;
        sg.type = SUBBAND_LL; sg.level = num_levels; sg.res = 0;
        sg.step = step;
        sg.ncbx = (ll_w + CB_DIM - 1) / CB_DIM;
        sg.ncby = (ll_h + CB_DIM - 1) / CB_DIM;
        sg.cb_start_idx = static_cast<int>(cb_infos.size());
        sg.cb_x0 = 0; sg.cb_y0 = 0;
        for (int cby = 0; cby < sg.ncby; cby++) {
            for (int cbx = 0; cbx < sg.ncbx; cbx++) {
                CodeBlockInfo cbi;
                cbi.x0 = static_cast<int16_t>(cbx * CB_DIM);
                cbi.y0 = static_cast<int16_t>(cby * CB_DIM);
                cbi.width  = static_cast<int16_t>(std::min(CB_DIM, ll_w - cbx * CB_DIM));
                cbi.height = static_cast<int16_t>(std::min(CB_DIM, ll_h - cby * CB_DIM));
                cbi.subband_type = SUBBAND_LL;
                cbi.level = static_cast<uint8_t>(num_levels);
                cbi.quant_step = step;
                cb_infos.push_back(cbi);
            }
        }
        subbands.push_back(sg);
    }

    /* Detail subbands */
    for (int l = num_levels; l >= 1; l--) {
        struct { int x0, y0, sw, sh, type; } defs[3] = {
            { w[l], 0,    w[l-1] - w[l], h[l],           SUBBAND_HL },
            { 0,    h[l], w[l],          h[l-1] - h[l],  SUBBAND_LH },
            { w[l], h[l], w[l-1] - w[l], h[l-1] - h[l],  SUBBAND_HH }
        };
        float weight = level_weight[std::min(l - 1, 5)];
        float step = base_step * weight;
        for (int s = 0; s < 3; s++) {
            SubbandGeom sg;
            sg.x0 = defs[s].x0; sg.y0 = defs[s].y0;
            sg.width = defs[s].sw; sg.height = defs[s].sh;
            sg.type = defs[s].type; sg.level = static_cast<uint8_t>(l - 1);
            sg.res = num_levels - l + 1;
            sg.step = step;
            sg.ncbx = (sg.width + CB_DIM - 1) / CB_DIM;
            sg.ncby = (sg.height + CB_DIM - 1) / CB_DIM;
            sg.cb_start_idx = static_cast<int>(cb_infos.size());
            sg.cb_x0 = 0; sg.cb_y0 = 0;
            float t1_step = step;
            if (defs[s].type == SUBBAND_HL || defs[s].type == SUBBAND_LH) t1_step = step * 2.0f;
            else if (defs[s].type == SUBBAND_HH) t1_step = step * 4.0f;
            for (int cby = 0; cby < sg.ncby; cby++) {
                for (int cbx = 0; cbx < sg.ncbx; cbx++) {
                    CodeBlockInfo cbi;
                    cbi.x0 = static_cast<int16_t>(sg.x0 + cbx * CB_DIM);
                    cbi.y0 = static_cast<int16_t>(sg.y0 + cby * CB_DIM);
                    cbi.width  = static_cast<int16_t>(std::min(CB_DIM, sg.width - cbx * CB_DIM));
                    cbi.height = static_cast<int16_t>(std::min(CB_DIM, sg.height - cby * CB_DIM));
                    cbi.subband_type = static_cast<uint8_t>(defs[s].type);
                    cbi.level = static_cast<uint8_t>(l - 1);
                    cbi.quant_step = t1_step;
                    cb_infos.push_back(cbi);
                }
            }
            subbands.push_back(sg);
        }
    }
}

/* BitWriter for J2K packet headers (with 0xFF byte-stuffing) */
struct BitWriter {
    std::vector<uint8_t>& buf;
    uint64_t acc;
    int      acc_n;
    bool     prev_ff;

    BitWriter(std::vector<uint8_t>& b) : buf(b), acc(0), acc_n(0), prev_ff(false) {}

    void write_bits(uint32_t val, int nbits) {
        if (nbits > 0 && nbits < 32) val &= (1u << nbits) - 1u;
        acc = (acc << nbits) | static_cast<uint64_t>(val);
        acc_n += nbits;
        while (acc_n >= 8) {
            if (prev_ff) {
                if (acc_n < 7) break;
                acc_n -= 7;
                uint8_t byte = static_cast<uint8_t>((acc >> acc_n) & 0x7Fu);
                buf.push_back(byte);
                prev_ff = (byte == 0xFF);
                continue;
            }
            while (acc_n >= 8) {
                acc_n -= 8;
                uint8_t byte = static_cast<uint8_t>((acc >> acc_n) & 0xFFu);
                buf.push_back(byte);
                if (byte == 0xFF) { prev_ff = true; break; }
            }
        }
    }

    void write_bit(int b) { write_bits(static_cast<uint32_t>(b & 1), 1); }

    void flush() {
        if (acc_n > 0) {
            int bits = prev_ff ? 7 : 8;
            uint8_t byte = static_cast<uint8_t>((acc << (bits - acc_n)) & (prev_ff ? 0x7Fu : 0xFFu));
            buf.push_back(byte);
            prev_ff = (byte == 0xFF);
            acc_n = 0; acc = 0;
        }
    }
};

/* Tag Tree (ITU-T T.800 B.10.2) */
struct TagTree {
    struct Node {
        int  value;
        int  low;
        bool known;
        int  parent;
    };
    std::vector<Node> nodes;
    std::vector<int>  level_off;
    std::vector<int>  level_w;
    int               num_leaves = 0;

    void build(int ncbx, int ncby) {
        nodes.clear(); level_off.clear(); level_w.clear();
        int w = ncbx, h = ncby, total = 0;
        std::vector<std::pair<int,int>> dims;
        while (true) {
            dims.push_back({w, h});
            if (w == 1 && h == 1) break;
            w = (w + 1) / 2; h = (h + 1) / 2;
        }
        int nlv = (int)dims.size();
        for (int lv = 0; lv < nlv; lv++) {
            level_off.push_back(total); level_w.push_back(dims[lv].first);
            total += dims[lv].first * dims[lv].second;
        }
        nodes.resize(total);
        num_leaves = dims[0].first * dims[0].second;
        for (int lv = 0; lv < nlv; lv++) {
            int wlv = dims[lv].first, hlv = dims[lv].second, off = level_off[lv];
            for (int j = 0; j < hlv; j++) for (int i = 0; i < wlv; i++) {
                Node& n = nodes[off + j * wlv + i];
                n.value = 0x7FFFFFFF; n.low = 0; n.known = false;
                n.parent = (lv + 1 < nlv)
                    ? level_off[lv+1] + (j/2) * dims[lv+1].first + (i/2) : -1;
            }
        }
    }

    void set_leaf(int leaf_idx, int value) {
        nodes[leaf_idx].value = value;
        int idx = leaf_idx;
        while (nodes[idx].parent != -1) {
            int par = nodes[idx].parent;
            if (value < nodes[par].value) { nodes[par].value = value; idx = par; }
            else break;
        }
    }

    void encode(BitWriter& bw, int leaf_idx, int threshold) {
        int path[32], plen = 0;
        for (int idx = leaf_idx; idx != -1; idx = nodes[idx].parent)
            path[plen++] = idx;
        int low = 0;
        for (int pi = plen - 1; pi >= 0; pi--) {
            Node& node = nodes[path[pi]];
            if (low > node.low) node.low = low; else low = node.low;
            while (low < threshold) {
                if (low >= node.value) {
                    if (!node.known) { bw.write_bit(1); node.known = true; }
                    break;
                }
                bw.write_bit(0);
                ++low;
            }
            node.low = low;
        }
    }
};

/* V198: PCRD-OPT slope-sort rate allocation + J2K codestream builder */
inline std::vector<uint8_t> build_ebcot_codestream(
    int width, int height, bool is_4k, bool is_3d,
    int num_levels, float base_step,
    const std::vector<SubbandGeom>& subbands,
    const uint8_t*  coded_data[3],
    const uint16_t* coded_len[3],
    const uint8_t*  num_passes[3],
    const uint16_t* pass_lengths[3],
    const uint8_t*  cb_num_bp[3],
    int64_t target_bytes,
    int cb_stride = CB_BUF_SIZE)
{
    std::vector<uint8_t> cs;
    cs.reserve(target_bytes > 0 ? target_bytes + 1024 : 1024*1024);

    auto w16 = [&](uint16_t v) { cs.push_back(v >> 8); cs.push_back(v & 0xFF); };
    auto w32 = [&](uint32_t v) { cs.push_back((v>>24)&0xFF); cs.push_back((v>>16)&0xFF);
                                  cs.push_back((v>>8)&0xFF);  cs.push_back(v&0xFF); };
    auto w8  = [&](uint8_t v)  { cs.push_back(v); };

    /* SOC */
    w16(J2K_SOC_M);

    /* SIZ */
    w16(J2K_SIZ_M);
    w16(2 + 2 + 32 + 2 + 3*3);
    w16(is_4k ? 0x0004 : 0x0003);
    w32(width); w32(height);
    w32(0); w32(0);
    w32(width); w32(height);
    w32(0); w32(0);
    w16(3);
    for (int c = 0; c < 3; c++) { w8(11); w8(1); w8(1); }

    /* COD */
    w16(J2K_COD_M);
    w16(2 + 1 + 4 + 5);
    w8(0x00);
    w8(0x00);
    w16(1);
    w8(0);
    w8(static_cast<uint8_t>(num_levels));
    w8(3);
    w8(3);
    w8(0x00);
    w8(0x00);

    /* QCD */
    {
        int nsb = 3 * num_levels + 1;
        w16(J2K_QCD_M);
        w16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        const int numgbits = is_4k ? 2 : 1;
        const uint8_t sqcd = static_cast<uint8_t>((numgbits << 5) | 0x02);
        w8(sqcd);
        for (int i = 0; i < nsb; i++) {
            float step_val = (i < static_cast<int>(subbands.size())) ? subbands[i].step : base_step;
            int log2s = static_cast<int>(std::floor(std::log2(std::max(step_val, 0.001f))));
            int eps = 12 - log2s;
            float denom = std::ldexp(1.0f, log2s);
            int man = static_cast<int>((step_val / denom - 1.0f) * 2048.0f);
            man = std::max(0, std::min(2047, man));
            w16(static_cast<uint16_t>((eps << 11) | man));
        }
    }

    /* PCRD-OPT rate allocation */
    size_t total_cbs = 0;
    if (!subbands.empty()) {
        const SubbandGeom& sb_last = subbands.back();
        total_cbs = static_cast<size_t>(sb_last.cb_start_idx)
                  + static_cast<size_t>(sb_last.ncbx) * sb_last.ncby;
    }
    std::vector<uint8_t>  pcrd_np_use[3];
    std::vector<uint16_t> pcrd_len_use[3];
    std::vector<size_t>   pcrd_res_body[3];
    for (int c = 0; c < 3; ++c) {
        pcrd_np_use[c].assign(total_cbs, 0);
        pcrd_len_use[c].assign(total_cbs, 0);
        pcrd_res_body[c].assign(num_levels + 1, 0);
    }

    const size_t per_comp_target = (target_bytes > 0)
        ? static_cast<size_t>(target_bytes / 3) : SIZE_MAX;

    /* V200: Precompute subband step_sq_i and bit-plane weights. */
    std::vector<int64_t> sb_step_sq(subbands.size());
    for (size_t sb = 0; sb < subbands.size(); ++sb)
        sb_step_sq[sb] = static_cast<int64_t>(subbands[sb].step * subbands[sb].step * (1 << 20));
    int64_t bp_weight[MAX_BPLANES];
    for (int bp = 0; bp < MAX_BPLANES; ++bp)
        bp_weight[bp] = 1ULL << (2 * bp);

    struct PassCand {
        int64_t  slope;
        uint16_t inc_len;
        uint16_t cb_idx;
        uint8_t  pass_idx;
    };

    auto insertion_sort_desc = [](PassCand* begin, PassCand* end) {
        for (PassCand* i = begin + 1; i < end; ++i) {
            PassCand key = *i;
            PassCand* j = i;
            while (j > begin && (j - 1)->slope < key.slope) {
                *j = *(j - 1);
                --j;
            }
            *j = key;
        }
    };

    auto pcrd_scan_comp = [&](int c) {
        size_t comp_bytes = 0;
        std::vector<uint8_t>  cur_np(total_cbs, 0);
        std::vector<uint16_t> cur_len(total_cbs, 0);

        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            int res = subbands[sb].res;
            int cb_start = subbands[sb].cb_start_idx;
            int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
            int64_t step_sq_i = sb_step_sq[sb];

            std::vector<PassCand> cands;
            cands.reserve(static_cast<size_t>(ncbs) * MAX_PASSES);

            for (int i = 0; i < ncbs; ++i) {
                int cb_idx = cb_start + i;
                uint8_t np = num_passes[c][cb_idx];
                uint16_t cb_len = coded_len[c][cb_idx];
                if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                    cb_len = static_cast<uint16_t>(cb_stride - 1);
                if (np == 0 || cb_len == 0) continue;
                const uint16_t* pl = pass_lengths[c]
                    + static_cast<size_t>(cb_idx) * MAX_PASSES;
                uint16_t prev_cum = 0;
                int64_t prev_slope = INT64_MAX;
                for (int p = 0; p < np; ++p) {
                    uint16_t cum = pl[p];
                    if (cum > cb_len) cum = cb_len;
                    uint16_t inc = cum - prev_cum;
                    prev_cum = cum;
                    if (inc == 0) continue;
                    int bp_idx = p / 3;
                    int64_t slope = (step_sq_i * bp_weight[bp_idx]) / inc;
                    if (slope > prev_slope) slope = prev_slope;
                    prev_slope = slope;
                    cands.push_back({slope, inc,
                        static_cast<uint16_t>(cb_idx), static_cast<uint8_t>(p)});
                }
            }

            /* Sort by slope descending. */
            if (cands.size() < 100) {
                insertion_sort_desc(cands.data(), cands.data() + cands.size());
            } else if (!cands.empty()) {
                std::sort(cands.begin(), cands.end(),
                    [](const PassCand& a, const PassCand& b) { return a.slope > b.slope; });
            }

            for (const PassCand& ca : cands) {
                if (static_cast<int>(cur_np[ca.cb_idx]) != ca.pass_idx) continue;
                size_t cost = ca.inc_len;
                if (comp_bytes + cost > per_comp_target) continue;
                const uint16_t* pl = pass_lengths[c]
                    + static_cast<size_t>(ca.cb_idx) * MAX_PASSES;
                uint16_t cum = pl[ca.pass_idx];
                uint16_t cb_len = coded_len[c][ca.cb_idx];
                if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                    cb_len = static_cast<uint16_t>(cb_stride - 1);
                if (cum > cb_len) cum = cb_len;
                cur_np[ca.cb_idx]  = static_cast<uint8_t>(ca.pass_idx + 1);
                cur_len[ca.cb_idx] = cum;
                pcrd_np_use[c][ca.cb_idx] = static_cast<uint8_t>(ca.pass_idx + 1);
                pcrd_len_use[c][ca.cb_idx] = cum;
                pcrd_res_body[c][res] += cost;
                comp_bytes += cost;
            }
        }
    };
    auto pcrd_f0 = std::async(std::launch::async, [&]() { pcrd_scan_comp(0); });
    auto pcrd_f1 = std::async(std::launch::async, [&]() { pcrd_scan_comp(1); });
    pcrd_scan_comp(2);
    pcrd_f0.wait();
    pcrd_f1.wait();

    const int numgbits_for_pmax = is_4k ? 2 : 1;
    std::vector<int> pre_sb_pmax(subbands.size());
    for (size_t sb = 0; sb < subbands.size(); ++sb) {
        float step = std::max(subbands[sb].step, 0.001f);
        int log2s = static_cast<int>(std::floor(std::log2f(step)));
        pre_sb_pmax[sb] = (12 - log2s) + numgbits_for_pmax - 1;
    }
    std::vector<TagTree> pre_incl_trees(subbands.size());
    std::vector<TagTree> pre_zbp_trees(subbands.size());
    for (size_t sb = 0; sb < subbands.size(); ++sb) {
        pre_incl_trees[sb].build(subbands[sb].ncbx, subbands[sb].ncby);
        pre_zbp_trees[sb].build(subbands[sb].ncbx, subbands[sb].ncby);
    }

    std::vector<std::vector<uint8_t>> pkt_by_res[3];
    for (int c = 0; c < 3; c++)
        pkt_by_res[c].resize(num_levels + 1);

    auto build_tp = [&](int comp) {
        size_t comp_bytes = 0;
        size_t comp_body_total = 0;
        for (int r = 0; r <= num_levels; ++r)
            comp_body_total += pcrd_res_body[comp][r];
        std::vector<uint8_t> pkt_header_buf;
        std::vector<uint8_t> pkt_body;
        pkt_header_buf.reserve(16384);
        pkt_body.reserve(std::max(size_t(512*1024), comp_body_total + 4096));

        TagTree incl_tree, zbp_tree;
        int max_ncbs = 0;
        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            int n = subbands[sb].ncbx * subbands[sb].ncby;
            if (n > max_ncbs) max_ncbs = n;
        }
        std::vector<uint8_t>  included(max_ncbs, 0);
        std::vector<uint16_t> cb_len_use(max_ncbs, 0);
        std::vector<uint8_t>  cb_np_use(max_ncbs, 0);

        for (int res = 0; res <= num_levels; res++) {
            pkt_header_buf.clear();
            pkt_body.clear();
            size_t res_body_total = pcrd_res_body[comp][res];
            pkt_body.reserve(std::max(pkt_body.capacity(), res_body_total + 256));

            BitWriter bw(pkt_header_buf);
            bw.write_bit(1);

            for (size_t sb = 0; sb < subbands.size(); sb++) {
                if (subbands[sb].res != res) continue;
                int ncbx = subbands[sb].ncbx, ncby = subbands[sb].ncby;
                int ncbs = ncbx * ncby;
                int cb_start = subbands[sb].cb_start_idx;
                int pmax = pre_sb_pmax[sb];

                incl_tree = pre_incl_trees[sb];
                zbp_tree = pre_zbp_trees[sb];

                if (ncbs > max_ncbs) {
                    max_ncbs = ncbs;
                    included.assign(ncbs, 0);
                    cb_len_use.assign(ncbs, 0);
                    cb_np_use.assign(ncbs, 0);
                } else {
                    std::memset(included.data(), 0, ncbs * sizeof(uint8_t));
                    std::memset(cb_len_use.data(), 0, ncbs * sizeof(uint16_t));
                    std::memset(cb_np_use.data(), 0, ncbs * sizeof(uint8_t));
                }

                for (int cbi = 0; cbi < ncbs; cbi++) {
                    int cb_idx = cb_start + cbi;
                    uint8_t np_use = pcrd_np_use[comp][cb_idx];
                    uint16_t len_use = pcrd_len_use[comp][cb_idx];
                    if (np_use == 0 || len_use == 0) {
                        incl_tree.set_leaf(cbi, 0x7FFFFFFF);
                        zbp_tree.set_leaf(cbi, pmax);
                        continue;
                    }
                    included[cbi] = true;
                    comp_bytes += len_use;
                    cb_len_use[cbi] = len_use;
                    cb_np_use[cbi]  = np_use;
                    incl_tree.set_leaf(cbi, 0);
                    int nb = (cb_num_bp != nullptr) ? static_cast<int>(cb_num_bp[comp][cb_idx]) : 0;
                    int z = pmax - nb;
                    if (z < 0) z = 0;
                    zbp_tree.set_leaf(cbi, z);
                }

                for (int cbi = 0; cbi < ncbs; cbi++) {
                    int cb_idx = cb_start + cbi;
                    incl_tree.encode(bw, cbi, 1);
                    if (!included[cbi]) continue;
                    zbp_tree.encode(bw, cbi, pmax);

                    uint8_t np = cb_np_use[cbi];
                    uint16_t len = cb_len_use[cbi];

                    if (np == 1)       bw.write_bit(0);
                    else if (np == 2)  bw.write_bits(2, 2);
                    else if (np <= 5)  bw.write_bits(0xC | (np - 3), 4);
                    else if (np <= 36) bw.write_bits(0x1E0 | (np - 6), 9);
                    else               bw.write_bits(0xFF80u | (unsigned(np) - 37u), 16);

                    int lblock = 3;
                    int floor_log2_np = (np <= 1) ? 0 : (31 - __builtin_clz(static_cast<unsigned>(np)));
                    int len_bits = lblock + floor_log2_np;
                    if (len_bits < 1) len_bits = 1;
                    while ((1 << len_bits) <= len) { bw.write_bit(1); lblock++; len_bits++; }
                    bw.write_bit(0);
                    bw.write_bits(len, len_bits);

                    const uint8_t* src = coded_data[comp] + (size_t)cb_idx * cb_stride + 1;
                    size_t pkt_off = pkt_body.size();
                    pkt_body.resize(pkt_off + len);
                    std::memcpy(pkt_body.data() + pkt_off, src, len);
                }
            }

            bw.flush();
            auto& dst = pkt_by_res[comp][res];
            size_t d0 = dst.size();
            dst.resize(d0 + pkt_header_buf.size() + pkt_body.size());
            std::memcpy(dst.data() + d0, pkt_header_buf.data(), pkt_header_buf.size());
            std::memcpy(dst.data() + d0 + pkt_header_buf.size(), pkt_body.data(), pkt_body.size());
        }
    };

    auto fut0 = std::async(std::launch::async, [&]() { build_tp(0); });
    auto fut1 = std::async(std::launch::async, [&]() { build_tp(1); });
    build_tp(2);
    fut0.wait();
    fut1.wait();

    /* Tile-parts */
    int n_tile_parts = is_4k ? 6 : 3;
    std::vector<const std::vector<uint8_t>*> ordered_pkts;
    ordered_pkts.reserve((num_levels + 1) * 3);
    for (int r = 0; r <= num_levels; r++)
        for (int c = 0; c < 3; c++)
            ordered_pkts.push_back(&pkt_by_res[c][r]);

    size_t total_pkt = 0;
    for (auto* p : ordered_pkts) total_pkt += p->size();

    std::vector<int> tp_first_pkt(n_tile_parts);
    std::vector<size_t> tp_pkt_bytes(n_tile_parts, 0);
    {
        size_t target_per_tp = (total_pkt + n_tile_parts - 1) / n_tile_parts;
        int pkt_idx = 0;
        for (int t = 0; t < n_tile_parts; ++t) {
            tp_first_pkt[t] = pkt_idx;
            size_t accum = 0;
            int last_t = (t == n_tile_parts - 1);
            while (pkt_idx < (int)ordered_pkts.size()
                   && (last_t || accum < target_per_tp)) {
                accum += ordered_pkts[pkt_idx]->size();
                ++pkt_idx;
            }
            tp_pkt_bytes[t] = accum;
        }
    }

    /* TLM */
    w16(J2K_TLM_M);
    w16(static_cast<uint16_t>(2 + 1 + 1 + n_tile_parts * (1 + 4)));
    w8(0); w8(0x50);
    for (int t = 0; t < n_tile_parts; ++t) {
        uint32_t tp_size = static_cast<uint32_t>(12 + 2 + tp_pkt_bytes[t]);
        w8(0); w32(tp_size);
    }

    /* Emit tile-parts */
    for (int t = 0; t < n_tile_parts; ++t) {
        uint32_t tp_size = static_cast<uint32_t>(12 + 2 + tp_pkt_bytes[t]);
        w16(J2K_SOT_M); w16(10);
        w16(0); w32(tp_size);
        w8(static_cast<uint8_t>(t)); w8(static_cast<uint8_t>(n_tile_parts));
        w16(J2K_SOD_M);
        int end_pkt = (t + 1 < n_tile_parts) ? tp_first_pkt[t + 1] : (int)ordered_pkts.size();
        for (int i = tp_first_pkt[t]; i < end_pkt; ++i)
            cs.insert(cs.end(), ordered_pkts[i]->begin(), ordered_pkts[i]->end());
    }

    w16(J2K_EOC_M);
    return cs;
}


/* ============================================================================
 * SECTION 7: DWT Kernel Forward Declarations
 *
 * These are provided by cuda_j2k_encoder.cu during compilation.
 * V18 does NOT duplicate the DWT kernels — it calls them via gpu_dwt97_level
 * and gpu_dwt97_level_fp32 which are defined in the main encoder.
 * ============================================================================ */

extern void gpu_dwt97_level(
    __half* d_half_a, __half* d_half_h, __half* d_half_work,
    const int32_t* d_input,
    int width, int height, int stride, int level, cudaStream_t st,
    bool skip_hdwt = false);

extern void gpu_dwt97_level_fp32(
    float* d_a_f, float* d_b_f, const int32_t* d_input,
    int width, int height, int stride, int level, cudaStream_t st,
    bool skip_hdwt = false);

/* Forward-declare the RGB+HDWT0 fused kernels used by V18 */
extern __global__ void kernel_rgb48_xyz_hdwt0_1ch_2row(
    const uint16_t* __restrict__ d_rgb16,
    const __half*   __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    __half* __restrict__ d_tmp, int ch,
    int width, int height, int rgb_stride, int dwt_stride);

extern __global__ void kernel_rgb48_xyz_hdwt0_1ch_2row_fp32(
    const uint16_t* __restrict__ d_rgb16,
    const float*    __restrict__ d_lut_in,
    const uint16_t* __restrict__ d_lut_out,
    const float*    __restrict__ d_matrix,
    float* __restrict__ d_tmp, int ch,
    int width, int height, int rgb_stride, int dwt_stride);


/* ============================================================================
 * SECTION 8: CudaJ2KEncoderImpl — Struct with all fields
 * ============================================================================ */

struct CudaJ2KEncoderImpl
{
    /* DWT buffers — FP16 for fast mode */
    __half*  d_a[3]  = {nullptr};
    __half*  d_b[3]  = {nullptr};

    /* V196: FP32 DWT buffers for high-precision correct mode */
    float*   d_a_f32[3] = {nullptr};
    float*   d_b_f32[3] = {nullptr};
    size_t   buf_pixels_f32 = 0;

    int32_t* d_in[3] = {nullptr};
    uint8_t* d_packed = nullptr;

    cudaStream_t stream[3] = {nullptr};
    size_t buf_pixels = 0;

    /* Colour conversion device buffers */
    uint16_t* d_rgb16   = nullptr;
    __half*   d_lut_in     = nullptr;
    float*    d_lut_in_f32 = nullptr;
    uint16_t* d_lut_out    = nullptr;
    float*    d_matrix     = nullptr;
    size_t    rgb_buf_pixels = 0;
    bool      colour_loaded  = false;

    /* EBCOT T1 buffers — pre-allocated pool */
    CodeBlockInfo* d_cb_info     = nullptr;
    uint8_t*  d_ebcot_data[3]    = {nullptr};
    uint16_t* d_ebcot_len[3]     = {nullptr};
    uint8_t*  d_ebcot_npasses[3] = {nullptr};
    uint16_t* d_ebcot_passlens[3]= {nullptr};
    uint8_t*  d_ebcot_numbp[3]   = {nullptr};
    uint8_t*  h_ebcot_data[3]    = {nullptr};
    uint16_t* h_ebcot_len[3]     = {nullptr};
    uint8_t*  h_ebcot_npasses[3] = {nullptr};
    uint16_t* h_ebcot_passlens[3]= {nullptr};
    uint8_t*  h_ebcot_numbp[3]   = {nullptr};
    int       ebcot_num_cbs      = 0;
    size_t    ebcot_pool_cbs     = 0;

    std::vector<SubbandGeom> ebcot_subbands;
    std::vector<CodeBlockInfo> ebcot_cb_table;

    /* V16: LRU cache for code-block tables */
    struct CachedCBTable {
        int width, height;
        float step;
        std::vector<CodeBlockInfo> cb_table;
        std::vector<SubbandGeom> subbands;
    };
    std::array<CachedCBTable, 4> cb_cache;
    int cb_cache_next = 0;

    /* V42: CUDA Graph per component (captures DWT+quantize+T1+D2H) */
    cudaGraphExec_t cg_v18[3] = {nullptr, nullptr, nullptr};
    int    cg_width       = 0;
    int    cg_height      = 0;
    size_t cg_per_comp   = 0;
    bool   cg_is_4k       = false;
    bool   cg_is_3d       = false;
    int    cg_target_bytes = 0;
    float  cg_base_step    = 0.0f;
    int    cg_num_cbs      = 0;
    bool   cg_fast_mode    = false;

    /* V42 fallback: direct launches when graph capture fails */
    bool   graphs_failed   = false;

    bool init() {
        for (int c = 0; c < 3; ++c)
            if (cudaStreamCreate(&stream[c]) != cudaSuccess) return false;
        return true;
    }

    void ensure_buffers(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels) return;
        cleanup_dwt_buffers();
        size_t pad = static_cast<size_t>(width) * 8 * sizeof(__half) + 64;
        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a[c],  pixels * sizeof(__half) + pad);
            cudaMalloc(&d_b[c],  pixels * sizeof(__half) + pad);
            cudaMalloc(&d_in[c], pixels * sizeof(int32_t));
        }
        cudaMalloc(&d_packed, pixels * 3 * sizeof(uint8_t));
        buf_pixels = pixels;
    }

    void ensure_buffers_f32(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= buf_pixels_f32) return;
        for (int c = 0; c < 3; ++c) {
            if (d_a_f32[c]) cudaFree(d_a_f32[c]);
            if (d_b_f32[c]) cudaFree(d_b_f32[c]);
        }
        size_t pad = static_cast<size_t>(width) * 8 * sizeof(float) + 64;
        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_a_f32[c], pixels * sizeof(float) + pad);
            cudaMalloc(&d_b_f32[c], pixels * sizeof(float) + pad);
        }
        buf_pixels_f32 = pixels;
    }

    void ensure_rgb_buffer(int width, int height) {
        size_t pixels = static_cast<size_t>(width) * height;
        if (pixels <= rgb_buf_pixels) return;
        if (d_rgb16) { cudaFree(d_rgb16); d_rgb16 = nullptr; }
        cudaMalloc(&d_rgb16, pixels * 3 * sizeof(uint16_t));
        rgb_buf_pixels = pixels;
    }

    void ensure_ebcot_pool(int num_cbs) {
        if (static_cast<size_t>(num_cbs) <= ebcot_pool_cbs) return;
        if (d_cb_info) cudaFree(d_cb_info);
        for (int c = 0; c < 3; ++c) {
            if (d_ebcot_data[c])     cudaFree(d_ebcot_data[c]);
            if (d_ebcot_len[c])      cudaFree(d_ebcot_len[c]);
            if (d_ebcot_npasses[c])   cudaFree(d_ebcot_npasses[c]);
            if (d_ebcot_passlens[c])  cudaFree(d_ebcot_passlens[c]);
            if (d_ebcot_numbp[c])    cudaFree(d_ebcot_numbp[c]);
            if (h_ebcot_data[c])     cudaFreeHost(h_ebcot_data[c]);
            if (h_ebcot_len[c])      cudaFreeHost(h_ebcot_len[c]);
            if (h_ebcot_npasses[c])   cudaFreeHost(h_ebcot_npasses[c]);
            if (h_ebcot_passlens[c])  cudaFreeHost(h_ebcot_passlens[c]);
            if (h_ebcot_numbp[c])    cudaFreeHost(h_ebcot_numbp[c]);
        }
        size_t alloc_cbs = static_cast<size_t>(num_cbs) * 5 / 4;
        cudaMalloc(&d_cb_info, alloc_cbs * sizeof(CodeBlockInfo));
        for (int c = 0; c < 3; ++c) {
            cudaMalloc(&d_ebcot_data[c],    alloc_cbs * CB_BUF_SIZE);
            cudaMalloc(&d_ebcot_len[c],     alloc_cbs * sizeof(uint16_t));
            cudaMalloc(&d_ebcot_npasses[c],  alloc_cbs * sizeof(uint8_t));
            cudaMalloc(&d_ebcot_passlens[c], alloc_cbs * MAX_PASSES * sizeof(uint16_t));
            cudaMalloc(&d_ebcot_numbp[c],   alloc_cbs * sizeof(uint8_t));
            cudaHostAlloc(&h_ebcot_data[c],    alloc_cbs * CB_BUF_SIZE, cudaHostAllocDefault);
            cudaHostAlloc(&h_ebcot_len[c],     alloc_cbs * sizeof(uint16_t), cudaHostAllocDefault);
            cudaHostAlloc(&h_ebcot_npasses[c],  alloc_cbs * sizeof(uint8_t), cudaHostAllocDefault);
            cudaHostAlloc(&h_ebcot_passlens[c], alloc_cbs * MAX_PASSES * sizeof(uint16_t), cudaHostAllocDefault);
            cudaHostAlloc(&h_ebcot_numbp[c],   alloc_cbs * sizeof(uint8_t), cudaHostAllocDefault);
        }
        ebcot_pool_cbs = alloc_cbs;
    }

    void upload_colour_params(GpuColourParams const& params) {
        if (d_lut_in)     cudaFree(d_lut_in);
        if (d_lut_in_f32) cudaFree(d_lut_in_f32);
        if (d_lut_out)    cudaFree(d_lut_out);
        if (d_matrix)     cudaFree(d_matrix);
        cudaMalloc(&d_lut_in,     4096 * sizeof(__half));
        cudaMalloc(&d_lut_in_f32, 4096 * sizeof(float));
        cudaMalloc(&d_lut_out,    4096 * sizeof(uint16_t));
        cudaMalloc(&d_matrix,     9 * sizeof(float));
        __half h_lut_in[4096];
        for (int i = 0; i < 4096; ++i) h_lut_in[i] = __float2half(params.lut_in[i]);
        cudaMemcpy(d_lut_in,     h_lut_in,        4096 * sizeof(__half),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_in_f32, params.lut_in,   4096 * sizeof(float),    cudaMemcpyHostToDevice);
        cudaMemcpy(d_lut_out,    params.lut_out,  4096 * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix,     params.matrix,   9 * sizeof(float),       cudaMemcpyHostToDevice);
        colour_loaded = true;
    }

    void destroy_v18_graphs() {
        for (int c = 0; c < 3; ++c) {
            if (cg_v18[c]) { cudaGraphExecDestroy(cg_v18[c]); cg_v18[c] = nullptr; }
        }
        cg_width = cg_height = 0;
        cg_per_comp = 0;
        cg_num_cbs = 0;
    }

    void cleanup_dwt_buffers() {
        for (int c = 0; c < 3; ++c) {
            if (d_a[c])  { cudaFree(d_a[c]);  d_a[c]  = nullptr; }
            if (d_b[c])  { cudaFree(d_b[c]);  d_b[c]  = nullptr; }
            if (d_in[c]) { cudaFree(d_in[c]); d_in[c] = nullptr; }
        }
        if (d_packed) { cudaFree(d_packed); d_packed = nullptr; }
        buf_pixels = 0;
    }

    ~CudaJ2KEncoderImpl() {
        destroy_v18_graphs();
        cleanup_dwt_buffers();
        for (int c = 0; c < 3; ++c) {
            if (d_a_f32[c]) cudaFree(d_a_f32[c]);
            if (d_b_f32[c]) cudaFree(d_b_f32[c]);
            if (stream[c])  cudaStreamDestroy(stream[c]);
        }
        if (d_rgb16)        cudaFree(d_rgb16);
        if (d_lut_in)       cudaFree(d_lut_in);
        if (d_lut_in_f32)   cudaFree(d_lut_in_f32);
        if (d_lut_out)      cudaFree(d_lut_out);
        if (d_matrix)       cudaFree(d_matrix);
        if (d_cb_info)      cudaFree(d_cb_info);
        for (int c = 0; c < 3; ++c) {
            if (d_ebcot_data[c])     cudaFree(d_ebcot_data[c]);
            if (d_ebcot_len[c])      cudaFree(d_ebcot_len[c]);
            if (d_ebcot_npasses[c])   cudaFree(d_ebcot_npasses[c]);
            if (d_ebcot_passlens[c])  cudaFree(d_ebcot_passlens[c]);
            if (d_ebcot_numbp[c])    cudaFree(d_ebcot_numbp[c]);
            if (h_ebcot_data[c])     cudaFreeHost(h_ebcot_data[c]);
            if (h_ebcot_len[c])      cudaFreeHost(h_ebcot_len[c]);
            if (h_ebcot_npasses[c])   cudaFreeHost(h_ebcot_npasses[c]);
            if (h_ebcot_passlens[c])  cudaFreeHost(h_ebcot_passlens[c]);
            if (h_ebcot_numbp[c])    cudaFreeHost(h_ebcot_numbp[c]);
        }
        buf_pixels_f32 = 0;
    }
};


/* ============================================================================
 * SECTION 9: Helper Functions
 * ============================================================================ */

static float
compute_base_step(int width, int height, size_t per_comp)
{
    size_t pixels = static_cast<size_t>(width) * height;
    float ratio = static_cast<float>(pixels) / std::max(per_comp, static_cast<size_t>(1));
    return std::clamp(ratio * 0.08f, 0.20f, 32.5f);
}

static bool
lookup_cb_table(CudaJ2KEncoderImpl* impl, int width, int height,
                int /*num_levels*/, float step, bool /*is_4k*/,
                std::vector<CodeBlockInfo>& cb_table,
                std::vector<SubbandGeom>& subbands)
{
    float step_q = std::round(step * 1000.0f) / 1000.0f;
    for (int i = 0; i < 4; ++i) {
        if (impl->cb_cache[i].width == width &&
            impl->cb_cache[i].height == height &&
            std::abs(impl->cb_cache[i].step - step_q) < 0.0005f) {
            cb_table = impl->cb_cache[i].cb_table;
            subbands = impl->cb_cache[i].subbands;
            return true;
        }
    }
    return false;
}

static void
store_cb_table(CudaJ2KEncoderImpl* impl, int width, int height,
               float step,
               const std::vector<CodeBlockInfo>& cb_table,
               const std::vector<SubbandGeom>& subbands)
{
    float step_q = std::round(step * 1000.0f) / 1000.0f;
    int idx = impl->cb_cache_next;
    impl->cb_cache[idx].width    = width;
    impl->cb_cache[idx].height   = height;
    impl->cb_cache[idx].step     = step_q;
    impl->cb_cache[idx].cb_table = cb_table;
    impl->cb_cache[idx].subbands = subbands;
    impl->cb_cache_next = (idx + 1) % 4;
}

static void swap_bufs(__half*& a, __half*& b) { __half* t = a; a = b; b = t; }
static void swap_bufs_f32(float*& a, float*& b) { float* t = a; a = b; b = t; }

/* V16: Consolidated D2H — 5 memcpys → 1 2D + 4 linear per component */
static void
ebcot_d2h_consolidated(CudaJ2KEncoderImpl* impl, int num_cbs, int cb_stride)
{
    for (int c = 0; c < 3; ++c) {
        cudaMemcpy2DAsync(impl->h_ebcot_data[c], cb_stride,
                          impl->d_ebcot_data[c], CB_BUF_SIZE,
                          cb_stride, num_cbs,
                          cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_len[c], impl->d_ebcot_len[c],
                        num_cbs * sizeof(uint16_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_npasses[c], impl->d_ebcot_npasses[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_numbp[c], impl->d_ebcot_numbp[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_passlens[c], impl->d_ebcot_passlens[c],
                        (size_t)num_cbs * MAX_PASSES * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, impl->stream[c]);
    }
}


/* ============================================================================
 * SECTION 10: rebuild_v42_comp_graphs — CUDA Graph Capture for V18
 *
 * Captures: DWT levels 1+ + quantize + EBCOT T1 + D2H per component.
 * Does NOT capture H2D or RGB+HDWT0 (those run before graph launch).
 * ============================================================================ */

static void
rebuild_v18_comp_graphs(
    CudaJ2KEncoderImpl* impl,
    int width, int height, int stride,
    size_t per_comp, float base_step, int num_cbs, int num_levels,
    bool is_4k, bool is_3d, bool fast_mode, bool use_fp32_dwt,
    int bp_skip, bool use_bypass, int max_cb_d2h)
{
    impl->destroy_v18_graphs();

    constexpr int EBCOT_THREADS = 64;
    int ebcot_grid = (num_cbs + EBCOT_THREADS - 1) / EBCOT_THREADS;

    for (int c = 0; c < 3; ++c) {
        float step = base_step * (c == 1 ? 1.0f : 1.1f);
        cudaGraph_t g;
        cudaError_t cap_err = cudaStreamBeginCapture(impl->stream[c],
            cudaStreamCaptureModeThreadLocal);
        if (cap_err != cudaSuccess) {
            impl->cg_v18[c] = nullptr;
            continue;
        }

        /* DWT levels 1+ */
        {
            int w = width, h = height;
            for (int level = 0; level < num_levels; ++level) {
                if (use_fp32_dwt) {
                    gpu_dwt97_level_fp32(impl->d_a_f32[c], impl->d_b_f32[c],
                                         impl->d_in[c], w, h, stride, level, impl->stream[c],
                                         level == 0);
                } else {
                    gpu_dwt97_level(impl->d_a[c], impl->d_b[c], nullptr,
                                    impl->d_in[c], w, h, stride, level, impl->stream[c],
                                    level == 0);
                }
                w = (w + 1) / 2;
                h = (h + 1) / 2;
            }
        }

        /* EBCOT T1 */
        if (fast_mode) {
            kernel_ebcot_t1<true, 12, __half><<<ebcot_grid, EBCOT_THREADS, 0, impl->stream[c]>>>(
                impl->d_a[c], stride,
                impl->d_cb_info, num_cbs,
                impl->d_ebcot_data[c], impl->d_ebcot_len[c],
                impl->d_ebcot_npasses[c], impl->d_ebcot_passlens[c],
                impl->d_ebcot_numbp[c], bp_skip, use_bypass);
        } else if (use_fp32_dwt) {
            kernel_ebcot_t1<false, 13, float><<<ebcot_grid, EBCOT_THREADS, 0, impl->stream[c]>>>(
                impl->d_a_f32[c], stride,
                impl->d_cb_info, num_cbs,
                impl->d_ebcot_data[c], impl->d_ebcot_len[c],
                impl->d_ebcot_npasses[c], impl->d_ebcot_passlens[c],
                impl->d_ebcot_numbp[c], bp_skip, use_bypass);
        } else {
            kernel_ebcot_t1<false, 13, __half><<<ebcot_grid, EBCOT_THREADS, 0, impl->stream[c]>>>(
                impl->d_a[c], stride,
                impl->d_cb_info, num_cbs,
                impl->d_ebcot_data[c], impl->d_ebcot_len[c],
                impl->d_ebcot_npasses[c], impl->d_ebcot_passlens[c],
                impl->d_ebcot_numbp[c], bp_skip, use_bypass);
        }

        /* D2H */
        cudaMemcpy2DAsync(impl->h_ebcot_data[c], max_cb_d2h,
                          impl->d_ebcot_data[c], CB_BUF_SIZE,
                          max_cb_d2h, num_cbs,
                          cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_len[c], impl->d_ebcot_len[c],
                        num_cbs * sizeof(uint16_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_npasses[c], impl->d_ebcot_npasses[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_numbp[c], impl->d_ebcot_numbp[c],
                        num_cbs * sizeof(uint8_t), cudaMemcpyDeviceToHost, impl->stream[c]);
        cudaMemcpyAsync(impl->h_ebcot_passlens[c], impl->d_ebcot_passlens[c],
                        (size_t)num_cbs * MAX_PASSES * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost, impl->stream[c]);

        cap_err = cudaStreamEndCapture(impl->stream[c], &g);
        if (cap_err != cudaSuccess) {
            impl->cg_v18[c] = nullptr;
            continue;
        }

        cudaError_t inst_err = cudaGraphInstantiate(&impl->cg_v18[c], g, nullptr, nullptr, 0);
        if (inst_err != cudaSuccess)
            impl->cg_v18[c] = nullptr;
        cudaGraphDestroy(g);
    }

    impl->cg_width       = width;
    impl->cg_height      = height;
    impl->cg_per_comp   = per_comp;
    impl->cg_is_4k       = is_4k;
    impl->cg_is_3d       = is_3d;
    impl->cg_num_cbs     = num_cbs;
    impl->cg_base_step   = base_step;
    impl->cg_fast_mode   = fast_mode;
}


/* ============================================================================
 * SECTION 11: encode_ebcot — Main Encoding Pipeline with CUDA Graphs
 *
 * Flow:
 *   1. H2D upload RGB48
 *   2. RGB+HDWT0 fused kernel (direct launch per component)
 *   3. If graphs valid: cudaGraphLaunch for DWT+T1+D2H
 *      If graphs invalid/failed: direct kernel launch fallback
 *   4. Stream sync + error check
 *   5. Adaptive retry decision
 *   6. CPU T2 assembly (PCRD-OPT slope-sort)
 * ============================================================================ */

std::vector<uint8_t>
CudaJ2KEncoder::encode_ebcot(
    const uint16_t* rgb16,
    int width, int height, int rgb_stride_pixels,
    int64_t bit_rate, int fps, bool is_3d, bool is_4k,
    bool fast_mode)
{
    if (!_initialized || !_colour_params_valid) return {};

    const float fast_step_mult    = fast_mode ? 3.0f : 1.0f;
    const float fast_bitrate_mult = fast_mode ? 0.5f : 1.0f;

    _impl->ensure_buffers(width, height);
    _impl->ensure_rgb_buffer(width, height);

    const bool use_fp32_dwt = !fast_mode;
    if (use_fp32_dwt) _impl->ensure_buffers_f32(width, height);

    const int num_levels = is_4k ? 6 : NUM_DWT_LEVELS;
    int stride = width;

    constexpr int EBCOT_THREADS = 64;
    int  bp_skip    = fast_mode ? 1 : 0;
    bool use_bypass = false;
    const int max_cb_d2h = fast_mode ? 640 : CB_BUF_SIZE;

    /* Step 1: H2D */
    size_t rgb_bytes = static_cast<size_t>(height) * rgb_stride_pixels * sizeof(uint16_t);
    cudaMemcpy(_impl->d_rgb16, rgb16, rgb_bytes, cudaMemcpyHostToDevice);

    /* Step 2: RGB+HDWT0 fused kernel (direct launch per component) */
    int rgb_grid_2row = (height + 1) / 2;

    if (use_fp32_dwt) {
        size_t ch_smem_f32 = static_cast<size_t>(2 * width) * sizeof(float);
        for (int c = 0; c < 3; ++c) {
            kernel_rgb48_xyz_hdwt0_1ch_2row_fp32<<<rgb_grid_2row, H_THREADS_FUSED,
                ch_smem_f32, _impl->stream[c]>>>(
                _impl->d_rgb16,
                _impl->d_lut_in_f32, _impl->d_lut_out, _impl->d_matrix,
                _impl->d_b_f32[c], c,
                width, height, rgb_stride_pixels, stride);
        }
    } else {
        size_t ch_smem = static_cast<size_t>(2 * width) * sizeof(__half);
        for (int c = 0; c < 3; ++c) {
            kernel_rgb48_xyz_hdwt0_1ch_2row<<<rgb_grid_2row, H_THREADS_FUSED,
                ch_smem, _impl->stream[c]>>>(
                _impl->d_rgb16,
                _impl->d_lut_in, _impl->d_lut_out, _impl->d_matrix,
                _impl->d_b[c], c,
                width, height, rgb_stride_pixels, stride);
        }
    }

    /* Step 3: Compute base step and build/lookup code-block table */
    int64_t frame_bits = bit_rate / fps;
    if (is_3d) frame_bits /= 2;
    int64_t target_bytes = static_cast<int64_t>(
        static_cast<double>(frame_bits / 8) * fast_bitrate_mult);

    float base_step = compute_base_step(width, height,
        static_cast<size_t>(target_bytes / 3)) * fast_step_mult;

    const int   adaptive_max_attempts = fast_mode ? 1 : 2;
    const float adaptive_thresh_low  = 0.005f;
    const float adaptive_thresh_high = 0.55f;
    float current_step = base_step;
    int   num_cbs      = _impl->ebcot_num_cbs;

    for (int attempt = 0; attempt < adaptive_max_attempts; ++attempt) {
        /* LRU-cached code-block table lookup */
        if (!lookup_cb_table(_impl.get(), width, height, num_levels,
                             current_step, is_4k,
                             _impl->ebcot_cb_table, _impl->ebcot_subbands)) {
            build_codeblock_table(width, height, stride, num_levels, current_step, is_4k,
                                  _impl->ebcot_cb_table, _impl->ebcot_subbands);
            store_cb_table(_impl.get(), width, height, current_step,
                          _impl->ebcot_cb_table, _impl->ebcot_subbands);
        }
        num_cbs = static_cast<int>(_impl->ebcot_cb_table.size());

        _impl->ensure_ebcot_pool(num_cbs);

        if (num_cbs != _impl->ebcot_num_cbs) {
            cudaMemcpy(_impl->d_cb_info, _impl->ebcot_cb_table.data(),
                       num_cbs * sizeof(CodeBlockInfo), cudaMemcpyHostToDevice);
        }
        _impl->ebcot_num_cbs = num_cbs;

        int ebcot_grid = (num_cbs + EBCOT_THREADS - 1) / EBCOT_THREADS;

        /* ====================================================================
         * CUDA Graph: attempt capture on first call, then launch graph.
         * Fall back to direct launches when capture fails.
         * ==================================================================== */
        size_t per_comp = static_cast<size_t>(target_bytes / 3);

        /* Check if graph needs rebuild */
        bool graph_valid = !_impl->graphs_failed &&
            _impl->cg_v18[0] != nullptr &&
            _impl->cg_width  == width  &&
            _impl->cg_height == height &&
            _impl->cg_num_cbs == num_cbs &&
            _impl->cg_per_comp == per_comp &&
            std::abs(_impl->cg_base_step - current_step) < 0.001f &&
            _impl->cg_is_4k  == is_4k   &&
            _impl->cg_is_3d  == is_3d   &&
            _impl->cg_fast_mode == fast_mode &&
            attempt == 0;  /* graphs are attempt-0 only */

        if (!graph_valid && !_impl->graphs_failed && attempt == 0) {
            /* Try to rebuild graphs */
            rebuild_v18_comp_graphs(_impl.get(), width, height, stride,
                per_comp, current_step, num_cbs, num_levels,
                is_4k, is_3d, fast_mode, use_fp32_dwt,
                bp_skip, use_bypass, max_cb_d2h);
            graph_valid = (_impl->cg_v18[0] != nullptr);
            if (!graph_valid) {
                _impl->graphs_failed = true;
                fprintf(stderr, "[V18] CUDA Graph capture failed, using direct launch fallback\n");
            }
        }

        if (graph_valid && !_impl->graphs_failed) {
            /* Launch per-component CUDA Graphs */
            for (int c = 0; c < 3; ++c)
                cudaGraphLaunch(_impl->cg_v18[c], _impl->stream[c]);
        } else {
            /* Direct launch fallback */
            for (int c = 0; c < 3; ++c) {
                int w = width, h = height;
                for (int level = 0; level < num_levels; ++level) {
                    if (use_fp32_dwt) {
                        gpu_dwt97_level_fp32(_impl->d_a_f32[c], _impl->d_b_f32[c],
                                             _impl->d_in[c], w, h, stride, level, _impl->stream[c],
                                             level == 0);
                    } else {
                        gpu_dwt97_level(_impl->d_a[c], _impl->d_b[c], nullptr,
                                        _impl->d_in[c], w, h, stride, level, _impl->stream[c],
                                        level == 0);
                    }
                    w = (w + 1) / 2;
                    h = (h + 1) / 2;
                }

                if (fast_mode) {
                    kernel_ebcot_t1<true, 12, __half><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                        _impl->d_a[c], stride,
                        _impl->d_cb_info, num_cbs,
                        _impl->d_ebcot_data[c], _impl->d_ebcot_len[c],
                        _impl->d_ebcot_npasses[c], _impl->d_ebcot_passlens[c],
                        _impl->d_ebcot_numbp[c], bp_skip, use_bypass);
                } else if (use_fp32_dwt) {
                    if (attempt == 0) {
                        kernel_ebcot_t1<false, 13, float><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                            _impl->d_a_f32[c], stride,
                            _impl->d_cb_info, num_cbs,
                            _impl->d_ebcot_data[c], _impl->d_ebcot_len[c],
                            _impl->d_ebcot_npasses[c], _impl->d_ebcot_passlens[c],
                            _impl->d_ebcot_numbp[c], bp_skip, use_bypass);
                    } else {
                        kernel_ebcot_t1<false, 16, float><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                            _impl->d_a_f32[c], stride,
                            _impl->d_cb_info, num_cbs,
                            _impl->d_ebcot_data[c], _impl->d_ebcot_len[c],
                            _impl->d_ebcot_npasses[c], _impl->d_ebcot_passlens[c],
                            _impl->d_ebcot_numbp[c], bp_skip, use_bypass);
                    }
                } else {
                    kernel_ebcot_t1<false, 13, __half><<<ebcot_grid, EBCOT_THREADS, 0, _impl->stream[c]>>>(
                        _impl->d_a[c], stride,
                        _impl->d_cb_info, num_cbs,
                        _impl->d_ebcot_data[c], _impl->d_ebcot_len[c],
                        _impl->d_ebcot_npasses[c], _impl->d_ebcot_passlens[c],
                        _impl->d_ebcot_numbp[c], bp_skip, use_bypass);
                }
            }

            /* D2H (direct path) */
            ebcot_d2h_consolidated(_impl.get(), num_cbs, max_cb_d2h);
        }

        /* Sync and error check */
        for (int c = 0; c < 3; ++c)
            cudaStreamSynchronize(_impl->stream[c]);
        {
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
                fprintf(stderr, "GPU pipeline error: %s\n", cudaGetErrorString(err));
        }

        /* Adaptive retry decision */
        if (attempt + 1 >= adaptive_max_attempts) break;
        int64_t total_bytes_used = 0;
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < num_cbs; ++i) {
                uint16_t len = _impl->h_ebcot_len[c][i];
                if (len > static_cast<uint16_t>(max_cb_d2h - 1))
                    len = static_cast<uint16_t>(max_cb_d2h - 1);
                total_bytes_used += len;
            }
        }
        double byte_ratio = static_cast<double>(total_bytes_used)
                          / static_cast<double>(target_bytes);
        if (byte_ratio < adaptive_thresh_low || byte_ratio >= adaptive_thresh_high)
            break;
        current_step *= 0.5f;
    }
    base_step = current_step;

    /* Step 7: CPU T2 assembly (V198 PCRD-OPT slope-sort) */
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

    return result;
}


/**
 * Upload colour conversion LUT+matrix to GPU device memory.
 */
void
CudaJ2KEncoder::set_colour_params(GpuColourParams const& params)
{
    if (!_initialized || !params.valid) return;
    _impl->upload_colour_params(params);
    _colour_params_valid = true;
}


/* Singleton */
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
