/*
 * GPU EBCOT T1 + MQ Arithmetic Coder for JPEG2000
 * Implements ITU-T T.800 Annexes C and D
 *
 * This file is included from both cuda_j2k_encoder.cu and slang_j2k_encoder_v17.cu.
 * All functions use __device__/__host__ qualifiers for GPU/CPU dual compilation.
 *
 * Architecture:
 *   - One thread per code-block (32x32 DWT coefficients)
 *   - MQ coder runs sequentially per thread (not parallelizable within a code-block)
 *   - Massive parallelism across ~7500 code-blocks per frame
 *   - T2 packet assembly on CPU after D2H transfer
 */

#ifndef GPU_EBCOT_H
#define GPU_EBCOT_H

#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>

/* ===== Constants ===== */

static constexpr int CB_DIM      = 32;   /* code-block dimension */
static constexpr int CB_PIXELS   = CB_DIM * CB_DIM;  /* 1024 */
static constexpr int STRIPE_H    = 4;    /* T1 stripe height */
static constexpr int MAX_BPLANES = 16;   /* max bit-planes for 12-bit + guard */
static constexpr int MAX_PASSES  = MAX_BPLANES * 3;  /* 3 passes per bit-plane */
static constexpr int CB_BUF_SIZE = 2048; /* max coded bytes per code-block — ~avg 500 bytes at 150Mbps */

/* Subband types */
static constexpr int SUBBAND_LL = 0;
static constexpr int SUBBAND_HL = 1;
static constexpr int SUBBAND_LH = 2;
static constexpr int SUBBAND_HH = 3;

/* T1 Context label indices (18 contexts for significance + sign + refinement) */
static constexpr int T1_CTXNO_ZC  = 0;   /* zero coding: 9 contexts (0-8) */
static constexpr int T1_CTXNO_SC  = 9;   /* sign coding: 5 contexts (9-13) */
static constexpr int T1_CTXNO_MR  = 14;  /* magnitude refinement: 3 contexts (14-16) */
static constexpr int T1_CTXNO_AGG = 17;  /* run-length / aggregation context */
static constexpr int T1_CTXNO_UNI = 18;  /* uniform context */
static constexpr int T1_NUM_CTXS  = 19;


/* ===== MQ Coder Probability Estimation Table (ITU-T T.800, Table C.2) ===== */

struct MQTableEntry {
    uint16_t qe;      /* probability of LPS */
    uint8_t  nmps;    /* next state after MPS */
    uint8_t  nlps;    /* next state after LPS */
    uint8_t  switchf; /* swap MPS/LPS symbols after LPS? */
};

/* 47 entries from the standard */
__device__ __constant__ static const MQTableEntry MQ_TABLE[47] = {
    {0x5601,  1,  1, 1}, {0x3401,  2,  6, 0}, {0x1801,  3,  9, 0}, {0x0AC1,  4, 12, 0},
    {0x0521,  5, 29, 0}, {0x0221, 38, 33, 0}, {0x5601,  7,  6, 1}, {0x5401,  8, 14, 0},
    {0x4801,  9, 14, 0}, {0x3801, 10, 14, 0}, {0x3001, 11, 17, 0}, {0x2401, 12, 18, 0},
    {0x1C01, 13, 20, 0}, {0x1601, 29, 21, 0}, {0x5601, 15, 14, 1}, {0x5401, 16, 14, 0},
    {0x5101, 17, 15, 0}, {0x4801, 18, 16, 0}, {0x3801, 19, 17, 0}, {0x3401, 20, 18, 0},
    {0x3001, 21, 19, 0}, {0x2801, 22, 19, 0}, {0x2401, 23, 20, 0}, {0x2201, 24, 21, 0},
    {0x1C01, 25, 22, 0}, {0x1801, 26, 23, 0}, {0x1601, 27, 24, 0}, {0x1401, 28, 25, 0},
    {0x1201, 29, 26, 0}, {0x1101, 30, 27, 0}, {0x0AC1, 31, 28, 0}, {0x09C1, 32, 29, 0},
    {0x08A1, 33, 30, 0}, {0x0521, 34, 31, 0}, {0x0441, 35, 32, 0}, {0x02A1, 36, 33, 0},
    {0x0221, 37, 34, 0}, {0x0141, 38, 35, 0}, {0x0111, 39, 36, 0}, {0x0085, 40, 37, 0},
    {0x0049, 41, 38, 0}, {0x0025, 42, 39, 0}, {0x0015, 43, 40, 0}, {0x0009, 44, 41, 0},
    {0x0005, 45, 42, 0}, {0x0001, 45, 43, 0}, {0x5601, 46, 46, 0}
};


/* ===== MQ Coder State ===== */

struct MQCoder {
    uint32_t A;        /* interval (16-bit logical, stored in 32-bit for arithmetic) */
    uint32_t C;        /* code register */
    int      CT;       /* counter */
    uint8_t* bp;       /* byte output pointer */
    uint8_t* start;    /* start of output buffer */
    /* Per-context state: index into MQ_TABLE + MPS symbol */
    uint8_t  index[T1_NUM_CTXS];
    uint8_t  mps[T1_NUM_CTXS];
};


/* ===== MQ Coder Functions (matching OpenJPEG's proven implementation) ===== */

__device__ static void mq_init(MQCoder* mq, uint8_t* buf) {
    mq->A  = 0x8000;
    mq->C  = 0;
    mq->CT = 12;
    /* bp starts at buf-1; first mq_byteout increments bp to buf[0].
     * buf[-1] is a sentinel that should be writable. We use buf[0] as sentinel
     * and start writing at buf[1], adjusting start accordingly. */
    buf[0] = 0;  /* sentinel byte for carry propagation */
    mq->bp    = buf;
    mq->start = buf;

    /* Context initialization per JPEG2000 Table D.7:
     * ZC (0-8): state 0, MPS=0
     * SC (9-13): state 0, MPS=0
     * MR (14-16): state 0, MPS=0
     * AGG (17): state 3, MPS=0
     * UNI (18): state 46, MPS=0
     * Context 0 (ZC first): state 4 per OpenJPEG */
    for (int i = 0; i < T1_NUM_CTXS; i++) {
        mq->index[i] = 0;
        mq->mps[i]   = 0;
    }
    mq->index[0]            = 4;  /* ZC context 0 starts at state 4 */
    mq->index[T1_CTXNO_AGG] = 3;
    mq->index[T1_CTXNO_UNI] = 46;
}

/* Byte output with carry propagation and 0xFF stuffing.
 * Follows OpenJPEG's opj_mqc_byteout() exactly. */
__device__ static void mq_byteout(MQCoder* mq) {
    if (*mq->bp == 0xFF) {
        mq->bp++;
        *mq->bp = static_cast<uint8_t>(mq->C >> 20);
        mq->C &= 0xFFFFF;
        mq->CT = 7;
    } else {
        if (mq->C & 0x8000000) {
            /* Carry propagation */
            (*mq->bp)++;
            mq->C &= 0x7FFFFFF;
            if (*mq->bp == 0xFF) {
                /* Carry caused previous byte to become 0xFF — use 7-bit mode */
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
        if (mq->CT == 0)
            mq_byteout(mq);
    } while (mq->A < 0x8000);
}

/* MQ encode matching OpenJPEG's opj_mqc_encode() */
__device__ static void mq_encode(MQCoder* mq, int ctx, int d) {
    uint8_t  curS = mq->index[ctx];
    uint16_t qe   = MQ_TABLE[curS].qe;
    int      mpsv = mq->mps[ctx];

    mq->A -= qe;

    if (d != mpsv) {
        /* LPS */
        if (mq->A < qe) {
            /* Conditional exchange: A is already the LPS interval */
            mq->C += qe;
        } else {
            /* Normal: assign LPS the lower interval */
            mq->A = qe;
        }
        mq->index[ctx] = MQ_TABLE[curS].nlps;
        if (MQ_TABLE[curS].switchf)
            mq->mps[ctx] = 1 - mpsv;
        mq_renorme(mq);
    } else {
        /* MPS */
        if (mq->A < 0x8000) {
            if (mq->A < qe) {
                /* Conditional exchange: swap intervals */
                mq->A = qe;
            } else {
                mq->C += qe;
            }
            mq->index[ctx] = MQ_TABLE[curS].nmps;
            mq_renorme(mq);
        } else {
            mq->C += qe;
        }
    }
}

/* MQ flush (Figure C.11: SETBITS + 2× BYTEOUT) */
__device__ static void mq_flush(MQCoder* mq) {
    /* SETBITS: set C to distinguish end-of-data from fill bits */
    uint32_t tempc = mq->C + mq->A;
    mq->C |= 0xFFFF;
    if (mq->C >= tempc)
        mq->C -= 0x8000;
    /* Transfer remaining bits */
    mq->C <<= mq->CT;
    mq_byteout(mq);
    mq->C <<= mq->CT;
    mq_byteout(mq);
    /* Don't include trailing 0xFF in output */
    if (*mq->bp == 0xFF)
        mq->bp--;
}


/* ===== T1 Context Determination (ITU-T T.800, Table D.1/D.2/D.3) ===== */
/* Optimized: sentinel-padded sigma array eliminates all boundary checks.
 * sigma_pad[0] = sentinel (all zeros), sigma_pad[1..cbh] = real rows,
 * sigma_pad[cbh+1] = sentinel (all zeros).
 * Columns are shifted by 1 bit: real column c is stored at bit (c+1).
 * Bits 0 and (cbw+1) are always 0 sentinels.
 * This eliminates 4 boundary checks per sig_bit() call (~245K calls). */

/* Padded sigma access: no bounds checks needed.
 * sigma_pad has cbh+2 rows; real row r maps to sigma_pad[r+1].
 * Real column c maps to bit (c+1).
 * Sentinels guarantee bit 0, bit cbw+1, row 0, and row cbh+1 are all zero. */
#define SIGP(sigma_pad, r, c)   (((sigma_pad)[(r)+1] >> ((c)+1)) & 1)

/* Bitwise neighbor-OR for all 32 columns of a given row.
 * Returns a bitmask where bit (c+1) is set if coefficient at real column c
 * has any significant neighbor. Eliminates 8 sig_bit calls per coefficient.
 * This computes the result for ALL columns in ~6 bitwise ops vs 8*32 function calls. */
__device__ static uint32_t has_sig_neighbor_mask(const uint32_t* sigma_pad, int r) {
    /* sigma_pad[r+1] = current row, sigma_pad[r] = north, sigma_pad[r+2] = south.
     * Real column c is at bit (c+1). Shift left = west neighbor, shift right = east neighbor. */
    uint32_t north = sigma_pad[r];      /* row above (r-1 in real coords) */
    uint32_t cur   = sigma_pad[r + 1];  /* current row */
    uint32_t south = sigma_pad[r + 2];  /* row below (r+1 in real coords) */

    /* OR of all 8 neighbors for each column position */
    uint32_t nbr = north | south                   /* N, S */
                 | (cur << 1) | (cur >> 1)          /* W, E */
                 | (north << 1) | (north >> 1)      /* NW, NE */
                 | (south << 1) | (south >> 1);     /* SW, SE */
    return nbr;
}

/* Compute sh (vertical=N+S), sv (horizontal=W+E), sd (diagonal=NW+NE+SW+SE)
 * for a single coefficient at real row r, real column c.
 * Uses padded sigma; no bounds checks. */
__device__ static void neighbor_counts(const uint32_t* sigma_pad, int r, int c,
                                        int* sh_out, int* sv_out, int* sd_out) {
    int pc = c + 1;  /* padded column */
    int pr = r + 1;  /* padded row */
    uint32_t north = sigma_pad[pr - 1];
    uint32_t south = sigma_pad[pr + 1];
    uint32_t cur   = sigma_pad[pr];

    *sh_out = ((north >> pc) & 1) + ((south >> pc) & 1);             /* N + S */
    *sv_out = ((cur >> (pc - 1)) & 1) + ((cur >> (pc + 1)) & 1);    /* W + E */
    *sd_out = ((north >> (pc - 1)) & 1) + ((north >> (pc + 1)) & 1) /* NW + NE */
            + ((south >> (pc - 1)) & 1) + ((south >> (pc + 1)) & 1);/* SW + SE */
}

/* Bitwise any-significant-neighbor test for a single coefficient.
 * Faster than OR-ing 8 sig_bit calls; no bounds checks. */
__device__ static int has_sig_neighbor(const uint32_t* sigma_pad, int r, int c) {
    int pc = c + 1;
    int pr = r + 1;
    uint32_t north = sigma_pad[pr - 1];
    uint32_t south = sigma_pad[pr + 1];
    uint32_t cur   = sigma_pad[pr];
    /* Check all 8 neighbors with bitwise ops — single expression, no branches */
    return (((north >> (pc - 1)) | (north >> pc) | (north >> (pc + 1)) |
             (cur >> (pc - 1)) | (cur >> (pc + 1)) |
             (south >> (pc - 1)) | (south >> pc) | (south >> (pc + 1))) & 1);
}

/* Zero-coding context LUT (Table D.1).
 * Index: subband * 45 + sh * 15 + sv * 5 + min(sd, 4)
 * sh in [0,2], sv in [0,2], sd in [0,4] — 4 * 3 * 3 * 5 = 180 entries.
 * Stored in __constant__ memory for fast broadcast across warps. */
__device__ __constant__ static const uint8_t ZC_LUT[180] = {
    /* LL subband (same as LH): sv primary, sh secondary.
     * Index: sh*15 + sv*5 + min(sd,4). Derived from Table D.1. */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 6, 7, 7, 7, 7,
    /* sh=0, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=1, sv=0, sd=0..4 */ 3, 4, 4, 4, 4,
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=1, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=0, sd=0..4 */ 5, 5, 5, 5, 5,
    /* sh=2, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* HL subband: sh primary (N+S), sv secondary (W+E). */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 3, 4, 4, 4, 4,
    /* sh=0, sv=2, sd=0..4 */ 5, 5, 5, 5, 5,
    /* sh=1, sv=0, sd=0..4 */ 6, 7, 7, 7, 7,
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=1, sv=2, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=2, sv=0, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=1, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* LH subband: sv primary, sh secondary (same table as LL). */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 6, 7, 7, 7, 7,
    /* sh=0, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=1, sv=0, sd=0..4 */ 3, 4, 4, 4, 4,
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=1, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=0, sd=0..4 */ 5, 5, 5, 5, 5,
    /* sh=2, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* HH subband: sd primary, hv=sh+sv secondary. */
    /* sh=0, sv=0, sd=0..4 */ 0, 3, 6, 8, 8,
    /* sh=0, sv=1, sd=0..4 */ 1, 4, 7, 8, 8,
    /* sh=0, sv=2, sd=0..4 */ 2, 5, 7, 8, 8,
    /* sh=1, sv=0, sd=0..4 */ 1, 4, 7, 8, 8,
    /* sh=1, sv=1, sd=0..4 */ 2, 5, 7, 8, 8,
    /* sh=1, sv=2, sd=0..4 */ 2, 5, 7, 8, 8,
    /* sh=2, sv=0, sd=0..4 */ 2, 5, 7, 8, 8,
    /* sh=2, sv=1, sd=0..4 */ 2, 5, 7, 8, 8,
    /* sh=2, sv=2, sd=0..4 */ 2, 5, 7, 8, 8
};

/* Zero-coding context via LUT — replaces t1_zero_context with a single table lookup.
 * Eliminates the switch/case and 8 sig_bit calls; neighbor counts are pre-computed. */
__device__ static int t1_zero_context_fast(const uint32_t* sigma_pad, int r, int c, int subband) {
    int sh, sv, sd;
    neighbor_counts(sigma_pad, r, c, &sh, &sv, &sd);
    int sd_clamp = (sd < 4) ? sd : 4;  /* clamp diagonal to [0,4] for LUT index */
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + sd_clamp];
}

/* Sign coding context (Table D.2) — returns context label offset (0-4) + XOR bit.
 * Optimized: uses padded sigma to eliminate bounds checks. Branchless where possible. */
__device__ static void t1_sign_context_fast(const uint32_t* sigma_pad, const uint32_t* sign_bits,
                                             int r, int c, int cbw, int cbh,
                                             int* ctx_out, int* xor_bit_out) {
    int pc = c + 1;  /* padded column */
    int pr = r + 1;  /* padded row */

    /* Horizontal contribution: W and E neighbors */
    uint32_t sig_w = (sigma_pad[pr] >> (pc - 1)) & 1;
    uint32_t sig_e = (sigma_pad[pr] >> (pc + 1)) & 1;
    /* sign_bits uses real indexing (not padded), but we guard with sig_w/sig_e */
    int hpos = 0, hneg = 0;
    if (sig_w) { if ((sign_bits[r] >> (c - 1)) & 1) hneg = 1; else hpos = 1; }
    if (sig_e) { if ((sign_bits[r] >> (c + 1)) & 1) hneg = 1; else hpos = 1; }
    int hc = hpos - hneg;

    /* Vertical contribution: N and S neighbors */
    uint32_t sig_n = (sigma_pad[pr - 1] >> pc) & 1;
    uint32_t sig_s = (sigma_pad[pr + 1] >> pc) & 1;
    int vpos = 0, vneg = 0;
    if (sig_n) { if ((sign_bits[r - 1] >> c) & 1) vneg = 1; else vpos = 1; }
    if (sig_s) { if ((sign_bits[r + 1] >> c) & 1) vneg = 1; else vpos = 1; }
    int vc = vpos - vneg;

    /* Normalize: if hc < 0, flip signs */
    if (hc < 0) { hc = -hc; vc = -vc; }

    /* Sign prediction + context via small LUT (3x3 = 9 entries, packed in code) */
    /* hc in {0,1}, vc in {-1,0,1} after normalization */
    if (hc == 0) {
        *xor_bit_out = (vc < 0) ? 1 : 0;
        *ctx_out = (vc == 0) ? 0 : 1;
    } else {
        *xor_bit_out = 0;
        *ctx_out = (vc < 0) ? 2 : (vc == 0) ? 3 : 4;
    }
}

/* Magnitude refinement context (Table D.3) — uses padded sigma, no bounds checks.
 * Returns context label offset (0-2). */
__device__ static int t1_mr_context_fast(const uint32_t* sigma_pad, const uint32_t* firstref_bits,
                                          int r, int c) {
    if (!((firstref_bits[r] >> c) & 1))
        return 2;  /* not first refinement */
    /* Any significant neighbor? Use bitwise OR — single expression, no branches */
    return has_sig_neighbor(sigma_pad, r, c) ? 1 : 0;
}

/* Legacy sig_bit kept for any external users (not used in hot path) */
__device__ static int sig_bit(const uint32_t* sigma_bits, int r, int c, int h) {
    if (r < 0 || r >= h || c < 0 || c >= CB_DIM) return 0;
    return (sigma_bits[r] >> c) & 1;
}


/* ===== Code-Block Info ===== */

struct CodeBlockInfo {
    int16_t  x0, y0;         /* top-left in DWT array (d_a[c]) */
    int16_t  width, height;  /* actual dimensions (<=32) */
    uint8_t  subband_type;   /* SUBBAND_LL/HL/LH/HH */
    uint8_t  level;          /* DWT level (0=finest detail) */
    float    quant_step;     /* quantization step for this subband */
};


/* ===== EBCOT T1 Kernel: one code-block per thread ===== */

/* V141: launch_bounds(64, 16) — 64 threads/block, 16 blocks/SM (1024/SM).
 * With 2KB mag[] per thread, smaller blocks cut L1 pressure.
 * Caller must launch with exactly 64 threads/block. */
__global__ __launch_bounds__(64, 16) void kernel_ebcot_t1(
    const __half* __restrict__ d_dwt,   /* DWT output coefficients (d_a[c]) */
    int dwt_stride,                      /* row stride of DWT array (= image width) */
    const CodeBlockInfo* __restrict__ d_cb_info,  /* code-block metadata */
    int num_cbs,                         /* number of code-blocks */
    uint8_t*  __restrict__ d_coded_data, /* output: coded bytes per CB (num_cbs * CB_BUF_SIZE) */
    uint16_t* __restrict__ d_coded_len,  /* output: actual coded length per CB */
    uint8_t*  __restrict__ d_num_passes, /* output: number of coding passes per CB */
    uint16_t* __restrict__ d_pass_lengths /* output: cumulative length at each pass (num_cbs * MAX_PASSES) */
)
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

    /* 1. Load and quantize DWT coefficients.
     * Pack sigma/sign/firstref/coded as uint32 bitfields (1 bit per column per row).
     *
     * OPTIMIZATION: sigma_pad[] has sentinel rows at [0] and [cbh+1] (all zeros)
     * and columns shifted by +1 (bit 0 and bit cbw+1 are sentinel zeros).
     * This eliminates ALL boundary checks from neighbor significance lookups. */
    int16_t mag[CB_DIM * CB_DIM];
    uint32_t sigma_pad[CB_DIM + 2]; /* padded: sigma_pad[0]=sentinel, [1..cbh]=real, [cbh+1]=sentinel */
    uint32_t sign_bits[CB_DIM];     /* sign: bit c = negative (real indexing, not padded) */
    uint32_t firstref_bits[CB_DIM]; /* first refinement: bit c = first time */
    uint32_t coded_bits[CB_DIM];    /* coded in current pass */

    sigma_pad[0] = 0;           /* north sentinel row */
    sigma_pad[cbh + 1] = 0;     /* south sentinel row */
    for (int r = 0; r < cbh; r++) {
        sigma_pad[r + 1] = 0;   /* real rows stored at [r+1], shifted by 1 bit for column sentinel */
        firstref_bits[r] = 0xFFFFFFFF; /* all 1s = first time */
    }

    float inv_step = __frcp_rn(cbi.quant_step);  /* fast reciprocal */
    int max_mag = 0;
    for (int r = 0; r < cbh; r++) {
        uint32_t sb = 0;
        const __half* row_ptr = d_dwt + (cbi.y0 + r) * dwt_stride + cbi.x0;
        for (int c = 0; c < cbw; c++) {
            float val = __half2float(__ldg(row_ptr + c));
            int q = __float2int_rn(fabsf(val) * inv_step);
            mag[r * cbw + c] = static_cast<int16_t>(q);
            if (val < 0.0f) sb |= (1u << c);
            max_mag |= q;  /* bitwise OR instead of branch — captures all bits */
        }
        sign_bits[r] = sb;
    }
    /* max_mag is now the bitwise OR of all magnitudes; num_bp is the position of the MSB */

    /* 2. Compute number of bit-planes using __clz (count leading zeros) */
    int num_bp = (max_mag > 0) ? (32 - __clz(max_mag)) : 0;
    if (num_bp == 0) {
        d_coded_len[cb_idx] = 0;
        d_num_passes[cb_idx] = 0;
        return;
    }

    /* 3. Initialize MQ coder */
    uint8_t* out_buf = d_coded_data + (size_t)cb_idx * CB_BUF_SIZE;
    out_buf[0] = 0;  /* sentinel for MQ byte stuffing */

    MQCoder mq;
    mq_init(&mq, out_buf);

    int total_passes = 0;
    uint16_t* pass_lens = d_pass_lengths + (size_t)cb_idx * MAX_PASSES;

    /* 4. Bit-plane coding loop.
     * Per JPEG2000 standard: first bit-plane has only cleanup pass (CUP).
     * Subsequent bit-planes: SPP → MRP → CUP.
     *
     * OPTIMIZATION SUMMARY:
     *   - sigma_pad[] eliminates all boundary checks (~245K sig_bit calls had 4 branches each)
     *   - has_sig_neighbor() replaces 8 sig_bit calls with 1 bitwise expression
     *   - t1_zero_context_fast() uses LUT instead of switch/case + 8 sig_bit calls
     *   - t1_sign_context_fast() uses padded sigma, no bounds checks
     *   - t1_mr_context_fast() uses bitwise neighbor OR, no bounds checks
     *   - Early pass skip: SPP/MRP skip when no significant coefficients exist yet
     *
     * Padded sigma convention: real row r -> sigma_pad[r+1], real col c -> bit (c+1).
     * Macro SIGP(sigma_pad, r, c) reads significance without bounds checks. */

    /* Track whether any coefficient is significant yet (for early pass skipping) */
    int any_significant = 0;

    for (int bp = num_bp - 1; bp >= 0; bp--) {
        bool first_bp = (bp == num_bp - 1);

        for (int r2 = 0; r2 < cbh; r2++) coded_bits[r2] = 0;

        /* --- Significance Propagation Pass (SPP) — skip for first bit-plane --- */
        if (!first_bp) {
        if (any_significant) {
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                uint32_t cmask_pad = 1u << (c + 1);
                uint32_t cmask = 1u << c;
                for (int r = stripe_y; r < stripe_end; r++) {
                    if (sigma_pad[r + 1] & cmask_pad) continue;
                    if (!has_sig_neighbor(sigma_pad, r, c)) continue;

                    int bit = (mag[r * cbw + c] >> bp) & 1;
                    int zc = t1_zero_context_fast(sigma_pad, r, c, cbi.subband_type);
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
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end SPP */

        /* --- Magnitude Refinement Pass (MRP) — skip for first bit-plane --- */
        if (!first_bp) {
        /* OPTIMIZATION D: Skip MRP entirely if no coefficient is significant yet */
        if (any_significant) {
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                uint32_t cmask_pad = 1u << (c + 1);
                for (int r = stripe_y; r < stripe_end; r++) {
                    if (!(sigma_pad[r + 1] & cmask_pad) || ((coded_bits[r] >> c) & 1)) continue;
                    int bit = (mag[r * cbw + c] >> bp) & 1;
                    int mr = t1_mr_context_fast(sigma_pad, firstref_bits, r, c);
                    mq_encode(&mq, T1_CTXNO_MR + mr, bit);
                    firstref_bits[r] &= ~(1u << c);
                }
            }
        }
        } /* end any_significant guard */
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end MRP */

        /* --- Cleanup Pass (CUP) — always runs, including first bit-plane --- */
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                uint32_t cmask_pad = 1u << (c + 1);
                uint32_t cmask = 1u << c;
                int all_zero = 1;
                int stripe_len = stripe_end - stripe_y;
                if (stripe_len == 4) {
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if ((sigma_pad[r + 1] & cmask_pad) || (coded_bits[r] & cmask)) { all_zero = 0; break; }
                        /* OPTIMIZATION A: Bitwise neighbor check */
                        if (has_sig_neighbor(sigma_pad, r, c)) { all_zero = 0; break; }
                    }
                } else { all_zero = 0; }

                if (all_zero) {
                    int any_sig = 0;
                    for (int r = stripe_y; r < stripe_end; r++)
                        if ((mag[r * cbw + c] >> bp) & 1) { any_sig = 1; break; }
                    mq_encode(&mq, T1_CTXNO_AGG, any_sig);
                    if (!any_sig) continue;
                    int run_len = 0;
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if ((mag[r * cbw + c] >> bp) & 1) break;
                        run_len++;
                    }
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
                        if ((sigma_pad[r + 1] & cmask_pad) || (coded_bits[r] & cmask)) continue;
                        int bit = (mag[r * cbw + c] >> bp) & 1;
                        int zc = t1_zero_context_fast(sigma_pad, r, c, cbi.subband_type);
                        mq_encode(&mq, T1_CTXNO_ZC + zc, bit);
                        if (bit) {
                            sigma_pad[r + 1] |= cmask_pad;
                            any_significant = 1;
                            t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                            mq_encode(&mq, T1_CTXNO_SC + sc_ctx, ((sign_bits[r] >> c) & 1) ^ xor_bit);
                        }
                        coded_bits[r] |= cmask;
                    }
                } else {
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if ((sigma_pad[r + 1] & cmask_pad) || (coded_bits[r] & cmask)) continue;
                        int bit = (mag[r * cbw + c] >> bp) & 1;
                        int zc = t1_zero_context_fast(sigma_pad, r, c, cbi.subband_type);
                        mq_encode(&mq, T1_CTXNO_ZC + zc, bit);
                        if (bit) {
                            sigma_pad[r + 1] |= cmask_pad;
                            any_significant = 1;
                            int sc_ctx, xor_bit;
                            t1_sign_context_fast(sigma_pad, sign_bits, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                            mq_encode(&mq, T1_CTXNO_SC + sc_ctx, ((sign_bits[r] >> c) & 1) ^ xor_bit);
                        }
                        coded_bits[r] |= cmask;
                    }
                }
            }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
    }

    /* 5. MQ flush */
    mq_flush(&mq);

    /* V133 OPT: No byte shift — T2 will read starting at buf[1] (skip sentinel byte).
     * This eliminates the per-CB byte-by-byte copy (up to 2KB/CB × 7500 CBs = 15MB/frame). */
    int coded_len = static_cast<int>(mq.bp - mq.start);
    if (coded_len < 0) coded_len = 0;
    if (coded_len > CB_BUF_SIZE - 1) coded_len = CB_BUF_SIZE - 1;

    /* Byte-stuffing safety: scan starting at buf[1] (actual coded data start).
     * Combined with length trim if last byte is 0xFF. */
    for (int i = 1; i < coded_len; i++) {
        if (out_buf[i] == 0xFF && i + 1 <= coded_len)
            out_buf[i + 1] &= 0x7F;
    }
    while (coded_len > 0 && out_buf[coded_len] == 0xFF)
        coded_len--;

    d_coded_len[cb_idx] = static_cast<uint16_t>(coded_len);
    d_num_passes[cb_idx] = static_cast<uint8_t>(total_passes);
    /* Pass lengths unchanged (still relative to buf[1..coded_len]) */
    for (int p = 0; p < total_passes; p++) {
        int pl = (pass_lens[p] > 0) ? (pass_lens[p] - 1) : 0;
        pass_lens[p] = static_cast<uint16_t>(min(pl, coded_len));
    }
}


#endif /* GPU_EBCOT_H */
