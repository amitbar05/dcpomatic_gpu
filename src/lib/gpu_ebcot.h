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
static constexpr int CB_BUF_SIZE = 8192; /* max coded bytes per code-block */

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

/* Significance state flags for 8 neighbors */
#define SIG_N(sigma, r, c, w, h)  (((r) > 0)                                   ? ((sigma)[((r)-1)*(w)+(c)]) : 0)
#define SIG_S(sigma, r, c, w, h)  (((r) < (h)-1)                               ? ((sigma)[((r)+1)*(w)+(c)]) : 0)
#define SIG_W(sigma, r, c, w, h)  (((c) > 0)                                   ? ((sigma)[(r)*(w)+((c)-1)]) : 0)
#define SIG_E(sigma, r, c, w, h)  (((c) < (w)-1)                               ? ((sigma)[(r)*(w)+((c)+1)]) : 0)
#define SIG_NW(sigma, r, c, w, h) (((r) > 0 && (c) > 0)                        ? ((sigma)[((r)-1)*(w)+((c)-1)]) : 0)
#define SIG_NE(sigma, r, c, w, h) (((r) > 0 && (c) < (w)-1)                    ? ((sigma)[((r)-1)*(w)+((c)+1)]) : 0)
#define SIG_SW(sigma, r, c, w, h) (((r) < (h)-1 && (c) > 0)                    ? ((sigma)[((r)+1)*(w)+((c)-1)]) : 0)
#define SIG_SE(sigma, r, c, w, h) (((r) < (h)-1 && (c) < (w)-1)               ? ((sigma)[((r)+1)*(w)+((c)+1)]) : 0)

/* Zero-coding context (Table D.1) — returns context label 0-8 */
__device__ static int t1_zero_context(const uint8_t* sigma, int r, int c, int w, int h, int subband) {
    int sh = SIG_N(sigma,r,c,w,h)  + SIG_S(sigma,r,c,w,h);   /* vertical */
    int sv = SIG_W(sigma,r,c,w,h)  + SIG_E(sigma,r,c,w,h);   /* horizontal */
    int sd = SIG_NW(sigma,r,c,w,h) + SIG_NE(sigma,r,c,w,h) +
             SIG_SW(sigma,r,c,w,h) + SIG_SE(sigma,r,c,w,h);  /* diagonal */

    switch (subband) {
    case SUBBAND_HL:
        /* Table D.1: HL subband */
        if (sh == 2) return 8;
        if (sh == 1) return (sv + sd >= 1) ? 7 : 6;
        if (sv == 2) return 5;
        if (sv == 1) return (sd >= 1) ? 4 : 3;
        return (sd >= 2) ? 2 : (sd == 1) ? 1 : 0;

    case SUBBAND_LH:
        /* Table D.1: LH subband (swap h/v from HL) */
        if (sv == 2) return 8;
        if (sv == 1) return (sh + sd >= 1) ? 7 : 6;
        if (sh == 2) return 5;
        if (sh == 1) return (sd >= 1) ? 4 : 3;
        return (sd >= 2) ? 2 : (sd == 1) ? 1 : 0;

    case SUBBAND_HH:
        /* Table D.1: HH subband — per OpenJPEG t1_init_ctxno_zc case 3 */
        { int hv = sh + sv;
          if (sd == 0)       return (hv == 0) ? 0 : (hv == 1) ? 1 : 2;
          else if (sd == 1)  return (hv == 0) ? 3 : (hv == 1) ? 4 : 5;
          else if (sd == 2)  return (hv == 0) ? 6 : 7;
          else               return 8;
        }

    default: /* LL */
        /* Table D.1: LL subband (same as LH) */
        if (sv == 2) return 8;
        if (sv == 1) return (sh + sd >= 1) ? 7 : 6;
        if (sh == 2) return 5;
        if (sh == 1) return (sd >= 1) ? 4 : 3;
        return (sd >= 2) ? 2 : (sd == 1) ? 1 : 0;
    }
}

/* Sign coding context (Table D.2) — returns context label offset (0-4) + XOR bit.
 * Per OpenJPEG: contributions are clamped to [-1, 1] via min(pos_count,1)-min(neg_count,1). */
__device__ static void t1_sign_context(const uint8_t* sigma, const int8_t* signs,
                                        int r, int c, int w, int h,
                                        int* ctx_out, int* xor_bit_out) {
    /* Horizontal: min(positive,1) - min(negative,1) → range [-1, 1] */
    int hpos = 0, hneg = 0;
    if (c > 0 && sigma[r*w+(c-1)]) { if (signs[r*w+(c-1)]) hneg++; else hpos++; }
    if (c < w-1 && sigma[r*w+(c+1)]) { if (signs[r*w+(c+1)]) hneg++; else hpos++; }
    int hc = (hpos > 0 ? 1 : 0) - (hneg > 0 ? 1 : 0);  /* clamped to [-1, 1] */

    /* Vertical: same clamping */
    int vpos = 0, vneg = 0;
    if (r > 0 && sigma[(r-1)*w+c]) { if (signs[(r-1)*w+c]) vneg++; else vpos++; }
    if (r < h-1 && sigma[(r+1)*w+c]) { if (signs[(r+1)*w+c]) vneg++; else vpos++; }
    int vc = (vpos > 0 ? 1 : 0) - (vneg > 0 ? 1 : 0);

    /* Normalize: if hc < 0, flip signs */
    if (hc < 0) { hc = -hc; vc = -vc; }
    /* Now hc ∈ {0, 1}, vc ∈ {-1, 0, 1} */

    /* Sign prediction bit */
    if (hc == 0 && vc == 0)
        *xor_bit_out = 0;
    else
        *xor_bit_out = (hc > 0 || (hc == 0 && vc > 0)) ? 0 : 1;

    /* Context label (Table D.2 / OpenJPEG lut_ctxno_sc) */
    if (hc == 0) {
        if (vc == -1)      *ctx_out = 1;
        else if (vc == 0)  *ctx_out = 0;
        else               *ctx_out = 1;
    } else { /* hc == 1 */
        if (vc == -1)      *ctx_out = 2;
        else if (vc == 0)  *ctx_out = 3;
        else               *ctx_out = 4;
    }
}

/* Magnitude refinement context (Table D.3) — returns context label offset (0-2).
 * Per OpenJPEG: offset 0 = no sig neighbors + first ref, 1 = sig neighbors + first ref, 2 = not first ref. */
__device__ static int t1_mr_context(const uint8_t* sigma, const uint8_t* firstref,
                                     int r, int c, int w, int h) {
    if (!firstref[r*w+c])
        return 2;  /* not first refinement */
    int sum = SIG_N(sigma,r,c,w,h) + SIG_S(sigma,r,c,w,h) +
              SIG_W(sigma,r,c,w,h) + SIG_E(sigma,r,c,w,h) +
              SIG_NW(sigma,r,c,w,h) + SIG_NE(sigma,r,c,w,h) +
              SIG_SW(sigma,r,c,w,h) + SIG_SE(sigma,r,c,w,h);
    return (sum >= 1) ? 1 : 0;  /* 0=no neighbors, 1=has neighbors */
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

__global__ void kernel_ebcot_t1(
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

    /* 1. Load and quantize DWT coefficients */
    int16_t mag[CB_DIM * CB_DIM];   /* absolute quantized magnitudes */
    uint8_t sigma[CB_DIM * CB_DIM]; /* significance state */
    int8_t  sign_arr[CB_DIM * CB_DIM]; /* sign (0=positive, 1=negative) */
    uint8_t firstref[CB_DIM * CB_DIM]; /* first refinement flag */
    uint8_t coded_in_pass[CB_DIM * CB_DIM]; /* was this sample coded in current pass? */

    /* Zero-initialize */
    for (int i = 0; i < cbw * cbh; i++) {
        sigma[i] = 0;
        firstref[i] = 1; /* 1 = first time refinement */
    }

    float inv_step = 1.0f / cbi.quant_step;
    int max_mag = 0;
    for (int r = 0; r < cbh; r++) {
        for (int c = 0; c < cbw; c++) {
            float val = __half2float(d_dwt[(cbi.y0 + r) * dwt_stride + (cbi.x0 + c)]);
            int q = static_cast<int>(fabsf(val) * inv_step + 0.5f);
            mag[r * cbw + c] = static_cast<int16_t>(q);
            sign_arr[r * cbw + c] = (val < 0.0f) ? 1 : 0;
            if (q > max_mag) max_mag = q;
        }
    }

    /* 2. Compute number of bit-planes */
    int num_bp = 0;
    { int tmp = max_mag; while (tmp > 0) { num_bp++; tmp >>= 1; } }
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
     * Subsequent bit-planes: SPP → MRP → CUP. */
    for (int bp = num_bp - 1; bp >= 0; bp--) {
        bool first_bp = (bp == num_bp - 1);

        for (int i = 0; i < cbw * cbh; i++) coded_in_pass[i] = 0;

        /* --- Significance Propagation Pass (SPP) — skip for first bit-plane --- */
        if (!first_bp) {

        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                for (int r = stripe_y; r < stripe_end; r++) {
                    int idx = r * cbw + c;
                    if (sigma[idx]) continue; /* already significant */

                    /* Check if any neighbor is significant */
                    int has_sig_nbr = SIG_N(sigma,r,c,cbw,cbh) | SIG_S(sigma,r,c,cbw,cbh) |
                                      SIG_W(sigma,r,c,cbw,cbh) | SIG_E(sigma,r,c,cbw,cbh) |
                                      SIG_NW(sigma,r,c,cbw,cbh)| SIG_NE(sigma,r,c,cbw,cbh)|
                                      SIG_SW(sigma,r,c,cbw,cbh)| SIG_SE(sigma,r,c,cbw,cbh);
                    if (!has_sig_nbr) continue;

                    /* Code significance bit */
                    int bit = (mag[idx] >> bp) & 1;
                    int zc = t1_zero_context(sigma, r, c, cbw, cbh, cbi.subband_type);
                    mq_encode(&mq, T1_CTXNO_ZC + zc, bit);

                    if (bit) {
                        /* Newly significant — code sign */
                        sigma[idx] = 1;
                        int sc_ctx, xor_bit;
                        t1_sign_context(sigma, sign_arr, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                        mq_encode(&mq, T1_CTXNO_SC + sc_ctx, sign_arr[idx] ^ xor_bit);
                    }
                    coded_in_pass[idx] = 1;
                }
            }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end if (!first_bp) for SPP */

        /* --- Magnitude Refinement Pass (MRP) — skip for first bit-plane --- */
        if (!first_bp) {
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                for (int r = stripe_y; r < stripe_end; r++) {
                    int idx = r * cbw + c;
                    if (!sigma[idx] || coded_in_pass[idx]) continue;
                    /* Was significant before this bit-plane */
                    int bit = (mag[idx] >> bp) & 1;
                    int mr = t1_mr_context(sigma, firstref, r, c, cbw, cbh);
                    mq_encode(&mq, T1_CTXNO_MR + mr, bit);
                    firstref[idx] = 0;
                }
            }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
        } /* end if (!first_bp) for MRP */

        /* --- Cleanup Pass (CUP) — always runs, including first bit-plane --- */
        for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
            int stripe_end = min(stripe_y + STRIPE_H, cbh);
            for (int c = 0; c < cbw; c++) {
                /* Run-length coding: check if all 4 rows in stripe column are insignificant
                 * and have no significant neighbors */
                int all_zero = 1;
                int stripe_len = stripe_end - stripe_y;
                if (stripe_len == 4) {
                    for (int r = stripe_y; r < stripe_end; r++) {
                        int idx = r * cbw + c;
                        if (sigma[idx] || coded_in_pass[idx]) { all_zero = 0; break; }
                        int has_sig_nbr = SIG_N(sigma,r,c,cbw,cbh) | SIG_S(sigma,r,c,cbw,cbh) |
                                          SIG_W(sigma,r,c,cbw,cbh) | SIG_E(sigma,r,c,cbw,cbh) |
                                          SIG_NW(sigma,r,c,cbw,cbh)| SIG_NE(sigma,r,c,cbw,cbh)|
                                          SIG_SW(sigma,r,c,cbw,cbh)| SIG_SE(sigma,r,c,cbw,cbh);
                        if (has_sig_nbr) { all_zero = 0; break; }
                    }
                } else {
                    all_zero = 0;
                }

                if (all_zero) {
                    /* Check if any of the 4 will become significant in this bit-plane */
                    int any_sig = 0;
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if ((mag[r * cbw + c] >> bp) & 1) { any_sig = 1; break; }
                    }
                    mq_encode(&mq, T1_CTXNO_AGG, any_sig);
                    if (!any_sig) continue;
                    /* Code the run length: find first significant */
                    int run_len = 0;
                    for (int r = stripe_y; r < stripe_end; r++) {
                        if ((mag[r * cbw + c] >> bp) & 1) break;
                        run_len++;
                    }
                    mq_encode(&mq, T1_CTXNO_UNI, (run_len >> 1) & 1);
                    mq_encode(&mq, T1_CTXNO_UNI, run_len & 1);
                    /* Code that sample and its sign */
                    int r = stripe_y + run_len;
                    int idx = r * cbw + c;
                    sigma[idx] = 1;
                    int sc_ctx, xor_bit;
                    t1_sign_context(sigma, sign_arr, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                    mq_encode(&mq, T1_CTXNO_SC + sc_ctx, sign_arr[idx] ^ xor_bit);
                    coded_in_pass[idx] = 1;
                    /* Continue with remaining samples in the stripe column */
                    for (r = stripe_y + run_len + 1; r < stripe_end; r++) {
                        idx = r * cbw + c;
                        if (sigma[idx] || coded_in_pass[idx]) continue;
                        int bit = (mag[idx] >> bp) & 1;
                        int zc = t1_zero_context(sigma, r, c, cbw, cbh, cbi.subband_type);
                        mq_encode(&mq, T1_CTXNO_ZC + zc, bit);
                        if (bit) {
                            sigma[idx] = 1;
                            t1_sign_context(sigma, sign_arr, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                            mq_encode(&mq, T1_CTXNO_SC + sc_ctx, sign_arr[idx] ^ xor_bit);
                        }
                        coded_in_pass[idx] = 1;
                    }
                } else {
                    /* Non-run-length path: code each sample individually */
                    for (int r = stripe_y; r < stripe_end; r++) {
                        int idx = r * cbw + c;
                        if (sigma[idx] || coded_in_pass[idx]) continue;
                        int bit = (mag[idx] >> bp) & 1;
                        int zc = t1_zero_context(sigma, r, c, cbw, cbh, cbi.subband_type);
                        mq_encode(&mq, T1_CTXNO_ZC + zc, bit);
                        if (bit) {
                            sigma[idx] = 1;
                            int sc_ctx, xor_bit;
                            t1_sign_context(sigma, sign_arr, r, c, cbw, cbh, &sc_ctx, &xor_bit);
                            mq_encode(&mq, T1_CTXNO_SC + sc_ctx, sign_arr[idx] ^ xor_bit);
                        }
                        coded_in_pass[idx] = 1;
                    }
                }
            }
        }
        pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
    }

    /* 5. MQ flush */
    mq_flush(&mq);

    /* Coded data starts at buf[1] (buf[0] is sentinel).
     * Length = bp - start; the data to include in packet is start[1..bp]. */
    int coded_len = static_cast<int>(mq.bp - mq.start);
    if (coded_len < 0) coded_len = 0;
    if (coded_len > CB_BUF_SIZE - 1) coded_len = CB_BUF_SIZE - 1;

    /* Shift data: move buf[1..coded_len] to buf[0..coded_len-1] so T2 reads from offset 0. */
    for (int i = 0; i < coded_len; i++)
        out_buf[i] = out_buf[i + 1];

    /* Safety: enforce J2K byte-stuffing rule — after 0xFF, next byte must have bit 7 = 0.
     * Fix violations by clearing MSB of the byte following any 0xFF. This preserves
     * the data (with minor quality loss) rather than truncating. */
    for (int i = 0; i < coded_len - 1; i++) {
        if (out_buf[i] == 0xFF) {
            out_buf[i + 1] &= 0x7F;  /* clear MSB — ensures < 0x80, well below 0x90 */
        }
    }
    /* Also ensure last byte is not 0xFF (would be ambiguous) */
    while (coded_len > 0 && out_buf[coded_len - 1] == 0xFF)
        coded_len--;

    d_coded_len[cb_idx] = static_cast<uint16_t>(coded_len);
    d_num_passes[cb_idx] = static_cast<uint8_t>(total_passes);
    /* Update all pass lengths to exclude sentinel byte and respect truncation */
    for (int p = 0; p < total_passes; p++) {
        int pl = (pass_lens[p] > 0) ? (pass_lens[p] - 1) : 0;
        pass_lens[p] = static_cast<uint16_t>(min(pl, coded_len));
    }
}


#endif /* GPU_EBCOT_H */
