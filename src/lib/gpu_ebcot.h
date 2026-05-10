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
/* V180: MAX_BPLANES must match the largest MAX_BP template arg used by callers.
 * V230: Raised to 17 to support step=0.0625 (third halving).  At step=0.0625,
 * LL5 step=0.040625 → q_max=4095/0.040625=100850 → 17 bit-planes needed.
 * V238: Raised to 18 to match the MAX_BP=18 kernel instantiation (V233).  At
 * step=0.03125, LL5 step=0.020312 → for very bright content q_max could reach
 * 18 bit-planes; with MAX_BPLANES=17 (MAX_PASSES=51) the kernel would write
 * pass_lens[51] one past the allocated array, corrupting the adjacent CB entry.
 * Formula: max passes = 3*MAX_BP - 2 = 3*18-2 = 52 ≤ MAX_PASSES=54. */
static constexpr int MAX_BPLANES = 18;
static constexpr int MAX_PASSES  = MAX_BPLANES * 3;  /* 3 passes per bit-plane */
/* V229: Raised from 2048 to 16384.  Dense checker patterns (checker_8 HL1/LH1/HH1) have
 * up to 13 bitplanes × ~700 bytes/bp ≈ 9100 bytes per code block.  The old 2048-byte
 * limit stopped coding after ~1280 bytes (guard = CB_BUF_SIZE - 768), truncating dense
 * CBs to ~1 bitplane and capping checker PSNR at 17–30 dB regardless of bitrate. */
static constexpr int CB_BUF_SIZE = 16384; /* max coded bytes per code-block */

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

/* V164: Pack all 5 fields into one uint32_t for a single constant-memory read per mq_encode.
 * Layout: bits[15:0]=qe, bits[21:16]=nmps, bits[27:22]=nlps, bit[28]=switchf.
 * nmps/nlps max = 46 < 64 → 6 bits each. qe max = 0x5601 → 16 bits. switchf = 0 or 1.
 * Reduces divergent constant-memory reads from 3 to 1 per mq_encode call. */
#define MQ_PACK(qe, nmps, nlps, sw) \
    (uint32_t)((qe) | ((uint32_t)(nmps) << 16) | ((uint32_t)(nlps) << 22) | ((uint32_t)(sw) << 28))

/* 47 entries from the standard */
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


/* ===== MQ Coder State ===== */

/* V161: Pack index (6 bits) + mps (1 bit) into one byte: ctx_packed = (index << 1) | mps.
 * This reduces LMEM reads in mq_encode from 2 (index[] + mps[]) to 1 (ctx_packed[]).
 * index values 0-46 fit in 6 bits (max 46 = 0x2E). The MPS bit occupies bit 0.
 * Halves context-state LMEM footprint: 19 bytes instead of 38. */
struct MQCoder {
    uint32_t A;        /* interval (16-bit logical, stored in 32-bit for arithmetic) */
    uint32_t C;        /* code register */
    int      CT;       /* counter */
    uint8_t* bp;       /* byte output pointer */
    uint8_t* start;    /* start of output buffer */
    /* V161: packed context state: bits 7-1 = MQ table index (0-46), bit 0 = MPS symbol */
    uint8_t  ctx_packed[T1_NUM_CTXS];
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

    /* Context initialization per JPEG2000 Table D.7.
     * V161: ctx_packed = (index << 1) | mps; all start at index=0, mps=0 */
    for (int i = 0; i < T1_NUM_CTXS; i++) mq->ctx_packed[i] = 0;
    mq->ctx_packed[0]            = 4 << 1;   /* ZC context 0: state 4, MPS=0 */
    mq->ctx_packed[T1_CTXNO_AGG] = 3 << 1;   /* AGG: state 3, MPS=0 */
    mq->ctx_packed[T1_CTXNO_UNI] = 46 << 1;  /* UNI: state 46, MPS=0 */
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

/* MQ encode matching OpenJPEG's opj_mqc_encode()
 * V161: ctx_packed = (index << 1) | mps — one LMEM read instead of two.
 * V164: MQ_TABLE packed into uint32_t — one constant-memory read replaces 3. */
__device__ static void mq_encode(MQCoder* mq, int ctx, int d) {
    uint8_t  packed = mq->ctx_packed[ctx];   /* V161: one LMEM read */
    uint32_t entry  = MQ_TABLE[packed >> 1]; /* V164: one packed constant-mem read */
    uint16_t qe     = static_cast<uint16_t>(entry);
    int      mpsv   = packed & 1;

    mq->A -= qe;

    if (d != mpsv) {
        /* LPS */
        if (mq->A < qe) {
            mq->C += qe;
        } else {
            mq->A = qe;
        }
        uint8_t nlps    = static_cast<uint8_t>((entry >> 22) & 0x3F);
        uint8_t switchf = static_cast<uint8_t>((entry >> 28) & 1);
        mq->ctx_packed[ctx] = static_cast<uint8_t>((nlps << 1) | (mpsv ^ switchf));
        mq_renorme(mq);
    } else {
        /* MPS */
        if (mq->A < 0x8000) {
            if (mq->A < qe) {
                mq->A = qe;
            } else {
                mq->C += qe;
            }
            uint8_t nmps = static_cast<uint8_t>((entry >> 16) & 0x3F);
            mq->ctx_packed[ctx] = static_cast<uint8_t>((nmps << 1) | mpsv);
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

/* V165: Lightweight MQ byte-align without SETBITS.
 * Used before bypass passes: flushes partial bits to output without
 * the SETBITS step (decoder uses pass length to find pass end). */
__device__ static void mq_align(MQCoder* mq) {
    mq->C <<= mq->CT;
    mq_byteout(mq);
    mq->C <<= mq->CT;
    mq_byteout(mq);
    if (*mq->bp == 0xFF) mq->bp--;
}

/* V178: Re-initialize MQ arithmetic state at a new byte position (after bypass segment).
 * Match OpenJPEG encoder restart: set bp = new_bp - 1 so first mq_byteout writes to new_bp.
 * NO sentinel byte written — the old approach (new_bp[0]=0; bp=new_bp) inserted an extra
 * 0x00 into the bitstream between bypass CUP and the next MQ SPP.  The decoder does not
 * expect this extra byte, causing a 1-byte offset per bypass bit-plane: AC patterns where
 * SPP codes real symbols produce ~12 dB PSNR; DC patterns (empty SPP) were unaffected.
 * bypass_flush guarantees the last bypass byte != 0xFF, so *(new_bp-1) != 0xFF and the
 * carry-propagation branch in mq_byteout cannot fire on the first write.
 * Contexts (ctx_packed[]) are preserved for compression continuity. */
__device__ static void mq_restart(MQCoder* mq, uint8_t* new_bp) {
    mq->bp = new_bp - 1;  /* first mq_byteout increments to new_bp — no extra sentinel byte */
    mq->A  = 0x8000;
    mq->C  = 0;
    mq->CT = 12;
}


/* ===== V165: JPEG2000 Selective Arithmetic Coding Bypass ===== */

/* Bypass coder state (J2K Part 1, C.3.8 BYPASS option).
 * Bits are packed MSB-first into bytes.  After a 0xFF output byte, cnt=1
 * (bit 7 of next byte reserved as stuffed-0); only 7 data bits fit in that
 * byte.  This matches the OpenJPEG decoder's opj_mqc_raw_decode behaviour. */
struct BypassCoder {
    uint8_t* bp;    /* next write position in output buffer */
    uint8_t  accum; /* bits accumulated MSB-first (incomplete byte) */
    int      cnt;   /* bits already in accum; 1 after 0xFF (stuffed-0 pending) */
};

/* V244: Emit one complete byte and set up next-byte state.
 * J2K raw coding (ISO 15444-1 C.3.8): after 0xFF, the next byte's bit 7
 * is a stuffed 0 — so only 7 data bits fit.  We model this by starting
 * with cnt=1 (bit 7 already "used" as 0) rather than writing a separate
 * 0x00 stuffing byte.  The decoder reads 0xFF with ct=7 (OpenJPEG
 * opj_mqc_raw_decode), skips bit 7 of the following byte, and reads
 * 7 data bits — matching this encoder convention exactly. */
__device__ static void bypass_emit(BypassCoder* bc, uint8_t out) {
    *bc->bp++ = out;
    bc->accum = 0;
    bc->cnt   = (out == 0xFF) ? 1 : 0;
}

/* Write one raw bit MSB-first. */
__device__ static void bypass_write(BypassCoder* bc, int bit) {
    bc->accum = static_cast<uint8_t>((bc->accum << 1) | bit);
    if (++bc->cnt == 8)
        bypass_emit(bc, bc->accum);
}

/* Flush remaining partial byte (zero-padded LSBs).
 * cnt=1 (pending stuffed-0 with no data bits) emits 0x00, which is valid. */
__device__ static void bypass_flush(BypassCoder* bc) {
    if (bc->cnt > 0) {
        *bc->bp++ = static_cast<uint8_t>(bc->accum << (8 - bc->cnt));
        bc->cnt   = 0;
    }
}

/* Write n bits (1 ≤ n ≤ 8) from 'bits', MSB-first (bit n-1 first).
 * Handles byte boundary crossing and J2K 0xFF stuffing correctly. */
__device__ static void bypass_write_bits(BypassCoder* bc, uint8_t bits, int n) {
    int space = 8 - bc->cnt;
    if (n <= space) {
        bc->accum = static_cast<uint8_t>((bc->accum << n) | bits);
        bc->cnt += n;
        if (bc->cnt == 8)
            bypass_emit(bc, bc->accum);
    } else {
        /* Split: complete current byte, then add remaining rem bits. */
        int rem = n - space;
        bypass_emit(bc, static_cast<uint8_t>((bc->accum << space) | (bits >> rem)));
        /* After bypass_emit: accum=0, cnt=0 (normal) or 1 (stuffed).
         * Merge rem leftover bits into the accumulator. */
        uint8_t lo = static_cast<uint8_t>(bits & ((1u << rem) - 1));
        bc->accum = static_cast<uint8_t>((bc->accum << rem) | lo);
        bc->cnt  += rem;
        /* If cnt reaches 8 (stuffed case, rem=7): emit the stuffed byte immediately.
         * That byte has bit 7=0 (stuffed) and bits 6..0 = lo, so lo ≤ 0x7F ≠ 0xFF. */
        if (bc->cnt == 8)
            bypass_emit(bc, bc->accum);
    }
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
__device__ static uint64_t has_sig_neighbor_mask(const uint64_t* sigma_pad, int r) {
    /* sigma_pad[r+1] = current row, sigma_pad[r] = north, sigma_pad[r+2] = south.
     * Real column c is at bit (c+1). Shift left = west neighbor, shift right = east neighbor. */
    uint64_t north = sigma_pad[r];      /* row above (r-1 in real coords) */
    uint64_t cur   = sigma_pad[r + 1];  /* current row */
    uint64_t south = sigma_pad[r + 2];  /* row below (r+1 in real coords) */

    /* OR of all 8 neighbors for each column position */
    uint64_t nbr = north | south                   /* N, S */
                 | (cur << 1) | (cur >> 1)          /* W, E */
                 | (north << 1) | (north >> 1)      /* NW, NE */
                 | (south << 1) | (south >> 1);     /* SW, SE */
    return nbr;
}

/* Compute sh (vertical=N+S), sv (horizontal=W+E), sd (diagonal=NW+NE+SW+SE)
 * for a single coefficient at real row r, real column c.
 * Uses padded sigma; no bounds checks. */
__device__ static void neighbor_counts(const uint64_t* sigma_pad, int r, int c,
                                        int* sh_out, int* sv_out, int* sd_out) {
    int pc = c + 1;  /* padded column */
    int pr = r + 1;  /* padded row */
    uint64_t north = sigma_pad[pr - 1];
    uint64_t south = sigma_pad[pr + 1];
    uint64_t cur   = sigma_pad[pr];

    *sh_out = ((north >> pc) & 1) + ((south >> pc) & 1);             /* N + S */
    *sv_out = ((cur >> (pc - 1)) & 1) + ((cur >> (pc + 1)) & 1);    /* W + E */
    *sd_out = ((north >> (pc - 1)) & 1) + ((north >> (pc + 1)) & 1) /* NW + NE */
            + ((south >> (pc - 1)) & 1) + ((south >> (pc + 1)) & 1);/* SW + SE */
}

/* Bitwise any-significant-neighbor test for a single coefficient.
 * Faster than OR-ing 8 sig_bit calls; no bounds checks.
 * sigma_pad is uint64_t so c=31 (pc=32) never overflows — bit 32 of a uint64_t is valid. */
__device__ static int has_sig_neighbor(const uint64_t* sigma_pad, int r, int c) {
    int pc = c + 1;
    int pr = r + 1;
    uint64_t north = sigma_pad[pr - 1];
    uint64_t south = sigma_pad[pr + 1];
    uint64_t cur   = sigma_pad[pr];
    return (((north >> (pc - 1)) | (north >> pc) | (north >> (pc + 1)) |
             (cur >> (pc - 1)) | (cur >> (pc + 1)) |
             (south >> (pc - 1)) | (south >> pc) | (south >> (pc + 1))) & 1);
}

/* Zero-coding context LUT (Table D.1).
 * Index: subband * 45 + sh * 15 + sv * 5 + min(sd, 4)
 * sh in [0,2], sv in [0,2], sd in [0,4] — 4 * 3 * 3 * 5 = 180 entries.
 * Stored in __constant__ memory for fast broadcast across warps. */
__device__ __constant__ static const uint8_t ZC_LUT[180] = {
    /* LL subband: h=sv(W+E) primary, v=sh(N+S) secondary. T.800 Table D.1.
     * Index: sh*15 + sv*5 + min(sd,4).
     * V184: H=1,V=0,D=0 → label 5 (was incorrectly 6). T.800 Table D.1 row
     * "1 0 0 → 5", "1 0 ≥1 → 6". The earlier "ctx 6 for all d" reading collapsed
     * H=1,V=0,D=0 into the same context as D≥1, which is the spec's distinction
     * for LL/LH. With this fix, flat_50000 PSNR jumps from 9 dB to ~30+ dB. */
    /* V184: full LL/LH/HL ZC_LUT rewrite per T.800 Table D.1.
     * In our coordinates sh = ΣV (N+S count), sv = ΣH (W+E count), sd = ΣD.
     * Spec mapping for LL/LH (k_LL(ΣH, ΣV, ΣD)):
     *   ΣH=0,ΣV=0: D=0→0, D=1→1, D≥2→2
     *   ΣH=0,ΣV=1: 3 (any D)
     *   ΣH=0,ΣV=2: 4 (any D)
     *   ΣH=1,ΣV=0: D=0→5, D≥1→6
     *   ΣH=1,ΣV≥1: 7
     *   ΣH=2,ΣV=*: 8
     * For HL the H↔V roles swap, so HL[sh,sv,sd] = k_LL(sh,sv,sd) — i.e. our sh
     * acts as H and sv as V (the reverse of LL/LH).  The previous tables used
     * an old/incorrect derivation that placed "5" where "4" belonged and
     * collapsed several distinct contexts, badly mismatching the OPJ decoder.
     * Fixing this jumps flat_50000 PSNR from 9 dB toward expected. */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 5, 6, 6, 6, 6,  /* H=1,V=0: D=0→5, D≥1→6 */
    /* sh=0, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,  /* H=2,V=0 → 8 */
    /* sh=1, sv=0, sd=0..4 */ 3, 3, 3, 3, 3,  /* H=0,V=1 → 3 */
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,  /* H=1,V=1 → 7 */
    /* sh=1, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,  /* H=2,V=1 → 8 */
    /* sh=2, sv=0, sd=0..4 */ 4, 4, 4, 4, 4,  /* H=0,V=2 → 4 */
    /* sh=2, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,  /* H=1,V=2 → 7 */
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,  /* H=2,V=2 → 8 */
    /* HL subband: H↔V swap.  HL[sh,sv,sd] = k_LL(sh,sv,sd) — sh is H, sv is V. */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 3, 3, 3, 3, 3,  /* HL: H=0,V=1 → 3 */
    /* sh=0, sv=2, sd=0..4 */ 4, 4, 4, 4, 4,  /* HL: H=0,V=2 → 4 */
    /* sh=1, sv=0, sd=0..4 */ 5, 6, 6, 6, 6,  /* HL: H=1,V=0: D=0→5, D≥1→6 */
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,  /* HL: H=1,V=1 → 7 */
    /* sh=1, sv=2, sd=0..4 */ 7, 7, 7, 7, 7,  /* HL: H=1,V=2 → 7 */
    /* sh=2, sv=0, sd=0..4 */ 8, 8, 8, 8, 8,  /* HL: H=2 → 8 */
    /* sh=2, sv=1, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* LH subband: same as LL (H is dominant). */
    /* sh=0, sv=0, sd=0..4 */ 0, 1, 2, 2, 2,
    /* sh=0, sv=1, sd=0..4 */ 5, 6, 6, 6, 6,  /* H=1,V=0: D=0→5, D≥1→6 */
    /* sh=0, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,  /* H=2 → 8 */
    /* sh=1, sv=0, sd=0..4 */ 3, 3, 3, 3, 3,  /* H=0,V=1 → 3 */
    /* sh=1, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=1, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* sh=2, sv=0, sd=0..4 */ 4, 4, 4, 4, 4,  /* H=0,V=2 → 4 */
    /* sh=2, sv=1, sd=0..4 */ 7, 7, 7, 7, 7,
    /* sh=2, sv=2, sd=0..4 */ 8, 8, 8, 8, 8,
    /* HH subband: sd primary, hv=sh+sv secondary (ITU-T T.800 Table D.1).
     * hv=0: d=0→0, d=1→3, d≥2→6.  hv=1: d=0→1, d=1→4, d≥2→7.  hv≥2: d=0→2, d=1→5, d≥2→8. */
    /* sh=0, sv=0, sd=0..4 */ 0, 3, 6, 6, 6,
    /* sh=0, sv=1, sd=0..4 */ 1, 4, 7, 7, 7,
    /* sh=0, sv=2, sd=0..4 */ 2, 5, 8, 8, 8,
    /* sh=1, sv=0, sd=0..4 */ 1, 4, 7, 7, 7,
    /* sh=1, sv=1, sd=0..4 */ 2, 5, 8, 8, 8,
    /* sh=1, sv=2, sd=0..4 */ 2, 5, 8, 8, 8,
    /* sh=2, sv=0, sd=0..4 */ 2, 5, 8, 8, 8,
    /* sh=2, sv=1, sd=0..4 */ 2, 5, 8, 8, 8,
    /* sh=2, sv=2, sd=0..4 */ 2, 5, 8, 8, 8
};

/* Zero-coding context via LUT — replaces t1_zero_context with a single table lookup.
 * Eliminates the switch/case and 8 sig_bit calls; neighbor counts are pre-computed. */
__device__ static int t1_zero_context_fast(const uint64_t* sigma_pad, int r, int c, int subband) {
    int sh, sv, sd;
    neighbor_counts(sigma_pad, r, c, &sh, &sv, &sd);
    int sd_clamp = (sd < 4) ? sd : 4;  /* clamp diagonal to [0,4] for LUT index */
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + sd_clamp];
}

/* V150: fused neighbor-check + ZC context.
 * Reads sigma_pad north/cur/south ONCE, checks for any significant neighbor,
 * and if found computes the ZC context in the same pass.  Returns -1 when
 * there is no significant neighbor (caller should skip the coefficient).
 * Saves 3 LMEM reads vs calling has_sig_neighbor() + t1_zero_context_fast(). */
__device__ static int t1_zc_if_neighbor(const uint64_t* sigma_pad, int r, int c, int subband) {
    int pc = c + 1, pr = r + 1;
    uint64_t north = sigma_pad[pr - 1];
    uint64_t cur   = sigma_pad[pr];
    uint64_t south = sigma_pad[pr + 1];

    /* Check any significant neighbor (same logic as has_sig_neighbor) */
    if (!(((north >> (pc-1)) | (north >> pc) | (north >> (pc+1)) |
           (cur >> (pc-1)) | (cur >> (pc+1)) |
           (south >> (pc-1)) | (south >> pc) | (south >> (pc+1))) & 1))
        return -1;  /* no significant neighbor */

    /* Compute neighbor counts for ZC context (reusing already-loaded rows) */
    int sh = ((north >> pc) & 1) + ((south >> pc) & 1);
    int sv = ((cur >> (pc-1)) & 1) + ((cur >> (pc+1)) & 1);
    int sd = ((north >> (pc-1)) & 1) + ((north >> (pc+1)) & 1)
           + ((south >> (pc-1)) & 1) + ((south >> (pc+1)) & 1);
    int sd_clamp = (sd < 4) ? sd : 4;
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + sd_clamp];
}

/* V169: fused ZC+SC context — reads sigma_pad ONCE for both zero-coding and sign coding.
 * Used in SPP when bit may be 1: computes ZC check and pre-loads sign context data.
 * Returns ZC context index (or -1 if no significant neighbor).
 * If ZC >= 0 and *sc_ctx_out is needed: caller provides sign_bits; sc/xor are filled.
 * Sign context is computed from sigma_pad values BEFORE sigma_pad update — valid because
 * the update sets bit (c+1) which is checked by NEITHER sig_w (bit c) NOR sig_e (bit c+2). */
__device__ static int t1_zc_sc_if_neighbor(const uint64_t* sigma_pad, const uint32_t* sign_bits,
                                            int r, int c, int subband,
                                            int* sc_ctx_out, int* xor_bit_out)
{
    int pc = c + 1, pr = r + 1;
    uint64_t north = sigma_pad[pr - 1];
    uint64_t cur   = sigma_pad[pr];
    uint64_t south = sigma_pad[pr + 1];

    if (!(((north >> (pc-1)) | (north >> pc) | (north >> (pc+1)) |
           (cur >> (pc-1)) | (cur >> (pc+1)) |
           (south >> (pc-1)) | (south >> pc) | (south >> (pc+1))) & 1))
        return -1;

    int sh = ((north >> pc) & 1) + ((south >> pc) & 1);
    int sv = ((cur >> (pc-1)) & 1) + ((cur >> (pc+1)) & 1);
    int sd = ((north >> (pc-1)) & 1) + ((north >> (pc+1)) & 1)
           + ((south >> (pc-1)) & 1) + ((south >> (pc+1)) & 1);
    int zc = ZC_LUT[subband * 45 + sh * 15 + sv * 5 + ((sd < 4) ? sd : 4)];

    /* Precompute sign context using already-loaded sigma rows */
    uint32_t sig_w = (cur >> (pc - 1)) & 1;
    uint32_t sig_e = (cur >> (pc + 1)) & 1;
    int hpos = 0, hneg = 0;
    if (sig_w) { if ((sign_bits[r] >> (c - 1)) & 1) hneg = 1; else hpos = 1; }
    if (sig_e) { if ((sign_bits[r] >> (c + 1)) & 1) hneg = 1; else hpos = 1; }
    int hc = hpos - hneg;
    uint32_t sig_n = (north >> pc) & 1;
    uint32_t sig_s = (south >> pc) & 1;
    int vpos = 0, vneg = 0;
    if (sig_n) { if ((sign_bits[r - 1] >> c) & 1) vneg = 1; else vpos = 1; }
    if (sig_s) { if ((sign_bits[r + 1] >> c) & 1) vneg = 1; else vpos = 1; }
    int vc = vpos - vneg;
    /* V179: bug fix — when original hc<0 we flip both contributions to land in the
     * canonical hc>=0 cases of Table D.2, but the predicted sign χ̃=1 must propagate
     * as xor_bit=1.  The hc==0 branch already handled vc<0 correctly.  The hc!=0
     * branch was always returning xor_bit=0, ignoring the original hc<0 flip — that
     * caused every sign coded with hc<0 (e.g. all coefficients in the all-negative
     * flat-image LL5 once their W neighbor went significant) to be decoded with the
     * wrong sign, producing the +/- alternation observed in diag_t1. */
    int flipped = 0;
    if (hc < 0) { hc = -hc; vc = -vc; flipped = 1; }
    if (hc == 0) { *xor_bit_out = (vc < 0) ? 1 : 0; *sc_ctx_out = (vc == 0) ? 0 : 1; }
    else         { *xor_bit_out = flipped;          *sc_ctx_out = (vc < 0) ? 2 : (vc == 0) ? 3 : 4; }
    return zc;
}

/* Sign coding context (Table D.2) — returns context label offset (0-4) + XOR bit.
 * Optimized: uses padded sigma to eliminate bounds checks. Branchless where possible. */
__device__ static void t1_sign_context_fast(const uint64_t* sigma_pad, const uint32_t* sign_bits,
                                             int r, int c, int cbw, int cbh,
                                             int* ctx_out, int* xor_bit_out) {
    int pc = c + 1;  /* padded column */
    int pr = r + 1;  /* padded row */

    /* Horizontal contribution: W and E neighbors */
    uint64_t sig_w = (sigma_pad[pr] >> (pc - 1)) & 1;
    uint64_t sig_e = (sigma_pad[pr] >> (pc + 1)) & 1;
    /* sign_bits uses real indexing (not padded), but we guard with sig_w/sig_e */
    int hpos = 0, hneg = 0;
    if (sig_w) { if ((sign_bits[r] >> (c - 1)) & 1) hneg = 1; else hpos = 1; }
    if (sig_e) { if ((sign_bits[r] >> (c + 1)) & 1) hneg = 1; else hpos = 1; }
    int hc = hpos - hneg;

    /* Vertical contribution: N and S neighbors */
    uint64_t sig_n = (sigma_pad[pr - 1] >> pc) & 1;
    uint64_t sig_s = (sigma_pad[pr + 1] >> pc) & 1;
    int vpos = 0, vneg = 0;
    if (sig_n) { if ((sign_bits[r - 1] >> c) & 1) vneg = 1; else vpos = 1; }
    if (sig_s) { if ((sign_bits[r + 1] >> c) & 1) vneg = 1; else vpos = 1; }
    int vc = vpos - vneg;

    /* V179: track flip so xor_bit reflects original hc<0 (predicted sign χ̃=1). */
    int flipped = 0;
    if (hc < 0) { hc = -hc; vc = -vc; flipped = 1; }

    if (hc == 0) {
        *xor_bit_out = (vc < 0) ? 1 : 0;
        *ctx_out = (vc == 0) ? 0 : 1;
    } else {
        *xor_bit_out = flipped;
        *ctx_out = (vc < 0) ? 2 : (vc == 0) ? 3 : 4;
    }
}

/* V169: ZC context from pre-loaded sigma rows (avoids LMEM re-read when caller has rows) */
__device__ static int t1_zc_from_rows(uint64_t north, uint64_t cur, uint64_t south,
                                       int c, int subband) {
    int pc = c + 1;
    int sh = ((north >> pc) & 1) + ((south >> pc) & 1);
    int sv = ((cur >> (pc-1)) & 1) + ((cur >> (pc+1)) & 1);
    int sd = ((north >> (pc-1)) & 1) + ((north >> (pc+1)) & 1)
           + ((south >> (pc-1)) & 1) + ((south >> (pc+1)) & 1);
    return ZC_LUT[subband * 45 + sh * 15 + sv * 5 + ((sd < 4) ? sd : 4)];
}

/* V169: SC context from pre-loaded sigma rows — same math as t1_sign_context_fast but
 * using caller-provided rows so the LMEM reads are shared with ZC context computation. */
__device__ static void t1_sc_from_rows(uint64_t north, uint64_t cur, uint64_t south,
                                        const uint32_t* sign_bits, int r, int c,
                                        int* ctx_out, int* xor_bit_out) {
    int pc = c + 1;
    uint32_t sig_w = (cur >> (pc - 1)) & 1;
    uint32_t sig_e = (cur >> (pc + 1)) & 1;
    int hpos = 0, hneg = 0;
    if (sig_w) { if ((sign_bits[r] >> (c - 1)) & 1) hneg = 1; else hpos = 1; }
    if (sig_e) { if ((sign_bits[r] >> (c + 1)) & 1) hneg = 1; else hpos = 1; }
    int hc = hpos - hneg;
    uint32_t sig_n = (north >> pc) & 1;
    uint32_t sig_s = (south >> pc) & 1;
    int vpos = 0, vneg = 0;
    if (sig_n) { if ((sign_bits[r - 1] >> c) & 1) vneg = 1; else vpos = 1; }
    if (sig_s) { if ((sign_bits[r + 1] >> c) & 1) vneg = 1; else vpos = 1; }
    int vc = vpos - vneg;
    /* V179: see t1_sign_context_fast for the fix rationale. */
    int flipped = 0;
    if (hc < 0) { hc = -hc; vc = -vc; flipped = 1; }
    if (hc == 0) { *xor_bit_out = (vc < 0) ? 1 : 0; *ctx_out = (vc == 0) ? 0 : 1; }
    else         { *xor_bit_out = flipped;          *ctx_out = (vc < 0) ? 2 : (vc == 0) ? 3 : 4; }
}

/* Magnitude refinement context (Table D.3) — uses padded sigma, no bounds checks.
 * Returns context label offset (0-2). */
__device__ static int t1_mr_context_fast(const uint64_t* sigma_pad, const uint32_t* firstref_bits,
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

/* V154: launch_bounds(64, 16); mag[] eliminated (LMEM 4704→2656 bytes).
 * V141: (64, 16) set after finding smaller blocks cut L1 pressure with 2KB mag[].
 * V143: kept after experiments with (64, 8).
 * V145: (128, 8) and (32, 32) both ~1% slower than (64, 16).
 * V155: tried (64, 32) for 32 blocks/SM — forced 32 regs, reduced ILP → +3ms. Reverted.
 * V156: bp_skip re-enabled with correct edge-case guard (always code >= 1 bit-plane).
 * V167d: launch_bounds(64) no minBlocks → 40 regs + spills → 48ms. Reverted.
 * V172: template<bool FAST4>: fast path (FAST4=true) clamps num_bp to 4, uses pre-built
 *       mag_bp[4][CB_DIM] (Pass 2) instead of per-iteration d_dwt re-read. At num_bp≤4,
 *       the compiler fully unrolls the bit-plane loop → 0 LMEM (same as V169 MAX_BPLANES=4).
 *       Correct path (FAST4=false): V171 col_mag_arr recompute, 800 bytes LMEM.
 *       Dispatch: fast flag → kernel_ebcot_t1<true>; else → kernel_ebcot_t1<false>.
 *       (64, 8) tested, no change: LMEM=800 bytes, 47 regs. Not from reg pressure.
 * V175: template<bool FAST4, int MAX_BP>: second template parameter controls compile-time
 *       upper bound on num_bp. FAST4=true: MAX_BP=4 (same as before). FAST4=false (correct):
 *       MAX_BP=7 — compiler sees num_bp ≤ 7 and can unroll the bit-plane loop. Empirical
 *       finding: all 2K blocks have ≤7 bp at 150Mbps (capping at 4 and 7 gave same output).
 *       Unrolled loop eliminates serial branch overhead, enables software pipelining.
 * Caller must launch with exactly 64 threads/block. */
/* V196: helper to load a single DWT coefficient as float irrespective of storage type.
 * For __half storage we go via __half2float; for float storage we just use __ldg.
 * Marked __forceinline__ so the dispatch is purely compile-time. */
template<typename DWT_T>
__device__ __forceinline__ float dwt_load(const DWT_T* p) { return __half2float(__ldg(p)); }  /* fp16 fallback (unused) */
template<>
__device__ __forceinline__ float dwt_load<float>(const float* p) { return __ldg(p); }

/* V246: kernel_ebcot_t1 templated on BYPASS (compile-time constant) so the
 * compiler can dead-code-eliminate all non-bypass/bypass branches, reducing
 * register pressure and enabling better ILP.  Previously use_bypass was a
 * runtime bool parameter; now BYPASS is a template constant. */
template<bool FAST4, bool BYPASS = false, int MAX_BP = (FAST4 ? 4 : 10), typename DWT_T = float>
__global__ __launch_bounds__(64, 16) void kernel_ebcot_t1(
    const DWT_T* __restrict__ d_dwt,    /* DWT output coefficients (d_a[c]) */
    int dwt_stride,                      /* row stride of DWT array (= image width) */
    const CodeBlockInfo* __restrict__ d_cb_info,  /* code-block metadata */
    int num_cbs,                         /* number of code-blocks */
    uint8_t*  __restrict__ d_coded_data, /* output: coded bytes per CB (num_cbs * CB_BUF_SIZE) */
    uint16_t* __restrict__ d_coded_len,  /* output: actual coded length per CB */
    uint8_t*  __restrict__ d_num_passes, /* output: number of coding passes per CB */
    uint16_t* __restrict__ d_pass_lengths, /* output: cumulative length at each pass (num_cbs * MAX_PASSES) */
    uint8_t*  __restrict__ d_num_bp,    /* output: number of coded bit-planes per CB (for T2 z-field) */
    float*    __restrict__ d_energy,    /* output: sum(coeff²) per CB for energy-based PCRD */
    int bp_skip = 0                     /* V143: skip this many LSB bit-planes (fast mode) */
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
     * V154: Two-pass over d_dwt to eliminate mag[] (saves 2048 bytes LMEM/thread).
     * Pass 1: compute sign_bits[], max_mag (→ num_bp).
     * Pass 2: re-read d_dwt, pack directly into mag_bp[].
     * d_dwt re-read is coalesced (global mem) and L2-cached; cheaper than
     * keeping a 2048-byte mag[] intermediary in LMEM.
     *
     * OPTIMIZATION: sigma_pad[] has sentinel rows at [0] and [cbh+1] (all zeros)
     * and columns shifted by +1 (bit 0 and bit cbw+1 are sentinel zeros).
     * This eliminates ALL boundary checks from neighbor significance lookups.
     *
     * V149: col_mag_arr[c] replaces mag[r*cbw+c] in the coding loop.
     * Layout: col_mag_arr[c] = uint32 with bit r set if coefficient (r,c)
     * has bit bp_idx set.  In the hot coding loop we load ONE uint32 per column
     * per bit-plane (replacing 4 strided int16 LMEM loads), then extract row
     * bits with register shifts — 32× fewer LMEM load instructions.
     * V171: mag_bp[] eliminated — recompute per-bit-plane col_mag_arr[] from d_dwt.
     * Reduces active LMEM from 1952→730 bytes (63% less), improving L2 hit rate. */
    uint64_t sigma_pad[CB_DIM + 2]; /* padded: sigma_pad[0]=sentinel, [1..cbh]=real, [cbh+1]=sentinel */
    uint32_t sign_bits[CB_DIM];     /* sign: bit c = negative (real indexing, not padded) */
    uint32_t firstref_bits[CB_DIM]; /* first refinement: bit c = first time */
    uint32_t coded_bits[CB_DIM];    /* coded in current pass */

    /* V158: initialize full sigma_pad (all CB_DIM+2 entries) so V158's per-stripe
     * precomputed mask can safely read sigma_pad[stripe_y+5] for any stripe including
     * the last incomplete stripe of small boundary code-blocks (e.g. cbh=2). */
    for (int i = 0; i < CB_DIM + 2; i++) sigma_pad[i] = 0;
    for (int r = 0; r < cbh; r++) firstref_bits[r] = 0xFFFFFFFF;

    float inv_step = __frcp_rn(cbi.quant_step);  /* fast reciprocal */

    /* Pass 1: compute sign_bits[], max_mag, energy_sum (for PCRD) */
    int max_mag = 0;
    float energy_sum = 0.0f;
    for (int r = 0; r < cbh; r++) {
        uint32_t sb = 0;
        const DWT_T* row_ptr = d_dwt + (cbi.y0 + r) * dwt_stride + cbi.x0;
        for (int c = 0; c < cbw; c++) {
            float val = dwt_load(row_ptr + c);
            int q = __float2int_rd(fabsf(val) * inv_step)  /* V211 */;
            if (val < 0.0f) sb |= (1u << c);
            max_mag |= q;  /* bitwise OR — captures all bits for num_bp */
            energy_sum += val * val;  /* V287: per-CB energy for PCRD distortion model */
        }
        sign_bits[r] = sb;
    }
    /* max_mag is now the bitwise OR of all magnitudes; num_bp is the position of the MSB */

    /* 2. Compute number of bit-planes using __clz (count leading zeros) */
    int num_bp = (max_mag > 0) ? (32 - __clz(max_mag)) : 0;
    /* V175: Use MAX_BP template parameter as compile-time loop bound for both modes.
     *   FAST4=true: MAX_BP=4 → compiler sees num_bp ≤ 4 → fully unrolls 4-iter loop.
     *   FAST4=false: MAX_BP=7 → compiler sees num_bp ≤ 7 → unrolls 7-iter loop.
     *   All 2K blocks have ≤7 bp at 150Mbps (empirically verified: capping at 7 and 10
     *   produce identical output size). Unrolled loop enables software pipelining. */
    if (num_bp > MAX_BP) num_bp = MAX_BP;
    d_energy[cb_idx] = energy_sum;  /* always store (even for zero-pass CBs) */
    if (num_bp == 0) {
        d_coded_len[cb_idx] = 0;
        d_num_passes[cb_idx] = 0;
        d_num_bp[cb_idx] = 0;
        return;
    }

    /* V303: CHUNK_BP chunked mag_bp_flat — reduce LMEM from MAX_BP*CB_DIM*4 to
     * CHUNK_BP*CB_DIM*4. Build CHUNK_BP bit-planes per d_dwt scan instead of all
     * MAX_BP at once.
     * V307: CHUNK_BP raised from 4 to 8.
     * V308: CHUNK_BP raised from 8 to 16.
     * V309: CHUNK_BP raised to MAX_BPLANES=18 — single d_dwt scan for ALL num_bp values.
     * This eliminates chunk-scan overhead entirely: the outer chunk loop always runs once.
     * LMEM: 18*32*4=2304B chunk + 272B sigma_pad + 384B sign/ref/coded = ~2960B/thread.
     * The compiler sees a fixed chunk_size=num_bp (bounded by the template MAX_BP) and
     * may reduce register count vs CHUNK_BP=16 (where the chunk loop runs 2× for MAX_BP=17/18),
     * potentially improving occupancy above the current 56%.
     * Key benefit vs V308: retry T1 runs at step=0.0625 (MAX_BP=17) and step=0.03125
     * (MAX_BP=18) also become single-scan — V308 still needed 2 scans for those. */
    constexpr int CHUNK_BP = MAX_BPLANES;
    uint32_t mag_bp_flat[CHUNK_BP * CB_DIM];

    /* 3. Initialize MQ coder */
    uint8_t* out_buf = d_coded_data + (size_t)cb_idx * CB_BUF_SIZE;
    out_buf[0] = 0;  /* sentinel for MQ byte stuffing */

    MQCoder mq;
    mq_init(&mq, out_buf);

    int total_passes = 0;
    uint16_t* pass_lens = d_pass_lengths + (size_t)cb_idx * MAX_PASSES;

    /* 4. Bit-plane coding loop — chunked.
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

    int any_significant = 0;

    /* V156: cap bp_skip so we always code at least 1 bit-plane. */
    int actual_bp_skip = (bp_skip > 0) ? (bp_skip < num_bp ? bp_skip : num_bp - 1) : 0;

    for (int chunk_top = num_bp - 1; chunk_top >= 0; chunk_top -= CHUNK_BP) {
        /* Build mag_bp_flat for bit-planes [chunk_top .. chunk_bot] (MSB to LSB). */
        int chunk_bot = chunk_top - CHUNK_BP + 1;
        if (chunk_bot < 0) chunk_bot = 0;
        int chunk_size = chunk_top - chunk_bot + 1;

        for (int ci = 0; ci < chunk_size; ci++)
            for (int c = 0; c < CB_DIM; c++)
                mag_bp_flat[ci * CB_DIM + c] = 0;

        for (int r = 0; r < cbh; r++) {
            uint32_t rmask = 1u << r;
            const DWT_T* row_ptr = d_dwt + (cbi.y0 + r) * dwt_stride + cbi.x0;
            for (int c = 0; c < cbw; c++) {
                int q = __float2int_rd(fabsf(dwt_load(row_ptr + c)) * inv_step)  /* V211 */;
                if (q == 0) continue;
                for (int ci = 0; ci < chunk_size; ci++) {
                    if ((q >> (chunk_top - ci)) & 1)
                        mag_bp_flat[ci * CB_DIM + c] |= rmask;
                }
            }
        }

        /* Process passes for each bit-plane in this chunk (MSB first). */
        for (int ci = 0; ci < chunk_size; ci++) {
            int bp = chunk_top - ci;

            /* V222: CB buffer overflow guard. */
            if (static_cast<int>(mq.bp - mq.start) >= CB_BUF_SIZE - 768) goto t1_done;
            bool first_bp = (bp == num_bp - 1);

            /* V243: BYPASS flag — J2K Part 1 C.3.8, verified against OpenJPEG t1.c line 2103:
             *   type = ((bpno_plus_one <= numbps-4) && (passtype < 2) && LAZY) ? RAW : MQ
             * passtype 0=SPP, 1=MRP are bypassed; passtype 2=CUP is ALWAYS MQ.
             * Bypass starts at the 5th bit-plane from MSB (bp < num_bp-4).
             * First 10 passes (1+3+3+3) are all MQ; then alternating bypass(SPP+MRP) + MQ(CUP). */
            bool bp_bypass = (BYPASS && bp < num_bp - 4);

            for (int r2 = 0; r2 < CB_DIM; r2++) coded_bits[r2] = 0;

            /* --- Significance Propagation Pass (SPP) — skip for first bit-plane --- */
            if (!first_bp) {
            if (bp_bypass) {
                /* V245: Bypass SPP — raw significance + raw sign bits.
                 * ISO 15444-1 C.3.8: in bypass mode the sign is encoded as the raw
                 * actual_sign bit (no XOR with sign prediction).  OpenJPEG's
                 * dec_sigpass_raw reads it directly without any XOR.  Using XOR here
                 * (V247 attempt) caused gradient patterns to regress from 88→41 dB. */
                BypassCoder bc;
                bc.bp = mq.bp + 1;  bc.accum = 0;  bc.cnt = 0;
                if (any_significant) {
                for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                    int stripe_end = min(stripe_y + STRIPE_H, cbh);
                    int stripe_len = stripe_end - stripe_y;
                    /* V245: all-significant column skip (same optimisation as MQ SPP V159). */
                    uint64_t spp_all_sig = sigma_pad[stripe_y + 1] & sigma_pad[stripe_y + 2]
                                         & sigma_pad[stripe_y + 3] & sigma_pad[stripe_y + 4];
                    uint32_t spp_sig[STRIPE_H] = {0, 0, 0, 0};
                    for (int ri = 0; ri < stripe_len; ri++)
                        spp_sig[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1);
                    for (int c = 0; c < cbw; c++) {
                        uint64_t cmask_pad = 1ull << (c + 1);
                        uint32_t cmask = 1u << c;
                        if (spp_all_sig & cmask_pad) continue;  /* all rows significant */
                        uint32_t col_mag = mag_bp_flat[ci * CB_DIM + c];
                        for (int r = stripe_y; r < stripe_end; r++) {
                            if (spp_sig[r - stripe_y] & cmask) continue;
                            if (!has_sig_neighbor(sigma_pad, r, c)) continue;
                            int bit = (col_mag >> r) & 1;
                            if (bit) {
                                int sign = (sign_bits[r] >> c) & 1;
                                bypass_write_bits(&bc, (uint8_t)((2u | (unsigned)sign)), 2); /* sig=1, raw sign */
                                sigma_pad[r + 1] |= cmask_pad;
                                any_significant = 1;
                            } else {
                                bypass_write(&bc, 0);
                            }
                            coded_bits[r] |= cmask;
                        }
                    }
                }
                }
                bypass_flush(&bc);
                mq.bp = bc.bp - 1;
            } else {
                /* MQ SPP */
                if (any_significant) {
                for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                    int stripe_end = min(stripe_y + STRIPE_H, cbh);
                    int stripe_len = stripe_end - stripe_y;

                    /* V159: SPP fast-skip: AND of all stripe rows — bit set = all 4 rows are significant. */
                    uint64_t spp_all_sig = sigma_pad[stripe_y + 1] & sigma_pad[stripe_y + 2]
                                         & sigma_pad[stripe_y + 3] & sigma_pad[stripe_y + 4];

                    /* V160: Per-row significance mask for row-level SPP skip. */
                    uint32_t spp_sig[STRIPE_H] = {0, 0, 0, 0};
                    for (int ri = 0; ri < stripe_len; ri++)
                        spp_sig[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1);

                    for (int c = 0; c < cbw; c++) {
                        uint64_t cmask_pad = 1ull << (c + 1);
                        uint32_t cmask = 1u << c;
                        if (spp_all_sig & cmask_pad) continue;
                        uint32_t col_mag = mag_bp_flat[ci * CB_DIM + c];
                        for (int r = stripe_y; r < stripe_end; r++) {
                            if (spp_sig[r - stripe_y] & cmask) continue;
                            /* V168: fused neighbor-check + ZC context — halves sigma_pad LMEM reads */
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
                } /* end any_significant */
                if (BYPASS) mq_flush(&mq);
            } /* end MQ SPP */
            pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
            if (!bp_bypass && BYPASS) mq_restart(&mq, mq.bp + 1);
            } /* end SPP */

            /* --- Magnitude Refinement Pass (MRP) --- */
            if (!first_bp) {
            if (any_significant) {
            if (bp_bypass) {
                /* V165: Bypass MRP — write raw refinement bits directly. No MQ, no context. */
                BypassCoder bc;
                bc.bp = mq.bp + 1;  bc.accum = 0;  bc.cnt = 0;
                for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                    int stripe_end = min(stripe_y + STRIPE_H, cbh);
                    int stripe_len_mrp = stripe_end - stripe_y;
                    uint32_t mrp_proc[STRIPE_H] = {0, 0, 0, 0};
                    for (int ri = 0; ri < stripe_len_mrp; ri++)
                        mrp_proc[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1) & ~coded_bits[stripe_y + ri];
                    /* V245: bit-scan over active MRP columns; pack row-bits per column
                     * into bypass_write_bits to replace up to STRIPE_H individual writes
                     * with one fused call (1–4 bits per column per stripe). */
                    uint32_t mrp_active = mrp_proc[0] | mrp_proc[1]
                                        | mrp_proc[2] | mrp_proc[3];
                    while (mrp_active) {
                        int c = __ffs(mrp_active) - 1;
                        mrp_active &= mrp_active - 1u;
                        uint32_t cmask = 1u << c;
                        uint32_t col_mag = mag_bp_flat[ci * CB_DIM + c];
                        /* Gather bits for this column's active rows MSB-first,
                         * then emit in one bypass_write_bits call. */
                        uint8_t bits = 0; int nbits = 0;
                        for (int r = stripe_y; r < stripe_end; r++) {
                            if (!(mrp_proc[r - stripe_y] & cmask)) continue;
                            bits = static_cast<uint8_t>((bits << 1) | ((col_mag >> r) & 1));
                            firstref_bits[r] &= ~(1u << c);
                            nbits++;
                        }
                        bypass_write_bits(&bc, bits, nbits);
                    }
                }
                bypass_flush(&bc);
                mq.bp = bc.bp - 1;  /* advance mq.bp to last written byte */
            } else {
                /* MQ MRP */
                for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                    int stripe_end = min(stripe_y + STRIPE_H, cbh);
                    int stripe_len_mrp = stripe_end - stripe_y;

                    /* V160: Precompute per-row "should process" mask for MRP. */
                    uint32_t mrp_proc[STRIPE_H] = {0, 0, 0, 0};
                    for (int ri = 0; ri < stripe_len_mrp; ri++)
                        mrp_proc[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1) & ~coded_bits[stripe_y + ri];
                    uint32_t mrp_colmask = mrp_proc[0] | mrp_proc[1] | mrp_proc[2] | mrp_proc[3];

                    for (int c = 0; c < cbw; c++) {
                        uint32_t cmask = 1u << c;
                        if (!(mrp_colmask & cmask)) continue;
                        uint32_t col_mag = mag_bp_flat[ci * CB_DIM + c];
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
            } /* end any_significant guard */
            /* V243: flush+restart MQ after non-bypass MRP. */
            if (!bp_bypass && BYPASS) { mq_flush(&mq); }
            pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
            if (!bp_bypass && BYPASS) { mq_restart(&mq, mq.bp + 1); }
            } /* end MRP */

            /* --- Cleanup Pass (CUP) — always MQ, even in bypass bit-planes.
             * V243: ISO 15444-1 C.3.8 / OpenJPEG t1.c: CUP (passtype=2) is never bypass.
             * When coming from bypass SPP+MRP, restart MQ before encoding. */
            if (bp_bypass && BYPASS) {
                mq_restart(&mq, mq.bp + 1);  /* start fresh MQ segment after bypass MRP */
            }
            {
                /* MQ CUP */
                for (int stripe_y = 0; stripe_y < cbh; stripe_y += STRIPE_H) {
                    int stripe_end = min(stripe_y + STRIPE_H, cbh);
                    int stripe_len_cup = stripe_end - stripe_y;

                    /* V158: Per-stripe fast_not_zero precompute for AGG run-length bypass. */
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

                    /* V160: Per-row skip mask for CUP inner row loop. */
                    uint32_t cup_skip[STRIPE_H] = {0, 0, 0, 0};
                    for (int ri = 0; ri < stripe_len_cup; ri++)
                        cup_skip[ri] = (uint32_t)(sigma_pad[stripe_y + 1 + ri] >> 1) | coded_bits[stripe_y + ri];

                    for (int c = 0; c < cbw; c++) {
                        uint64_t cmask_pad = 1ull << (c + 1);
                        uint32_t cmask = 1u << c;
                        int all_zero;

                        if (fast_not_zero & cmask_pad) {
                            all_zero = 0;
                        } else {
                            all_zero = (stripe_len_cup == 4) ? 1 : 0;
                            /* V168: when !any_significant, sigma_pad is all-zero → has_sig_neighbor
                             * always false, cup_skip always zero — skip inner loop entirely. */
                            if (all_zero && any_significant) {
                                for (int r = stripe_y; r < stripe_end; r++) {
                                    if (cup_skip[r - stripe_y] & cmask) { all_zero = 0; break; }
                                    if (has_sig_neighbor(sigma_pad, r, c)) { all_zero = 0; break; }
                                }
                            }
                        }

                        uint32_t col_mag = mag_bp_flat[ci * CB_DIM + c];

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
            } /* end MQ CUP */
            /* V243: CUP always terminates with MQ flush for TERMALL; restart for next bit-plane. */
            if (BYPASS) { mq_flush(&mq); }
            pass_lens[total_passes++] = static_cast<uint16_t>(mq.bp - mq.start + 1);
            if (BYPASS) { mq_restart(&mq, mq.bp + 1); }

            /* V156: exit after coding the last wanted bit-plane. */
            if (actual_bp_skip > 0 && bp == actual_bp_skip) goto t1_done;
        } /* end per-bit-plane inner loop */
    } /* end chunk outer loop */
    t1_done:;

    /* 5. Final flush.
     * V242 BYPASS+RESTART: every pass already flushed in the loop — no final flush needed.
     * Non-bypass mode: standard MQ flush of the continuous arithmetic stream. */
    {
        if (!BYPASS)
            mq_flush(&mq);
        while (mq.bp > mq.start && *mq.bp == 0xFF) mq.bp--;
    }

    /* V133 OPT: No byte shift — T2 will read starting at buf[1] (skip sentinel byte).
     * This eliminates the per-CB byte-by-byte copy (up to 2KB/CB × 7500 CBs = 15MB/frame). */
    int coded_len = static_cast<int>(mq.bp - mq.start);
    if (coded_len < 0) coded_len = 0;
    if (coded_len > CB_BUF_SIZE - 1) coded_len = CB_BUF_SIZE - 1;

    /* Trim trailing 0xFF so the last byte can't merge with a marker. */
    while (coded_len > 0 && out_buf[coded_len] == 0xFF)
        coded_len--;

    d_coded_len[cb_idx] = static_cast<uint16_t>(coded_len);
    d_num_passes[cb_idx] = static_cast<uint8_t>(total_passes);
    /* V156: d_num_bp stays = num_bp (z = pmax - num_bp in T2 packet header).
     * z describes the coefficient bit-depth (content + quant step), NOT how many
     * passes were included.  Changing z breaks the decoder's CB-header parsing. */
    d_num_bp[cb_idx]    = static_cast<uint8_t>(num_bp);
    /* Pass lengths unchanged (still relative to buf[1..coded_len]) */
    for (int p = 0; p < total_passes; p++) {
        int pl = (pass_lens[p] > 0) ? (pass_lens[p] - 1) : 0;
        pass_lens[p] = static_cast<uint16_t>(min(pl, coded_len));
    }
}


/* V240: GPU compact-copy kernel — packs max_cb_d2h bytes from each CB's
 * CB_BUF_SIZE-strided slot in src into a contiguous dst (dst_stride stride).
 * One warp (32 threads) per code block; uint32_t copies for efficiency.
 * Replaces cudaMemcpy2DAsync for sparse content to eliminate the
 * pathological per-row DMA overhead from low width/pitch ratios.
 * Launched as <<<num_cbs, 32, 0, stream>>>.  dst_stride must be a multiple of 4. */
__global__ void kernel_compact_ebcot(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__       dst,
    int num_cbs, int dst_stride, int src_stride)
{
    int cb = blockIdx.x;
    if (cb >= num_cbs) return;
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src + (size_t)cb * src_stride);
    uint32_t*       d = reinterpret_cast<uint32_t*>      (dst + (size_t)cb * dst_stride);
    int words = dst_stride / 4;
    for (int i = threadIdx.x; i < words; i += 32)
        d[i] = s[i];
}


#endif /* GPU_EBCOT_H */
