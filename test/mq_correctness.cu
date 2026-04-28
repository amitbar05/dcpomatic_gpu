/*
 * MQ coder correctness test.
 *
 * Drive the GPU MQ coder with a deterministic (ctx, decision) sequence,
 * then decode the output bytes with a CPU reference MQ decoder that
 * mirrors OpenJPEG's opj_mqc_decode/bytein verbatim. The test passes
 * if every decoded bit matches the encoded bit and the decoded length
 * equals the encoded length.
 *
 * If this fails, the bug is in the GPU MQ coder. If it passes, MQ is
 * correct and the T1 PSNR fault must lie in higher layers (context
 * generation, pass scheduling, codestream framing).
 *
 * Build:
 *   nvcc -O2 -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -o test/mq_correctness test/mq_correctness.cu -lcudart
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "lib/gpu_ebcot.h"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1);} } while(0)

/* Single-thread kernel exercising the GPU MQ coder under test. */
__global__ void k_mq_encode(const int* ctxs, const int* ds, int n,
                             uint8_t* out_buf, int* out_len)
{
    MQCoder mq;
    mq_init(&mq, out_buf);
    for (int i = 0; i < n; ++i)
        mq_encode(&mq, ctxs[i], ds[i]);
    mq_flush(&mq);
    /* mq->bp points at the last written byte (mq_flush may have backed up
     * over a trailing 0xFF). Length = (bp - start) + 1 if start..bp written.
     * In gpu_ebcot.h, mq_init sets buf[0]=0 sentinel and start=buf, bp=buf.
     * After encoding, valid output bytes are start+1 .. bp inclusive. */
    *out_len = static_cast<int>(mq.bp - mq.start);   /* number of bytes after sentinel */
}

/* CPU reference MQ table (numeric values match MQ_TABLE in gpu_ebcot.h). */
struct MqRow { uint16_t qe; uint8_t nmps, nlps, sw; };
static const MqRow REF[47] = {
    {0x5601,1,1,1},{0x3401,2,6,0},{0x1801,3,9,0},{0x0AC1,4,12,0},
    {0x0521,5,29,0},{0x0221,38,33,0},{0x5601,7,6,1},{0x5401,8,14,0},
    {0x4801,9,14,0},{0x3801,10,14,0},{0x3001,11,17,0},{0x2401,12,18,0},
    {0x1C01,13,20,0},{0x1601,29,21,0},{0x5601,15,14,1},{0x5401,16,14,0},
    {0x5101,17,15,0},{0x4801,18,16,0},{0x3801,19,17,0},{0x3401,20,18,0},
    {0x3001,21,19,0},{0x2801,22,19,0},{0x2401,23,20,0},{0x2201,24,21,0},
    {0x1C01,25,22,0},{0x1801,26,23,0},{0x1601,27,24,0},{0x1401,28,25,0},
    {0x1201,29,26,0},{0x1101,30,27,0},{0x0AC1,31,28,0},{0x09C1,32,29,0},
    {0x08A1,33,30,0},{0x0521,34,31,0},{0x0441,35,32,0},{0x02A1,36,33,0},
    {0x0221,37,34,0},{0x0141,38,35,0},{0x0111,39,36,0},{0x0085,40,37,0},
    {0x0049,41,38,0},{0x0025,42,39,0},{0x0015,43,40,0},{0x0009,44,41,0},
    {0x0005,45,42,0},{0x0001,45,43,0},{0x5601,46,46,0}
};

/* CPU reference decoder — mirrors opj_mqc_decode + opj_mqc_bytein exactly. */
struct MqDec {
    const uint8_t* bp;     /* pointer to current byte (NOT consumed yet) */
    const uint8_t* end;    /* one-past-last */
    uint32_t C;
    uint32_t A;
    int      CT;
    uint8_t  ctx_idx[T1_NUM_CTXS];
    uint8_t  ctx_mps[T1_NUM_CTXS];
};

static void mq_dec_bytein(MqDec* d) {
    if (d->bp != d->end) {
        uint32_t c = (d->bp + 1 != d->end) ? *(d->bp + 1) : 0xFFu;
        if (*d->bp == 0xFF) {
            if (c > 0x8F) {
                d->C += 0xFF00;
                d->CT = 8;
            } else {
                d->bp++;
                d->C += c << 9;
                d->CT = 7;
            }
        } else {
            d->bp++;
            d->C += c << 8;
            d->CT = 8;
        }
    } else {
        d->C += 0xFF00;
        d->CT = 8;
    }
}

static void mq_dec_init(MqDec* d, const uint8_t* buf, int len) {
    d->bp  = buf;
    d->end = buf + len;
    d->C   = (len == 0) ? (0xFFu << 16) : ((uint32_t)*buf << 16);
    mq_dec_bytein(d);
    d->C <<= 7;
    d->CT -= 7;
    d->A   = 0x8000;
    for (int i = 0; i < T1_NUM_CTXS; ++i) { d->ctx_idx[i] = 0; d->ctx_mps[i] = 0; }
    d->ctx_idx[0]            = 4;
    d->ctx_idx[T1_CTXNO_AGG] = 3;
    d->ctx_idx[T1_CTXNO_UNI] = 46;
}

static void mq_dec_renorm(MqDec* d) {
    int safety = 64;
    do {
        if (d->CT == 0) mq_dec_bytein(d);
        d->A <<= 1;
        d->C <<= 1;
        d->CT--;
        if (--safety <= 0) {
            fprintf(stderr, "ABORT: renorm runaway (A=0x%X CT=%d C=0x%X bp_off=%ld)\n",
                    d->A, d->CT, d->C, (long)(d->bp - (d->end - (d->end - d->bp))));
            std::exit(2);
        }
    } while ((d->A & 0x8000) == 0);
}

/* Mirrors opj_mqc_decode + opj_mqc_lpsexchange + opj_mqc_mpsexchange. */
static int mq_dec_decode(MqDec* d, int ctx) {
    uint8_t  idx = d->ctx_idx[ctx];
    uint8_t  mps = d->ctx_mps[ctx];
    const MqRow& r = REF[idx];
    uint16_t qe = r.qe;
    int ret;

    d->A -= qe;
    if ((d->C >> 16) < qe) {
        /* LPS exchange */
        if (d->A < qe) {
            ret = mps;
            d->ctx_idx[ctx] = r.nmps;
        } else {
            ret = 1 - mps;
            d->ctx_idx[ctx] = r.nlps;
            if (r.sw) d->ctx_mps[ctx] ^= 1;
        }
        d->A = qe;
        mq_dec_renorm(d);
    } else {
        d->C -= ((uint32_t)qe) << 16;
        if ((d->A & 0x8000) == 0) {
            /* MPS exchange */
            if (d->A < qe) {
                ret = 1 - mps;
                d->ctx_idx[ctx] = r.nlps;
                if (r.sw) d->ctx_mps[ctx] ^= 1;
            } else {
                ret = mps;
                d->ctx_idx[ctx] = r.nmps;
            }
            mq_dec_renorm(d);
        } else {
            ret = mps;
        }
    }
    return ret;
}

/* --- Test driver --- */
struct Trial { const char* name; std::vector<int> ctxs, ds; };

static void gen_random(Trial& t, int n, unsigned seed) {
    t.ctxs.resize(n); t.ds.resize(n);
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        t.ctxs[i] = s % T1_NUM_CTXS;
        t.ds[i]   = (s >> 16) & 1;
    }
}

static int run_trial(const Trial& t) {
    int n = (int)t.ctxs.size();
    int *d_c, *d_d, *d_len;
    uint8_t* d_buf;
    CK(cudaMalloc(&d_c, n*sizeof(int)));
    CK(cudaMalloc(&d_d, n*sizeof(int)));
    CK(cudaMalloc(&d_buf, CB_BUF_SIZE));
    CK(cudaMalloc(&d_len, sizeof(int)));
    CK(cudaMemcpy(d_c, t.ctxs.data(), n*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_d, t.ds.data(),   n*sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(d_buf, 0, CB_BUF_SIZE));

    k_mq_encode<<<1,1>>>(d_c, d_d, n, d_buf, d_len);
    CK(cudaDeviceSynchronize());

    int len = 0;
    CK(cudaMemcpy(&len, d_len, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> bytes(len + 1);
    CK(cudaMemcpy(bytes.data(), d_buf, len + 1, cudaMemcpyDeviceToHost));
    cudaFree(d_c); cudaFree(d_d); cudaFree(d_buf); cudaFree(d_len);

    /* Encoded data starts at offset 1: byte 0 is the carry sentinel from mq_init. */
    const uint8_t* enc = bytes.data() + 1;
    int            enc_len = len;       /* bp-start counted bytes after sentinel */

    MqDec dec; mq_dec_init(&dec, enc, enc_len);
    int errs = 0, first_err = -1;
    for (int i = 0; i < n; ++i) {
        int got = mq_dec_decode(&dec, t.ctxs[i]);
        if (got != t.ds[i]) {
            if (first_err < 0) first_err = i;
            errs++;
        }
    }
    printf("[%-22s] n=%-5d coded=%-4d  %s",
           t.name, n, enc_len, errs ? "FAIL" : "PASS");
    if (errs) printf("  errors=%d first_diff_at=%d", errs, first_err);
    printf("\n");
    return errs == 0 ? 0 : 1;
}

int main() {
    int fails = 0;
    Trial t;

    t.name = "tiny_zero8";
    t.ctxs.assign(8, 0); t.ds.assign(8, 0);  fails += run_trial(t);

    t.name = "all_zero_ctx0";
    t.ctxs.assign(1024, 0); t.ds.assign(1024, 0);  fails += run_trial(t);

    t.name = "all_one_ctx0";
    t.ctxs.assign(1024, 0); t.ds.assign(1024, 1);  fails += run_trial(t);

    t.name = "alt_ctx0";
    t.ctxs.assign(1024, 0); t.ds.resize(1024);
    for (int i = 0; i < 1024; ++i) t.ds[i] = i & 1;
    fails += run_trial(t);

    for (unsigned seed : {1u, 2u, 42u, 12345u, 0xBEEFu}) {
        char buf[64]; snprintf(buf, 64, "rand_seed_%u", seed);
        Trial r; r.name = buf;
        gen_random(r, 4096, seed);
        fails += run_trial(r);
    }

    t.name = "longrun_zeros";
    t.ctxs.assign(8000, 0); t.ds.assign(8000, 0);  fails += run_trial(t);

    t.name = "ctx18_uni_random";
    t.ctxs.assign(2000, T1_CTXNO_UNI); t.ds.resize(2000);
    { unsigned s=99; for (int i=0;i<2000;++i){ s=s*1103515245u+12345u; t.ds[i]=(s>>16)&1; } }
    fails += run_trial(t);

    t.name = "ctx_sweep";
    t.ctxs.resize(T1_NUM_CTXS * 100); t.ds.resize(t.ctxs.size());
    for (size_t i=0;i<t.ctxs.size();++i){ t.ctxs[i] = (int)(i % T1_NUM_CTXS); t.ds[i] = (i>>1)&1; }
    fails += run_trial(t);

    printf("\n=== MQ correctness: %s (failures=%d) ===\n",
           fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
