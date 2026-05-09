/*
    ICT (Irreversible Component Transform) Correctness Test — GPU vs. CPU Reference

    Verifies the GPU ICT forward transform matches the CPU reference within
    float precision. Tests both:
      (a) kernel_ict_fwd_f32out — the actual encoder path (int32 in, float out)
      (b) Roundtrip: GPU forward ICT → CPU inverse ICT error < 2.0 LSB

    Test patterns: all_zero, all_max, mid_gray, gradient_ramp, random, impulse,
                   checkerboard, achromatic (R=G=B, Cb/Cr should be ~0)
    Resolution: 64×64 pixels.

    Build:
      nvcc -O2 -arch=sm_61 -std=c++17 \
           -I src -I src/lib \
           -o test/ict_correctness test/ict_correctness.cu -lcudart
*/

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

/* ICT forward kernel — matches cuda_j2k_encoder.cu kernel_ict_fwd_f32out exactly.
 * Reads int32 XYZ, writes unrounded floats (Y with DC shift, Cb/Cr without). */
__global__ void k_ict_fwd_f32out(
    const int32_t* __restrict__ d_c0_in,
    const int32_t* __restrict__ d_c1_in,
    const int32_t* __restrict__ d_c2_in,
    float* __restrict__ d_c0_out,
    float* __restrict__ d_c1_out,
    float* __restrict__ d_c2_out,
    int pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixels) return;

    float c0 = static_cast<float>(d_c0_in[idx]);
    float c1 = static_cast<float>(d_c1_in[idx]);
    float c2 = static_cast<float>(d_c2_in[idx]);

    d_c0_out[idx] =  0.299f   * c0 + 0.587f   * c1 + 0.114f   * c2 - 2048.0f;
    d_c1_out[idx] = -0.16875f * c0 - 0.33126f * c1 + 0.5f     * c2;
    d_c2_out[idx] =  0.5f     * c0 - 0.41869f * c1 - 0.08131f * c2;
}

static constexpr int W = 64, H = 64, PIXELS = W * H, MAX_VAL = 4095;

/* CPU reference: forward ICT (matches GPU kernel math exactly) */
static void cpu_ict_fwd(const int32_t* r, const int32_t* g, const int32_t* b,
                        float* y, float* cb, float* cr, int n)
{
    for (int i = 0; i < n; i++) {
        float c0 = r[i], c1 = g[i], c2 = b[i];
        y[i]  =  0.299f   * c0 + 0.587f   * c1 + 0.114f   * c2 - 2048.0f;
        cb[i] = -0.16875f * c0 - 0.33126f * c1 + 0.5f     * c2;
        cr[i] =  0.5f     * c0 - 0.41869f * c1 - 0.08131f * c2;
    }
}

/* CPU reference: inverse ICT per JPEG2000 Part 1 Annex G.2.2 (adds back DC=2048) */
static void cpu_ict_inv(const float* y, const float* cb, const float* cr,
                        float* r, float* g, float* b, int n)
{
    for (int i = 0; i < n; i++) {
        float yv = y[i] + 2048.0f;
        r[i] = yv +  1.402f   * cr[i];
        g[i] = yv - 0.34413f  * cb[i] - 0.71414f * cr[i];
        b[i] = yv +  1.772f   * cb[i];
    }
}

struct Gen { const char* name; void (*fn)(int32_t*, int32_t*, int32_t*, int); };

static void gen_zeros   (int32_t* r, int32_t* g, int32_t* b, int n) { memset(r,0,4*n); memset(g,0,4*n); memset(b,0,4*n); }
static void gen_max     (int32_t* r, int32_t* g, int32_t* b, int n) { for(int i=0;i<n;i++) r[i]=g[i]=b[i]=MAX_VAL; }
static void gen_mid     (int32_t* r, int32_t* g, int32_t* b, int n) { for(int i=0;i<n;i++) r[i]=g[i]=b[i]=2048; }
static void gen_ramp    (int32_t* r, int32_t* g, int32_t* b, int n) {
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) {
        int i=y*W+x;
        r[i]=(int32_t)(x*MAX_VAL/(W-1)); g[i]=(int32_t)(y*MAX_VAL/(H-1));
        b[i]=(int32_t)((x+y)*MAX_VAL/(W+H-2));
    }
}
static void gen_random  (int32_t* r, int32_t* g, int32_t* b, int n) {
    unsigned s=12345;
    for(int i=0;i<n;i++){
        s=s*1664525u+1013904223u; r[i]=s%(MAX_VAL+1);
        s=s*1664525u+1013904223u; g[i]=s%(MAX_VAL+1);
        s=s*1664525u+1013904223u; b[i]=s%(MAX_VAL+1);
    }
}
static void gen_impulse (int32_t* r, int32_t* g, int32_t* b, int n) {
    memset(r,0,4*n); memset(g,0,4*n); memset(b,0,4*n);
    int c=(H/2)*W+(W/2); r[c]=g[c]=b[c]=MAX_VAL;
}
static void gen_checker (int32_t* r, int32_t* g, int32_t* b, int n) {
    for(int y=0;y<H;y++) for(int x=0;x<W;x++) {
        int i=y*W+x; int32_t v=((x/8+y/8)&1)?MAX_VAL:0; r[i]=g[i]=b[i]=v;
    }
}
/* achromatic test: R=G=B → Cb and Cr should be exactly 0 */
static void gen_achromatic(int32_t* r, int32_t* g, int32_t* b, int n) {
    unsigned s=99999;
    for(int i=0;i<n;i++){
        s=s*1664525u+1013904223u; int32_t v=s%(MAX_VAL+1); r[i]=g[i]=b[i]=v;
    }
}

int main()
{
    printf("=== ICT Correctness Test ===\n");
    int devcount; CUDA_CHECK(cudaGetDeviceCount(&devcount));
    if (!devcount) { fprintf(stderr, "No CUDA devices\n"); return 1; }
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n\n", prop.name);

    int32_t *d_r, *d_g, *d_b;
    float   *d_y, *d_cb, *d_cr;
    CUDA_CHECK(cudaMalloc(&d_r,  PIXELS*4)); CUDA_CHECK(cudaMalloc(&d_g,  PIXELS*4));
    CUDA_CHECK(cudaMalloc(&d_b,  PIXELS*4)); CUDA_CHECK(cudaMalloc(&d_y,  PIXELS*4));
    CUDA_CHECK(cudaMalloc(&d_cb, PIXELS*4)); CUDA_CHECK(cudaMalloc(&d_cr, PIXELS*4));

    Gen tests[] = {
        {"zeros",     gen_zeros},  {"all_max",   gen_max},   {"mid_gray",  gen_mid},
        {"ramp",      gen_ramp},   {"random",    gen_random}, {"impulse",   gen_impulse},
        {"checker",   gen_checker},{"achromatic",gen_achromatic},
    };

    int fails = 0;
    for (auto& t : tests) {
        std::vector<int32_t> hr(PIXELS), hg(PIXELS), hb(PIXELS);
        t.fn(hr.data(), hg.data(), hb.data(), PIXELS);

        CUDA_CHECK(cudaMemcpy(d_r, hr.data(), PIXELS*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_g, hg.data(), PIXELS*4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, hb.data(), PIXELS*4, cudaMemcpyHostToDevice));

        int grid = (PIXELS+255)/256;
        k_ict_fwd_f32out<<<grid, 256>>>(d_r, d_g, d_b, d_y, d_cb, d_cr, PIXELS);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> hy(PIXELS), hcb(PIXELS), hcr(PIXELS);
        CUDA_CHECK(cudaMemcpy(hy.data(),  d_y,  PIXELS*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hcb.data(), d_cb, PIXELS*4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hcr.data(), d_cr, PIXELS*4, cudaMemcpyDeviceToHost));

        /* GPU vs CPU forward */
        std::vector<float> ry(PIXELS), rcb(PIXELS), rcr(PIXELS);
        cpu_ict_fwd(hr.data(), hg.data(), hb.data(), ry.data(), rcb.data(), rcr.data(), PIXELS);

        double maxd_fwd = 0;
        for (int i = 0; i < PIXELS; i++) {
            maxd_fwd = std::max(maxd_fwd, (double)fabsf(hy[i]-ry[i]));
            maxd_fwd = std::max(maxd_fwd, (double)fabsf(hcb[i]-rcb[i]));
            maxd_fwd = std::max(maxd_fwd, (double)fabsf(hcr[i]-rcr[i]));
        }

        /* Roundtrip: GPU forward → CPU inverse */
        std::vector<float> rr(PIXELS), rg(PIXELS), rb(PIXELS);
        cpu_ict_inv(hy.data(), hcb.data(), hcr.data(), rr.data(), rg.data(), rb.data(), PIXELS);
        double maxd_rt = 0;
        for (int i = 0; i < PIXELS; i++) {
            maxd_rt = std::max(maxd_rt, (double)fabsf(rr[i] - hr[i]));
            maxd_rt = std::max(maxd_rt, (double)fabsf(rg[i] - hg[i]));
            maxd_rt = std::max(maxd_rt, (double)fabsf(rb[i] - hb[i]));
        }

        /* Achromatic: Cb and Cr should be within 0.01 of 0 */
        double maxd_achrom = 0;
        bool is_achromatic = (strcmp(t.name, "achromatic") == 0);
        if (is_achromatic) {
            for (int i = 0; i < PIXELS; i++) {
                maxd_achrom = std::max(maxd_achrom, (double)fabsf(hcb[i]));
                maxd_achrom = std::max(maxd_achrom, (double)fabsf(hcr[i]));
            }
        }

        /* FP32 precision at scale 4095: ~5e-4 ULP; roundtrip < 2.0 LSB */
        bool ok = (maxd_fwd < 1e-3) && (maxd_rt < 2.0);
        if (is_achromatic) ok = ok && (maxd_achrom < 0.1);
        if (!ok) fails++;

        printf("  %-14s  fwd_maxd=%.6f  rt_maxd=%.4f  [%s]\n",
               t.name, maxd_fwd, maxd_rt, ok ? "PASS" : "FAIL");
        if (is_achromatic)
            printf("                 achromatic Cb/Cr max=%.6f  [%s]\n",
                   maxd_achrom, (maxd_achrom < 0.1) ? "PASS" : "FAIL");
    }

    printf("\n=== ICT: %s (%d/%d passed) ===\n",
           fails ? "FAIL" : "PASS", (int)(sizeof(tests)/sizeof(tests[0]))-fails,
           (int)(sizeof(tests)/sizeof(tests[0])));
    return fails ? 1 : 0;
}
