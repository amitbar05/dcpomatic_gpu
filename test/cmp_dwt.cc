/*
 * Compare GPU DWT output to OPJ-compatible (standard) reference.
 * For a known input, dump LL5 coefficient stats from GPU and compute reference.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include "lib/cuda_j2k_encoder.h"

static const int W = 2048, H = 1080;

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i / 4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

int main() {
    CudaJ2KEncoder enc;
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    int input_val = 50000;
    std::vector<uint16_t> rgb((size_t)W*H*3, uint16_t(input_val));
    printf("=== Flat input %d → expected: LL5 = constant, all detail = 0 ===\n", input_val);
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);

    /* d_a layout after DWT: nested rectangles.
     * LL5 is at [0..64) x [0..34); origin (0,0).
     * HL5 (level 5 horizontal detail): rows [0..34), cols [64..128).
     * LH5: rows [34..68), cols [0..64).
     * HH5: rows [34..68), cols [64..128). */
    int ll5_w = (W + 31) / 32;  /* 64 */
    int ll5_h = (H + 31) / 32;  /* 34 */

    std::vector<float> dwt = enc.debug_get_dwt_output(1, W, H);  /* Y component */
    auto at = [&](int x, int y) -> float {
        return dwt[(size_t)y * W + x];
    };

    /* LL5 stats */
    {
        float mn=1e9, mx=-1e9, sum=0, sumsq=0;
        for (int y = 0; y < ll5_h; ++y)
            for (int x = 0; x < ll5_w; ++x) {
                float v = at(x, y);
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v; sumsq += v*v;
            }
        size_t n = ll5_w * ll5_h;
        printf("LL5 (Y comp): min=%.2f max=%.2f mean=%.2f std=%.2f n=%zu\n",
               mn, mx, sum/n, std::sqrt(sumsq/n - (sum/n)*(sum/n)), n);
    }

    /* HL5 stats: cols [ll5_w..2*ll5_w), rows [0..ll5_h) */
    {
        float mn=1e9, mx=-1e9, sum=0, sumsq=0;
        for (int y = 0; y < ll5_h; ++y)
            for (int x = ll5_w; x < 2*ll5_w; ++x) {
                float v = at(x, y);
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v; sumsq += v*v;
            }
        size_t n = ll5_w * ll5_h;
        printf("HL5 (Y comp): min=%.2f max=%.2f mean=%.2f std=%.2f n=%zu\n",
               mn, mx, sum/n, std::sqrt(sumsq/n - (sum/n)*(sum/n)), n);
    }

    /* HL1 stats: cols [W/2..W), rows [0..H/2) */
    {
        float mn=1e9, mx=-1e9, sum=0, sumsq=0; int nz=0;
        for (int y = 0; y < H/2; ++y)
            for (int x = W/2; x < W; ++x) {
                float v = at(x, y);
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v; sumsq += v*v;
                if (std::abs(v) > 1) ++nz;
            }
        size_t n = (size_t)(W/2) * (H/2);
        printf("HL1 (Y comp): min=%.2f max=%.2f mean=%.2f std=%.2f n=%zu nz_count=%d (%.1f%%)\n",
               mn, mx, sum/n, std::sqrt(sumsq/n - (sum/n)*(sum/n)), n, nz, 100.0*nz/n);
    }

    /* Print right cols of HL1 (rows 0..H/2, cols W-5..W) — H-DWT right boundary */
    printf("\nHL1 right cols (should be ~0 for flat input):\n");
    for (int xx = W - 5; xx < W; ++xx) {
        printf("  HL1 x=%d: ", xx);
        for (int yy = 0; yy < H/2; yy += 100) printf("%.2f ", at(xx, yy));
        printf("\n");
    }
    /* Print top rows */
    printf("\nLH1 top rows (should be ~0):\n");
    for (int yy = H/2; yy < H/2 + 5; ++yy) {
        printf("  LH1 y=%d: ", yy);
        for (int xx = 0; xx < W/2; xx += 256) printf("%.2f ", at(xx, yy));
        printf("\n");
    }
    /* Print bottom 5 rows of LH1 (rows H/2..H-1, cols 0..W/2): rows where boundary may have artifacts */
    printf("\nLH1 bottom rows (should be ~0 for flat input):\n");
    for (int yy = H - 5; yy < H; ++yy) {
        printf("  LH1 y=%d (raw): ", yy);
        for (int xx = 0; xx < W/2; xx += 256) printf("%.2f ", at(xx, yy));
        printf("\n");
    }
    printf("\nHL1 bottom rows (should be ~0 for flat input):\n");
    for (int yy = H/2 - 5; yy < H/2; ++yy) {
        printf("  HL1 y=%d (raw): ", yy);
        for (int xx = W/2; xx < W; xx += 256) printf("%.2f ", at(xx, yy));
        printf("\n");
    }
    printf("\nHH1 bottom rows (should be ~0 for flat input):\n");
    for (int yy = H - 5; yy < H; ++yy) {
        printf("  HH1 y=%d (raw): ", yy);
        for (int xx = W/2; xx < W; xx += 256) printf("%.2f ", at(xx, yy));
        printf("\n");
    }

    /* For an h_bars pattern with values y_lo=625-2048=-1423, y_hi=3125-2048=1077,
     * after standard 9/7 forward DWT (with K factors), the LL band of a constant region
     * should be (constant - DC_level=2048) ≈ -1423 or 1077 depending on x position.
     * For our normalized DWT (gain=1), LL5 should cover ~32-px blocks → fully -1423 or 1077.
     * Per-LL5-coefficient: depending on whether the 32-px block falls in dark or bright bar. */
    printf("\nReference (post DC level shift): y_lo=-1423, y_hi=1077\n");
    printf("Each LL5 coefficient covers a 32x32 image block. For 256-px stripes, every LL5\n");
    printf("column maps to one full bar so LL5 should alternate between -1423 and 1077\n");
    printf("at column boundaries every 8 columns (= 256/32).\n");
    return 0;
}
