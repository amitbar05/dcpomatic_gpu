/* Benchmark fast vs correct mode end-to-end timing. */
#include <chrono>
#include <cstdio>
#include <vector>
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
    if (!enc.is_initialized()) return 1;
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    /* Realistic content: gradient with some noise */
    std::vector<uint16_t> rgb((size_t)W*H*3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            unsigned s = (unsigned)(x*7919u + y*6151u);
            s = s * 1664525u + 1013904223u;
            uint16_t v = uint16_t(8000 + (x * 35000ll / (W-1)) + ((s >> 17) & 0x7FF) - 1024);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
        }

    using Clk = std::chrono::high_resolution_clock;
    /* Warmup */
    for (int i = 0; i < 3; ++i) enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, false);

    auto t0 = Clk::now();
    int N = 10;
    for (int i = 0; i < N; ++i) {
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, false);
    }
    auto t1 = Clk::now();
    double ms_correct = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() / 1000.0 / N;
    auto cs_correct = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, false);

    /* Warmup fast */
    for (int i = 0; i < 3; ++i) enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, true);
    auto t2 = Clk::now();
    for (int i = 0; i < N; ++i) {
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, true);
    }
    auto t3 = Clk::now();
    double ms_fast = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count() / 1000.0 / N;
    auto cs_fast = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, true);

    printf("CORRECT: %.2f ms/frame  size=%zu\n", ms_correct, cs_correct.size());
    printf("FAST:    %.2f ms/frame  size=%zu  (speedup %.2fx)\n",
           ms_fast, cs_fast.size(), ms_correct / ms_fast);
    return 0;
}
