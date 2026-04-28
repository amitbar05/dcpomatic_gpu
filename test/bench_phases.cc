/*
    Benchmark encode_ebcot with phase timing.  Uses CUDA events for GPU
    pieces and chrono for CPU T2.  Runs N=20 frames, ignores the first as
    warmup.
*/
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "lib/cuda_j2k_encoder.h"

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count() / 1000.0;
}

static void
build_params(GpuColourParams& p)
{
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i/4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0] = 0.4124f; p.matrix[1] = 0.3576f; p.matrix[2] = 0.1805f;
    p.matrix[3] = 0.2126f; p.matrix[4] = 0.7152f; p.matrix[5] = 0.0722f;
    p.matrix[6] = 0.0193f; p.matrix[7] = 0.1192f; p.matrix[8] = 0.9505f;
    p.valid = true;
}

int main(int argc, char** argv)
{
    const int W = 2048, H = 1080, FPS = 24, N = 20;
    const int64_t BR = 150'000'000;
    bool fast = (argc > 1 && std::string(argv[1]) == "fast");

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb(size_t(W) * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t i = (size_t(y) * W + x) * 3;
            float fx = x/float(W), fy = y/float(H);
            rgb[i+0] = uint16_t((0.5f + 0.5f*std::sin(fx*20.f))*60000.f);
            rgb[i+1] = uint16_t((0.5f + 0.5f*std::sin(fy*20.f + 1.f))*60000.f);
            rgb[i+2] = uint16_t((0.5f + 0.5f*std::sin((fx+fy)*15.f + 2.f))*60000.f);
        }

    /* Warm up: 10 frames so GPU/DRAM reaches thermal steady state */
    for (int i = 0; i < 10; ++i)
        (void) enc.encode_ebcot(rgb.data(), W, H, W*3, BR, FPS, false, false, fast);

    double total_ms = 0;
    size_t out_bytes = 0;
    for (int i = 0; i < N; ++i) {
        auto t = Clock::now();
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, FPS, false, false, fast);
        total_ms += ms_since(t);
        out_bytes = cs.size();
    }
    std::printf("mode=%s: %.2f ms/frame avg, %zu bytes\n",
                fast ? "fast" : "correct", total_ms / N, out_bytes);
    return 0;
}
