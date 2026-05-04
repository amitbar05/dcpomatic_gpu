/*
    CUDA J2K Encoder Performance Benchmark
    Measures correct mode and fast mode throughput for 2K and 4K.
    
    Build:
      nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
           -I/home/amit/dcp-o-matic-gpu/src -I/home/amit/dcp-o-matic-gpu/src/lib \
           -o test/bench_cuda_j2k test/bench_cuda_j2k.cu \
           src/lib/cuda_j2k_encoder.cu -lcudart
*/

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

#include "lib/cuda_j2k_encoder.h"

using Clock = std::chrono::high_resolution_clock;

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) {
        p.lut_in[i]  = i / 4095.f;
        p.lut_out[i] = static_cast<uint16_t>(i);
    }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

static void gen_sine(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float fx = x / float(w), fy = y / float(h);
            size_t i = ((size_t)y*w + x)*3;
            p[i+0] = uint16_t((0.5f + 0.5f*sinf(fx*20.f))*60000.f);
            p[i+1] = uint16_t((0.5f + 0.5f*sinf(fy*20.f+1.f))*60000.f);
            p[i+2] = uint16_t((0.5f + 0.5f*sinf((fx+fy)*15.f+2.f))*60000.f);
        }
}

static double bench(CudaJ2KEncoder& enc, int w, int h, int64_t br, int fps,
                     bool is_4k, bool fast, int warmup, int iters)
{
    std::vector<uint16_t> rgb((size_t)w * h * 3);
    gen_sine(rgb.data(), w, h);

    /* Warmup */
    for (int i = 0; i < warmup; ++i)
        enc.encode_ebcot(rgb.data(), w, h, w*3, br, fps, false, is_4k, fast);

    /* Measure */
    auto t0 = Clock::now();
    size_t total_bytes = 0;
    for (int i = 0; i < iters; ++i) {
        auto cs = enc.encode_ebcot(rgb.data(), w, h, w*3, br, fps, false, is_4k, fast);
        total_bytes += cs.size();
    }
    auto t1 = Clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = total_ms / iters;
    double fps_out = iters * 1000.0 / total_ms;
    double avg_kb = total_bytes / (double)iters / 1024.0;

    printf("  %s %s: %8.2f ms/frame  %7.1f fps  %8.1f KB/frame\n",
           fast ? "FAST" : "CORR",
           is_4k ? "4K" : "2K",
           avg_ms, fps_out, avg_kb);
    return avg_ms;
}

int main() {
    printf("=== CUDA J2K Encoder Benchmark ===\n\n");

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) {
        printf("FAIL: encoder init failed\n");
        return 1;
    }
    GpuColourParams cp; build_params(cp); enc.set_colour_params(cp);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d MB VRAM)\n\n", prop.name, prop.totalGlobalMem / (1024*1024));

    const int warmup = 3, iters = 20;

    /* 2K @ 150 Mbps */
    printf("2K DCI (2048x1080) @ 150 Mbps / 24 fps:\n");
    double c2k = bench(enc, 2048, 1080, 150000000LL, 24, false, false, warmup, iters);
    double f2k = bench(enc, 2048, 1080, 150000000LL, 24, false, true,  warmup, iters);
    printf("  Speedup (fast/correct): %.2fx\n\n", c2k / f2k);

    /* 4K @ 250 Mbps */
    printf("4K DCI (4096x2160) @ 250 Mbps / 24 fps:\n");
    double c4k = bench(enc, 4096, 2160, 250000000LL, 24, true, false, warmup, iters);
    double f4k = bench(enc, 4096, 2160, 250000000LL, 24, true, true,  warmup, iters);
    printf("  Speedup (fast/correct): %.2fx\n\n", c4k / f4k);

    printf("DONE\n");
    return 0;
}
