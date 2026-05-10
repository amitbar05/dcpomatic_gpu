/*
 * Per-pattern timing benchmark for V314 sig_frac step pre-selection.
 * Runs N=10 frames per pattern, reports ms/frame avg.
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -o test/bench_patterns_v314 test/bench_patterns_v314.cc \
 *        src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
#include "lib/cuda_j2k_encoder.h"

static const int W = 2048, H = 1080;
static const int64_t BR = 150'000'000;
static const int FPS = 24, N = 10;

using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t).count() / 1000.0;
}

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i/4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

static void bench(CudaJ2KEncoder& enc, const char* name,
                  std::function<uint16_t(int,int)> fn)
{
    std::vector<uint16_t> rgb(size_t(W)*H*3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = fn(x,y);
            rgb[(size_t(y)*W+x)*3+0] = v;
            rgb[(size_t(y)*W+x)*3+1] = v;
            rgb[(size_t(y)*W+x)*3+2] = v;
        }
    /* warmup */
    for (int i = 0; i < 3; ++i)
        (void) enc.encode_ebcot(rgb.data(), W, H, W*3, BR, FPS, false, false);
    double total = 0;
    for (int i = 0; i < N; ++i) {
        auto t = Clock::now();
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, FPS, false, false);
        total += ms_since(t);
        (void)cs;
    }
    printf("  %-26s  %6.1f ms/frame\n", name, total/N);
}

int main() {
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    printf("Per-pattern timing (W=%d H=%d, BR=150Mbps, N=%d frames each)\n", W, H, N);
    bench(enc, "flat_30000",       [](int,int){ return 30000; });
    bench(enc, "h_gradient_full",  [](int x,int){ return uint16_t(x*60000ll/(W-1)); });
    bench(enc, "v_gradient_full",  [](int,int y){ return uint16_t(y*60000ll/(H-1)); });
    bench(enc, "two_value_split",  [](int x,int){ return uint16_t(x < W/2 ? 20000 : 40000); });
    bench(enc, "h_bars_8",         [](int x,int){ return uint16_t(((x/256)%2)?50000:10000); });
    bench(enc, "v_bars_8",         [](int,int y){ return uint16_t(((y/135)%2)?50000:10000); });
    bench(enc, "hl_bars_64",       [](int x,int){ return uint16_t(((x/64)&1)?50000:10000); });
    bench(enc, "lh_bars_64",       [](int,int y){ return uint16_t(((y/64)&1)?50000:10000); });
    bench(enc, "checker_64",       [](int x,int y){ return uint16_t((((x/64)+(y/64))&1)?50000:10000); });
    bench(enc, "hh1_pixel_checker",[](int x,int y){ return uint16_t(((x+y)&1)?50000:10000); });
    bench(enc, "photo_synth",      [](int x,int y){
        float cx=(x-W/2)/float(W/2), cy=(y-H/2)/float(H/2);
        float r=std::sqrt(cx*cx+cy*cy);
        unsigned s=(unsigned)(x*1543u+y*7919u); s=s*1664525u+1013904223u;
        int noise=((s>>17)&0xFF)-128;
        int base=int(40000+20000*(1.0f-r));
        if ((x/256)==4&&(y/270)==2) base=60000;
        int v=base+noise;
        if (v<1000) v=1000; if (v>65000) v=65000;
        return uint16_t(v);
    });
    bench(enc, "checker_8",        [](int x,int y){ return uint16_t((((x/8)+(y/8))&1)?60000:4000); });
    bench(enc, "noise_small",      [](int x,int y){
        unsigned s=(unsigned)(x*17u+y*31u); s=s*1664525u+1013904223u;
        return uint16_t(30000+((s>>17)&0x3FF)-512);
    });
    return 0;
}
