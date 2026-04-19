/*
    V134 verification: exercise CudaJ2KEncoder::encode_ebcot in both
    "correct" (fast_mode=false) and "fast/lossy" (fast_mode=true) modes,
    check both outputs are valid J2K codestreams.

    Build:
      nvcc -c -std=c++17 -O2 src/lib/cuda_j2k_encoder.cu \
           -Isrc -Ideps/install/include -Ideps/install/include/libdcp-1.0 \
           -o /tmp/cuda_j2k_encoder.o
      g++ -std=c++17 -O2 test/verify_fast_mode.cc /tmp/cuda_j2k_encoder.o \
           -Isrc -Ideps/install/include \
           -lcudart -L/usr/local/cuda/lib64 -o test/verify_fast_mode

    Usage:
      ./test/verify_fast_mode
*/

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "lib/cuda_j2k_encoder.h"

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

static bool
is_valid_j2k(const std::vector<uint8_t>& cs, const char* label)
{
    if (cs.size() < 8) {
        std::cout << "  " << label << ": FAIL (too small: " << cs.size() << " bytes)\n";
        return false;
    }
    bool soc = cs[0] == 0xFF && cs[1] == 0x4F;
    bool eoc = cs[cs.size() - 2] == 0xFF && cs[cs.size() - 1] == 0xD9;
    bool siz = false, cod = false, qcd = false, sot = false, sod = false;
    for (size_t i = 2; i + 1 < cs.size(); ++i) {
        if (cs[i] != 0xFF) continue;
        uint8_t m = cs[i + 1];
        if (m == 0x51) siz = true;
        else if (m == 0x52) cod = true;
        else if (m == 0x5C) qcd = true;
        else if (m == 0x90) sot = true;
        else if (m == 0x93) { sod = true; break; }
    }
    bool ok = soc && siz && cod && qcd && sot && sod && eoc;
    std::cout << "  " << label
              << ": SOC=" << soc << " SIZ=" << siz << " COD=" << cod
              << " QCD=" << qcd << " SOT=" << sot << " SOD=" << sod
              << " EOC=" << eoc << " size=" << cs.size()
              << "  => " << (ok ? "VALID J2K" : "INVALID") << "\n";
    return ok;
}

static void
build_identity_colour_params(GpuColourParams& p)
{
    for (int i = 0; i < 4096; ++i) {
        float v = i / 4095.0f;
        p.lut_in[i]  = v;
        p.lut_out[i] = static_cast<uint16_t>(i);
    }
    /* near-identity matrix (keeps energy in Y; X/Z small) so each component
     * gets enough variance to exercise EBCOT bit-planes. */
    p.matrix[0] = 0.4124f; p.matrix[1] = 0.3576f; p.matrix[2] = 0.1805f;
    p.matrix[3] = 0.2126f; p.matrix[4] = 0.7152f; p.matrix[5] = 0.0722f;
    p.matrix[6] = 0.0193f; p.matrix[7] = 0.1192f; p.matrix[8] = 0.9505f;
    p.valid = true;
}

int main()
{
    const int W = 2048, H = 1080, FPS = 24;
    const int64_t BITRATE = 150'000'000;

    std::cout << "=== V134 fast_mode verification ===\n";
    std::cout << "Resolution: " << W << "x" << H << "\n";

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) {
        std::cerr << "CudaJ2KEncoder failed to initialize\n";
        return 1;
    }

    GpuColourParams p;
    build_identity_colour_params(p);
    enc.set_colour_params(p);

    /* Synthetic RGB48 frame — sine gradients, ~12-bit range */
    std::vector<uint16_t> rgb(static_cast<size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            size_t i = static_cast<size_t>(y) * W + x;
            float fx = x / static_cast<float>(W);
            float fy = y / static_cast<float>(H);
            rgb[i * 3 + 0] = static_cast<uint16_t>(
                (0.5f + 0.5f * std::sin(fx * 20.0f)) * 60000.0f);
            rgb[i * 3 + 1] = static_cast<uint16_t>(
                (0.5f + 0.5f * std::sin(fy * 20.0f + 1.0f)) * 60000.0f);
            rgb[i * 3 + 2] = static_cast<uint16_t>(
                (0.5f + 0.5f * std::sin((fx + fy) * 15.0f + 2.0f)) * 60000.0f);
        }
    }

    /* Warmup: first frame pays GPU buffer init cost. */
    (void) enc.encode_ebcot(rgb.data(), W, H, W * 3, BITRATE, FPS, false, false, false);
    (void) enc.encode_ebcot(rgb.data(), W, H, W * 3, BITRATE, FPS, false, false, true);

    /* Correct / verifiable path (average over N frames). */
    const int N = 10;
    auto t0 = Clock::now();
    std::vector<uint8_t> cs_correct;
    for (int i = 0; i < N; ++i)
        cs_correct = enc.encode_ebcot(rgb.data(), W, H, W * 3,
                                       BITRATE, FPS, false, false, /*fast_mode=*/false);
    auto t1 = Clock::now();
    bool ok_correct = is_valid_j2k(cs_correct, "correct/verifiable");

    auto t2 = Clock::now();
    std::vector<uint8_t> cs_fast;
    for (int i = 0; i < N; ++i)
        cs_fast = enc.encode_ebcot(rgb.data(), W, H, W * 3,
                                    BITRATE, FPS, false, false, /*fast_mode=*/true);
    auto t3 = Clock::now();
    bool ok_fast = is_valid_j2k(cs_fast, "fast/lossy     ");

    double ms_correct = duration_cast<microseconds>(t1 - t0).count() / 1000.0 / N;
    double ms_fast    = duration_cast<microseconds>(t3 - t2).count() / 1000.0 / N;

    std::cout << "\nTiming:\n";
    std::cout << "  correct/verifiable: " << ms_correct << " ms, "
              << cs_correct.size() << " bytes\n";
    std::cout << "  fast/lossy        : " << ms_fast << " ms, "
              << cs_fast.size() << " bytes\n";
    if (cs_fast.size() > 0 && cs_correct.size() > 0) {
        std::cout << "  fast is "
                  << (ms_correct / std::max(ms_fast, 0.001)) << "x faster, "
                  << (100.0 * cs_fast.size() / cs_correct.size()) << "% of the size\n";
    }

    /* Write files so we can opj_decompress them separately. */
    {
        std::ofstream o("/tmp/gpu_correct.j2c", std::ios::binary);
        o.write(reinterpret_cast<const char*>(cs_correct.data()), cs_correct.size());
    }
    {
        std::ofstream o("/tmp/gpu_fast.j2c", std::ios::binary);
        o.write(reinterpret_cast<const char*>(cs_fast.data()), cs_fast.size());
    }
    std::cout << "Wrote /tmp/gpu_correct.j2c and /tmp/gpu_fast.j2c\n";

    int rc = 0;
    if (!ok_correct) { std::cerr << "CORRECT path produced invalid J2K\n"; rc = 2; }
    if (!ok_fast)    { std::cerr << "FAST path produced invalid J2K\n"; rc = 3; }
    if (rc == 0) std::cout << "\nBoth paths OK.\n";
    return rc;
}
