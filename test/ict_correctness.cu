/*
    ICT (Irreversible Component Transform) Correctness Test — GPU vs. CPU Reference

    Verifies that the GPU `kernel_ict_fwd<int32_t>` forward ICT + CPU reference
    inverse ICT roundtrip is correct to within ~1.5 LSB error (max error < 2.0)
    for all test patterns.  The forward ICT operates in float internally (as
    implemented in the kernel) and rounds to int32_t; the CPU inverse uses the
    complementary float coefficients from JPEG2000 Part 1 Annex G.2.2.

    Test patterns:
        all_zero, all_max, mid_gray, gradient_ramp, random, impulse, checkerboard
    Resolution: 64×64 pixels (small enough to run fast, large enough to exercise
                GPU parallelism).

    Build:
      nvcc -O2 -arch=sm_61 -std=c++17 \
           -I/home/amit/dcp-o-matic-gpu/src \
           -I/home/amit/dcp-o-matic-gpu/src/lib \
           -o test/ict_correctness test/ict_correctness.cu \
           src/lib/cuda_j2k_encoder.cu -lcudart

    NOTE: The forward ICT kernel lives in cuda_j2k_encoder.cu.  The test is
    compiled together with that source file, so the linker resolves
    `kernel_ict_fwd<int32_t>` from the same translation unit.
*/

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>

// -----------------------------------------------------------------------
// CUDA error-check helper
// -----------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                         __FILE__, __LINE__, cudaGetErrorString(_e));         \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// -----------------------------------------------------------------------
// Forward declaration of the GPU kernel (defined in cuda_j2k_encoder.cu)
// -----------------------------------------------------------------------
template<typename T>
__global__ void kernel_ict_fwd(
    T* __restrict__ d_c0,       // in/out: component 0 (X → Y)
    T* __restrict__ d_c1,       // in/out: component 1 (Y → Cb)
    T* __restrict__ d_c2,       // in/out: component 2 (Z → Cr)
    int pixels,
    int stride);

// -----------------------------------------------------------------------
// Test constants
// -----------------------------------------------------------------------
static constexpr int W = 64;
static constexpr int H = 64;
static constexpr int PIXELS = W * H;
static constexpr int MAX_VAL = 4095;
static constexpr int MID_VAL = 2048;

// -----------------------------------------------------------------------
// CPU reference: inverse ICT per JPEG2000 Part 1 Annex G.2.2
//
// The forward ICT (done in the GPU kernel) is:
//   Y  =  0.299    * X + 0.587    * Y + 0.114    * Z
//   Cb = -0.16875  * X - 0.33126  * Y + 0.5      * Z + 2048
//   Cr =  0.5      * X - 0.41869  * Y - 0.08131  * Z + 2048
//
// The inverse ICT (JPEG2000 Part 1 Annex G.2.2, Eq. G-7..G-9) is:
//   R = Y                     + 1.402   * (Cr - 2048)
//   G = Y - 0.34413 * (Cb - 2048) - 0.71414 * (Cr - 2048)
//   B = Y + 1.772   * (Cb - 2048)
//
// Coefficients are exact per the JPEG2000 standard.
// -----------------------------------------------------------------------
static void cpu_inverse_ict_single(
    float y, float cb, float cr,
    float& r, float& g, float& b)
{
    float cb_off = cb - 2048.0f;
    float cr_off = cr - 2048.0f;

    r = y + 1.402f * cr_off;
    g = y - 0.34413f * cb_off - 0.71414f * cr_off;
    b = y + 1.772f * cb_off;
}

static void cpu_inverse_ict(
    const int32_t* y_plane,
    const int32_t* cb_plane,
    const int32_t* cr_plane,
    float* r_out,
    float* g_out,
    float* b_out,
    int pixels)
{
    for (int i = 0; i < pixels; ++i) {
        float y  = static_cast<float>(y_plane[i]);
        float cb = static_cast<float>(cb_plane[i]);
        float cr = static_cast<float>(cr_plane[i]);
        cpu_inverse_ict_single(y, cb, cr, r_out[i], g_out[i], b_out[i]);
    }
}

// -----------------------------------------------------------------------
// Test pattern generators (fill int32_t planar buffers)
// -----------------------------------------------------------------------

// All zero
static void gen_all_zero(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        x[i] = 0; y[i] = 0; z[i] = 0;
    }
}

// All max
static void gen_all_max(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        x[i] = MAX_VAL; y[i] = MAX_VAL; z[i] = MAX_VAL;
    }
}

// Mid-gray
static void gen_mid_gray(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        x[i] = MID_VAL; y[i] = MID_VAL; z[i] = MID_VAL;
    }
}

// Gradient ramp: X varies 0..MAX_VAL across width, Y varies 0..MAX_VAL across height,
// Z is diagonal blend.
static void gen_gradient_ramp(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            int idx = row * W + col;
            x[idx] = static_cast<int32_t>(
                static_cast<float>(col) / static_cast<float>(W - 1) * MAX_VAL + 0.5f);
            y[idx] = static_cast<int32_t>(
                static_cast<float>(row) / static_cast<float>(H - 1) * MAX_VAL + 0.5f);
            z[idx] = static_cast<int32_t>(
                (static_cast<float>(col + row) / static_cast<float>(W + H - 2)) * MAX_VAL + 0.5f);
        }
    }
}

// Random 1000 pixels (but we fill the whole 64x64 plane — the test is over all pixels)
static void gen_random(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    unsigned seed = 12345;
    for (int i = 0; i < pixels; ++i) {
        seed = seed * 1664525u + 1013904223u;
        x[i] = static_cast<int32_t>(seed % (MAX_VAL + 1));
        seed = seed * 1664525u + 1013904223u;
        y[i] = static_cast<int32_t>(seed % (MAX_VAL + 1));
        seed = seed * 1664525u + 1013904223u;
        z[i] = static_cast<int32_t>(seed % (MAX_VAL + 1));
    }
}

// Single impulse: all-zero except one pixel at max
static void gen_impulse(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        x[i] = 0; y[i] = 0; z[i] = 0;
    }
    int center = (H / 2) * W + (W / 2);
    x[center] = MAX_VAL;
    y[center] = MAX_VAL;
    z[center] = MAX_VAL;
}

// Checkerboard: alternating max and 0 in 8-pixel blocks
static void gen_checkerboard(int32_t* x, int32_t* y, int32_t* z, int pixels) {
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            int idx = row * W + col;
            int block = (col / 8) + (row / 8);
            int32_t val = (block & 1) ? MAX_VAL : 0;
            x[idx] = val;
            y[idx] = val;
            z[idx] = val;
        }
    }
}

// -----------------------------------------------------------------------
// Quality metrics
// -----------------------------------------------------------------------
static double compute_psnr(double mse, int max_val) {
    if (mse < 1e-12) return 999.0;
    return 20.0 * std::log10(static_cast<double>(max_val) / std::sqrt(mse));
}

struct ICTResult {
    std::string pattern_name;
    double max_err[3];    // per-component max absolute error
    double mean_err[3];   // per-component mean absolute error
    double psnr[3];       // per-component PSNR
    bool passed;
};

// -----------------------------------------------------------------------
// Test runner: allocates GPU memory, launches kernel, runs CPU inverse,
//              compares results.
// -----------------------------------------------------------------------
static ICTResult run_ict_test(
    const std::string& name,
    void (*gen)(int32_t*, int32_t*, int32_t*, int))
{
    ICTResult res;
    res.pattern_name = name;

    // Allocate host buffers
    std::vector<int32_t> h_x(PIXELS), h_y(PIXELS), h_z(PIXELS);
    gen(h_x.data(), h_y.data(), h_z.data(), PIXELS);

    // Allocate GPU memory
    int32_t *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, PIXELS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_y, PIXELS * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_z, PIXELS * sizeof(int32_t)));

    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), PIXELS * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), PIXELS * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z.data(), PIXELS * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Launch forward ICT on GPU: 256 threads per block
    int grid = (PIXELS + 255) / 256;
    kernel_ict_fwd<int32_t><<<grid, 256>>>(
        d_x, d_y, d_z, PIXELS, W);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy ICT output back to host
    std::vector<int32_t> h_y_out(PIXELS), h_cb_out(PIXELS), h_cr_out(PIXELS);
    CUDA_CHECK(cudaMemcpy(h_y_out.data(), d_x, PIXELS * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cb_out.data(), d_y, PIXELS * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cr_out.data(), d_z, PIXELS * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    // Apply CPU reference inverse ICT
    std::vector<float> h_r(PIXELS), h_g(PIXELS), h_b(PIXELS);
    cpu_inverse_ict(h_y_out.data(), h_cb_out.data(), h_cr_out.data(),
                    h_r.data(), h_g.data(), h_b.data(), PIXELS);

    // Compare reconstructed RGB against original XYZ
    // (In our test convention, X→R, Y→G, Z→B)
    // Convert original int32_t to float for comparison:
    std::vector<float> h_xf(PIXELS), h_yf(PIXELS), h_zf(PIXELS);
    for (int i = 0; i < PIXELS; ++i) {
        h_xf[i] = static_cast<float>(h_x[i]);
        h_yf[i] = static_cast<float>(h_y[i]);
        h_zf[i] = static_cast<float>(h_z[i]);
    }

    const float* recon[3] = { h_r.data(), h_g.data(), h_b.data() };
    const float* orig[3]  = { h_xf.data(), h_yf.data(), h_zf.data() };

    // Compute per-component error metrics
    for (int c = 0; c < 3; ++c) {
        double sum_err = 0.0;
        double sum_sq_err = 0.0;
        double max_err = 0.0;

        for (int i = 0; i < PIXELS; ++i) {
            double err = std::fabs(static_cast<double>(recon[c][i]) -
                                   static_cast<double>(orig[c][i]));
            sum_err += err;
            sum_sq_err += err * err;
            if (err > max_err) max_err = err;
        }

        res.max_err[c]  = max_err;
        res.mean_err[c] = sum_err / static_cast<double>(PIXELS);
        double mse = sum_sq_err / static_cast<double>(PIXELS);
        res.psnr[c] = compute_psnr(mse, MAX_VAL);
    }

    // Pass/fail: max_error < 2.0 (roundtrip through irreversible transform with
    // float→int32 rounding; ~1.5 LSB worst-case is expected) and PSNR > 72 dB.
    res.passed = true;
    for (int c = 0; c < 3; ++c) {
        if (res.max_err[c] >= 2.0 || res.psnr[c] <= 72.0) {
            res.passed = false;
        }
    }

    return res;
}

// -----------------------------------------------------------------------
// Helper: print a single result
// -----------------------------------------------------------------------
static void print_ict_result(const ICTResult& r) {
    const char* comp_names[3] = { "R(X)", "G(Y)", "B(Z)" };
    std::printf("  %-20s : ", r.pattern_name.c_str());
    for (int c = 0; c < 3; ++c) {
        std::printf("%s max=%.4f mean=%.4f PSNR=%.1f dB",
                    comp_names[c], r.max_err[c], r.mean_err[c], r.psnr[c]);
        if (c < 2) std::printf(" | ");
    }
    std::printf("  [%s]\n", r.passed ? "PASS" : "FAIL");
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main() {
    std::printf("=== ICT (Irreversible Component Transform) Correctness Test ===\n");
    std::printf("Resolution: %d x %d pixels\n", W, H);
    std::printf("Forward ICT: GPU (float internal, int32_t in/out)\n");
    std::printf("Inverse ICT: CPU reference (JPEG2000 Part 1 Annex G.2.2)\n");
    std::printf("Pass criteria: max absolute error < 2.0, PSNR > 72 dB (all channels)\n");
    std::printf("NOTE: ICT is *irreversible* — float→int32 rounding in forward pass\n");
    std::printf("      introduces up to ~1.5 LSB error after inverse transform.\n\n");

    // Check GPU availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "ERROR: No CUDA devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d, %zu MB)\n\n",
                prop.name, prop.major, prop.minor,
                prop.totalGlobalMem / (1024 * 1024));

    // Run all tests
    struct TestSpec {
        const char* name;
        void (*gen)(int32_t*, int32_t*, int32_t*, int);
    };

    TestSpec tests[] = {
        { "all_zero",        gen_all_zero },
        { "all_max",         gen_all_max },
        { "mid_gray",        gen_mid_gray },
        { "gradient_ramp",   gen_gradient_ramp },
        { "random",          gen_random },
        { "impulse",         gen_impulse },
        { "checkerboard",    gen_checkerboard },
    };

    std::vector<ICTResult> results;
    int passed_count = 0;

    for (const auto& t : tests) {
        ICTResult r = run_ict_test(t.name, t.gen);
        results.push_back(r);
        print_ict_result(r);
        if (r.passed) ++passed_count;
    }

    // Summary
    int total = static_cast<int>(results.size());
    std::printf("\n=== Summary: %d/%d tests PASSED ===\n", passed_count, total);

    if (passed_count == total) {
        std::printf("All ICT roundtrip tests passed — GPU forward + CPU inverse ICT is correct within FP32 tolerance.\n");
        return 0;
    } else {
        std::printf("SOME TESTS FAILED — roundtrip error exceeds tolerance.\n");
        return 1;
    }
}
