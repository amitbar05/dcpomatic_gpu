/*
 * Rate-Distortion Correctness Test
 *
 * Validates that the GPU encoder's rate control and PCRD produce quality
 * meeting J2K standard expectations. PSNR is measured in 12-bit XYZ domain:
 *   - Reference: rgb_to_xyz_y_12bit mirrors the encoder's color pipeline
 *   - Decoded:   openjpeg comps[1] (CIE Y component of XYZ output)
 *
 * Encoder color pipeline (mirrored by rgb_to_xyz_y_12bit):
 *   1. v12 = min(v16 >> 4, 4095)          -- 16-bit to 12-bit
 *   2. rgb_norm = v12 / 4095.0            -- LUT linearisation
 *   3. Y = 0.2126*R + 0.7152*G + 0.0722*B -- sRGB -> CIE Y
 *   4. encoded_y = (int)(Y * 4095.5)      -- scale to 12-bit int
 *
 * All input patterns use 16-bit values; the encoder uses v16 >> 4 internally.
 * Use v16 = v12 * 16 to specify a known 12-bit gray level.
 *
 * Tests:
 *   1. Rate compliance: codestream size within target bounds
 *   2. PSNR monotone non-decreasing as bitrate increases
 *   3. Minimum PSNR at 150 Mbps vs known-good baselines
 *   4. Near-lossless quality at 250 Mbps
 *   5. DC preservation: flat images decode to correct mean
 *   6. Max per-pixel error bounded
 *   7. XYZ component PSNR balance for achromatic content
 *   8. Determinism: same input always produces same output
 *   9. Different inputs produce different codestreams
 *  10. Flat images encoded losslessly at all tested bitrates
 *
 * Build:
 *   nvcc -O2 --use_fast_math -arch=sm_61 -std=c++17 \
 *       -I src -I src/lib -I/usr/include/openjpeg-2.5 \
 *       -o test/rate_distortion_correctness \
 *       test/rate_distortion_correctness.cc \
 *       src/lib/cuda_j2k_encoder.cu \
 *       -lcudart -lopenjp2
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <numeric>
#include <unistd.h>
#include <openjpeg.h>
#include "lib/cuda_j2k_encoder.h"

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("  PASS: %s\n", msg); }
    else       { ++g_fail; printf("  FAIL: %s\n", msg); }
}

/* ===== Encoder setup ===== */

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

static const int W = 2048, H = 1080, FPS = 24;

static std::vector<uint8_t> encode(CudaJ2KEncoder& enc,
    const std::vector<uint16_t>& frame, int mbps)
{
    return enc.encode_ebcot(frame.data(), W, H, W * 3,
                            (int64_t)mbps * 1000000, FPS, false, false);
}

/* ===== Color pipeline reference ===== */
/* Mirrors encoder kernel exactly: v16>>4 -> float -> XYZ matrix -> 12-bit int.
 * Derived from psnr_battery.cc's rgb_to_xyz_y_12bit (calibrated to match GPU
 * floating-point rounding; static_cast<int>(y * 4095.5f) matches the kernel). */

static inline int rgb_to_xyz_y_12bit(int r16, int g16, int b16) {
    int r12 = std::min(r16 >> 4, 4095);
    int g12 = std::min(g16 >> 4, 4095);
    int b12 = std::min(b16 >> 4, 4095);
    float r = r12 / 4095.f, g = g12 / 4095.f, b = b12 / 4095.f;
    float y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    if (y < 0.f) y = 0.f; if (y > 1.f) y = 1.f;
    int yv = static_cast<int>(y * 4095.5f);
    return std::min(yv, 4095);
}

static inline void rgb_to_xyz_12bit(int r16, int g16, int b16,
                                     int& x12, int& y12, int& z12) {
    int r12 = std::min(r16 >> 4, 4095);
    int g12 = std::min(g16 >> 4, 4095);
    int b12 = std::min(b16 >> 4, 4095);
    float r = r12/4095.f, g = g12/4095.f, b = b12/4095.f;
    float xf = 0.4124f*r + 0.3576f*g + 0.1805f*b;
    float yf = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    float zf = 0.0193f*r + 0.1192f*g + 0.9505f*b;
    auto cs = [](float v) -> int {
        if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
        return std::min((int)(v * 4095.5f), 4095);
    };
    x12 = cs(xf); y12 = cs(yf); z12 = cs(zf);
}

/* ===== OPJ decode ===== */

struct DecodeResult {
    bool ok;
    int w, h, nc;
    std::vector<std::vector<int32_t>> comps;
};

/* Renamed to avoid shadowing the OPJ library opj_decode() function. */
static DecodeResult opj_decode_cs(const std::vector<uint8_t>& cs) {
    DecodeResult r; r.ok = false; r.w = r.h = r.nc = 0;
    char tmp[64]; strcpy(tmp, "/tmp/rdcorr_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return r;
    write(fd, cs.data(), cs.size()); close(fd);

    opj_dparameters_t p; opj_set_default_decoder_parameters(&p);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec) { unlink(tmp); return r; }
    opj_set_info_handler(codec,  nullptr, nullptr);
    opj_set_warning_handler(codec, nullptr, nullptr);
    opj_set_error_handler(codec, nullptr, nullptr);
    opj_setup_decoder(codec, &p);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    if (!st) { opj_destroy_codec(codec); unlink(tmp); return r; }
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) &&
              opj_decode(codec, st, img) &&
              opj_end_decompress(codec, st);
    if (ok && img) {
        r.ok = true; r.w = (int)img->x1; r.h = (int)img->y1;
        r.nc = (int)img->numcomps;
        r.comps.resize(r.nc);
        for (int c = 0; c < r.nc; ++c) {
            int n = (int)(img->comps[c].w * img->comps[c].h);
            r.comps[c].assign(img->comps[c].data, img->comps[c].data + n);
        }
        opj_image_destroy(img);
    }
    opj_stream_destroy(st); opj_destroy_codec(codec); unlink(tmp);
    return r;
}

/* ===== PSNR helpers (12-bit XYZ domain, PEAK=4095) ===== */

static double compute_psnr_y(const std::vector<uint16_t>& frame,
                              const DecodeResult& r)
{
    if (!r.ok || r.nc < 2) return 0.0;
    int n = W * H;
    double mse = 0.0;
    for (int i = 0; i < n; ++i) {
        int ref = rgb_to_xyz_y_12bit(frame[i*3], frame[i*3+1], frame[i*3+2]);
        double e = ref - (double)r.comps[1][i];
        mse += e * e;
    }
    mse /= n;
    /* Use 999.0 for lossless so that truly lossless results (∞ dB) sort above
     * any finite PSNR value (e.g., 101 dB) in monotonicity checks. */
    if (mse < 1e-12) return 999.0;
    return 10.0 * std::log10(4095.0 * 4095.0 / mse);
}

static double compute_psnr_comp(const std::vector<int32_t>& dec,
                                 const std::vector<int>& ref)
{
    int n = (int)std::min(dec.size(), ref.size());
    double mse = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = ref[i] - (double)dec[i];
        mse += e * e;
    }
    mse /= n;
    if (mse < 1e-12) return 999.0;
    return 10.0 * std::log10(4095.0 * 4095.0 / mse);
}

/* ===== Pattern generators ===== */
/* All values are 16-bit; encoder uses v16 >> 4 internally. */

static std::vector<uint16_t> make_flat(uint16_t v16) {
    return std::vector<uint16_t>((size_t)W * H * 3, v16);
}

static std::vector<uint16_t> make_h_gradient() {
    std::vector<uint16_t> f((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = (uint16_t)(x * 60000ll / (W - 1));
            for (int c = 0; c < 3; ++c) f[((size_t)y * W + x) * 3 + c] = v;
        }
    return f;
}

static std::vector<uint16_t> make_checker(int sz,
                                           uint16_t hi = 50000,
                                           uint16_t lo = 10000) {
    std::vector<uint16_t> f((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = (((x / sz) + (y / sz)) & 1) ? hi : lo;
            for (int c = 0; c < 3; ++c) f[((size_t)y * W + x) * 3 + c] = v;
        }
    return f;
}

static std::vector<uint16_t> make_noise(uint32_t seed) {
    std::vector<uint16_t> f((size_t)W * H * 3);
    uint32_t s = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            s = s * 1664525u + 1013904223u;
            int v = 30000 + (int)((s >> 17) & 0x3FF) - 512;
            uint16_t v16 = (uint16_t)std::max(0, std::min(65535, v));
            for (int c = 0; c < 3; ++c) f[((size_t)y * W + x) * 3 + c] = v16;
        }
    return f;
}

static std::vector<uint16_t> make_photo_synth() {
    std::vector<uint16_t> f((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float cx = (x - W/2.f) / (W/2.f), cy = (y - H/2.f) / (H/2.f);
            float radial = std::sqrt(cx*cx + cy*cy);
            unsigned s = (unsigned)(x*1543u + y*7919u);
            s = s * 1664525u + 1013904223u;
            int noise = (int)((s >> 17) & 0xFF) - 128;
            int base = (int)(40000 + 20000 * (1.0f - radial));
            if ((x/256) == 4 && (y/270) == 2) base = 60000;
            int v = base + noise;
            v = std::max(1000, std::min(65000, v));
            for (int c = 0; c < 3; ++c) f[((size_t)y * W + x) * 3 + c] = (uint16_t)v;
        }
    return f;
}

/* ===== Tests ===== */

static void test_rate_compliance(CudaJ2KEncoder& enc) {
    printf("\n--- Rate Compliance (codestream size within target bounds) ---\n");
    struct {
        const char* name;
        std::function<std::vector<uint16_t>()> gen;
        bool high_entropy; /* expect codestream to fill budget */
    } pats[] = {
        { "flat_30000",  []{ return make_flat(30000); },         false },
        { "h_gradient",  []{ return make_h_gradient(); },        false },
        { "checker_8",   []{ return make_checker(8, 60000, 4000); }, true },
        { "noise",       []{ return make_noise(42); },           true },
    };
    const int bitrates[] = { 50, 100, 150, 250 };
    for (auto& pat : pats) {
        auto frame = pat.gen();
        for (int mbps : bitrates) {
            int64_t target = (int64_t)mbps * 1000000 / FPS / 8;
            auto cs = encode(enc, frame, mbps);
            int64_t sz = (int64_t)cs.size();
            /* Upper bound: never exceed target + 5% */
            bool upper_ok = sz <= target + target / 20;
            /* Lower bound for high-entropy patterns: should use ≥ 90% of budget */
            bool lower_ok = !pat.high_entropy || (sz >= target * 9 / 10);
            /* For all patterns: codestream must be non-empty J2K (at least 500 B) */
            bool nonempty = sz >= 500;
            bool ok = upper_ok && lower_ok && nonempty;
            char msg[128];
            snprintf(msg, sizeof(msg),
                     "%s @ %dMbps: cs=%lldB target=%lldB (%.1f%%)",
                     pat.name, mbps, (long long)sz, (long long)target,
                     100.0 * sz / target);
            CHECK(ok, msg);
        }
    }
}

static void test_psnr_monotone(CudaJ2KEncoder& enc) {
    printf("\n--- PSNR Monotone Non-Decreasing with Bitrate ---\n");
    struct { const char* name; std::function<std::vector<uint16_t>()> gen; } pats[] = {
        { "flat_30000",   []{ return make_flat(30000); } },
        { "h_gradient",   []{ return make_h_gradient(); } },
        { "checker_8",    []{ return make_checker(8, 60000, 4000); } },
        { "checker_64",   []{ return make_checker(64); } },
        { "noise",        []{ return make_noise(42); } },
        { "photo_synth",  []{ return make_photo_synth(); } },
    };
    const int rates[] = { 50, 100, 150, 250 };
    for (auto& pat : pats) {
        auto frame = pat.gen();
        double prev = -1.0;
        bool monotone = true;
        for (int mbps : rates) {
            auto cs = encode(enc, frame, mbps);
            auto r = opj_decode_cs(cs);
            if (!r.ok) { monotone = false; break; }
            double q = compute_psnr_y(frame, r);
            /* 2 dB tolerance: allows small non-monotonicity from base_step
             * discretisation at specific bitrate transitions. */
            if (q < prev - 2.0) { monotone = false; break; }
            prev = q;
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "%s PSNR non-decreasing 50→250 Mbps", pat.name);
        CHECK(monotone, msg);
    }
}

static void test_minimum_psnr(CudaJ2KEncoder& enc) {
    printf("\n--- Minimum PSNR at 150 Mbps (vs known-good baselines) ---\n");
    struct {
        const char* name;
        std::function<std::vector<uint16_t>()> gen;
        double min_psnr;
    } tests[] = {
        /* Baseline from psnr_battery: flat ~99, gradient ~99, checker_8 ~16.6,
         * checker_64 ~31.9, noise_small ~50.8, photo_synth ~61.7 */
        { "flat_30000",  []{ return make_flat(30000); },         95.0 },
        { "h_gradient",  []{ return make_h_gradient(); },        90.0 },
        { "checker_8",   []{ return make_checker(8, 60000, 4000); }, 13.0 },
        { "checker_64",  []{ return make_checker(64); },         25.0 },
        { "noise",       []{ return make_noise(42); },           45.0 },
        { "photo_synth", []{ return make_photo_synth(); },       50.0 },
    };
    for (auto& t : tests) {
        auto frame = t.gen();
        auto cs = encode(enc, frame, 150);
        auto r = opj_decode_cs(cs);
        double q = r.ok ? compute_psnr_y(frame, r) : 0.0;
        char msg[128];
        snprintf(msg, sizeof(msg), "%s @ 150 Mbps: PSNR=%.1f dB (min %.1f)",
                 t.name, q, t.min_psnr);
        CHECK(q >= t.min_psnr, msg);
    }
}

static void test_high_bitrate_quality(CudaJ2KEncoder& enc) {
    printf("\n--- High Bitrate (250 Mbps) Near-Lossless Quality ---\n");
    struct {
        const char* name;
        std::function<std::vector<uint16_t>()> gen;
        double min_psnr;
    } tests[] = {
        { "flat_10000",  []{ return make_flat(10000); },    99.0 },
        { "flat_30000",  []{ return make_flat(30000); },    99.0 },
        { "flat_50000",  []{ return make_flat(50000); },    99.0 },
        { "h_gradient",  []{ return make_h_gradient(); },   90.0 },
        { "checker_64",  []{ return make_checker(64); },    30.0 },
    };
    for (auto& t : tests) {
        auto frame = t.gen();
        auto cs = encode(enc, frame, 250);
        auto r = opj_decode_cs(cs);
        double q = r.ok ? compute_psnr_y(frame, r) : 0.0;
        char msg[128];
        snprintf(msg, sizeof(msg), "%s @ 250 Mbps: PSNR=%.1f dB (min %.1f)",
                 t.name, q, t.min_psnr);
        CHECK(q >= t.min_psnr, msg);
    }
}

static void test_dc_preservation(CudaJ2KEncoder& enc) {
    printf("\n--- DC (Mean Value) Preservation for Flat Images ---\n");
    /* Expected Y = rgb_to_xyz_y_12bit(v16,v16,v16) ≈ v16>>4 (since sRGB Y row sums to 1) */
    const uint16_t v16vals[] = { 0, 10000, 16384, 32768, 50000, 65520 };
    for (uint16_t v16 : v16vals) {
        int expected_y = rgb_to_xyz_y_12bit(v16, v16, v16);
        auto frame = make_flat(v16);
        auto cs = encode(enc, frame, 150);
        auto r = opj_decode_cs(cs);
        if (!r.ok || r.nc < 2) { CHECK(false, "flat dc: decode failed"); continue; }
        double sum = 0.0;
        for (auto x : r.comps[1]) sum += x;
        double mean = sum / (W * H);
        bool ok = std::abs(mean - expected_y) < 2.0;
        char msg[80];
        snprintf(msg, sizeof(msg), "flat_v16=%u: mean_Y=%.2f expected=%d",
                 (unsigned)v16, mean, expected_y);
        CHECK(ok, msg);
    }
}

static void test_max_pixel_error(CudaJ2KEncoder& enc) {
    printf("\n--- Maximum Per-Pixel Y Error Bounded ---\n");
    struct {
        const char* name;
        std::function<std::vector<uint16_t>()> gen;
        int max_err;
    } tests[] = {
        { "flat_30000",  []{ return make_flat(30000); },         1 },   /* flat is lossless */
        { "h_gradient",  []{ return make_h_gradient(); },       50 },   /* smooth ramp */
        { "checker_64",  []{ return make_checker(64); },      2000 },   /* hard edge pattern */
    };
    for (auto& t : tests) {
        auto frame = t.gen();
        auto cs = encode(enc, frame, 150);
        auto r = opj_decode_cs(cs);
        if (!r.ok || r.nc < 2) { CHECK(false, "max_pixel_err: decode failed"); continue; }
        int max_e = 0;
        for (int i = 0; i < W * H; ++i) {
            int ref = rgb_to_xyz_y_12bit(frame[i*3], frame[i*3+1], frame[i*3+2]);
            int e = std::abs(ref - (int)r.comps[1][i]);
            max_e = std::max(max_e, e);
        }
        bool ok = max_e <= t.max_err;
        char msg[128];
        snprintf(msg, sizeof(msg), "%s @ 150 Mbps: max_Y_err=%d (limit %d)",
                 t.name, max_e, t.max_err);
        CHECK(ok, msg);
    }
}

static void test_xyz_component_balance(CudaJ2KEncoder& enc) {
    printf("\n--- XYZ Component PSNR Balance (achromatic R=G=B input) ---\n");
    /* For R=G=B: X≈0.9505*v12, Y=v12, Z≈0.989*v12 in 12-bit. All 3 components
     * are coded independently with similar magnitudes → PSNR spread < 15 dB. */
    auto frame = make_checker(8, 60000, 4000);
    auto cs = encode(enc, frame, 150);
    auto r = opj_decode_cs(cs);
    if (!r.ok || r.nc < 3) {
        CHECK(false, "component balance: decode failed or < 3 comps");
        return;
    }
    std::vector<int> ref_x(W*H), ref_y(W*H), ref_z(W*H);
    for (int i = 0; i < W*H; ++i) {
        int x12, y12, z12;
        rgb_to_xyz_12bit(frame[i*3], frame[i*3+1], frame[i*3+2], x12, y12, z12);
        ref_x[i] = x12; ref_y[i] = y12; ref_z[i] = z12;
    }
    double qx = compute_psnr_comp(r.comps[0], ref_x);
    double qy = compute_psnr_comp(r.comps[1], ref_y);
    double qz = compute_psnr_comp(r.comps[2], ref_z);
    printf("    X=%.1f Y=%.1f Z=%.1f dB\n", qx, qy, qz);
    double qmin = std::min({qx, qy, qz});
    double qmax = std::max({qx, qy, qz});
    CHECK(qmax - qmin < 15.0, "X/Y/Z PSNR spread < 15 dB for achromatic input");
    CHECK(qy > 13.0, "Y PSNR > 13 dB for checker_8 @ 150 Mbps");
}

static void test_determinism(CudaJ2KEncoder& enc) {
    printf("\n--- Determinism (same input → same output) ---\n");
    auto frame = make_checker(8, 60000, 4000);
    auto cs1 = encode(enc, frame, 150);
    auto cs2 = encode(enc, frame, 150);
    auto cs3 = encode(enc, frame, 150);
    CHECK(cs1 == cs2, "Run 1 == Run 2 (byte-identical)");
    CHECK(cs2 == cs3, "Run 2 == Run 3 (byte-identical)");
}

static void test_different_inputs_different_output(CudaJ2KEncoder& enc) {
    printf("\n--- Different Inputs Produce Different Outputs ---\n");
    auto f1 = make_flat(30000);
    auto f2 = make_checker(8, 60000, 4000);
    auto f3 = make_noise(42);
    auto cs1 = encode(enc, f1, 150);
    auto cs2 = encode(enc, f2, 150);
    auto cs3 = encode(enc, f3, 150);
    CHECK(cs1 != cs2, "flat vs checker produce different codestreams");
    CHECK(cs2 != cs3, "checker vs noise produce different codestreams");
}

static void test_lossless_flat(CudaJ2KEncoder& enc) {
    printf("\n--- Flat Images Losslessly Coded at All Tested Bitrates ---\n");
    /* Flat images have zero AC DWT coefficients → trivially lossless at any bitrate.
     * Tolerance ±1 covers CPU vs GPU floating-point rounding in color pipeline. */
    const uint16_t v16vals[] = { 0, 10000, 32768, 50000, 65520 };
    const int bitrates[] = { 50, 100, 150, 250 };
    for (uint16_t v16 : v16vals) {
        int expected_y = rgb_to_xyz_y_12bit(v16, v16, v16);
        for (int mbps : bitrates) {
            auto frame = make_flat(v16);
            auto cs = encode(enc, frame, mbps);
            auto r = opj_decode_cs(cs);
            bool ok = false;
            if (r.ok && r.nc >= 2) {
                ok = true;
                for (int i = 0; i < W * H && ok; ++i)
                    if (std::abs((int)r.comps[1][i] - expected_y) > 1) ok = false;
            }
            char msg[72];
            snprintf(msg, sizeof(msg), "flat_v16=%u @ %dMbps: lossless (exp_Y=%d)",
                     (unsigned)v16, mbps, expected_y);
            CHECK(ok, msg);
        }
    }
}

int main() {
    printf("=== Rate-Distortion Correctness Test (W=%d H=%d) ===\n\n", W, H);

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) {
        fprintf(stderr, "CUDA encoder init failed\n");
        return 1;
    }
    GpuColourParams p;
    build_params(p);
    enc.set_colour_params(p);

    test_rate_compliance(enc);
    test_psnr_monotone(enc);
    test_minimum_psnr(enc);
    test_high_bitrate_quality(enc);
    test_dc_preservation(enc);
    test_max_pixel_error(enc);
    test_xyz_component_balance(enc);
    test_determinism(enc);
    test_different_inputs_different_output(enc);
    test_lossless_flat(enc);

    printf("\n=== Rate-Distortion Correctness: %s (%d/%d passed) ===\n",
           g_fail == 0 ? "PASS" : "FAIL",
           g_pass, g_pass + g_fail);
    return g_fail > 0 ? 1 : 0;
}
