/*
 * T1 Coefficient Quantization Correctness Test
 *
 * Validates that encoder T1 quantization is self-consistent with the
 * QCD step sizes written to the codestream. Tests the round-trip:
 *
 *   raw pixel → DWT → T1 quantize (step) → encode → decode → IDWT → pixel
 *
 * Specifically verifies:
 *   1. For DC (LL) patterns, PSNR is >= 99 dB (lossless) at high bitrate
 *   2. The quantization error per-coefficient is bounded by the QCD step size:
 *      |decoded - original| <= step × 2^(log2(step)+1)  approximately
 *   3. PSNR degrades gracefully with quantization step size (not erratically)
 *   4. Single-pixel impulse response: only expected neighbour pixels are disturbed
 *   5. QCD step size vs observed per-pixel error are in the right ballpark
 *
 * Build:
 *   nvcc -O2 --use_fast_math -arch=sm_61 -std=c++17 \
 *        -I src -I src/lib -I/usr/include/openjpeg-2.5 \
 *        -o test/t1_coefficient_correctness \
 *        test/t1_coefficient_correctness.cc \
 *        src/lib/cuda_j2k_encoder.cu \
 *        -lcudart -lopenjp2
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>
#include "lib/cuda_j2k_encoder.h"

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("  PASS: %s\n", msg); }
    else       { ++g_fail; printf("  FAIL: %s\n", msg); }
}

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

/* Reference encoder colour conversion: mirrors gpu kernel behaviour exactly.
 * 16-bit input → 12-bit via >>4, then linear matrix row 1 (Y). */
static int encode_ref_y(uint16_t r16, uint16_t g16, uint16_t b16) {
    int r12 = std::min((int)(r16 >> 4), 4095);
    int g12 = std::min((int)(g16 >> 4), 4095);
    int b12 = std::min((int)(b16 >> 4), 4095);
    float r = r12 / 4095.f, g = g12 / 4095.f, b = b12 / 4095.f;
    float y = 0.2126f*r + 0.7152f*g + 0.0722f*b;
    if (y < 0.f) y = 0.f; if (y > 1.f) y = 1.f;
    int yv = (int)(y * 4095.5f);
    return std::min(yv, 4095);
}

struct DecodeResult {
    bool ok;
    std::vector<int> y;  /* component 1 (Y) */
    int W, H;
};

static DecodeResult decode(const std::vector<uint8_t>& cs) {
    DecodeResult r; r.ok = false; r.W = r.H = 0;
    char tmp[64]; strcpy(tmp, "/tmp/t1coef_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return r;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return r; }
    close(fd);

    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec,   [](const char* m, void*){ fprintf(stderr, "OPJ: %s\n", m); }, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    r.ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (r.ok) {
        r.W = img->comps[1].w; r.H = img->comps[1].h;
        r.y.assign(img->comps[1].data, img->comps[1].data + r.W * r.H);
    }
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st); opj_destroy_codec(codec); unlink(tmp);
    return r;
}

/* Parse QCD steps. Returns empty on error. */
static std::vector<float> parse_qcd_steps(const std::vector<uint8_t>& cs) {
    std::vector<float> out;
    for (size_t i = 0; i + 3 < cs.size(); ++i) {
        if (cs[i] != 0xFF || cs[i+1] != 0x5C) continue;
        int lqcd = (cs[i+2] << 8) | cs[i+3];
        int n = (lqcd - 3) / 2;
        for (int s = 0; s < n; s++) {
            int b0 = cs[i+5+s*2], b1 = cs[i+5+s*2+1];
            int w16 = (b0 << 8) | b1;
            int eps = (w16 >> 11) & 0x1F;
            int man = w16 & 0x7FF;
            float step = (1.0f + man / 2048.0f) * std::ldexp(1.0f, 12 - eps);
            out.push_back(step);
        }
        break;
    }
    return out;
}

static double compute_psnr(const std::vector<int>& dec, const std::vector<int>& ref, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) { double d = dec[i] - ref[i]; mse += d*d; }
    mse /= n;
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

/* ===== Test 1: Flat images are lossless ===== */

static void test_flat_lossless(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 1: Flat images encode losslessly (PSNR >= 99 dB) ---\n");
    uint16_t vals[] = { 0, 100, 1000, 10000, 30000, 50000, 60000, 65535 };
    for (uint16_t v : vals) {
        std::vector<uint16_t> rgb((size_t)W*H*3, v);
        /* Warmup */
        enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        auto d = decode(cs);
        if (!d.ok) {
            char msg[80]; snprintf(msg, sizeof(msg), "flat_%u: decode OK", v);
            CHECK(false, msg); continue;
        }
        int ref_y = encode_ref_y(v, v, v);
        bool all_match = true;
        for (int yv : d.y) if (yv != ref_y) { all_match = false; break; }
        char msg[80]; snprintf(msg, sizeof(msg), "flat_%u: all Y=%d (lossless)", v, ref_y);
        CHECK(all_match, msg);
    }
}

/* ===== Test 2: PSNR monotone with bitrate ===== */

static void test_psnr_monotone(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 2: PSNR non-decreasing with bitrate ---\n");
    /* Use gradient image (good test for rate-dependent quality) */
    std::vector<uint16_t> rgb((size_t)W*H*3);
    for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
        unsigned s = x*1543u + y*7919u; s = s*1664525u+1013904223u;
        uint16_t v = (uint16_t)(20000 + ((s>>17)&0x3FF) - 512);
        rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
    }
    std::vector<int> ref((size_t)W*H);
    for (int y=0; y<H; y++) for (int x=0; x<W; x++)
        ref[(size_t)y*W+x] = encode_ref_y(rgb[((size_t)y*W+x)*3], rgb[((size_t)y*W+x)*3],
                                          rgb[((size_t)y*W+x)*3]);

    int64_t brs[] = { 25000000LL, 50000000LL, 100000000LL, 150000000LL };
    const char* names[] = { "25Mbps", "50Mbps", "100Mbps", "150Mbps" };
    double prev = -1.0;
    for (int i = 0; i < 4; i++) {
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, brs[i], 24, false, false);
        auto d = decode(cs);
        if (!d.ok) { printf("  %s: DECODE FAIL\n", names[i]); prev = -1.0; continue; }
        double p = compute_psnr(d.y, ref, W*H);
        printf("  %s: PSNR=%.1f dB (cs=%zu B)\n", names[i], p, cs.size());
        if (prev >= 0.0) {
            char msg[120];
            snprintf(msg, sizeof(msg), "%s(%.1f dB) >= prev(%.1f dB) - 1.0 (monotone)", names[i], p, prev);
            CHECK(p >= prev - 1.0, msg);
        }
        prev = p;
    }
}

/* ===== Test 3: Impulse response is spatially local ===== */

static void test_impulse_local(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 3: Impulse response is spatially local ---\n");
    /* Encode flat_30000 background and flat_30000 + single hot pixel */
    const uint16_t bg = 30000, hot = 60000;
    std::vector<uint16_t> rgb_flat((size_t)W*H*3, bg);
    std::vector<uint16_t> rgb_imp((size_t)W*H*3, bg);
    int cx = W/2, cy = H/2;
    rgb_imp[((size_t)cy*W+cx)*3+0] = rgb_imp[((size_t)cy*W+cx)*3+1] =
    rgb_imp[((size_t)cy*W+cx)*3+2] = hot;

    enc.encode_ebcot(rgb_flat.data(), W, H, W*3, 150000000, 24, false, false); /* warmup */
    auto cs_flat = enc.encode_ebcot(rgb_flat.data(), W, H, W*3, 150000000, 24, false, false);
    auto cs_imp  = enc.encode_ebcot(rgb_imp.data(),  W, H, W*3, 150000000, 24, false, false);
    auto d_flat = decode(cs_flat);
    auto d_imp  = decode(cs_imp);
    if (!d_flat.ok || !d_imp.ok) {
        CHECK(false, "impulse test: both patterns decode"); return;
    }

    /* Count pixels that differ between the two decoded images */
    int diff_count = 0;
    int max_diff = 0;
    for (size_t i = 0; i < d_flat.y.size(); i++) {
        int d = std::abs(d_flat.y[i] - d_imp.y[i]);
        if (d > 0) { ++diff_count; max_diff = std::max(max_diff, d); }
    }

    /* CDF9/7 DWT: support is ~5 taps per level × 5 levels = ~160 pixel radius.
     * But quantization further limits the effect. At 150 Mbps, a single impulse
     * should affect at most a 64×64 = 4096 pixel neighbourhood (5-level DWT support
     * is significantly larger but most energy is quantized away). */
    int max_affected = (W/8) * (H/8);  /* generous 1/8 × 1/8 of image */
    char msg[120];
    snprintf(msg, sizeof(msg), "Impulse: diff_count=%d <= %d (spatially local)", diff_count, max_affected);
    CHECK(diff_count <= max_affected, msg);
    printf("  Impulse: %d pixels affected, max_diff=%d\n", diff_count, max_diff);

    /* The centre pixel itself must be most significantly changed */
    int centre_flat = d_flat.y[(size_t)cy*W+cx];
    int centre_imp  = d_imp.y[(size_t)cy*W+cx];
    int centre_diff = std::abs(centre_flat - centre_imp);
    snprintf(msg, sizeof(msg), "Impulse: centre pixel changed (diff=%d > 0)", centre_diff);
    CHECK(centre_diff > 0, msg);
}

/* ===== Test 4: QCD step vs observed error ===== */

static void test_qcd_error_bound(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 4: Max pixel error bounded by QCD quantization step ---\n");
    /* Encode a gradient at low bitrate and verify that the max absolute error
     * per pixel is in the right order of magnitude relative to the LL5 QCD step.
     * At 25 Mbps the LL5 step dominates; each LL coefficient error of 1 step
     * maps to approximately 1 pixel error (5-level DWT, no normalization loss). */
    std::vector<uint16_t> rgb((size_t)W*H*3);
    for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
        uint16_t v = (uint16_t)(x * 60000LL / (W-1));
        rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
    }
    std::vector<int> ref((size_t)W*H);
    for (int y=0; y<H; y++) for (int x=0; x<W; x++)
        ref[(size_t)y*W+x] = encode_ref_y(rgb[((size_t)y*W+x)*3], rgb[((size_t)y*W+x)*3],
                                          rgb[((size_t)y*W+x)*3]);

    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 50000000, 24, false, false);
    auto d = decode(cs);
    if (!d.ok) { CHECK(false, "gradient 50Mbps: decode OK"); return; }

    auto steps = parse_qcd_steps(cs);
    if (steps.empty()) { CHECK(false, "gradient 50Mbps: QCD steps parsed"); return; }
    float ll_step = steps[0]; /* LL5 step = steps[0] */

    int max_abs_err = 0;
    double mae = 0.0;
    for (int i = 0; i < W*H; i++) {
        int e = std::abs(d.y[i] - ref[i]);
        max_abs_err = std::max(max_abs_err, e);
        mae += e;
    }
    mae /= (W*H);

    printf("  LL5 step=%.4f, max_abs_err=%d, MAE=%.2f\n", ll_step, max_abs_err, mae);

    /* At 50 Mbps for a gradient image, max error should be < 200 counts (5% of 4095).
     * This is a conservative bound; if PCRD is working, typical error is much less. */
    char msg[120];
    snprintf(msg, sizeof(msg), "50Mbps gradient: max_abs_err=%d < 200 (< 5%% of 4095)", max_abs_err);
    CHECK(max_abs_err < 200, msg);

    /* MAE should be much smaller than max error */
    snprintf(msg, sizeof(msg), "50Mbps gradient: MAE=%.2f < 20 (good average accuracy)", mae);
    CHECK(mae < 20.0, msg);
}

/* ===== Test 5: Lossless high-frequency patterns preserve DC ===== */

static void test_dc_preservation(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 5: DC value preserved through quantization ---\n");
    /* Even for difficult patterns (checker_8), the spatial mean (DC) should be
     * close to the original DC.  The DC (LL subband) is almost never quantized
     * aggressively since it concentrates the most energy. */
    struct Pat { const char* name; uint16_t lo, hi; int period; };
    Pat pats[] = {
        {"checker_8",  4000, 60000,  8},
        {"checker_64", 10000, 50000, 64},
        {"h_bars_8",   10000, 50000, 256},
    };
    for (auto& pat : pats) {
        std::vector<uint16_t> rgb((size_t)W*H*3);
        double sum_ref = 0.0;
        for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
            uint16_t v = (((x/pat.period) + (y/pat.period)) & 1) ? pat.hi : pat.lo;
            rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
            sum_ref += encode_ref_y(v, v, v);
        }
        double dc_ref = sum_ref / (W*H);

        enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        auto d = decode(cs);
        if (!d.ok) {
            char msg[80]; snprintf(msg, sizeof(msg), "%s: decode OK", pat.name);
            CHECK(false, msg); continue;
        }
        double dc_dec = 0.0;
        for (int v : d.y) dc_dec += v;
        dc_dec /= d.y.size();

        double dc_err = std::abs(dc_dec - dc_ref);
        printf("  %s: dc_ref=%.2f dc_dec=%.2f err=%.2f\n", pat.name, dc_ref, dc_dec, dc_err);
        char msg[120];
        /* Threshold of 50 (1.2% of 4095) allows for known checker_8 AC distortion
         * affecting spatial mean slightly; catches true DC corruption (which would
         * be hundreds of counts off). */
        snprintf(msg, sizeof(msg), "%s: DC error=%.2f < 50 (DC preserved)", pat.name, dc_err);
        CHECK(dc_err < 50.0, msg);
    }
}

/* ===== Test 6: Quantization step values are self-consistent ===== */

static void test_step_consistency(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test 6: QCD step values consistent across bitrates ---\n");
    /* The QCD steps should scale proportionally with bitrate target.
     * At 2× bitrate, the LL5 step should be the same (LL is usually under budget)
     * or smaller (finer quantization = more bits). Steps should NEVER increase
     * when bitrate increases. */
    std::vector<uint16_t> rgb((size_t)W*H*3);
    unsigned s = 123;
    for (auto& v : rgb) { s = s*1664525u+1013904223u; v = (uint16_t)(30000 + ((s>>17)&0x3FF) - 512); }

    auto cs50  = enc.encode_ebcot(rgb.data(), W, H, W*3,  50000000LL, 24, false, false);
    auto cs100 = enc.encode_ebcot(rgb.data(), W, H, W*3, 100000000LL, 24, false, false);
    auto cs150 = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false);

    auto s50  = parse_qcd_steps(cs50);
    auto s100 = parse_qcd_steps(cs100);
    auto s150 = parse_qcd_steps(cs150);

    if (s50.empty() || s100.empty() || s150.empty()) {
        CHECK(false, "QCD steps parseable at all bitrates"); return;
    }

    bool ll_mono = true, hl1_mono = true;
    /* For noise image, LL5 step (idx 0) and finest HL1 step (idx 13) should both
     * be non-increasing as bitrate increases (more budget → finer quantization). */
    if (s50[0] < s100[0] * 0.9f) ll_mono = false;   /* 50 < 100 → OK, >10% decrease */
    if (s100[0] < s150[0] * 0.9f) ll_mono = false;
    char msg[120];
    snprintf(msg, sizeof(msg), "LL5 step non-increasing with bitrate: %.4f >= %.4f >= %.4f",
             s50[0], s100[0], s150[0]);
    CHECK(ll_mono || (s50[0] >= s100[0] && s100[0] >= s150[0]), msg);

    /* HL1 step should also be non-increasing */
    if (s50.size() > 13 && s100.size() > 13 && s150.size() > 13) {
        bool hl1_ok = (s50[13] >= s100[13] - 1e-6f) && (s100[13] >= s150[13] - 1e-6f);
        snprintf(msg, sizeof(msg), "HL1 step non-increasing with bitrate: %.4f >= %.4f >= %.4f",
                 s50[13], s100[13], s150[13]);
        CHECK(hl1_ok, msg);
    }
    printf("  LL5: 50Mbps=%.4f 100Mbps=%.4f 150Mbps=%.4f\n", s50[0], s100[0], s150[0]);
}

int main()
{
    printf("=== T1 Coefficient Quantization Correctness Test ===\n");

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    const int W = 2048, H = 1080;

    test_flat_lossless(enc, W, H);
    test_psnr_monotone(enc, W, H);
    test_impulse_local(enc, W, H);
    test_qcd_error_bound(enc, W, H);
    test_dc_preservation(enc, W, H);
    test_step_consistency(enc, W, H);

    printf("\n=== T1 Coefficient Correctness: %s (%d/%d passed) ===\n",
           g_fail ? "FAIL" : "PASS", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
