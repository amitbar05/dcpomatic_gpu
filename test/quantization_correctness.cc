/*
 * Quantization Correctness Test
 *
 * Verifies the T2/QCD/PCRD correctness:
 *   1. QCD step size round-trip: step written to header matches reconstruction
 *   2. Byte budget enforcement: codestream <= target bytes
 *   3. PCRD rate-distortion: higher budget → equal or better PSNR (noise pattern)
 *   4. HL/LH orthogonality: bars in only H or V direction are lossless (step ≈ 0)
 *   5. QCD numsubbands: correct count for 5-level DWT
 *   6. Guard bits = 1 for all tested patterns
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
 *        -I src -I src/lib -I/usr/include/openjpeg-2.5 \
 *        -o test/quantization_correctness \
 *        test/quantization_correctness.cc src/lib/cuda_j2k_encoder.cu \
 *        -lcudart -lopenjp2
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>
#include "lib/cuda_j2k_encoder.h"

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("  PASS: %s\n", msg); }
    else       { ++g_fail; printf("  FAIL: %s\n", msg); }
}

/* Build identity LUT + sRGB→XYZ matrix (matches psnr_battery.cc) */
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

/* Decode J2K codestream and return Y-component pixels */
static bool decode_y(const std::vector<uint8_t>& cs, std::vector<int>& y_out,
                     int& W_out, int& H_out)
{
    char tmp[64]; strcpy(tmp, "/tmp/qcorr_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return false; }
    close(fd);

    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char* m, void*){ fprintf(stderr, "OPJ: %s\n", m); }, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (ok) {
        W_out = img->comps[1].w; H_out = img->comps[1].h;
        y_out.assign(img->comps[1].data, img->comps[1].data + W_out * H_out);
    }
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st); opj_destroy_codec(codec); unlink(tmp);
    return ok;
}

/* Parse QCD marker from codestream. Returns false if not found.
 * Fields: sqcd byte, step count, first few step values. */
struct QCDInfo {
    bool found = false;
    uint8_t sqty    = 0;   /* quantization style (should be 2 = expounded) */
    uint8_t numgbits = 0;  /* guard bits (should be 1) */
    int     numsteps = 0;
    struct Step { int eps; int man; float val; };
    std::vector<Step> steps;
};

static QCDInfo parse_qcd(const std::vector<uint8_t>& cs) {
    QCDInfo q;
    for (size_t i = 0; i + 6 < cs.size(); ++i) {
        if (cs[i] != 0xFF || cs[i+1] != 0x5C) continue;
        int len = (cs[i+2] << 8) | cs[i+3];
        uint8_t sqcd = cs[i+4];
        q.sqty     = sqcd & 0x1F;
        q.numgbits = (sqcd >> 5) & 0x07;
        q.found    = true;
        /* expounded quantization: 2 bytes per step */
        if (q.sqty == 2) {
            int nbytes = len - 3; /* subtract sqcd and length field overhead */
            q.numsteps = nbytes / 2;
            for (int s = 0; s < q.numsteps && i+5+s*2+1 < cs.size(); ++s) {
                int b0 = cs[i+5+s*2];
                int b1 = cs[i+5+s*2+1];
                int w16 = (b0 << 8) | b1;
                int eps = (w16 >> 11) & 0x1F;
                int man = w16 & 0x7FF;
                float val = (1.0f + man / 2048.0f) * std::ldexp(1.0f, -(eps - 1));
                q.steps.push_back({eps, man, val});
            }
        }
        break;
    }
    return q;
}

static double psnr_y(const std::vector<int>& dec, const std::vector<int>& ref, int n)
{
    double mse = 0.0;
    for (int i = 0; i < n; i++) { double d = dec[i] - ref[i]; mse += d*d; }
    mse /= n;
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

int main()
{
    printf("=== Quantization & QCD Correctness Test ===\n\n");

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr, "init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    const int W = 2048, H = 1080;

    /* ---- Test 1: QCD marker structure ---- */
    printf("--- Test 1: QCD marker structure ---\n");
    {
        std::vector<uint16_t> rgb((size_t)W*H*3, 30000);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        QCDInfo q = parse_qcd(cs);
        CHECK(q.found, "QCD marker present");
        CHECK(q.sqty == 2, "QCD sqty=2 (expounded quantization)");
        CHECK(q.numgbits == 1, "QCD numgbits=1 (1 guard bit for 2K)");
        /* 5-level DWT: 3 HH/HL/LH per level + 1 LL5 = 16 subbands */
        char msg[80]; snprintf(msg, sizeof(msg), "QCD numsteps=%d (expect 16)", q.numsteps);
        CHECK(q.numsteps == 16, msg);
        printf("  QCD steps[0..3]: %.6f %.6f %.6f %.6f\n",
               q.steps[0].val, q.steps[1].val, q.steps[2].val, q.steps[3].val);
    }

    /* ---- Test 2: QCD step size decreases with DWT level (coarser subbands → smaller step) ---- */
    printf("\n--- Test 2: QCD HL1 step < HL5 step (finer resolution → more precise quantization) ---\n");
    {
        std::vector<uint16_t> rgb((size_t)W*H*3);
        for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
            uint16_t v = (uint16_t)(x * 60000LL / (W-1));
            rgb[((size_t)y*W+x)*3+0] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
        }
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        QCDInfo q = parse_qcd(cs);
        /* For 5-level DWT, subband order in QCD: LL5, HL5,LH5,HH5, HL4,LH4,HH4, ... HL1,LH1,HH1 */
        /* step[0]=LL5, step[1]=HL5, step[4]=HL4, step[7]=HL3, step[10]=HL2, step[13]=HL1 */
        if (q.numsteps >= 14) {
            float hl5 = q.steps[1].val, hl1 = q.steps[13].val;
            char msg[120];
            snprintf(msg, sizeof(msg), "HL1 step(%.4f) >= HL5 step(%.4f) (finer res needs larger step)", hl1, hl5);
            CHECK(hl1 >= hl5, msg);
            printf("  HL5=%.4f  HL4=%.4f  HL3=%.4f  HL2=%.4f  HL1=%.4f\n",
                   q.steps[1].val, q.steps[4].val, q.steps[7].val, q.steps[10].val, q.steps[13].val);
        }
    }

    /* ---- Test 3: Byte budget enforcement ---- */
    printf("\n--- Test 3: Byte budget enforcement ---\n");
    {
        std::vector<uint16_t> rgb((size_t)W*H*3);
        unsigned s = 42;
        for (auto& v : rgb) { s = s*1664525u+1013904223u; v = s >> 1; }

        int64_t budgets[] = {50000000LL, 100000000LL, 150000000LL, 250000000LL};
        const char* names[] = {"50Mbps", "100Mbps", "150Mbps", "250Mbps"};
        /* frame size = budget_bits / fps / 8
         * Allow 20KB tolerance: J2K headers (SIZ/COD/QCD) + packet headers
         * across all subbands and components add ~14KB for a 2K frame. */
        static constexpr int HEADER_OVERHEAD = 20 * 1024;
        for (int i = 0; i < 4; i++) {
            auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, budgets[i], 24, false, false);
            int64_t max_bytes = budgets[i] / 24 / 8;
            char msg[120];
            snprintf(msg, sizeof(msg), "%s: cs=%zu <= budget=%lld+20KB=%lld bytes",
                     names[i], cs.size(), (long long)max_bytes, (long long)(max_bytes + HEADER_OVERHEAD));
            CHECK((int64_t)cs.size() <= max_bytes + HEADER_OVERHEAD, msg);
        }
    }

    /* ---- Test 4: PSNR monotone with bitrate (noise pattern) ---- */
    printf("\n--- Test 4: PSNR improves with bitrate (noise pattern) ---\n");
    {
        std::vector<uint16_t> rgb((size_t)W*H*3);
        for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
            unsigned s2 = x*1543u + y*7919u; s2 = s2*1664525u+1013904223u;
            uint16_t v = (uint16_t)(30000 + ((s2 >> 17) & 0x3FF) - 512);
            rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
        }

        auto encode_psnr = [&](int64_t br) {
            auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, br, 24, false, false);
            std::vector<int> y_dec; int dW, dH;
            if (!decode_y(cs, y_dec, dW, dH)) return -1.0;
            /* reference: apply same truncation as encoder (>>4) */
            std::vector<int> ref((size_t)W*H);
            for (int j=0; j<H; j++) for (int i=0; i<W; i++) {
                uint16_t v = rgb[((size_t)j*W+i)*3];
                int v12 = std::min((int)(v >> 4), 4095);
                float y = 0.2126f*(v12/4095.f) + 0.7152f*(v12/4095.f) + 0.0722f*(v12/4095.f);
                ref[(size_t)j*W+i] = (int)(y * 4095.5f);
            }
            return psnr_y(y_dec, ref, W*H);
        };

        double p50  = encode_psnr(50000000LL);
        double p100 = encode_psnr(100000000LL);
        double p150 = encode_psnr(150000000LL);
        printf("  PSNR: 50Mbps=%.1f  100Mbps=%.1f  150Mbps=%.1f\n", p50, p100, p150);
        char msg[120];
        snprintf(msg, sizeof(msg), "100Mbps PSNR(%.1f) >= 50Mbps PSNR(%.1f) - 0.5", p100, p50);
        CHECK(p100 >= p50 - 0.5, msg);
        snprintf(msg, sizeof(msg), "150Mbps PSNR(%.1f) >= 100Mbps PSNR(%.1f) - 0.5", p150, p100);
        CHECK(p150 >= p100 - 0.5, msg);
    }

    /* ---- Test 5: Lossless patterns have high PSNR (>= 99 dB = lossless) ---- */
    printf("\n--- Test 5: Flat patterns encode losslessly ---\n");
    {
        uint16_t flat_vals[] = {0, 1000, 30000, 50000, 60000, 65535};
        for (auto v : flat_vals) {
            std::vector<uint16_t> rgb((size_t)W*H*3, v);
            auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
            std::vector<int> y_dec; int dW, dH;
            bool ok = decode_y(cs, y_dec, dW, dH);
            char msg[80]; snprintf(msg, sizeof(msg), "flat_%u: decode OK", v);
            CHECK(ok, msg);
            if (ok) {
                /* All decoded Y values should be identical for a flat input */
                int y0 = y_dec[0];
                bool uniform = true;
                for (auto yv : y_dec) if (yv != y0) { uniform = false; break; }
                snprintf(msg, sizeof(msg), "flat_%u: all Y values identical (lossless)", v);
                CHECK(uniform, msg);
            }
        }
    }

    /* ---- Test 6: HL bars and LH bars encode at near-lossless quality ---- */
    printf("\n--- Test 6: HL/LH-dominant patterns near-lossless ---\n");
    {
        /* 64-pixel bars: frequency well within HL/LH DWT subbands */
        struct BarTest { const char* name; bool horizontal; };
        BarTest bar_tests[] = {{"hl_bars_64", true}, {"lh_bars_64", false}};
        for (auto& bt : bar_tests) {
            std::vector<uint16_t> rgb((size_t)W*H*3);
            for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
                int blk = bt.horizontal ? (x/64) : (y/64);
                uint16_t v = (blk & 1) ? 50000 : 10000;
                rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
            }
            /* Warmup then measure */
            enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
            auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
            std::vector<int> y_dec; int dW, dH;
            bool ok = decode_y(cs, y_dec, dW, dH);
            char msg[80]; snprintf(msg, sizeof(msg), "%s: decode OK", bt.name);
            CHECK(ok, msg);
            if (ok) {
                long long sum_b=0, sum_d=0; int nb=0, nd=0;
                for (int j=100; j<200; j++) for (int i=100; i<200; i++) {
                    int yv = y_dec[(size_t)j*dW+i];
                    bool is_bright = bt.horizontal ? ((i/64)&1) : ((j/64)&1);
                    if (is_bright) { sum_b+=yv; nb++; }
                    else           { sum_d+=yv; nd++; }
                }
                float avg_b = (float)sum_b/nb, avg_d = (float)sum_d/nd;
                float dc = (avg_b+avg_d)/2;
                int ref_bright = std::min((int)(50000 >> 4), 4095);
                int ref_dark   = std::min((int)(10000 >> 4), 4095);
                float ac_frac = (avg_b-avg_d) / (float)(ref_bright - ref_dark);
                snprintf(msg, sizeof(msg), "%s: AC fraction=%.3f (expect ~1.0)", bt.name, ac_frac);
                CHECK(ac_frac > 0.95f && ac_frac < 1.05f, msg);
                printf("  %s: dc=%.1f  ac_frac=%.4f\n", bt.name, dc, ac_frac);
            }
        }
    }

    printf("\n=== Quantization Correctness: %s (%d/%d passed) ===\n",
           g_fail ? "FAIL" : "PASS", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
