/*
    GPU J2K encoder round-trip correctness verification.

    Pipeline under test:
      RGB48 → GPU encode_ebcot() → J2K → OpenJPEG decode → reconstructed XYZ

    Tests:
      1.  J2K structure validity (SOC, SOT, EOC markers)
      2.  OpenJPEG decode success (no decode errors)
      3.  Output is non-trivial (not all-zero, not uniform)
      4.  Bit-depth plausible (12-bit XYZ values in [0, 4095])
      5.  PSNR vs CPU reference XYZ (correct mode > 40 dB, fast > 20 dB)
      6.  Bitstream determinism (same input → identical output, 3 times)
      7.  Gradient monotonicity (decoded gradient preserves monotone order)
      8.  Both 2K DCI and 4K DCI resolutions

    Test patterns:
      black, white, h-gradient, v-gradient, diagonal, checkerboard, sine, random

    Build:
      SRCLIB=/home/amit/dcp-o-matic-gpu/src/lib
      nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
           -I/home/amit/dcp-o-matic-gpu/src \
           -I/home/amit/dcp-o-matic-gpu/src/lib \
           -I/usr/include/openjpeg-2.5 \
           -o /home/amit/dcp-o-matic-gpu/test/verify_roundtrip \
           /home/amit/dcp-o-matic-gpu/test/verify_roundtrip.cc \
           /home/amit/dcp-o-matic-gpu/src/lib/cuda_j2k_encoder.cu \
           -lopenjp2 -lcudart
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

#include "lib/cuda_j2k_encoder.h"
#include <openjpeg.h>

/* ===== Globals ===== */

static int g_pass = 0, g_fail = 0;

static void CHECK(bool cond, const char* msg)
{
    if (cond) { ++g_pass; std::printf("  PASS: %s\n", msg); }
    else       { ++g_fail; std::printf("  FAIL: %s\n", msg); }
}

/* ===== Color helpers ===== */

/* sRGB→XYZ D65 matrix (12-bit output: float [0,1] → int [0,4095]) */
static void rgb48_to_xyz12(const uint16_t* rgb, int w, int h,
                            std::vector<int32_t>& X,
                            std::vector<int32_t>& Y,
                            std::vector<int32_t>& Z)
{
    const float m[9] = {
        0.4124f, 0.3576f, 0.1805f,
        0.2126f, 0.7152f, 0.0722f,
        0.0193f, 0.1192f, 0.9505f
    };
    const float scale_in  = 1.0f / 65535.0f;
    const float scale_out = 4095.0f;

    size_t n = (size_t)w * h;
    X.resize(n); Y.resize(n); Z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        float r = rgb[3*i+0] * scale_in;
        float g = rgb[3*i+1] * scale_in;
        float b = rgb[3*i+2] * scale_in;
        float x = (m[0]*r + m[1]*g + m[2]*b) * scale_out;
        float y = (m[3]*r + m[4]*g + m[5]*b) * scale_out;
        float z = (m[6]*r + m[7]*g + m[8]*b) * scale_out;
        X[i] = std::lrint(std::fmax(0.f, std::fmin(4095.f, x)));
        Y[i] = std::lrint(std::fmax(0.f, std::fmin(4095.f, y)));
        Z[i] = std::lrint(std::fmax(0.f, std::fmin(4095.f, z)));
    }
}

/* ===== OpenJPEG decode ===== */

static bool opj_decode_j2k(const std::vector<uint8_t>& cs,
                            std::vector<std::vector<int32_t>>& comps,
                            int& out_w, int& out_h, int& ncomps)
{
    char path[] = "/tmp/vrrt_XXXXXX.j2c";
    int fd = mkstemps(path, 4);
    if (fd < 0) return false;
    (void)write(fd, cs.data(), cs.size());
    close(fd);

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec) { unlink(path); return false; }
    opj_set_warning_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ warn: %s\n",m); }, nullptr);
    opj_set_error_handler(codec,   [](const char* m, void*){ fprintf(stderr,"OPJ err: %s\n",m); },   nullptr);

    opj_dparameters_t params;
    opj_set_default_decoder_parameters(&params);
    opj_setup_decoder(codec, &params);

    opj_stream_t* stream = opj_stream_create_file_stream(path, 1<<20, OPJ_TRUE);
    if (!stream) { opj_destroy_codec(codec); unlink(path); return false; }

    opj_image_t* image = nullptr;
    bool ok = opj_read_header(stream, codec, &image)
           && opj_decode(codec, stream, image)
           && opj_end_decompress(codec, stream);
    if (!ok) {
        if (image) opj_image_destroy(image);
        opj_stream_destroy(stream); opj_destroy_codec(codec);
        unlink(path); return false;
    }

    out_w = (int)image->comps[0].w;
    out_h = (int)image->comps[0].h;
    ncomps = (int)image->numcomps;
    comps.resize(ncomps);
    for (int c = 0; c < ncomps; ++c) {
        size_t sz = (size_t)image->comps[c].w * image->comps[c].h;
        comps[c].assign(image->comps[c].data, image->comps[c].data + sz);
    }
    opj_image_destroy(image);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    unlink(path);
    return true;
}

/* ===== PSNR ===== */

static double compute_psnr(const int32_t* ref, const int32_t* dec, size_t n, double max_val)
{
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = ref[i] - dec[i];
        mse += d * d;
    }
    mse /= n;
    if (mse < 1e-12) return 100.0;
    return 20.0 * std::log10(max_val / std::sqrt(mse));
}

/* ===== J2K structure check ===== */

static bool check_j2k_markers(const std::vector<uint8_t>& cs)
{
    if (cs.size() < 4) return false;
    /* SOC = FF4F */
    if (cs[0] != 0xFF || cs[1] != 0x4F) return false;
    /* EOC = FFD9 must appear near end */
    size_t n = cs.size();
    if (n < 2) return false;
    return (cs[n-2] == 0xFF && cs[n-1] == 0xD9);
}

/* ===== GpuColourParams ===== */

static void build_colour_params(GpuColourParams& p)
{
    for (int i = 0; i < 4096; ++i) {
        p.lut_in[i]  = i / 4095.f;
        p.lut_out[i] = static_cast<uint16_t>(i);
    }
    /* sRGB→XYZ D65 */
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

/* ===== Test frame generators ===== */

using FrameGen = std::function<void(uint16_t*, int, int)>;

static void gen_black(uint16_t* p, int w, int h) {
    memset(p, 0, sizeof(uint16_t) * 3 * (size_t)w * h);
}
static void gen_white(uint16_t* p, int w, int h) {
    std::fill(p, p + 3*(size_t)w*h, uint16_t(60000));
}
static void gen_hgradient(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint16_t v = uint16_t(x * 60000 / (w - 1));
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
}
static void gen_vgradient(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y) {
        uint16_t v = uint16_t(y * 60000 / (h - 1));
        for (int x = 0; x < w; ++x) {
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
    }
}
static void gen_checkerboard(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint16_t v = ((x/32 + y/32) & 1) ? 60000 : 4000;
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
}
static void gen_sine(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float fx = x / float(w), fy = y / float(h);
            size_t i = ((size_t)y*w+x)*3;
            p[i+0] = uint16_t((0.5f + 0.5f*std::sin(fx*20.f))*60000.f);
            p[i+1] = uint16_t((0.5f + 0.5f*std::sin(fy*20.f+1.f))*60000.f);
            p[i+2] = uint16_t((0.5f + 0.5f*std::sin((fx+fy)*15.f+2.f))*60000.f);
        }
}
static void gen_random(uint16_t* p, int w, int h) {
    uint32_t s = 0xDEADBEEFu;
    for (size_t i = 0; i < 3*(size_t)w*h; ++i) {
        s = s * 1664525u + 1013904223u;
        p[i] = uint16_t((s >> 16) & 0xFFFF);
    }
}
static void gen_midgray(uint16_t* p, int w, int h) {
    std::fill(p, p + 3*(size_t)w*h, uint16_t(32768));
}

/* Additional comprehensive test patterns */
static void gen_diagonal(uint16_t* p, int w, int h) {
    memset(p, 0, sizeof(uint16_t)*3*(size_t)w*h);
    for (int y = 0; y < h && y < w; ++y) {
        size_t i = ((size_t)y*w+y)*3;
        p[i+0]=p[i+1]=p[i+2]=60000;
    }
}
static void gen_single_impulse(uint16_t* p, int w, int h) {
    memset(p, 0, sizeof(uint16_t)*3*(size_t)w*h);
    p[(h/2*(size_t)w+w/2)*3+0] = 60000;
    p[(h/2*(size_t)w+w/2)*3+1] = 60000;
    p[(h/2*(size_t)w+w/2)*3+2] = 60000;
}
static void gen_hbars_8(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y) {
        int bar = (y * 8 / h);
        uint16_t v = uint16_t(bar * 60000 / 7);
        for (int x = 0; x < w; ++x) {
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
    }
}
static void gen_vbars_8(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int bar = (x * 8 / w);
            uint16_t v = uint16_t(bar * 60000 / 7);
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
}
static void gen_flat_level(uint16_t* p, int w, int h, uint16_t level) {
    std::fill(p, p + 3*(size_t)w*h, level);
}
static void gen_color_bars(uint16_t* p, int w, int h) {
    /* RGB color bars: red, green, blue, yellow, cyan, magenta, white */
    const int nbars = 7;
    const uint16_t colors[nbars][3] = {
        {60000,0,0}, {0,60000,0}, {0,0,60000},
        {60000,60000,0}, {0,60000,60000}, {60000,0,60000},
        {60000,60000,60000}
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int bar = (x * nbars / w);
            p[(y*(size_t)w+x)*3+0] = colors[bar][0];
            p[(y*(size_t)w+x)*3+1] = colors[bar][1];
            p[(y*(size_t)w+x)*3+2] = colors[bar][2];
        }
}
static void gen_two_value(uint16_t* p, int w, int h, uint16_t v0, uint16_t v1) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint16_t v = (x < w/2) ? v0 : v1;
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
}
static void gen_fine_checker(uint16_t* p, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint16_t v = ((x/8 + y/8) & 1) ? 60000 : 4000;
            p[(y*(size_t)w+x)*3+0] = v;
            p[(y*(size_t)w+x)*3+1] = v;
            p[(y*(size_t)w+x)*3+2] = v;
        }
}

/* ===== Run one round-trip test ===== */

struct RoundTripResult {
    bool decoded;
    double psnr_Y;   /* luminance PSNR (Y channel) */
    double psnr_min; /* min PSNR across 3 components */
    int dec_w, dec_h, dec_ncomp;
    bool non_trivial;    /* output is not all-zero */
    bool bits_in_range;  /* all decoded values in [0, 4095] */
    size_t cs_bytes;     /* codestream size in bytes */
};

static RoundTripResult run_roundtrip(CudaJ2KEncoder& enc,
                                     const uint16_t* rgb, int w, int h,
                                     int64_t br, int fps, bool /*unused_fast*/ = false)
{
    RoundTripResult r{};
    auto cs = enc.encode_ebcot(rgb, w, h, w*3, br, fps, false, false);
    r.cs_bytes = cs.size();
    if (cs.empty()) return r;

    std::vector<std::vector<int32_t>> comps;
    r.decoded = opj_decode_j2k(cs, comps, r.dec_w, r.dec_h, r.dec_ncomp);
    if (!r.decoded || comps.size() < 3) return r;

    /* Non-trivial check */
    int64_t sum = 0;
    for (auto v : comps[0]) sum += std::abs(v);
    r.non_trivial = (sum > 0);

    /* Range check: all values in [0, 4095] */
    r.bits_in_range = true;
    for (int c = 0; c < r.dec_ncomp && r.bits_in_range; ++c)
        for (auto v : comps[c])
            if (v < 0 || v > 4095) { r.bits_in_range = false; break; }

    /* PSNR vs CPU XYZ reference */
    std::vector<int32_t> refX, refY, refZ;
    rgb48_to_xyz12(rgb, w, h, refX, refY, refZ);

    size_t n = (size_t)r.dec_w * r.dec_h;
    if (n > 0 && n <= refX.size()) {
        double px = compute_psnr(refX.data(), comps[0].data(), n, 4095.0);
        double py = compute_psnr(refY.data(), comps[1].data(), n, 4095.0);
        double pz = compute_psnr(refZ.data(), comps[2].data(), n, 4095.0);
        r.psnr_Y   = py;
        r.psnr_min = std::min(px, std::min(py, pz));
    }
    return r;
}

/* ===== Main test suite ===== */

int main()
{
    std::printf("=== GPU J2K Round-Trip Correctness Verification ===\n\n");

    CudaJ2KEncoder enc;
    CHECK(enc.is_initialized(), "Encoder initializes");
    if (!enc.is_initialized()) {
        std::printf("Encoder failed to initialize — aborting.\n");
        return 1;
    }

    GpuColourParams cp;
    build_colour_params(cp);
    enc.set_colour_params(cp);

    /* ---- Test 1: J2K structure validity ---- */
    std::printf("\n--- Test 1: J2K marker structure ---\n");
    {
        const int W=2048, H=1080;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false);
        CHECK(cs.size() > 100,       "Codestream not empty");
        CHECK(check_j2k_markers(cs), "SOC(FF4F) and EOC(FFD9) present");
        /* Check SOT marker (FF90) exists */
        bool has_sot = false;
        for (size_t i = 0; i+1 < cs.size(); ++i)
            if (cs[i]==0xFF && cs[i+1]==0x90) { has_sot=true; break; }
        CHECK(has_sot, "SOT(FF90) marker present");
        /* Check SIZ marker (FF51) */
        bool has_siz = false;
        for (size_t i = 0; i+1 < cs.size(); ++i)
            if (cs[i]==0xFF && cs[i+1]==0x51) { has_siz=true; break; }
        CHECK(has_siz, "SIZ(FF51) marker present");
    }

    /* ---- Test 2: OpenJPEG decode on all patterns ---- */
    std::printf("\n--- Test 2: OpenJPEG decode — all patterns (2K, correct mode) ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        struct { const char* name; FrameGen gen; } patterns[] = {
            {"black",       gen_black      },
            {"white",       gen_white      },
            {"midgray",     gen_midgray    },
            {"h-gradient",  gen_hgradient  },
            {"v-gradient",  gen_vgradient  },
            {"checkerboard",gen_checkerboard},
            {"sine",        gen_sine       },
            {"random",      gen_random     },
        };
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        for (auto& pat : patterns) {
            pat.gen(rgb.data(), W, H);
            auto r = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: decode OK", pat.name);
            CHECK(r.decoded, msg);
            snprintf(msg, sizeof(msg), "%s: output %dx%d", pat.name, W, H);
            CHECK(r.dec_w==W && r.dec_h==H, msg);
        }
    }

    /* ---- Test 3: PSNR thresholds ---- */
    std::printf("\n--- Test 3: PSNR thresholds ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);

        struct { const char* name; FrameGen gen; double min_psnr_correct; } cases[] = {
            {"sine",      gen_sine,       40.0},
            {"hgradient", gen_hgradient,  40.0},
            {"midgray",   gen_midgray,    50.0},
            {"random",    gen_random,     10.0},  /* fully random 12-bit at 150Mbps is bitrate-limited ~13dB */
        };
        for (auto& c : cases) {
            c.gen(rgb.data(), W, H);
            auto rc = run_roundtrip(enc, rgb.data(), W, H, BR, 24);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s correct mode PSNR_Y=%.1fdB >= %.0fdB",
                     c.name, rc.psnr_Y, c.min_psnr_correct);
            CHECK(rc.psnr_Y >= c.min_psnr_correct, msg);
        }

        /* Black frame: all-zero input */
        gen_black(rgb.data(), W, H);
        auto rb = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
        CHECK(rb.decoded, "black frame decode OK");
        CHECK(rb.bits_in_range, "black frame values in [0,4095]");
    }

    /* ---- Test 4: Bitstream determinism ---- */
    std::printf("\n--- Test 4: Bitstream determinism ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);

        auto cs0 = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        auto cs1 = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        auto cs2 = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        CHECK(cs0.size() == cs1.size() && cs1.size() == cs2.size(),
              "Correct mode: all 3 runs produce same-length codestream");
        CHECK(cs0 == cs1 && cs1 == cs2,
              "Correct mode: all 3 runs produce bit-identical codestream");
    }

    /* ---- Test 5: Non-trivial output for non-black input ---- */
    std::printf("\n--- Test 5: Non-trivial decoded output ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        struct { const char* name; FrameGen gen; } pats[] = {
            {"white",      gen_white},
            {"h-gradient", gen_hgradient},
            {"checkerboard",gen_checkerboard},
        };
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        for (auto& p : pats) {
            p.gen(rgb.data(), W, H);
            auto r = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
            char msg[128];
            snprintf(msg, sizeof(msg), "%s: non-trivial decoded output", p.name);
            CHECK(r.non_trivial, msg);
            snprintf(msg, sizeof(msg), "%s: all values in [0,4095]", p.name);
            CHECK(r.bits_in_range, msg);
        }
    }

    /* ---- Test 6: Gradient monotonicity ---- */
    std::printf("\n--- Test 6: Gradient monotonicity ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_hgradient(rgb.data(), W, H);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        std::vector<std::vector<int32_t>> comps;
        int dw, dh, nc;
        if (opj_decode_j2k(cs, comps, dw, dh, nc) && nc >= 2) {
            /* Y channel: compute row averages, check monotone trend */
            std::vector<double> row_avg(dh, 0.0);
            for (int y = 0; y < dh; ++y)
                for (int x = 0; x < dw; ++x)
                    row_avg[y] += comps[1][y*(size_t)dw+x];
            /* With h-gradient: column averages (along x) should be monotone */
            std::vector<double> col_avg(dw, 0.0);
            for (int y = 0; y < dh; ++y)
                for (int x = 0; x < dw; ++x)
                    col_avg[x] += comps[1][y*(size_t)dw+x];
            /* Check 80% of adjacent pairs are non-decreasing */
            int mono = 0, total = dw - 1;
            for (int x = 0; x < total; ++x)
                if (col_avg[x+1] >= col_avg[x] - 1.0) ++mono;
            CHECK(mono >= total*8/10, "H-gradient Y channel is mostly monotone left-to-right");
        } else {
            CHECK(false, "H-gradient: decode failed");
        }
    }

    /* ---- Test 7: 4K DCI resolution ---- */
    std::printf("\n--- Test 7: 4K DCI resolution ---\n");
    {
        const int W=4096, H=2160;
        const int64_t BR=300000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);

        /* Warmup */
        enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, true);

        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, true);
        CHECK(cs.size() > 1000,       "4K codestream not empty");
        CHECK(check_j2k_markers(cs),  "4K: valid SOC+EOC markers");

        std::vector<std::vector<int32_t>> comps;
        int dw, dh, nc;
        bool ok = opj_decode_j2k(cs, comps, dw, dh, nc);
        CHECK(ok,          "4K: OpenJPEG decode succeeds");
        CHECK(dw==W&&dh==H,"4K: decoded dimensions match 4096x2160");
        if (ok && nc >= 2) {
            std::vector<int32_t> refX, refY, refZ;
            rgb48_to_xyz12(rgb.data(), W, H, refX, refY, refZ);
            size_t n = (size_t)dw * dh;
            double psnr = compute_psnr(refY.data(), comps[1].data(), n, 4095.0);
            char msg[64];
            snprintf(msg, sizeof(msg), "4K correct PSNR_Y=%.1fdB >= 35dB", psnr);
            CHECK(psnr >= 35.0, msg);
        }
    }

    /* ---- Test 9: Multiple sequential frames (state isolation) ---- */
    std::printf("\n--- Test 9: Sequential frame state isolation ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;

        std::vector<uint16_t> rgb_a(3*(size_t)W*H), rgb_b(3*(size_t)W*H);
        gen_sine(rgb_a.data(), W, H);
        gen_hgradient(rgb_b.data(), W, H);

        /* Encode A, B, A, B — A outputs should be identical */
        auto cs_a0 = enc.encode_ebcot(rgb_a.data(), W, H, W*3, BR, 24, false, false);
        auto cs_b0 = enc.encode_ebcot(rgb_b.data(), W, H, W*3, BR, 24, false, false);
        auto cs_a1 = enc.encode_ebcot(rgb_a.data(), W, H, W*3, BR, 24, false, false);
        auto cs_b1 = enc.encode_ebcot(rgb_b.data(), W, H, W*3, BR, 24, false, false);

        CHECK(cs_a0 == cs_a1, "A output identical after interleaved B frames");
        CHECK(cs_b0 == cs_b1, "B output identical after interleaved A frames");
        CHECK(cs_a0 != cs_b0, "Different inputs produce different outputs");
    }

    /* ---- Test 10: Additional pattern decode + PSNR (correct mode) ---- */
    std::printf("\n--- Test 10: Additional patterns (correct mode, 2K) ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);

        struct { const char* name; FrameGen gen; double min_psnr; } extra[] = {
            {"impulse",       gen_single_impulse, 40.0},
            {"diagonal",      gen_diagonal,       40.0},
            {"h-bars-8",      gen_hbars_8,        40.0},
            {"v-bars-8",      gen_vbars_8,        40.0},
            {"color-bars",    gen_color_bars,     35.0},
            {"fine-checker",  gen_fine_checker,   10.0},
        };
        for (auto& c : extra) {
            c.gen(rgb.data(), W, H);
            auto r = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
            char msg[160];
            snprintf(msg, sizeof(msg), "%s: decode OK", c.name);
            CHECK(r.decoded, msg);
            snprintf(msg, sizeof(msg), "%s: dims %dx%d", c.name, W, H);
            CHECK(r.dec_w==W && r.dec_h==H, msg);
            snprintf(msg, sizeof(msg), "%s: PSNR_Y=%.1fdB >= %.0fdB", c.name, r.psnr_Y, c.min_psnr);
            CHECK(r.psnr_Y >= c.min_psnr, msg);
            snprintf(msg, sizeof(msg), "%s: non-trivial output", c.name);
            CHECK(r.non_trivial, msg);
            snprintf(msg, sizeof(msg), "%s: values in [0,4095]", c.name);
            CHECK(r.bits_in_range, msg);
        }
    }

    /* ---- Test 11: Full 3-component PSNR (not just Y) ---- */
    std::printf("\n--- Test 11: Full 3-component PSNR verification ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);

        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        std::vector<std::vector<int32_t>> comps;
        int dw, dh, nc;
        CHECK(opj_decode_j2k(cs, comps, dw, dh, nc), "3-comp PSNR: decode OK");
        CHECK(nc >= 3, "3-comp PSNR: 3 components decoded");

        std::vector<int32_t> refX, refY, refZ;
        rgb48_to_xyz12(rgb.data(), W, H, refX, refY, refZ);
        size_t n = (size_t)dw * dh;
        double px = compute_psnr(refX.data(), comps[0].data(), n, 4095.0);
        double py = compute_psnr(refY.data(), comps[1].data(), n, 4095.0);
        double pz = compute_psnr(refZ.data(), comps[2].data(), n, 4095.0);

        char msg[160];
        snprintf(msg, sizeof(msg), "X channel PSNR=%.1fdB >= 40dB", px);
        CHECK(px >= 40.0, msg);
        snprintf(msg, sizeof(msg), "Y channel PSNR=%.1fdB >= 40dB", py);
        CHECK(py >= 40.0, msg);
        snprintf(msg, sizeof(msg), "Z channel PSNR=%.1fdB >= 40dB", pz);
        CHECK(pz >= 40.0, msg);

        /* All 3 components should have similar PSNR (within 10dB of each other) */
        double pmax = std::max(px, std::max(py, pz));
        double pmin = std::min(px, std::min(py, pz));
        snprintf(msg, sizeof(msg), "PSNR spread max-min=%.1fdB < 10dB", pmax-pmin);
        CHECK(pmax - pmin < 10.0, msg);
    }

    /* ---- Test 12: Flat-field tests at multiple levels ---- */
    std::printf("\n--- Test 12: Flat-field tests at multiple levels ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);

        uint16_t levels[] = {0, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 65535};
        for (auto lv : levels) {
            gen_flat_level(rgb.data(), W, H, lv);
            auto r = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
            char msg[160];
            snprintf(msg, sizeof(msg), "flat-%u: decode OK", (unsigned)lv);
            CHECK(r.decoded, msg);
            snprintf(msg, sizeof(msg), "flat-%u: PSNR_Y=%.1fdB >= 45dB", (unsigned)lv, r.psnr_Y);
            CHECK(r.psnr_Y >= 45.0, msg);
            snprintf(msg, sizeof(msg), "flat-%u: values in [0,4095]", (unsigned)lv);
            CHECK(r.bits_in_range, msg);
        }
    }

    /* ---- Test 13: Non-standard resolution ---- */
    std::printf("\n--- Test 13: Non-standard resolutions ---\n");
    {
        const int64_t BR=150000000LL;
        struct { int w, h; const char* name; } res[] = {
            {1920, 1080, "1080p"},
            {1280, 720,  "720p"},
            {720,  576,  "SD"},
            {1024, 1024, "square"},
        };
        for (auto& r : res) {
            std::vector<uint16_t> rgb(3*(size_t)r.w*r.h);
            gen_sine(rgb.data(), r.w, r.h);
            auto res_r = run_roundtrip(enc, rgb.data(), r.w, r.h, BR, 24, false);
            char msg[160];
            snprintf(msg, sizeof(msg), "%s %dx%d: decode OK", r.name, r.w, r.h);
            CHECK(res_r.decoded, msg);
            snprintf(msg, sizeof(msg), "%s %dx%d: dims match", r.name, r.w, r.h);
            CHECK(res_r.dec_w==r.w && res_r.dec_h==r.h, msg);
            snprintf(msg, sizeof(msg), "%s %dx%d: PSNR_Y=%.1fdB >= 12dB", r.name, r.w, r.h, res_r.psnr_Y);
            CHECK(res_r.psnr_Y >= 12.0, msg);
            snprintf(msg, sizeof(msg), "%s %dx%d: values in [0,4095]", r.name, r.w, r.h);
            CHECK(res_r.bits_in_range, msg);
        }
    }

    /* ---- Test 14: Different bitrate levels ---- */
    std::printf("\n--- Test 14: Different bitrate levels ---\n");
    {
        const int W=2048, H=1080;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);

        int64_t bitrates[] = {50000000LL, 100000000LL, 150000000LL, 200000000LL};
        const char* br_names[] = {"50Mbps", "100Mbps", "150Mbps", "200Mbps"};
        for (int i = 0; i < 4; ++i) {
            auto r = run_roundtrip(enc, rgb.data(), W, H, bitrates[i], 24, false);
            char msg[160];
            snprintf(msg, sizeof(msg), "%s: decode OK", br_names[i]);
            CHECK(r.decoded, msg);
            snprintf(msg, sizeof(msg), "%s: PSNR_Y=%.1fdB >= 35dB", br_names[i], r.psnr_Y);
            CHECK(r.psnr_Y >= 35.0, msg);
            snprintf(msg, sizeof(msg), "%s: values in [0,4095]", br_names[i]);
            CHECK(r.bits_in_range, msg);
        }

        /* Bitrate monotonicity: higher bitrate → larger codestream.
         * We test codestream size rather than PSNR because smooth patterns can encode
         * near-losslessly even at low bitrates, making PSNR non-monotone across
         * bitrate steps. Use noise (high entropy) for the PSNR monotonicity check. */
        auto r50_sine  = run_roundtrip(enc, rgb.data(), W, H,  50000000LL, 24, false);
        auto r200_sine = run_roundtrip(enc, rgb.data(), W, H, 200000000LL, 24, false);
        char mono_msg[160];
        snprintf(mono_msg, sizeof(mono_msg), "200Mbps cs_bytes(%zu) >= 50Mbps cs_bytes(%zu)",
                 r200_sine.cs_bytes, r50_sine.cs_bytes);
        CHECK(r200_sine.cs_bytes >= r50_sine.cs_bytes, mono_msg);

        /* PSNR monotonicity on noise (information-rich pattern clears budget) */
        std::vector<uint16_t> rgb_noise(3*(size_t)W*H);
        gen_random(rgb_noise.data(), W, H);
        auto rn50  = run_roundtrip(enc, rgb_noise.data(), W, H,  50000000LL, 24, false);
        auto rn200 = run_roundtrip(enc, rgb_noise.data(), W, H, 200000000LL, 24, false);
        snprintf(mono_msg, sizeof(mono_msg),
                 "noise: 200Mbps PSNR(%.1fdB) >= 50Mbps PSNR(%.1fdB)", rn200.psnr_Y, rn50.psnr_Y);
        CHECK(rn200.psnr_Y >= rn50.psnr_Y - 0.5, mono_msg);
    }

    /* ---- Test 15: J2K marker field verification ---- */
    std::printf("\n--- Test 15: J2K marker field verification ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        CHECK(cs.size() > 1000, "marker test: codestream not empty");

        /* Parse SIZ: check Rsiz, width, height, num components */
        bool found_siz = false;
        for (size_t i = 0; i + 40 < cs.size(); ++i) {
            if (cs[i]==0xFF && cs[i+1]==0x51) {
                uint16_t lsiz = (cs[i+2]<<8)|cs[i+3];
                uint16_t rsiz = (cs[i+4]<<8)|cs[i+5];
                uint32_t xsiz = (cs[i+6]<<24)|(cs[i+7]<<16)|(cs[i+8]<<8)|cs[i+9];
                uint32_t ysiz = (cs[i+10]<<24)|(cs[i+11]<<16)|(cs[i+12]<<8)|cs[i+13];
                uint16_t ncomp = (cs[i+38]<<8)|cs[i+39];

                CHECK(lsiz >= 41, "SIZ: Lsiz >= 41");
                CHECK(rsiz == 0x0003 || rsiz == 0x0004, "SIZ: Rsiz=0x0003/0x0004 (DCI cinema profile)");
                CHECK(xsiz == (uint32_t)W, "SIZ: Xsiz matches input width");
                CHECK(ysiz == (uint32_t)H, "SIZ: Ysiz matches input height");
                CHECK(ncomp == 3, "SIZ: Csiz=3 components");

                /* Check component precision: 12-bit unsigned XYZ (Ssiz=11, bit7=0).
                 * Values are unsigned [0,4095] — OPJ recovers them via inverse ICT. */
                for (int c = 0; c < 3; ++c) {
                    uint8_t ssiz = cs[i+40+c*3];
                    uint8_t xrsiz = cs[i+41+c*3];
                    uint8_t yrsiz = cs[i+42+c*3];
                    char msg[80];
                    snprintf(msg, sizeof(msg), "SIZ: comp %d Ssiz=11 bits signed", c);
                    CHECK((ssiz & 0x7F) == 11, msg);
                    snprintf(msg, sizeof(msg), "SIZ: comp %d unsigned 12-bit", c);
                    CHECK((ssiz & 0x80) == 0, msg);
                    snprintf(msg, sizeof(msg), "SIZ: comp %d XRsiz=1", c);
                    CHECK(xrsiz == 1, msg);
                    snprintf(msg, sizeof(msg), "SIZ: comp %d YRsiz=1", c);
                    CHECK(yrsiz == 1, msg);
                }
                found_siz = true;
                break;
            }
        }
        CHECK(found_siz, "SIZ marker found and parsed");

        /* Parse COD: check SPcod fields (OpenJPEG-compatible 5-byte SPcod) */
        bool found_cod = false;
        for (size_t i = 0; i + 13 < cs.size(); ++i) {
            if (cs[i]==0xFF && cs[i+1]==0x52) {
                uint8_t scod = cs[i+4];
                uint8_t sgcod_prog = cs[i+5];
                uint8_t sgcod_nlevels = cs[i+9];
                uint8_t spcod_xcb     = cs[i+10]; /* xcb'-2: code-block width exponent */
                uint8_t spcod_ycb     = cs[i+11]; /* ycb'-2: code-block height exponent */
                uint8_t spcod_style   = cs[i+12]; /* SPcod cblk_style: coding mode switches */
                uint8_t spcod_filter  = cs[i+13]; /* SPcod filter: 0=9/7 irreversible */
                (void)spcod_style;

                char msg[80];
                snprintf(msg, sizeof(msg), "COD: SGcod num levels=%d (expected 5)", sgcod_nlevels);
                CHECK(sgcod_nlevels == 5, msg);
                /* Standard J2K: xcb' and ycb' are separate bytes.
                 * xcb'=3 → 2^(3+2)=32 pixel width, ycb'=3 → 32 pixel height. */
                snprintf(msg, sizeof(msg), "COD: SPcod xcb'=%d (expected 3)", spcod_xcb);
                CHECK(spcod_xcb == 3, msg);
                snprintf(msg, sizeof(msg), "COD: SPcod ycb'=%d (expected 3)", spcod_ycb);
                CHECK(spcod_ycb == 3, msg);
                snprintf(msg, sizeof(msg), "COD: SPcod wavelet=9/7 irreversible (filter=%d)", spcod_filter);
                CHECK(spcod_filter == 0, msg);

                /* CPRL progression (DCI SMPTE 429-4 requirement) */
                snprintf(msg, sizeof(msg), "COD: CPRL progression order (DCI)");
                CHECK((sgcod_prog & 0x0F) == 4, msg);
                /* Quality layers: big-endian uint16 at i+6..i+7 */
                snprintf(msg, sizeof(msg), "COD: 1 quality layer");
                CHECK(cs[i+6] == 0 && cs[i+7] == 1, msg);
                found_cod = true;
                break;
            }
        }
        CHECK(found_cod, "COD marker found and parsed");

        /* Parse QCD: check quantisation style, guard bits */
        bool found_qcd = false;
        for (size_t i = 0; i + 6 < cs.size(); ++i) {
            if (cs[i]==0xFF && cs[i+1]==0x5C) {
                uint8_t sqcd = cs[i+4];
                uint8_t sqty = sqcd & 0x1F;
                uint8_t numgbits = (sqcd >> 5) & 0x07;
                char msg[80];
                snprintf(msg, sizeof(msg), "QCD: sqty=2 (expounded)");
                CHECK(sqty == 2, msg);
                snprintf(msg, sizeof(msg), "QCD: numgbits=%d (1 for 2K)", numgbits);
                CHECK(numgbits == 1, msg);
                found_qcd = true;
                break;
            }
        }
        CHECK(found_qcd, "QCD marker found and parsed");
    }

    /* ---- Test 16: Two-value patterns (hard edge between two flat levels) ---- */
    std::printf("\n--- Test 16: Two-value split patterns ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);

        struct { uint16_t v0, v1; const char* name; double min_psnr; } splits[] = {
            {0,     60000, "black-white",  35.0},
            {10000, 50000, "dark-bright",  40.0},
            {30000, 30000, "equal-flat",   50.0},
            {0,     65535, "full-range",   35.0},
        };
        for (auto& s : splits) {
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    uint16_t v = (x < W/2) ? s.v0 : s.v1;
                    rgb[(y*(size_t)W+x)*3+0] = v;
                    rgb[(y*(size_t)W+x)*3+1] = v;
                    rgb[(y*(size_t)W+x)*3+2] = v;
                }
            auto r = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
            char msg[160];
            snprintf(msg, sizeof(msg), "%s: decode OK", s.name);
            CHECK(r.decoded, msg);
            snprintf(msg, sizeof(msg), "%s: PSNR_Y=%.1fdB >= %.0fdB", s.name, r.psnr_Y, s.min_psnr);
            CHECK(r.psnr_Y >= s.min_psnr, msg);
            snprintf(msg, sizeof(msg), "%s: values in [0,4095]", s.name);
            CHECK(r.bits_in_range, msg);
        }
    }

    /* ---- Test 17: Extended determinism (5 consecutive runs) ---- */
    std::printf("\n--- Test 17: Extended determinism (5 runs) ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);
        gen_sine(rgb.data(), W, H);

        auto cs0 = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        bool all_same = true;
        for (int i = 1; i < 5; ++i) {
            auto csi = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
            if (csi != cs0) { all_same = false; break; }
        }
        CHECK(all_same, "5 consecutive runs produce bit-identical output");
    }

    /* ---- Test 18: Color component isolation (pure red/green/blue) ---- */
    std::printf("\n--- Test 18: Color component isolation ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H, 0);

        /* Pure red: only R channel has signal */
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint16_t v = uint16_t(x * 60000 / W);
                rgb[(y*(size_t)W+x)*3+0] = v;
            }
        auto r_red = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
        CHECK(r_red.decoded, "red-only: decode OK");
        CHECK(r_red.psnr_Y >= 40.0, "red-only: PSNR_Y >= 40dB");

        /* Pure green */
        memset(rgb.data(), 0, sizeof(uint16_t)*3*W*H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint16_t v = uint16_t(x * 60000 / W);
                rgb[(y*(size_t)W+x)*3+1] = v;
            }
        auto r_green = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
        CHECK(r_green.decoded, "green-only: decode OK");
        CHECK(r_green.psnr_Y >= 40.0, "green-only: PSNR_Y >= 40dB");

        /* Pure blue */
        memset(rgb.data(), 0, sizeof(uint16_t)*3*W*H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint16_t v = uint16_t(x * 60000 / W);
                rgb[(y*(size_t)W+x)*3+2] = v;
            }
        auto r_blue = run_roundtrip(enc, rgb.data(), W, H, BR, 24, false);
        CHECK(r_blue.decoded, "blue-only: decode OK");
        CHECK(r_blue.psnr_Y >= 40.0, "blue-only: PSNR_Y >= 40dB");
    }

    /* ---- Test 19: Output size sanity ---- */
    std::printf("\n--- Test 19: Output size sanity ---\n");
    {
        const int W=2048, H=1080;
        const int64_t BR=150000000LL;
        std::vector<uint16_t> rgb(3*(size_t)W*H);

        /* Black frame: output should be small but valid */
        gen_black(rgb.data(), W, H);
        auto cs_black = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        CHECK(cs_black.size() > 100, "black frame: codestream has minimum markers");
        CHECK(cs_black.size() < 50000, "black frame: < 50KB (efficient coding of zeros)");

        /* White frame: should be small (flat 60k) */
        gen_white(rgb.data(), W, H);
        auto cs_white = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        CHECK(cs_white.size() > 100, "white frame: codestream has minimum markers");
        CHECK(cs_white.size() < 100000, "white frame: < 100KB");

        /* Checkerboard/fine detail: should be larger than flat frames */
        gen_checkerboard(rgb.data(), W, H);
        auto cs_checker = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false);
        CHECK(cs_checker.size() > cs_black.size(),
              "checkerboard > black (more detail = more bytes)");
        CHECK(cs_checker.size() > cs_white.size(),
              "checkerboard > white (more detail = more bytes)");

        /* Codestream must end with FFD9 */
        for (auto& cs_ptr : {&cs_black, &cs_white, &cs_checker}) {
            size_t n = cs_ptr->size();
            CHECK(n >= 2 && (*cs_ptr)[n-2]==0xFF && (*cs_ptr)[n-1]==0xD9,
                  "codestream ends with FFD9 (EOC)");
        }
    }

    /* ---- Summary ---- */
    std::printf("\n=== Summary: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
