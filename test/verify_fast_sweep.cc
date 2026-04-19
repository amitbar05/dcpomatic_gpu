/*
    Sweep many frame styles through encode_ebcot(fast_mode=true) and verify
    each output passes dcp::verify_j2k AND decodes cleanly in OpenJPEG.
    This catches content-dependent codestream bugs the sine-gradient test
    would miss.
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "lib/cuda_j2k_encoder.h"

#include <dcp/array_data.h>
#include <dcp/verify.h>
#include <dcp/verify_j2k.h>
#include <openjpeg-2.5/openjpeg.h>

struct MemBuf { const uint8_t* data; size_t size; size_t pos; };
static OPJ_SIZE_T mb_read(void* dst, OPJ_SIZE_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    if (mb->pos >= mb->size) return (OPJ_SIZE_T) -1;
    size_t avail = mb->size - mb->pos, take = n < avail ? n : avail;
    memcpy(dst, mb->data + mb->pos, take); mb->pos += take; return take;
}
static OPJ_OFF_T mb_skip(OPJ_OFF_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    OPJ_OFF_T a = OPJ_OFF_T(mb->size) - OPJ_OFF_T(mb->pos), t = n < a ? n : a;
    mb->pos += t; return t;
}
static OPJ_BOOL mb_seek(OPJ_OFF_T n, void* ud) {
    auto* mb = static_cast<MemBuf*>(ud);
    if (n > OPJ_OFF_T(mb->size)) return OPJ_FALSE;
    mb->pos = n; return OPJ_TRUE;
}
static void quiet_cb(const char*, void*) {}
static thread_local std::string* g_opj_msg = nullptr;
static void collect_err_cb(const char* m, void*) { if (g_opj_msg) *g_opj_msg += m; }

static bool
openjpeg_decodes(const std::vector<uint8_t>& cs, std::string* err = nullptr)
{
    if (err) { err->clear(); g_opj_msg = err; }
    MemBuf mb { cs.data(), cs.size(), 0 };
    opj_stream_t* str = opj_stream_default_create(OPJ_TRUE);
    opj_stream_set_read_function(str, mb_read);
    opj_stream_set_skip_function(str, mb_skip);
    opj_stream_set_seek_function(str, mb_seek);
    opj_stream_set_user_data(str, &mb, nullptr);
    opj_stream_set_user_data_length(str, cs.size());

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_info_handler(codec, quiet_cb, nullptr);
    opj_set_warning_handler(codec, err ? collect_err_cb : quiet_cb, nullptr);
    opj_set_error_handler(codec, err ? collect_err_cb : quiet_cb, nullptr);

    opj_dparameters_t dp; opj_set_default_decoder_parameters(&dp);
    bool ok = false;
    opj_image_t* img = nullptr;
    if (opj_setup_decoder(codec, &dp)
        && opj_read_header(str, codec, &img)
        && opj_decode(codec, str, img)
        && opj_end_decompress(codec, str)) {
        ok = img && img->numcomps == 3;
    }
    if (img) opj_image_destroy(img);
    opj_destroy_codec(codec);
    opj_stream_destroy(str);
    if (err) g_opj_msg = nullptr;
    return ok;
}

static void
build_identity_colour_params(GpuColourParams& p)
{
    for (int i = 0; i < 4096; ++i) {
        p.lut_in[i]  = i / 4095.0f;
        p.lut_out[i] = static_cast<uint16_t>(i);
    }
    p.matrix[0] = 0.4124f; p.matrix[1] = 0.3576f; p.matrix[2] = 0.1805f;
    p.matrix[3] = 0.2126f; p.matrix[4] = 0.7152f; p.matrix[5] = 0.0722f;
    p.matrix[6] = 0.0193f; p.matrix[7] = 0.1192f; p.matrix[8] = 0.9505f;
    p.valid = true;
}

using FrameFn = void(*)(std::vector<uint16_t>&, int, int);

static void f_black   (std::vector<uint16_t>& b, int, int)   { std::fill(b.begin(), b.end(), 0); }
static void f_white   (std::vector<uint16_t>& b, int, int)   { std::fill(b.begin(), b.end(), 65535); }
static void f_mid_gray(std::vector<uint16_t>& b, int, int)   { std::fill(b.begin(), b.end(), 32768); }
static void f_near_black(std::vector<uint16_t>& b, int, int) {
    for (size_t i = 0; i < b.size(); ++i) b[i] = (i % 11 == 0) ? 32 : 0;
}
static void f_one_pixel(std::vector<uint16_t>& b, int W, int H) {
    std::fill(b.begin(), b.end(), 0);
    size_t p = (size_t(H/2) * W + W/2) * 3;
    b[p] = b[p+1] = b[p+2] = 60000;
}
static void f_noise(std::vector<uint16_t>& b, int, int) {
    std::mt19937 rng(12345);
    for (auto& v : b) v = rng() & 0xFFFF;
}
static void f_stripes(std::vector<uint16_t>& b, int W, int H) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t p = (size_t(y) * W + x) * 3;
            uint16_t v = (x / 16) & 1 ? 60000 : 2000;
            b[p] = b[p+1] = b[p+2] = v;
        }
}
static void f_sine(std::vector<uint16_t>& b, int W, int H) {
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            size_t p = (size_t(y) * W + x) * 3;
            float fx = x / float(W), fy = y / float(H);
            b[p]   = uint16_t((0.5f + 0.5f*std::sin(fx*20.f)) * 60000.f);
            b[p+1] = uint16_t((0.5f + 0.5f*std::sin(fy*20.f + 1.f)) * 60000.f);
            b[p+2] = uint16_t((0.5f + 0.5f*std::sin((fx+fy)*15.f + 2.f)) * 60000.f);
        }
}
static void f_dark_sparse(std::vector<uint16_t>& b, int W, int H) {
    std::mt19937 rng(7);
    std::fill(b.begin(), b.end(), 0);
    for (int i = 0; i < 100; ++i) {
        int x = rng() % W, y = rng() % H;
        size_t p = (size_t(y) * W + x) * 3;
        b[p] = b[p+1] = b[p+2] = 50000;
    }
}

struct Case { const char* name; FrameFn fn; };

int main()
{
    const int W = 2048, H = 1080;
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { std::fprintf(stderr, "CUDA init failed\n"); return 2; }
    GpuColourParams p; build_identity_colour_params(p); enc.set_colour_params(p);

    Case cases[] = {
        {"black",        f_black},
        {"white",        f_white},
        {"mid_gray",     f_mid_gray},
        {"near_black",   f_near_black},
        {"one_pixel",    f_one_pixel},
        {"noise",        f_noise},
        {"stripes",      f_stripes},
        {"sine",         f_sine},
        {"dark_sparse",  f_dark_sparse},
    };

    std::vector<uint16_t> rgb(size_t(W) * H * 3);
    int fails = 0;
    std::printf("%-14s  %-7s  %10s  %-12s  %-11s\n",
                "case", "mode", "bytes", "dcp::verify", "opj decode");
    std::printf("-------------------------------------------------------------\n");

    for (auto& c : cases) {
        c.fn(rgb, W, H);
        for (int fast = 0; fast <= 1; ++fast) {
            /* Warmup */
            (void) enc.encode_ebcot(rgb.data(), W, H, W*3, 150'000'000, 24, false, false, (bool)fast);
            auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150'000'000, 24, false, false, (bool)fast);

            auto data = std::make_shared<dcp::ArrayData>(cs.data(), (int)cs.size());
            std::vector<dcp::VerificationNote> notes;
            dcp::verify_j2k(data, 0, 0, 24, notes);
            bool vok = notes.empty();
            std::string derr;
            bool dok = openjpeg_decodes(cs, &derr);

            std::printf("%-14s  %-7s  %10zu  %-12s  %-11s\n",
                        c.name, fast ? "fast" : "correct", cs.size(),
                        vok ? "OK" : "FAIL", dok ? "OK" : "FAIL");
            if (!vok) {
                for (auto const& n : notes) {
                    std::printf("    [code=%d", int(n.code()));
                    if (n.note()) std::printf(" note=\"%s\"", n.note()->c_str());
                    std::printf("]\n");
                }
                fails++;
            }
            if (!dok) { fails++; if (!derr.empty()) std::printf("    opj: %s", derr.c_str()); }
        }
    }

    std::printf("\n%s\n", fails == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return fails;
}
