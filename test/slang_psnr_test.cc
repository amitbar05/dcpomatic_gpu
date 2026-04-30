/*
 * Slang J2K Encoder PSNR Test — minimal correctness baseline.
 *
 * Encodes a small set of synthetic patterns through SlangJ2KEncoder,
 * decodes via OpenJPEG, prints PSNR. Mirrors the cuda_j2k psnr_battery
 * but limited to the patterns Slang can handle without DCI compliance.
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/slang_psnr_test \
 *        test/slang_psnr_test.cc \
 *        src/lib/slang_j2k_encoder_v17.cu \
 *        -lcudart -lopenjp2
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>

#include "lib/cuda_j2k_encoder.h"
#include "lib/slang_j2k_encoder.h"

static const int W = 2048, H = 1080;

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i / 4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

static bool opj_decode_cs(const std::vector<uint8_t>& cs,
                          std::vector<int>& y_out, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/slang_psnr_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) {
        close(fd); unlink(tmp); return false;
    }
    close(fd);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (ok) {
        W_out = img->comps[0].w; H_out = img->comps[0].h;
        if (img->numcomps >= 2)
            y_out.assign(img->comps[1].data, img->comps[1].data + W_out*H_out);
    }
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st);
    opj_destroy_codec(codec);
    unlink(tmp);
    return ok;
}

static double psnr(const std::vector<int>& dec, int dW, int dH,
                   const std::vector<int>& ref)
{
    if (dec.empty() || ref.empty() || dec.size() != ref.size()) return -1;
    double sse = 0;
    for (size_t i = 0; i < dec.size(); ++i) {
        double e = double(dec[i] - ref[i]);
        sse += e * e;
    }
    double mse = sse / dec.size();
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

int main() {
    SlangJ2KEncoder enc;
    if (!enc.is_initialized()) { std::fprintf(stderr, "Slang init failed\n"); return 1; }
    GpuColourParams cp; build_params(cp); enc.set_colour_params(cp);

    auto encode_pattern = [&](const char* name, auto pat_fn) {
        std::vector<uint16_t> rgb((size_t)W * H * 3);
        std::vector<int>      ref_y((size_t)W * H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint16_t v = pat_fn(x, y);
                rgb[((size_t)y*W + x)*3 + 0] = v;
                rgb[((size_t)y*W + x)*3 + 1] = v;
                rgb[((size_t)y*W + x)*3 + 2] = v;
                ref_y[(size_t)y*W + x] = std::min(int(v) >> 4, 4095);
            }

        /* prime + encode (Slang pipelines, returns previous frame) */
        enc.encode_from_rgb48(rgb.data(), W, H, W*3, 150000000LL, 24, false, false);
        auto cs = enc.encode_from_rgb48(rgb.data(), W, H, W*3, 150000000LL, 24, false, false);
        if (cs.empty()) cs = enc.flush();
        if (cs.empty()) {
            std::printf("  %-22s NO_OUTPUT\n", name);
            return;
        }
        std::vector<int> dec_y; int dW=0, dH=0;
        if (!opj_decode_cs(cs, dec_y, dW, dH)) {
            std::printf("  %-22s DECODE_FAIL  cs=%zuB\n", name, cs.size());
            return;
        }
        double p = psnr(dec_y, dW, dH, ref_y);
        std::printf("  %-22s cs=%-7zuB  PSNR_Y = %5.2f dB\n", name, cs.size(), p);
    };

    std::printf("Slang V17 PSNR battery (2K @ 150 Mbps):\n");
    encode_pattern("flat_30000",      [](int,int){return uint16_t(30000);});
    encode_pattern("flat_50000",      [](int,int){return uint16_t(50000);});
    encode_pattern("h_gradient",      [](int x,int){return uint16_t(uint64_t(x)*60000ull/(W-1));});
    encode_pattern("h_bars_8",        [](int x,int){return uint16_t(((x/256)%2)?50000:10000);});
    encode_pattern("v_bars_8",        [](int,int y){return uint16_t(((y/135)%2)?50000:10000);});
    encode_pattern("checker_64",      [](int x,int y){return uint16_t((((x/64)+(y/64))&1)?50000:10000);});
    encode_pattern("two_value_split", [](int x,int){return uint16_t(x<W/2?20000:40000);});
    encode_pattern("single_impulse",  [](int x,int y){return (x==W/2&&y==H/2)?uint16_t(50000):uint16_t(30000);});

    return 0;
}
