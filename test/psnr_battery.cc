/*
 * PSNR test battery.
 *
 * Encodes a fixed 2K image with the CUDA J2K encoder (encode_ebcot path),
 * decodes with OpenJPEG, and reports PSNR_Y for each of several controlled
 * test patterns. Used to characterise the remaining T1 correctness bug:
 * different patterns stress different code paths (flat → MSB CUP only;
 * impulse → single significant CB; vertical/horizontal bars → 1-D structure;
 * checkerboard → mixed signs; gradient → varying magnitudes).
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/psnr_battery \
 *        test/psnr_battery.cc src/lib/cuda_j2k_encoder.cu \
 *        -lcudart -lopenjp2
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>
#include "lib/cuda_j2k_encoder.h"

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

static bool opj_decode(const std::vector<uint8_t>& cs,
                       std::vector<std::vector<int>>& comps, int& W, int& H,
                       int cp_reduce = 0)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/psbat_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return false; }
    close(fd);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    params.cp_reduce = cp_reduce;
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ: %s\n",m); }, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (ok) {
        W = img->comps[0].w; H = img->comps[0].h;
        comps.resize(img->numcomps);
        for (int c = 0; c < (int)img->numcomps; c++)
            comps[c].assign(img->comps[c].data, img->comps[c].data + W*H);
    }
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st);
    opj_destroy_codec(codec);
    unlink(tmp);
    return ok;
}

/* Approximate sRGB → CIE XYZ Y component for an R=G=B input.  Matches
 * the LUT/matrix in build_params (identity LUT, ITU-R BT.709 matrix). */
static inline int rgb_to_xyz_y_12bit(int r16, int g16, int b16) {
    float rf = r16 / 65535.f, gf = g16 / 65535.f, bf = b16 / 65535.f;
    float y = 0.2126f*rf + 0.7152f*gf + 0.0722f*bf;
    int yv = static_cast<int>(y * 4095.f + 0.5f);
    if (yv < 0) yv = 0; if (yv > 4095) yv = 4095;
    return yv;
}

struct Pattern {
    const char* name;
    std::function<uint16_t(int x, int y)> rgb;
};

static int W = 2048, H = 1080;

static double psnr(const std::vector<int>& dec, std::function<int(int,int)> ref) {
    double mse = 0.0;
    size_t n = (size_t)W * H;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double d = dec[(size_t)y * W + x] - ref(x, y);
            mse += d * d;
        }
    mse /= (double)n;
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

static void run(CudaJ2KEncoder& enc, const char* name,
                std::function<uint16_t(int,int)> rgb_for_xy)
{
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = rgb_for_xy(x, y);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
        }
    /* warmup + measured */
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, false);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false, false);

    std::vector<std::vector<int>> comps; int dW, dH;
    if (!opj_decode(cs, comps, dW, dH, 0)) {
        printf("  %-26s  DECODE FAIL  (cs=%zu B)\n", name, cs.size());
        return;
    }
    auto ref_y = [&](int x, int y){
        uint16_t v = rgb_for_xy(x, y);
        return rgb_to_xyz_y_12bit(v, v, v);
    };
    double p = psnr(comps[1], ref_y);
    printf("  %-26s  cs=%-7zu B  PSNR_Y = %5.1f dB\n", name, cs.size(), p);
}

int main()
{
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    printf("PSNR battery (W=%d H=%d, BR=150Mbps)\n", W, H);

    run(enc, "flat_30000",          [](int,int){ return 30000; });
    run(enc, "flat_50000",          [](int,int){ return 50000; });
    run(enc, "flat_5000",           [](int,int){ return 5000; });
    run(enc, "h_gradient_full",     [](int x,int){ return uint16_t(x * 60000ll / (W-1)); });
    run(enc, "v_gradient_full",     [](int,int y){ return uint16_t(y * 60000ll / (H-1)); });
    run(enc, "h_bars_8",            [](int x,int){ return uint16_t(((x/256) % 2) ? 50000 : 10000); });
    run(enc, "v_bars_8",            [](int,int y){ return uint16_t(((y/135) % 2) ? 50000 : 10000); });
    run(enc, "checker_64",          [](int x,int y){ return uint16_t((((x/64)+(y/64))&1) ? 50000 : 10000); });
    run(enc, "single_impulse",      [](int x,int y){ return (x==W/2 && y==H/2) ? 60000 : 30000; });
    run(enc, "noise_small",         [](int x,int y){
        unsigned s = (unsigned)(x * 17u + y * 31u);
        s = s * 1664525u + 1013904223u;
        return uint16_t(30000 + ((s >> 17) & 0x3FF) - 512);
    });
    run(enc, "ramp_small_range",    [](int x,int){ return uint16_t(20000 + x * 10000ll / (W-1)); });
    run(enc, "two_value_split",     [](int x,int){ return uint16_t(x < W/2 ? 20000 : 40000); });
    return 0;
}
