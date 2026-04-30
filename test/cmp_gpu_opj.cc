/*
 * Side-by-side PSNR comparison: GPU encoder vs OpenJPEG reference encoder.
 * Same XYZ inputs, same target byte budget, same decode pipeline.
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/cmp_gpu_opj \
 *        test/cmp_gpu_opj.cc src/lib/cuda_j2k_encoder.cu \
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

static const int W = 2048, H = 1080;
static const int64_t BR = 150000000LL;        /* 150 Mbps */
static const int64_t TARGET_BYTES = BR / 24 / 8;  /* per-frame byte budget */

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

/* Encoder-matched reference: 16-bit input -> 12-bit truncate, linear LUT,
 * matrix, saturate, output truncate.  Mirrors kernel_rgb48_xyz_hdwt0_1ch_2row.
 * Old version used (rf*4095+0.5) which differed by 1 unit on some inputs and
 * gave spurious 72 dB ceilings on flat patterns the encoder actually handles
 * losslessly. */
static inline int matrix_y_12bit(int r12, int g12, int b12, float c0, float c1, float c2) {
    float r = r12 / 4095.f, g = g12 / 4095.f, b = b12 / 4095.f;
    float v = c0*r + c1*g + c2*b;
    if (v < 0.f) v = 0.f; if (v > 1.f) v = 1.f;
    int yv = static_cast<int>(v * 4095.5f);
    if (yv > 4095) yv = 4095;
    return yv;
}

static inline int rgb_to_xyz_y_12bit(int r16, int g16, int b16) {
    return matrix_y_12bit(std::min(r16 >> 4, 4095),
                          std::min(g16 >> 4, 4095),
                          std::min(b16 >> 4, 4095),
                          0.2126f, 0.7152f, 0.0722f);
}

static inline int rgb_to_xyz_x_12bit(int r16, int g16, int b16) {
    return matrix_y_12bit(std::min(r16 >> 4, 4095),
                          std::min(g16 >> 4, 4095),
                          std::min(b16 >> 4, 4095),
                          0.4124f, 0.3576f, 0.1805f);
}

static inline int rgb_to_xyz_z_12bit(int r16, int g16, int b16) {
    return matrix_y_12bit(std::min(r16 >> 4, 4095),
                          std::min(g16 >> 4, 4095),
                          std::min(b16 >> 4, 4095),
                          0.0193f, 0.1192f, 0.9505f);
}

static bool opj_decode(const std::vector<uint8_t>& cs,
                       std::vector<std::vector<int>>& comps, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/cmpdc_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return false; }
    close(fd);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ: %s\n",m); }, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (ok) {
        W_out = img->comps[0].w; H_out = img->comps[0].h;
        comps.resize(img->numcomps);
        for (int c = 0; c < (int)img->numcomps; c++)
            comps[c].assign(img->comps[c].data, img->comps[c].data + W_out*H_out);
    }
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st);
    opj_destroy_codec(codec);
    unlink(tmp);
    return ok;
}

/* Encode with OPJ given a pre-computed XYZ frame (3 components, 12-bit unsigned).
 * Mirrors what the GPU encoder does: 3 components, prec=12, 5 levels, irreversible 9/7.
 * target_bytes = approximate compressed budget. */
static bool opj_encode_xyz(const std::vector<int>& X, const std::vector<int>& Y, const std::vector<int>& Z,
                           int target_bytes, int num_levels,
                           std::vector<uint8_t>& out)
{
    opj_cparameters_t params;
    opj_set_default_encoder_parameters(&params);
    params.irreversible      = 1;
    params.numresolution     = num_levels + 1;
    params.cp_disto_alloc    = 1;
    params.tcp_numlayers     = 1;
    params.tcp_rates[0]      = (W * H * 3 * 1.5f) / float(target_bytes);  /* uncompressed/compressed */
    params.cblockw_init      = 32;
    params.cblockh_init      = 32;
    params.prog_order        = OPJ_LRCP;

    opj_image_cmptparm_t parm[3];
    for (int c = 0; c < 3; ++c) {
        parm[c].dx = parm[c].dy = 1;
        parm[c].w = W; parm[c].h = H;
        parm[c].x0 = parm[c].y0 = 0;
        parm[c].prec = parm[c].bpp = 12;
        parm[c].sgnd = 0;
    }
    opj_image_t* img = opj_image_create(3, parm, OPJ_CLRSPC_SYCC);
    img->x0 = img->y0 = 0; img->x1 = W; img->y1 = H;
    for (size_t i = 0; i < (size_t)W*H; ++i) {
        img->comps[0].data[i] = X[i];
        img->comps[1].data[i] = Y[i];
        img->comps[2].data[i] = Z[i];
    }

    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char* m, void*){ fprintf(stderr,"OPJ_ENC: %s\n",m); }, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_encoder(codec, &params, img);

    char tmp[64]; std::strcpy(tmp, "/tmp/cmpec_XXXXXX");
    int fd = mkstemp(tmp); close(fd);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 0);
    bool ok = opj_start_compress(codec, img, st) &&
              opj_encode(codec, st) &&
              opj_end_compress(codec, st);
    opj_stream_destroy(st); opj_destroy_codec(codec); opj_image_destroy(img);

    if (ok) {
        FILE* f = fopen(tmp, "rb");
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); rewind(f);
        out.resize(sz);
        (void)fread(out.data(), 1, sz, f); fclose(f);
    }
    unlink(tmp);
    return ok;
}

struct Pat { const char* name; std::function<uint16_t(int,int)> rgb; };

static double psnr_y(const std::vector<int>& dec, std::function<uint16_t(int,int)> rgb_fn) {
    double mse = 0.0;
    size_t n = (size_t)W * H;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = rgb_fn(x, y);
            double d = dec[(size_t)y * W + x] - rgb_to_xyz_y_12bit(v,v,v);
            mse += d*d;
        }
    mse /= (double)n;
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

int main()
{
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"GPU init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    Pat patterns[] = {
        { "flat_30000",     [](int,int){ return uint16_t(30000); } },
        { "flat_50000",     [](int,int){ return uint16_t(50000); } },
        { "h_gradient",     [](int x,int){ return uint16_t(x * 60000ll / (W-1)); } },
        { "h_bars_8",       [](int x,int){ return uint16_t(((x/256) % 2) ? 50000 : 10000); } },
        { "v_bars_8",       [](int,int y){ return uint16_t(((y/135) % 2) ? 50000 : 10000); } },
        { "checker_64",     [](int x,int y){ return uint16_t((((x/64)+(y/64))&1) ? 50000 : 10000); } },
        { "two_value_split",[](int x,int){ return uint16_t(x < W/2 ? 20000 : 40000); } },
    };

    printf("Pattern             |   GPU PSNR_Y   GPU bytes |   OPJ PSNR_Y   OPJ bytes |  delta\n");
    printf("--------------------+--------------------------+--------------------------+--------\n");

    for (auto& pat : patterns) {
        std::vector<uint16_t> rgb((size_t)W * H * 3);
        std::vector<int> Xv((size_t)W*H), Yv((size_t)W*H), Zv((size_t)W*H);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                uint16_t v = pat.rgb(x, y);
                rgb[((size_t)y*W + x)*3 + 0] = v;
                rgb[((size_t)y*W + x)*3 + 1] = v;
                rgb[((size_t)y*W + x)*3 + 2] = v;
                Xv[(size_t)y*W+x] = rgb_to_xyz_x_12bit(v,v,v);
                Yv[(size_t)y*W+x] = rgb_to_xyz_y_12bit(v,v,v);
                Zv[(size_t)y*W+x] = rgb_to_xyz_z_12bit(v,v,v);
            }

        /* GPU */
        enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false, false);
        auto cs_gpu = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false, false);
        std::vector<std::vector<int>> dec_gpu; int dW, dH;
        double psnr_g = -1;
        size_t sz_gpu = cs_gpu.size();
        if (opj_decode(cs_gpu, dec_gpu, dW, dH)) psnr_g = psnr_y(dec_gpu[1], pat.rgb);

        /* OPJ */
        std::vector<uint8_t> cs_opj;
        double psnr_o = -1;
        size_t sz_opj = 0;
        if (opj_encode_xyz(Xv, Yv, Zv, TARGET_BYTES, 5, cs_opj)) {
            sz_opj = cs_opj.size();
            std::vector<std::vector<int>> dec_opj;
            if (opj_decode(cs_opj, dec_opj, dW, dH)) psnr_o = psnr_y(dec_opj[1], pat.rgb);
        }

        double delta = psnr_g - psnr_o;
        printf("%-19s |   %5.1f dB    %7zu B |   %5.1f dB    %7zu B | %+5.1f dB\n",
               pat.name, psnr_g, sz_gpu, psnr_o, sz_opj, delta);
    }
    return 0;
}
