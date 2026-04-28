/*
 * Diagnose at which DWT resolution the GPU encoder's output diverges from CPU.
 * Encodes one pattern with both GPU and OPJ, then decodes each at every cp_reduce
 * level (0=full, ..., N=LL only) and reports PSNR at each.
 *
 * If GPU LL-only PSNR ≈ OPJ LL-only PSNR but full PSNR is much worse,
 * then the bug is in detail-band encoding (HL/LH/HH).
 *
 * Build: nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib
 *        -I/usr/include/openjpeg-2.5 -o test/diag_reso test/diag_reso.cc
 *        src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
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
static const int64_t BR = 150000000LL;
static const int NLEVELS = 5;

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
static inline int rgb_to_y(int v) {
    return int(v / 65535.f * 4095.f + 0.5f);   /* y == r==g==b case (matrix row sums to 1) */
}
static inline int rgb_to_x(int v) { float vf=v/65535.f; return int((0.4124f+0.3576f+0.1805f)*vf*4095.f+0.5f); }
static inline int rgb_to_z(int v) { float vf=v/65535.f; return int((0.0193f+0.1192f+0.9505f)*vf*4095.f+0.5f); }

static bool opj_decode(const std::vector<uint8_t>& cs, int cp_reduce,
                       std::vector<std::vector<int>>& comps, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/diagr_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return false; }
    close(fd);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    params.cp_reduce = cp_reduce;
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char*, void*){}, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
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

static bool opj_encode_xyz(const std::vector<int>& X, const std::vector<int>& Y, const std::vector<int>& Z,
                           int target_bytes, std::vector<uint8_t>& out)
{
    opj_cparameters_t params;
    opj_set_default_encoder_parameters(&params);
    params.irreversible      = 1;
    params.numresolution     = NLEVELS + 1;
    params.cp_disto_alloc    = 1;
    params.tcp_numlayers     = 1;
    params.tcp_rates[0]      = (W * H * 3 * 1.5f) / float(target_bytes);
    params.cblockw_init      = 32;
    params.cblockh_init      = 32;
    params.prog_order        = OPJ_LRCP;

    opj_image_cmptparm_t parm[3];
    for (int c = 0; c < 3; ++c) {
        parm[c].dx = parm[c].dy = 1; parm[c].w = W; parm[c].h = H;
        parm[c].x0 = parm[c].y0 = 0; parm[c].prec = parm[c].bpp = 12; parm[c].sgnd = 0;
    }
    opj_image_t* img = opj_image_create(3, parm, OPJ_CLRSPC_SYCC);
    img->x0 = img->y0 = 0; img->x1 = W; img->y1 = H;
    for (size_t i = 0; i < (size_t)W*H; ++i) {
        img->comps[0].data[i] = X[i];
        img->comps[1].data[i] = Y[i];
        img->comps[2].data[i] = Z[i];
    }
    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec, [](const char*, void*){}, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_encoder(codec, &params, img);
    char tmp[64]; std::strcpy(tmp, "/tmp/diagre_XXXXXX");
    int fd = mkstemp(tmp); close(fd);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 0);
    bool ok = opj_start_compress(codec, img, st) &&
              opj_encode(codec, st) &&
              opj_end_compress(codec, st);
    opj_stream_destroy(st); opj_destroy_codec(codec); opj_image_destroy(img);
    if (ok) {
        FILE* f = fopen(tmp, "rb");
        fseek(f, 0, SEEK_END); size_t sz = ftell(f); rewind(f);
        out.resize(sz); (void)fread(out.data(), 1, sz, f); fclose(f);
    }
    unlink(tmp); return ok;
}

static double psnr_at_resolution(const std::vector<int>& dec, int dW, int dH,
                                 std::function<int(int,int)> ref_full_y) {
    /* dec dimensions are reduced by 2^level relative to W x H.
     * Each decoded pixel corresponds to a 2^level x 2^level block in original space.
     * For LL-only decoding, each decoded pixel ~= mean of original block (after low-pass). */
    int level = 0;
    int t = W;
    while (t > dW) { t /= 2; ++level; }
    int factor = 1 << level;
    double mse = 0.0;
    size_t n = (size_t)dW * dH;
    for (int y = 0; y < dH; ++y)
        for (int x = 0; x < dW; ++x) {
            int xb = x * factor, yb = y * factor;
            int sum = 0, cnt = 0;
            for (int dy = 0; dy < factor && yb+dy < H; ++dy)
                for (int dx = 0; dx < factor && xb+dx < W; ++dx) { sum += ref_full_y(xb+dx, yb+dy); ++cnt; }
            int ref = (cnt > 0) ? sum / cnt : 0;
            double d = dec[(size_t)y*dW + x] - ref;
            mse += d*d;
        }
    mse /= (double)n;
    if (mse < 1e-12) return 99.0;
    return 20.0 * std::log10(4095.0 / std::sqrt(mse));
}

int main(int argc, char** argv) {
    const char* pat_name = (argc > 1) ? argv[1] : "h_bars_8";
    auto rgb_fn = [&](int x, int y) -> uint16_t {
        if (!std::strcmp(pat_name, "flat"))           return 30000;
        if (!std::strcmp(pat_name, "h_bars_8"))       return uint16_t(((x/256) % 2) ? 50000 : 10000);
        if (!std::strcmp(pat_name, "v_bars_8"))       return uint16_t(((y/135) % 2) ? 50000 : 10000);
        if (!std::strcmp(pat_name, "checker_64"))     return uint16_t((((x/64)+(y/64))&1) ? 50000 : 10000);
        if (!std::strcmp(pat_name, "two_value"))      return uint16_t(x < W/2 ? 20000 : 40000);
        if (!std::strcmp(pat_name, "h_gradient"))     return uint16_t(x * 60000ll / (W-1));
        return 30000;
    };

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"GPU init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb((size_t)W * H * 3);
    std::vector<int> Xv((size_t)W*H), Yv((size_t)W*H), Zv((size_t)W*H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = rgb_fn(x, y);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
            Xv[(size_t)y*W+x] = rgb_to_x(v);
            Yv[(size_t)y*W+x] = rgb_to_y(v);
            Zv[(size_t)y*W+x] = rgb_to_z(v);
        }
    auto ref_y = [&](int x, int y){ return rgb_to_y(rgb_fn(x, y)); };

    enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false, false);
    auto cs_gpu = enc.encode_ebcot(rgb.data(), W, H, W*3, BR, 24, false, false, false);
    std::vector<uint8_t> cs_opj;
    opj_encode_xyz(Xv, Yv, Zv, BR/24/8, cs_opj);

    printf("Pattern: %s\n", pat_name);
    printf("                 GPU (size=%zu)         OPJ (size=%zu)\n", cs_gpu.size(), cs_opj.size());
    printf("cp_reduce=N      dim          PSNR_Y   dim          PSNR_Y\n");
    for (int r = NLEVELS; r >= 0; --r) {
        std::vector<std::vector<int>> dec_g, dec_o;
        int dW=0, dH=0;
        double pg = -1, po = -1;
        if (opj_decode(cs_gpu, r, dec_g, dW, dH)) pg = psnr_at_resolution(dec_g[1], dW, dH, ref_y);
        int gW = dW, gH = dH;
        if (opj_decode(cs_opj, r, dec_o, dW, dH)) po = psnr_at_resolution(dec_o[1], dW, dH, ref_y);
        printf("  %d              %4dx%-4d   %5.1f   %4dx%-4d   %5.1f\n",
               r, gW, gH, pg, dW, dH, po);
    }
    return 0;
}
