/*
 * Diagnose checker_64 PSNR issue.  Encodes checker pattern, decodes, prints
 * error spatial distribution to identify whether errors are concentrated at
 * cell boundaries (DWT issue) or uniform (T1 quantisation issue).
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/debug_checker \
 *        test/debug_checker.cc src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>
#include "lib/cuda_j2k_encoder.h"

static const int W = 2048, H = 1080;

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i / 4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

static bool opj_decode_cs(const std::vector<uint8_t>& cs,
                          std::vector<std::vector<int>>& comps, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/debug_chk_XXXXXX");
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

int main() {
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) return 1;
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = (((x/64)+(y/64))&1) ? 50000 : 10000;
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
        }

    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    printf("encoded cs_size=%zu B\n", cs.size());

    std::vector<std::vector<int>> dec; int dW, dH;
    if (!opj_decode_cs(cs, dec, dW, dH)) { fprintf(stderr,"decode fail\n"); return 1; }

    /* Reference: 12-bit Y for each pixel. matrix Y row sums to 1.0 → Y = v_12bit. */
    auto ref_y = [](int x, int y) -> int {
        int v = (((x/64)+(y/64))&1) ? 50000 : 10000;
        int v12 = std::min(v >> 4, 4095);  /* matches encoder */
        return v12;
    };

    /* Look at row 320 (inside a row of cells) and at cell-boundary row 64.
     * Print pixels near a vertical boundary x=64 (transition 10000→50000). */
    printf("\n--- Row 320 (y=320, in middle of a cell row) near x=60..70 (cell boundary at 64): ---\n");
    printf("    x:");
    for (int x = 56; x < 80; ++x) printf(" %5d", x);
    printf("\n");
    printf("  ref:");
    for (int x = 56; x < 80; ++x) printf(" %5d", ref_y(x, 320));
    printf("\n");
    printf("  dec:");
    for (int x = 56; x < 80; ++x) printf(" %5d", dec[1][320*dW + x]);
    printf("\n");
    printf("  err:");
    for (int x = 56; x < 80; ++x) printf(" %5d", dec[1][320*dW + x] - ref_y(x, 320));
    printf("\n");

    /* PSNR by region: inside cells vs near boundaries. */
    long long sse_in = 0, sse_bnd = 0;
    int n_in = 0, n_bnd = 0;
    int min_e = INT32_MAX, max_e = INT32_MIN;
    for (int y = 0; y < dH; ++y)
        for (int x = 0; x < dW; ++x) {
            int e = dec[1][y*dW + x] - ref_y(x, y);
            if (e < min_e) min_e = e;
            if (e > max_e) max_e = e;
            int dx = x % 64, dy = y % 64;
            int near_bnd = (dx < 8 || dx >= 56 || dy < 8 || dy >= 56);
            long long se = (long long)e * e;
            if (near_bnd) { sse_bnd += se; ++n_bnd; }
            else          { sse_in  += se; ++n_in;  }
        }
    auto psnr_of = [](long long sse, int n) {
        if (n <= 0 || sse <= 0) return 99.0;
        double mse = (double)sse / n;
        return 20.0 * std::log10(4095.0 / std::sqrt(mse));
    };
    printf("\nerror summary: min_err=%d max_err=%d\n", min_e, max_e);
    printf("  near-boundary (within 8px of cell edge):  PSNR = %.2f dB  (n=%d)\n",
           psnr_of(sse_bnd, n_bnd), n_bnd);
    printf("  inside cells (>=8px from edge):           PSNR = %.2f dB  (n=%d)\n",
           psnr_of(sse_in, n_in), n_in);

    /* Distribution of decoded values at cell interiors (should be exactly two
     * values: ref(10000)=625 and ref(50000)=3125). */
    int hist_lo = 0, hist_hi = 0;
    long long sum_lo = 0, sum_hi = 0;
    int n_lo = 0, n_hi = 0;
    for (int y = 16; y < dH - 16; y += 4)
        for (int x = 16; x < dW - 16; x += 4) {
            int dx = x % 64, dy = y % 64;
            if (dx < 16 || dx >= 48 || dy < 16 || dy >= 48) continue;
            int r = ref_y(x, y);
            int d = dec[1][y*dW + x];
            if (r == 625)  { sum_lo += d; ++n_lo; if (d != 625) ++hist_lo; }
            if (r == 3125) { sum_hi += d; ++n_hi; if (d != 3125) ++hist_hi; }
        }
    printf("\ncell-interior decoded values:\n");
    printf("  ref=625 (low):  mean_dec=%.1f, off-target=%d/%d\n",
           n_lo > 0 ? (double)sum_lo/n_lo : 0, hist_lo, n_lo);
    printf("  ref=3125 (hi):  mean_dec=%.1f, off-target=%d/%d\n",
           n_hi > 0 ? (double)sum_hi/n_hi : 0, hist_hi, n_hi);

    return 0;
}
