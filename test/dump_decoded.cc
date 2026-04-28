/*
 * Dump decoded Y values for the bars pattern from both GPU and OPJ.
 * Prints first 600 columns of row 0 to characterize the error pattern.
 *
 * Build: nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib
 *        -I/usr/include/openjpeg-2.5 -o test/dump_decoded test/dump_decoded.cc
 *        src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
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

static bool opj_decode(const std::vector<uint8_t>& cs,
                       std::vector<std::vector<int>>& comps, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/dump_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    if (write(fd, cs.data(), cs.size()) != (ssize_t)cs.size()) { close(fd); unlink(tmp); return false; }
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
    if (!enc.is_initialized()) { fprintf(stderr,"GPU init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    /* checker_64: 2D 64-pixel checker */
    auto bar_val = [](int x, int y) -> uint16_t {
        return uint16_t((((x/64)+(y/64))&1) ? 50000 : 10000);
    };

    std::vector<uint16_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = bar_val(x, y);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
        }

    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);

    std::vector<std::vector<int>> dec; int dW, dH;
    if (!opj_decode(cs, dec, dW, dH)) { fprintf(stderr,"decode fail\n"); return 1; }

    /* Reference Y values: input=10000 → Y=624.8 (12-bit), input=50000 → Y=3124. */
    int y_lo = int(10000.f / 65535.f * 4095.f + 0.5f);  /* ≈ 625 */
    int y_hi = int(50000.f / 65535.f * 4095.f + 0.5f);  /* ≈ 3125 */
    printf("Reference: bar A (x in [0,256)..[512,768)..) = Y_lo = %d\n", y_lo);
    printf("Reference: bar B (x in [256,512)..[768,1024)..) = Y_hi = %d\n", y_hi);
    printf("\nRow 0, columns 0..600 (showing bar transitions at x=256, 512, 768):\n");

    auto get_y = [&](int x, int y) { return dec[1][(size_t)y * dW + x]; };

    int show_cols[] = {0, 100, 200, 250, 255, 256, 257, 260, 300, 400,
                       500, 510, 511, 512, 513, 520, 600, 700, 750, 760, 767, 768, 769, 800, 900, 1000};
    printf("    x   |  GPU Y   ref Y\n");
    for (int i = 0; i < (int)(sizeof(show_cols)/sizeof(show_cols[0])); ++i) {
        int x = show_cols[i];
        int ref = ((x/256) % 2) ? y_hi : y_lo;
        printf("  %4d  |  %5d   %5d  %s\n",
               x, get_y(x, 0), ref,
               std::abs(get_y(x, 0) - ref) > 50 ? "(*)" : "");
    }

    /* Per-bar mean vs ref */
    printf("\nPer-bar mean (column avg over all rows):\n");
    for (int b = 0; b < 8; ++b) {
        long sum = 0; int cnt = 0;
        for (int y = 0; y < dH; ++y)
            for (int x = b*256+50; x < (b+1)*256-50 && x < dW; ++x) { sum += get_y(x, y); ++cnt; }
        int ref = (b % 2) ? y_hi : y_lo;
        printf("  bar %d (x in [%d,%d)): mean=%.1f  ref=%d  err=%+.1f\n",
               b, b*256+50, (b+1)*256-50, double(sum)/cnt, ref, double(sum)/cnt - ref);
    }

    /* Inspect last few rows */
    printf("\nLast 5 rows, every 200 columns:\n");
    for (int yy = dH - 5; yy < dH; ++yy) {
        printf("  row %d: ", yy);
        for (int xx = 0; xx < dW; xx += 200) printf("[x=%d]=%d ", xx, get_y(xx, yy));
        printf("\n");
    }
    printf("\nFirst 3 rows, every 200 columns:\n");
    for (int yy = 0; yy < 3; ++yy) {
        printf("  row %d: ", yy);
        for (int xx = 0; xx < dW; xx += 200) printf("[x=%d]=%d ", xx, get_y(xx, yy));
        printf("\n");
    }

    /* Find max error pixels */
    int max_err = 0, max_x = 0, max_y = 0;
    long long total_sq_err = 0;
    int hist[200] = {0}; /* err magnitude histogram, 0..199+ */
    for (int y = 0; y < dH; ++y) {
        for (int x = 0; x < dW; ++x) {
            int ref = (((x/64)+(y/64))&1) ? y_hi : y_lo;
            int err = std::abs(get_y(x, y) - ref);
            if (err > max_err) { max_err = err; max_x = x; max_y = y; }
            total_sq_err += (long long)err * err;
            hist[std::min(err, 199)]++;
        }
    }
    printf("\nMax abs err: %d at (x=%d, y=%d). dec=%d ref=%d\n",
           max_err, max_x, max_y, get_y(max_x, max_y),
           ((max_x/256) % 2) ? y_hi : y_lo);
    double mse = double(total_sq_err) / (dW * (long long)dH);
    printf("RMSE = %.2f, PSNR = %.1f dB\n", std::sqrt(mse),
           20.0 * std::log10(4095.0 / std::sqrt(mse)));

    /* Top 10 error magnitudes by count */
    printf("Error magnitude histogram (top entries):\n");
    int total = dW * dH;
    int shown = 0;
    for (int e = 0; e < 200 && shown < 15; ++e) {
        if (hist[e] > total / 1000) {  /* > 0.1% */
            printf("  err=%-3d: %8d pixels (%.2f%%)\n", e, hist[e], 100.0 * hist[e] / total);
            shown++;
        }
    }
    /* Show all non-trivial error counts */
    printf("All non-zero error counts:\n");
    for (int e = 0; e < 200; ++e) {
        if (hist[e] > 0) printf("  err=%-3d: %d\n", e, hist[e]);
    }
    return 0;
}
