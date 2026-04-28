/* Dump decoded Y values for a flat input. Investigate the 48.8 dB ceiling. */
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

static bool opj_dec(const std::vector<uint8_t>& cs, std::vector<std::vector<int>>& comps, int& W_out, int& H_out)
{
    char tmp[64]; std::strcpy(tmp, "/tmp/dfXXXXXX");
    int fd = mkstemp(tmp); write(fd, cs.data(), cs.size()); close(fd);
    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
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
    opj_stream_destroy(st); opj_destroy_codec(codec);
    unlink(tmp); return ok;
}

int main(int argc, char** argv) {
    int input_val = (argc > 1) ? std::atoi(argv[1]) : 50000;
    CudaJ2KEncoder enc; GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb((size_t)W*H*3, uint16_t(input_val));
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);

    std::vector<std::vector<int>> dec; int dW, dH;
    if (!opj_dec(cs, dec, dW, dH)) { fprintf(stderr,"dec fail\n"); return 1; }

    int ref_y = int(input_val / 65535.f * 4095.f + 0.5f);
    printf("Input %d -> Y reference = %d\n", input_val, ref_y);
    printf("Codestream %zu bytes\n", cs.size());

    /* Histogram of decoded Y values */
    int hist[8192] = {0};
    int mn = 99999, mx = -99999; long sum = 0;
    for (int i = 0; i < dW*dH; ++i) {
        int v = dec[1][i];
        if (v >= 0 && v < 8192) hist[v]++;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
    printf("decoded Y: min=%d max=%d mean=%.2f\n", mn, mx, double(sum)/(dW*dH));

    printf("histogram (counts per Y value, range %d..%d):\n", mn, mx);
    for (int v = mn; v <= mx; ++v) {
        if (hist[v] > 0) printf("  Y=%d: %d (%.2f%%)\n", v, hist[v], 100.0*hist[v]/(dW*dH));
    }

    /* Sample 5 spots: corners + center */
    auto y_at = [&](int x, int y) { return dec[1][(size_t)y*dW + x]; };
    printf("Samples: (0,0)=%d (W-1,0)=%d (W/2,H/2)=%d (0,H-1)=%d (W-1,H-1)=%d\n",
           y_at(0,0), y_at(W-1,0), y_at(W/2,H/2), y_at(0,H-1), y_at(W-1,H-1));
    return 0;
}
