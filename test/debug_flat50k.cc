/*
 * Diagnose why flat_50000 (constant gray) is not lossless under V196 FP32 path.
 *
 * Compares decoded GPU output vs the original 12-bit XYZ value for a flat
 * input. Reports min/max/distribution of decoded Y per row band so we can
 * tell whether errors cluster at boundaries (DWT) or spread evenly (T1
 * quantization).
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/debug_flat50k \
 *        test/debug_flat50k.cc src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
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
    char tmp[64]; std::strcpy(tmp, "/tmp/debug_flat_XXXXXX");
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

static void test_constant(uint16_t input_val) {
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"GPU init failed\n"); return; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb((size_t)W * H * 3, input_val);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);

    std::vector<std::vector<int>> dec; int dW, dH;
    if (!opj_decode_cs(cs, dec, dW, dH)) { fprintf(stderr,"decode fail\n"); return; }

    /* Compute expected 12-bit value matching what the encoder writes:
     * 12-bit input v12 = (input_val >> 4) clamped 4095. R=G=B=v12.
     * After matrix Y = 0.2126*r + 0.7152*g + 0.0722*b; R=G=B → Y = (sum)*v12.
     * Our matrix sums to ~1.0 for Y row, so Y_input ≈ v12.
     * Output LUT is identity → expected_Y = v12.
     */
    int v12 = std::min<int>(input_val >> 4, 4095);
    float yf = (0.2126f + 0.7152f + 0.0722f) * (v12 / 4095.f);
    int expected_Y = static_cast<int>(yf * 4095.5f);
    if (expected_Y > 4095) expected_Y = 4095;

    printf("=== input_val=%u (v12=%d) expected_Y=%d  cs_size=%zu B ===\n",
           input_val, v12, expected_Y, cs.size());

    /* Per-row error stats for Y component (component 1). */
    long long sse = 0;
    int min_e = INT32_MAX, max_e = INT32_MIN;
    int n_exact = 0, n_total = 0;
    int err_hist[8] = {0};  /* 0, 1, 2, 4, 8, 16, 32, >32 */
    for (int y = 0; y < dH; ++y) {
        for (int x = 0; x < dW; ++x) {
            int v = dec[1][y * dW + x];
            int e = v - expected_Y;
            sse += (long long)e * e;
            if (e < min_e) min_e = e;
            if (e > max_e) max_e = e;
            if (e == 0) ++n_exact;
            int ae = std::abs(e);
            int b = (ae == 0) ? 0 :
                    (ae <= 1) ? 1 :
                    (ae <= 2) ? 2 :
                    (ae <= 4) ? 3 :
                    (ae <= 8) ? 4 :
                    (ae <= 16) ? 5 :
                    (ae <= 32) ? 6 : 7;
            err_hist[b]++;
            ++n_total;
        }
    }
    double mse = (double)sse / n_total;
    double psnr = (mse > 0) ? 10.0 * std::log10(4095.0 * 4095.0 / mse) : 99.0;
    printf("  PSNR_Y = %.2f dB  (n=%d, exact=%d, min_err=%d max_err=%d)\n",
           psnr, n_total, n_exact, min_e, max_e);
    printf("  err histogram: 0=%d  1=%d  2=%d  3-4=%d  5-8=%d  9-16=%d  17-32=%d  >32=%d\n",
           err_hist[0], err_hist[1], err_hist[2], err_hist[3],
           err_hist[4], err_hist[5], err_hist[6], err_hist[7]);

    /* Sample row 0 and row H/2 first 16 pixels. */
    printf("  row 0 first 16 Y values:");
    for (int x = 0; x < 16; ++x) printf(" %d", dec[1][x]);
    printf("\n");
    printf("  row %d first 16 Y values:", H/2);
    for (int x = 0; x < 16; ++x) printf(" %d", dec[1][(H/2) * dW + x]);
    printf("\n");
}

int main() {
    test_constant(30000);
    test_constant(50000);
    test_constant(5000);
    test_constant(40000);
    return 0;
}
