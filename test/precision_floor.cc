/* Where does the 60 dB precision floor come from?
 * Encode-then-decode, compute coefficient-domain error vs DWT pre-quantization values.
 */
#include <cmath>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include "lib/cuda_j2k_encoder.h"

static const int W = 2048, H = 1080;
static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i / 4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

int main() {
    CudaJ2KEncoder enc; GpuColourParams p; build_params(p); enc.set_colour_params(p);
    auto bar_val = [](int x, int) -> uint16_t {
        return uint16_t(((x/256) % 2) ? 50000 : 10000);
    };
    std::vector<uint16_t> rgb((size_t)W*H*3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = bar_val(x, y);
            for (int c = 0; c < 3; ++c) rgb[((size_t)y*W + x)*3 + c] = v;
        }
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    std::vector<float> dwt = enc.debug_get_dwt_output(1, W, H);

    /* Compute reference DWT-9/7 of input on CPU FP32. */
    int input_y = int(50000/65535.f * 4095.f + 0.5f);
    int input_y_lo = int(10000/65535.f * 4095.f + 0.5f);
    /* Simple test: dump LL5 max abs and detail max abs */
    float ll5_max = 0, hl1_max = 0, hh1_max = 0;
    int ll5_w = (W + 31) / 32, ll5_h = (H + 31) / 32;
    for (int y = 0; y < ll5_h; ++y) for (int x = 0; x < ll5_w; ++x)
        ll5_max = std::max(ll5_max, std::fabs(dwt[(size_t)y*W + x]));
    for (int y = 0; y < H/2; ++y) for (int x = W/2; x < W; ++x)
        hl1_max = std::max(hl1_max, std::fabs(dwt[(size_t)y*W + x]));
    for (int y = H/2; y < H; ++y) for (int x = W/2; x < W; ++x)
        hh1_max = std::max(hh1_max, std::fabs(dwt[(size_t)y*W + x]));
    printf("Pre-encode DWT magnitudes: LL5=%.2f  HL1=%.2f  HH1=%.2f\n", ll5_max, hl1_max, hh1_max);
    printf("Y reference values: lo=%d hi=%d (after DC shift: %d, %d)\n",
           input_y_lo, input_y, input_y_lo - 2048, input_y - 2048);
    /* If FP16 noise floor, max coefficient mag should round to nearest representable.
     * For HL1 around 1500, FP16 step is 1.0 (range 1024..2048). */
    return 0;
}
