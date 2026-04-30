/*
 * diag_packet_bytes.cc — Parse the J2K codestream produced by the GPU encoder
 * and print bytes per (resolution, component) packet for a given pattern.
 *
 * Used to diagnose where bytes are being allocated for checker_64 vs other
 * patterns where PCRD walks LL→HL/LH/HH coarse-to-fine in raster order.
 *
 * Build:
 *   nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib \
 *        -I/usr/include/openjpeg-2.5 -o test/diag_packet_bytes \
 *        test/diag_packet_bytes.cc src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2
 *
 * Run:
 *   ./test/diag_packet_bytes [pattern_name]
 *   pattern_name ∈ {checker_64, photo_synth, h_bars_8, flat_30000, ...}
 */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "lib/cuda_j2k_encoder.h"

static const int W = 2048, H = 1080;
static const int NLEVELS = 5;

static void build_params(GpuColourParams& p) {
    for (int i = 0; i < 4096; ++i) { p.lut_in[i] = i / 4095.f; p.lut_out[i] = uint16_t(i); }
    p.matrix[0]=0.4124f; p.matrix[1]=0.3576f; p.matrix[2]=0.1805f;
    p.matrix[3]=0.2126f; p.matrix[4]=0.7152f; p.matrix[5]=0.0722f;
    p.matrix[6]=0.0193f; p.matrix[7]=0.1192f; p.matrix[8]=0.9505f;
    p.valid = true;
}

static uint16_t pat_value(const char* name, int x, int y) {
    if (!std::strcmp(name, "flat_30000"))      return 30000;
    if (!std::strcmp(name, "flat_50000"))      return 50000;
    if (!std::strcmp(name, "flat_5000"))       return 5000;
    if (!std::strcmp(name, "h_bars_8"))        return uint16_t(((x/256) % 2) ? 50000 : 10000);
    if (!std::strcmp(name, "v_bars_8"))        return uint16_t(((y/135) % 2) ? 50000 : 10000);
    if (!std::strcmp(name, "checker_64"))      return uint16_t((((x/64)+(y/64))&1) ? 50000 : 10000);
    if (!std::strcmp(name, "h_gradient"))      return uint16_t((uint64_t)x * 60000ull / (W-1));
    if (!std::strcmp(name, "v_gradient"))      return uint16_t((uint64_t)y * 60000ull / (H-1));
    if (!std::strcmp(name, "ramp_small_range"))return uint16_t(20000 + (x % 16));
    if (!std::strcmp(name, "two_value_split")) return uint16_t(x < W/2 ? 20000 : 40000);
    if (!std::strcmp(name, "single_impulse"))  return (x == W/2 && y == H/2) ? 50000 : 30000;
    if (!std::strcmp(name, "noise_small")) {
        unsigned u = uint32_t(y*W + x) * 2654435761u;
        return uint16_t(30000 + ((u >> 16) % 31));
    }
    if (!std::strcmp(name, "photo_synth")) {
        float fx = float(x)/W, fy = float(y)/H;
        float v = 0.5f + 0.3f*std::sin(8.0f*fx) + 0.15f*std::sin(40.0f*(fx+0.7f*fy));
        if (v < 0) v = 0; if (v > 1) v = 1;
        return uint16_t(v * 60000.f);
    }
    return 30000;
}

/* Find marker position; returns offset of marker (the 0xFF byte) or SIZE_MAX. */
static size_t find_marker(const std::vector<uint8_t>& cs, size_t start, uint16_t marker) {
    for (size_t i = start; i + 1 < cs.size(); ++i) {
        uint16_t m = (uint16_t(cs[i]) << 8) | cs[i+1];
        if (m == marker) return i;
    }
    return SIZE_MAX;
}

int main(int argc, char** argv) {
    const char* pat = (argc > 1) ? argv[1] : "checker_64";

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { std::fprintf(stderr, "GPU init failed\n"); return 1; }
    GpuColourParams cp; build_params(cp); enc.set_colour_params(cp);

    std::vector<uint16_t> rgb((size_t)W * H * 3);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = pat_value(pat, x, y);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
        }

    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    std::printf("pattern=%s cs_size=%zu B\n", pat, cs.size());

    /* Walk SOT markers. Each tile-part contains a slice of the LRCP packet
     * stream. We can't easily decompose individual packets without their
     * lengths, but we can sum bytes per tile-part — and since LRCP order is
     * (res 0 c0 c1 c2)(res 1 c0 c1 c2)... we can deduce roughly which
     * subbands fall in which TP based on byte fractions. Simpler: report
     * the tile-part split itself so we see how the packets distribute. */

    /* Find SIZ → COD → QCD → TLM → SOT(s) → SOD/EOC */
    size_t pos = 0;
    /* SOC */
    if (cs.size() < 2 || cs[0] != 0xFF || cs[1] != 0x4F) {
        std::fprintf(stderr, "no SOC\n"); return 1;
    }
    pos = 2;

    /* Walk markers until we hit SOT */
    int n_tps = 0;
    std::vector<uint32_t> tp_sizes;
    std::vector<size_t> tp_data_offsets;
    while (pos + 1 < cs.size()) {
        uint8_t a = cs[pos], b = cs[pos+1];
        if (a != 0xFF) { ++pos; continue; }
        uint16_t m = (uint16_t(a) << 8) | b;
        if (m == 0xFFD9) break;  /* EOC */
        if (m == 0xFF90) {
            /* SOT */
            uint16_t lsot = (uint16_t(cs[pos+2]) << 8) | cs[pos+3];
            uint16_t isot = (uint16_t(cs[pos+4]) << 8) | cs[pos+5];
            uint32_t psot = (uint32_t(cs[pos+6])<<24)|(uint32_t(cs[pos+7])<<16)
                          |(uint32_t(cs[pos+8])<<8) | uint32_t(cs[pos+9]);
            uint8_t tpsot = cs[pos+10], tnsot = cs[pos+11];
            tp_sizes.push_back(psot);
            ++n_tps;
            std::printf("  TP %d/%d: size=%u tile=%u\n",
                tpsot, tnsot, psot, isot);
            (void)lsot;
            /* Skip past SOT (12 bytes) and SOD marker (2 bytes) — packet bytes follow */
            pos += 12;
            if (pos + 1 < cs.size() && cs[pos] == 0xFF && cs[pos+1] == 0x93) {
                pos += 2;
                tp_data_offsets.push_back(pos);
                /* Jump to next marker after psot bytes from SOT start */
                pos += (psot - 14);
            }
        } else {
            /* Generic marker: read 2-byte length, advance */
            if (pos + 3 >= cs.size()) break;
            uint16_t len = (uint16_t(cs[pos+2]) << 8) | cs[pos+3];
            pos += 2 + len;
        }
    }
    std::printf("  total tile-parts: %d  (sum=%zu bytes)\n",
        n_tps,
        [&](){ size_t s=0; for(auto v:tp_sizes) s+=v; return s; }());

    return 0;
}
