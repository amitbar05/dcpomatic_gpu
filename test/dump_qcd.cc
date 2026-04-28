/*
 * Dump QCD step values from a J2K codestream.
 * Decodes (eps<<11)|man encoding per T.800 A.6.4 to actual step.
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

static void dump_qcd(const std::vector<uint8_t>& cs, const char* who) {
    /* Find QCD marker (0xFF 0x5C). */
    for (size_t i = 0; i + 2 < cs.size(); ++i) {
        if (cs[i] == 0xFF && cs[i+1] == 0x5C) {
            uint16_t lqcd = (cs[i+2] << 8) | cs[i+3];
            uint8_t sqcd  = cs[i+4];
            int sqty = sqcd & 0x1F;
            int numgbits = (sqcd >> 5) & 0x7;
            printf("[%s] QCD: lqcd=%u sqcd=0x%02X (sqty=%d %s, gbits=%d)\n",
                   who, lqcd, sqcd, sqty,
                   sqty == 0 ? "none" : (sqty == 1 ? "scalar derived" : "scalar expounded"),
                   numgbits);
            int n = (lqcd - 3) / 2;  /* number of subbands when expounded */
            if (sqty == 1) n = 1;
            for (int k = 0; k < n; ++k) {
                uint16_t val = (cs[i+5 + 2*k] << 8) | cs[i+6 + 2*k];
                int eps = (val >> 11) & 0x1F;
                int man = val & 0x7FF;
                /* OPJ formula: stepsize = (1+man/2048) * 2^(Rb-eps), Rb=12. */
                float step = (1.0f + man/2048.0f) * std::ldexp(1.0f, 12 - eps);
                /* band->numbps = eps + numgbits - 1 (for irreversible) */
                int band_numbps = eps + numgbits - 1;
                printf("  subband[%d] eps=%d man=%4d step=%.4f band->numbps=%d\n",
                       k, eps, man, step, band_numbps);
            }
            return;
        }
    }
    printf("[%s] QCD not found\n", who);
}

static bool opj_encode_xyz(const std::vector<int>& X, const std::vector<int>& Y, const std::vector<int>& Z,
                           int target_bytes, std::vector<uint8_t>& out)
{
    opj_cparameters_t params;
    opj_set_default_encoder_parameters(&params);
    params.irreversible = 1; params.numresolution = 6;
    params.cp_disto_alloc = 1; params.tcp_numlayers = 1;
    params.tcp_rates[0] = (W * H * 3 * 1.5f) / float(target_bytes);
    params.cblockw_init = 32; params.cblockh_init = 32;
    params.prog_order = OPJ_LRCP;
    opj_image_cmptparm_t parm[3];
    for (int c = 0; c < 3; ++c) {
        parm[c].dx = parm[c].dy = 1; parm[c].w = W; parm[c].h = H;
        parm[c].x0 = parm[c].y0 = 0; parm[c].prec = parm[c].bpp = 12; parm[c].sgnd = 0;
    }
    opj_image_t* img = opj_image_create(3, parm, OPJ_CLRSPC_SYCC);
    img->x0 = img->y0 = 0; img->x1 = W; img->y1 = H;
    for (size_t i = 0; i < (size_t)W*H; ++i) {
        img->comps[0].data[i] = X[i]; img->comps[1].data[i] = Y[i]; img->comps[2].data[i] = Z[i];
    }
    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    opj_setup_encoder(codec, &params, img);
    char tmp[64]; std::strcpy(tmp, "/tmp/qcdo_XXXXXX");
    int fd = mkstemp(tmp); close(fd);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 0);
    bool ok = opj_start_compress(codec, img, st) &&
              opj_encode(codec, st) &&
              opj_end_compress(codec, st);
    opj_stream_destroy(st); opj_destroy_codec(codec); opj_image_destroy(img);
    if (ok) {
        FILE* f = fopen(tmp, "rb"); fseek(f, 0, SEEK_END);
        size_t sz = ftell(f); rewind(f);
        out.resize(sz); (void)fread(out.data(), 1, sz, f); fclose(f);
    }
    unlink(tmp); return ok;
}

int main() {
    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr,"GPU init failed\n"); return 1; }
    GpuColourParams p; build_params(p); enc.set_colour_params(p);

    std::vector<uint16_t> rgb((size_t)W * H * 3);
    std::vector<int> Xv((size_t)W*H), Yv((size_t)W*H), Zv((size_t)W*H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            uint16_t v = uint16_t(((x/256) % 2) ? 50000 : 10000);
            rgb[((size_t)y*W + x)*3 + 0] = v;
            rgb[((size_t)y*W + x)*3 + 1] = v;
            rgb[((size_t)y*W + x)*3 + 2] = v;
            float vf = v / 65535.f;
            Xv[(size_t)y*W+x] = int((0.4124f+0.3576f+0.1805f)*vf*4095.f+0.5f);
            Yv[(size_t)y*W+x] = int(vf*4095.f+0.5f);  /* Y row sums to 1 */
            Zv[(size_t)y*W+x] = int((0.0193f+0.1192f+0.9505f)*vf*4095.f+0.5f);
        }
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    auto cs_gpu = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000LL, 24, false, false, false);
    std::vector<uint8_t> cs_opj;
    opj_encode_xyz(Xv, Yv, Zv, 150000000/24/8, cs_opj);

    /* JPEG2000 subband order in QCD: LL[N], HL[N],LH[N],HH[N], HL[N-1],LH[N-1],HH[N-1], ..., HL[1],LH[1],HH[1].
     * For 5 levels: 1+3*5 = 16 subbands. */
    printf("=== Subband index legend (5 levels):\n");
    printf("  0: LL5\n");
    int idx = 1;
    for (int lvl = 5; lvl >= 1; --lvl) {
        printf("  %d: HL%d   %d: LH%d   %d: HH%d\n", idx, lvl, idx+1, lvl, idx+2, lvl);
        idx += 3;
    }
    printf("\n");
    dump_qcd(cs_gpu, "GPU");
    printf("\n");
    dump_qcd(cs_opj, "OPJ");
    return 0;
}
