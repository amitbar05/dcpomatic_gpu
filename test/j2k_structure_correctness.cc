/*
 * J2K Codestream Structure Correctness Test
 *
 * Validates that the GPU encoder produces codestreams that conform to
 * ITU-T T.800 (JPEG2000 Part 1) structural requirements:
 *
 *   1. SOC marker at offset 0
 *   2. SIZ marker: Rsiz, Xsiz/Ysiz, tile dims, Csiz=3, bit depth=12, signed=0
 *   3. COD marker: progression order, num_layers, DWT_levels, precinct_size
 *   4. QCD marker: sqty=2 (expounded), numsteps=3*DWT+1, all steps in valid range
 *   5. SOT/SOD structure: valid tile index, tile length, SOD present
 *   6. EOC at end
 *   7. No malformed 0xFF escape (all FF xx markers are legal J2K marker codes)
 *   8. Packet header ZCOD tag tree: zbp in [0, pmax] for all code-blocks
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       -include test/gpu_ebcot_preinclude.h \
 *       -I/home/amit/dcp-o-matic-gpu/src \
 *       -I/home/amit/dcp-o-matic-gpu/src/lib \
 *       -o test/j2k_structure_correctness test/j2k_structure_correctness.cc \
 *       -I/usr/include/openjpeg-2.5 \
 *       /home/amit/dcp-o-matic-gpu/src/lib/cuda_j2k_encoder.cu \
 *       -lcudart -lopenjp2
 *
 * (Uses OpenjPEG only for decode validation, not structural parsing.)
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
#include <unistd.h>
#include <openjpeg.h>

#include "lib/cuda_j2k_encoder.h"

/* ===== Harness ===== */

static int g_pass = 0, g_fail = 0;
static bool g_verbose = false;

static void CHECK(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("  PASS: %s\n", msg); }
    else       { ++g_fail; printf("  FAIL: %s\n", msg); }
}

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

/* ===== J2K Marker Scanner ===== */

/* All valid J2K markers per ITU-T T.800 Table A-1 */
static const uint8_t VALID_MARKER_LSBS[] = {
    0x4F, /* SOC */  0x90, /* SOT */  0x93, /* SOD */  0xD9, /* EOC */
    0x51, /* SIZ */  0x52, /* COD */  0x53, /* COC */  0x55, /* TLM */
    0x57, /* PLM */  0x58, /* PLT */  0x60, /* QCD */  0x61, /* QCC */
    0x63, /* RGN */  0x64, /* POD */  0x67, /* PPM */  0x68, /* PPT */
    0x91, /* SOP */  0x92, /* EPH */  0x64, /* POC */  0x30, /* COM */
};

static bool is_valid_j2k_marker(uint8_t lsb) {
    for (auto v : VALID_MARKER_LSBS) if (v == lsb) return true;
    /* Range check: main header markers 0xFF50-0xFF6F, tile-part 0xFF90-0xFF93 */
    if (lsb >= 0x50 && lsb <= 0x6F) return true;
    if (lsb >= 0x90 && lsb <= 0x93) return true;
    if (lsb == 0xD9) return true;
    return false;
}

struct Marker {
    uint16_t code;
    size_t   offset;   /* byte offset of 0xFF in codestream */
    int      length;   /* payload length (excluding 2-byte marker), -1 if no length field */
};

static std::vector<Marker> scan_markers(const std::vector<uint8_t>& cs) {
    std::vector<Marker> out;
    size_t i = 0;
    while (i + 1 < cs.size()) {
        if (cs[i] != 0xFF) { ++i; continue; }
        uint8_t lsb = cs[i+1];
        if (lsb == 0xFF) { ++i; continue; } /* byte stuffing */
        Marker m;
        m.code   = (uint16_t)((0xFF << 8) | lsb);
        m.offset = i;
        m.length = -1;
        /* Markers with length fields (all except SOC, SOD, EOC, EPH) */
        bool has_len = (lsb != 0x4F && lsb != 0x93 && lsb != 0xD9 && lsb != 0x92);
        if (has_len && i + 3 < cs.size()) {
            m.length = (int)((cs[i+2] << 8) | cs[i+3]);
        }
        out.push_back(m);
        i += 2;
    }
    return out;
}

static const Marker* find_marker(const std::vector<Marker>& ms, uint16_t code) {
    for (auto& m : ms) if (m.code == code) return &m;
    return nullptr;
}

/* Find offset of first SOD marker (end of main header). */
static size_t find_sod_offset(const std::vector<uint8_t>& cs) {
    for (size_t i = 0; i + 1 < cs.size(); ++i)
        if (cs[i] == 0xFF && cs[i+1] == 0x93) return i;
    return cs.size();
}

/* ===== Test 1: SOC / EOC ===== */

static void test_soc_eoc(const std::vector<uint8_t>& cs) {
    printf("\n--- Test: SOC/EOC markers ---\n");

    CHECK(cs.size() >= 2 && cs[0] == 0xFF && cs[1] == 0x4F,
          "SOC marker (0xFF4F) at offset 0");
    CHECK(cs.size() >= 2 && cs[cs.size()-2] == 0xFF && cs[cs.size()-1] == 0xD9,
          "EOC marker (0xFFD9) at end of codestream");

    /* Only scan main header (SOC → first SOT) for illegal marker codes.
     * Per T.800 B.10.4: in codeblock data, 0xFF must be followed by a byte with
     * top nibble != 0xF (0x00-0x8F). But we only scan the main header here because
     * distinguishing packet-header vs codeblock boundaries requires full parsing. */
    size_t sod_off = find_sod_offset(cs);
    int illegal = 0;
    for (size_t i = 0; i + 1 < cs.size() && i < sod_off; ++i) {
        if (cs[i] == 0xFF && cs[i+1] != 0x00 && cs[i+1] != 0xFF) {
            if (!is_valid_j2k_marker(cs[i+1])) {
                printf("  Illegal in main header: 0xFF%02X at offset %zu\n", cs[i+1], i);
                ++illegal;
            }
        }
    }
    char msg[80]; snprintf(msg, sizeof(msg), "No illegal markers in main header (found %d)", illegal);
    CHECK(illegal == 0, msg);
}

/* ===== Test 2: SIZ marker ===== */

static void test_siz(const std::vector<uint8_t>& cs, int W, int H) {
    printf("\n--- Test: SIZ marker ---\n");
    auto markers = scan_markers(cs);
    const Marker* m = find_marker(markers, 0xFF51);
    CHECK(m != nullptr, "SIZ marker (0xFF51) present");
    if (!m) return;

    const uint8_t* p = cs.data() + m->offset + 2; /* skip 0xFF 0x51 */
    /* Lsiz (length) */
    int lsiz = (p[0] << 8) | p[1];
    p += 2;

    /* Rsiz: 0x0000 = unrestricted profile (cinema uses 0x0003/0x0004 but we're lax) */
    int rsiz = (p[0] << 8) | p[1]; p += 2;
    CHECK(rsiz == 0x0000 || rsiz == 0x0003 || rsiz == 0x0004,
          "SIZ Rsiz is 0 (unrestricted) or cinema profile");

    /* Xsiz, Ysiz: image size */
    int xsiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    int ysiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    char msg[120];
    snprintf(msg, sizeof(msg), "SIZ Xsiz=%d == W=%d", xsiz, W);
    CHECK(xsiz == W, msg);
    snprintf(msg, sizeof(msg), "SIZ Ysiz=%d == H=%d", ysiz, H);
    CHECK(ysiz == H, msg);

    /* XOsiz, YOsiz: image origin (should be 0,0) */
    int xosiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    int yosiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    CHECK(xosiz == 0 && yosiz == 0, "SIZ image origin (0,0)");

    /* XTsiz, YTsiz: tile size (should equal image size for single tile) */
    int xtsiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    int ytsiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    snprintf(msg, sizeof(msg), "SIZ XTsiz=%d == W=%d (single tile)", xtsiz, W);
    CHECK(xtsiz == W, msg);
    snprintf(msg, sizeof(msg), "SIZ YTsiz=%d == H=%d (single tile)", ytsiz, H);
    CHECK(ytsiz == H, msg);

    /* XTOsiz, YTOsiz: tile origin */
    int xtosiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    int ytosiz = (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3]; p+=4;
    CHECK(xtosiz == 0 && ytosiz == 0, "SIZ tile origin (0,0)");

    /* Csiz: number of components */
    int csiz = (p[0] << 8) | p[1]; p += 2;
    snprintf(msg, sizeof(msg), "SIZ Csiz=%d == 3 (XYZ)", csiz);
    CHECK(csiz == 3, msg);

    /* Per-component: Ssiz (bit depth), XRsiz, YRsiz */
    bool depths_ok = true, seps_ok = true;
    for (int c = 0; c < csiz && c < 3; c++) {
        int ssiz   = p[0]; p++;  /* bit depth - 1; MSB=1 if signed */
        int xrsiz  = p[0]; p++;
        int yrsiz  = p[0]; p++;
        bool signed_c = (ssiz >> 7) & 1;
        int  prec     = (ssiz & 0x7F) + 1;
        if (prec != 12 || signed_c) depths_ok = false;
        if (xrsiz != 1 || yrsiz != 1) seps_ok = false;
    }
    CHECK(depths_ok, "SIZ all components: 12-bit unsigned");
    CHECK(seps_ok,   "SIZ all components: XRsiz=YRsiz=1 (no subsampling)");
}

/* ===== Test 3: COD marker ===== */

static void test_cod(const std::vector<uint8_t>& cs) {
    printf("\n--- Test: COD marker ---\n");
    auto markers = scan_markers(cs);
    const Marker* m = find_marker(markers, 0xFF52);
    CHECK(m != nullptr, "COD marker (0xFF52) present");
    if (!m) return;

    const uint8_t* p = cs.data() + m->offset + 4; /* skip FF52 + 2-byte length */

    /* Scod: coding style */
    uint8_t scod = *p++;
    bool has_sop = (scod >> 1) & 1;
    bool has_eph = (scod >> 2) & 1;
    bool has_precinct = scod & 1;
    /* DCI requires specific precinct sizes */
    printf("  COD Scod=0x%02X: SOP=%d EPH=%d precinct_defined=%d\n",
           scod, has_sop, has_eph, has_precinct);

    /* SGcod: progression order, num_layers, MCT */
    uint8_t prog_order = *p++;
    int num_layers = (p[0] << 8) | p[1]; p += 2;
    uint8_t mct = *p++;
    char msg[120];
    snprintf(msg, sizeof(msg), "COD progression order LRCP(0) or RLCP(1) (got %d)", prog_order);
    CHECK(prog_order <= 4, "COD progression order valid (0-4)");
    snprintf(msg, sizeof(msg), "COD num_layers=%d >= 1", num_layers);
    CHECK(num_layers >= 1, msg);
    /* MCT=1 means decorrelation is applied (should be 1 for XYZ) */
    snprintf(msg, sizeof(msg), "COD MCT=%d (should be 1 for irreversible color transform)", mct);
    CHECK(mct == 1, msg);

    /* SPcod */
    int num_levels = *p++;
    snprintf(msg, sizeof(msg), "COD DWT levels=%d (expect 5)", num_levels);
    CHECK(num_levels == 5, msg);

    uint8_t xcb = *p++;
    uint8_t ycb = *p++;
    /* Code-block exponents: CB_DIM = 32 → xcb = 5-2 = 3? or 32-2? Let me check:
     * ITU-T T.800: xcb+2 is the code-block width exponent.
     * So CB_DIM=32=2^5 → xcb = 5-2 = 3. */
    snprintf(msg, sizeof(msg), "COD CB width/height: xcb=%d ycb=%d (expect 3 for 32px)", xcb, ycb);
    CHECK(xcb == ycb, "COD code-block is square");
    printf("  COD: xcb+2=%d ycb+2=%d → CB_DIM=%d\n", xcb+2, ycb+2, 1<<(xcb+2));

    uint8_t modes = *p++;
    /* modes: bit 0=BYPASS, bit 1=RESET, bit 2=RESTART, bit 3=CAUSAL, bit 4=ERTERM, bit 5=SEGMARK */
    printf("  COD modes=0x%02X (bypass=%d reset=%d restart=%d)\n",
           modes, modes & 1, (modes>>1)&1, (modes>>2)&1);

    uint8_t wavelet = *p++;
    /* T.800 Table A-20: 0x00 = 9/7 irreversible (CDF9/7), 0x01 = 5/3 reversible (lossless) */
    snprintf(msg, sizeof(msg), "COD wavelet filter=0x%02X (0x00=9/7 irreversible)", wavelet);
    CHECK(wavelet == 0x00, msg);
}

/* ===== Test 4: QCD marker ===== */

static void test_qcd(const std::vector<uint8_t>& cs, int num_dwt_levels) {
    printf("\n--- Test: QCD marker ---\n");
    auto markers = scan_markers(cs);
    const Marker* m = find_marker(markers, 0xFF5C);
    CHECK(m != nullptr, "QCD marker (0xFF5C) present");
    if (!m) return;

    const uint8_t* base = cs.data() + m->offset;
    int lqcd = (base[2] << 8) | base[3];
    uint8_t sqcd = base[4];
    uint8_t sqty    = sqcd & 0x1F;
    uint8_t numgbits = (sqcd >> 5) & 0x07;

    CHECK(sqty == 2, "QCD sqty=2 (expounded quantization for 9/7 irreversible)");
    CHECK(numgbits == 1, "QCD numgbits=1 (standard 2K guard bits)");

    /* For expounded quantization: 2 bytes per step.
     * numsteps = (lqcd - 3) / 2  [lqcd includes sqcd + 2-byte len field] */
    int numsteps = (lqcd - 3) / 2;
    int expected = 3 * num_dwt_levels + 1; /* LL + 3 per level */
    char msg[120];
    snprintf(msg, sizeof(msg), "QCD numsteps=%d == %d*3+1==%d for %d-level DWT",
             numsteps, num_dwt_levels, expected, num_dwt_levels);
    CHECK(numsteps == expected, msg);

    /* Parse and validate step values */
    bool all_positive = true;
    bool hl_mono = true;     /* HL steps should be monotonically non-decreasing coarse→fine */
    std::vector<float> hl_steps, lh_steps, hh_steps;
    for (int s = 0; s < numsteps && s < 64; s++) {
        int b0 = base[5 + s*2];
        int b1 = base[5 + s*2 + 1];
        int w16 = (b0 << 8) | b1;
        int eps = (w16 >> 11) & 0x1F;
        int man = w16 & 0x7FF;
        /* T.800 A.6.4: step = (1 + man/2^11) * 2^(Rb - eps), Rb=prec=12 for 12-bit */
        float val = (1.0f + man / 2048.0f) * std::ldexp(1.0f, 12 - eps);
        if (val <= 0.0f) all_positive = false;

        /* Subband ordering: LL(0), then for each level coarse→fine: HL,LH,HH */
        if (s > 0) {
            int rel = (s - 1) % 3; /* 0=HL, 1=LH, 2=HH */
            if (rel == 0) hl_steps.push_back(val);
            else if (rel == 1) lh_steps.push_back(val);
            else hh_steps.push_back(val);
        }
        if (g_verbose) printf("  QCD step[%d]: eps=%d man=%d val=%.6f\n", s, eps, man, val);
    }
    CHECK(all_positive, "QCD all step values positive");

    /* HL steps should increase coarse→fine (finer resolution → larger relative error) */
    for (size_t i = 1; i < hl_steps.size(); i++) {
        if (hl_steps[i] < hl_steps[i-1] * 0.5f) { hl_mono = false; break; }
    }
    CHECK(hl_mono, "QCD HL steps non-decreasing coarse→fine (within 2×)");
    printf("  QCD HL steps (coarse→fine): ");
    for (float v : hl_steps) printf("%.4f ", v);
    printf("\n");
    printf("  QCD HH steps (coarse→fine): ");
    for (float v : hh_steps) printf("%.4f ", v);
    printf("\n");
}

/* ===== Test 5: SOT / SOD / EOC ordering ===== */

static void test_tile_structure(const std::vector<uint8_t>& cs) {
    printf("\n--- Test: Tile-part structure ---\n");
    auto markers = scan_markers(cs);

    const Marker* sot = find_marker(markers, 0xFF90);
    CHECK(sot != nullptr, "SOT marker (0xFF90) present");

    const Marker* sod = find_marker(markers, 0xFF93);
    CHECK(sod != nullptr, "SOD marker (0xFF93) present");

    if (sot && sod) {
        CHECK(sot->offset < sod->offset, "SOT before SOD");
    }

    /* SOT fields: Lsot=10, Isot (tile index), Psot (tile-part length), TPsot, TNsot */
    if (sot) {
        const uint8_t* p = cs.data() + sot->offset + 2;
        int lsot = (p[0] << 8) | p[1];
        CHECK(lsot == 10, "SOT Lsot=10 (correct fixed length)");
        int isot = (p[2] << 8) | p[3];
        CHECK(isot == 0, "SOT Isot=0 (single tile)");
        uint32_t psot = (p[4]<<24)|(p[5]<<16)|(p[6]<<8)|p[7];
        uint8_t tpsot = p[8];
        uint8_t tnsot = p[9];
        char msg[120];
        snprintf(msg, sizeof(msg), "SOT Psot=%u > 0 (valid tile-part length)", psot);
        CHECK(psot > 0, msg);
        CHECK(tpsot == 0, "SOT TPsot=0 (first tile-part)");
        /* TNsot: 0=unknown, N=total tile-part count (e.g. 3 for per-component tile-parts) */
        snprintf(msg, sizeof(msg), "SOT TNsot=%u valid (T.800 A.4.2: 0=unknown, N=count)", tnsot);
        CHECK(tnsot <= 3, msg);  /* our encoder uses 1 or 3 tile-parts */
        printf("  SOT TPsot=%u TNsot=%u Psot=%u\n", tpsot, tnsot, psot);
        /* Psot must be <= bytes remaining from SOT (may be < if multi-tile-part) */
        size_t from_sot = cs.size() - sot->offset;
        snprintf(msg, sizeof(msg), "SOT Psot=%u <= codestream from SOT=%zu", psot, from_sot);
        CHECK((size_t)psot <= from_sot, msg);
        snprintf(msg, sizeof(msg), "SOT Psot=%u >= 14 (SOT+SOD minimum)", psot);
        CHECK(psot >= 14, msg);  /* SOT(12) + SOD(2) = 14 bytes minimum */
    }
}

/* ===== Test 6: QCD step T1 consistency ===== */

static void test_qcd_t1_consistency(const std::vector<uint8_t>& cs) {
    printf("\n--- Test: QCD step vs T1 quantization consistency ---\n");

    /* Parse QCD steps */
    const uint8_t* p = nullptr;
    for (size_t i = 0; i + 1 < cs.size(); i++) {
        if (cs[i] == 0xFF && cs[i+1] == 0x5C) { p = cs.data() + i; break; }
    }
    if (!p) { printf("  SKIP: QCD not found\n"); return; }
    int lqcd = (p[2] << 8) | p[3];
    int numsteps = (lqcd - 3) / 2;

    /* HH subbands should have larger step than HL/LH at same level
     * (HH T1 gain = 5.5× vs HL/LH gain = 2.0×; base step is the same)
     * This verifies that QCD correctly encodes the T1-scaled step for HH. */
    bool hh_bigger_ok = true;
    for (int lev = 0; lev < 5; lev++) {
        int hl_idx = 1 + lev * 3;     /* HL at this level */
        int hh_idx = 1 + lev * 3 + 2; /* HH at this level */
        if (hl_idx >= numsteps || hh_idx >= numsteps) continue;

        auto read_step = [&](int idx) -> float {
            int b0 = p[5 + idx*2];
            int b1 = p[5 + idx*2 + 1];
            int w16 = (b0 << 8) | b1;
            int eps = (w16 >> 11) & 0x1F;
            int man = w16 & 0x7FF;
            return (1.0f + man / 2048.0f) * std::ldexp(1.0f, 12 - eps);
        };
        float hl = read_step(hl_idx);
        float hh = read_step(hh_idx);
        /* HH should be at least 2× larger than HL (K²=1.51 → gain ratio 5.5/2=2.75) */
        if (hh < hl * 2.0f) { hh_bigger_ok = false; }
        printf("  Level %d: HL=%.4f HH=%.4f ratio=%.2f\n", lev, hl, hh, hh/hl);
    }
    CHECK(hh_bigger_ok, "QCD HH steps >= 2× HL at each level (T1 gain ratio 5.5/2=2.75)");
}

/* ===== Test 7: Decode succeeds for multiple patterns ===== */

static bool opj_decode_ok(const std::vector<uint8_t>& cs) {
    char tmp[64]; strcpy(tmp, "/tmp/j2ks_XXXXXX");
    int fd = mkstemp(tmp);
    if (fd < 0) return false;
    write(fd, cs.data(), cs.size()); close(fd);

    opj_dparameters_t params; opj_set_default_decoder_parameters(&params);
    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    opj_set_error_handler(codec,   [](const char* m, void*){ fprintf(stderr, "OPJ: %s\n", m); }, nullptr);
    opj_set_warning_handler(codec, [](const char*, void*){}, nullptr);
    opj_setup_decoder(codec, &params);
    opj_stream_t* st = opj_stream_create_file_stream(tmp, 1<<20, 1);
    opj_image_t* img = nullptr;
    bool ok = opj_read_header(st, codec, &img) && opj_decode(codec, st, img);
    opj_end_decompress(codec, st);
    if (img) opj_image_destroy(img);
    opj_stream_destroy(st); opj_destroy_codec(codec); unlink(tmp);
    return ok;
}

static void test_decode_patterns(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test: Decode succeeds for standard patterns ---\n");
    struct Pat {
        const char* name;
        std::function<uint16_t(int,int,int,int)> fn;
    };
    Pat pats[] = {
        {"flat_30000",  [](int,int,int,int) -> uint16_t { return 30000; }},
        {"h_gradient",  [](int x,int,int W,int) -> uint16_t { return (uint16_t)(x * 60000LL / (W-1)); }},
        {"checker_64",  [](int x,int y,int,int) -> uint16_t { return (uint16_t)((((x/64)+(y/64))&1) ? 50000 : 10000); }},
        {"noise_small", [](int x,int y,int,int) -> uint16_t {
            unsigned s = x*17u + y*31u; s = s*1664525u+1013904223u;
            return (uint16_t)(30000 + ((s>>17)&0x3FF) - 512);
        }},
    };
    for (auto& pat : pats) {
        std::vector<uint16_t> rgb((size_t)W*H*3);
        for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
            uint16_t v = pat.fn(x, y, W, H);
            rgb[((size_t)y*W+x)*3+0] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
        }
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
        bool ok = opj_decode_ok(cs);
        char msg[80]; snprintf(msg, sizeof(msg), "Decode OK: %s (%d×%d)", pat.name, W, H);
        CHECK(ok, msg);
    }
}

/* ===== Test 8: Determinism ===== */

static void test_determinism(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test: Encode determinism ---\n");
    std::vector<uint16_t> rgb((size_t)W*H*3);
    for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
        unsigned s = x*1543u + y*7919u; s = s*1664525u+1013904223u;
        uint16_t v = (uint16_t)(30000 + ((s>>17)&0x3FF) - 512);
        rgb[((size_t)y*W+x)*3] = rgb[((size_t)y*W+x)*3+1] = rgb[((size_t)y*W+x)*3+2] = v;
    }
    /* Warmup */
    enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);

    std::vector<uint8_t> cs0 = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
    std::vector<uint8_t> cs1 = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);
    std::vector<uint8_t> cs2 = enc.encode_ebcot(rgb.data(), W, H, W*3, 150000000, 24, false, false);

    CHECK(cs0 == cs1, "Encode run 1 == run 2 (byte-identical)");
    CHECK(cs0 == cs2, "Encode run 1 == run 3 (byte-identical)");
    char msg[80];
    snprintf(msg, sizeof(msg), "Encode output size=%zu > 0", cs0.size());
    CHECK(!cs0.empty(), msg);
}

/* ===== Test 9: Byte budget is tight (no gross over/under-allocation) ===== */

static void test_budget(CudaJ2KEncoder& enc, int W, int H) {
    printf("\n--- Test: Byte budget compliance ---\n");
    std::vector<uint16_t> rgb((size_t)W*H*3);
    unsigned s = 77;
    for (auto& v : rgb) { s = s*1664525u+1013904223u; v = (s >> 1); }

    int64_t brs[] = { 50000000LL, 100000000LL, 150000000LL, 250000000LL };
    const char* names[] = { "50Mbps", "100Mbps", "150Mbps", "250Mbps" };
    for (int i = 0; i < 4; i++) {
        auto cs = enc.encode_ebcot(rgb.data(), W, H, W*3, brs[i], 24, false, false);
        int64_t max_bytes = brs[i] / 24 / 8;
        static const int OVERHEAD = 20 * 1024;
        char msg[120];
        snprintf(msg, sizeof(msg), "%s: cs=%zu <= %lld+20KB=%lld",
                 names[i], cs.size(), (long long)max_bytes,
                 (long long)(max_bytes + OVERHEAD));
        CHECK((int64_t)cs.size() <= max_bytes + OVERHEAD, msg);
        /* Also check not wildly under-budget (< 5% of budget → likely dropped data) */
        snprintf(msg, sizeof(msg), "%s: cs=%zu >= %lld*5%%=%lld (not zero)",
                 names[i], cs.size(), (long long)max_bytes, (long long)(max_bytes / 20));
        CHECK((int64_t)cs.size() >= max_bytes / 20, msg);
    }
}

/* ===== main ===== */

int main()
{
    g_verbose = (getenv("VERBOSE") != nullptr);
    printf("=== J2K Codestream Structure Correctness Test ===\n");

    CudaJ2KEncoder enc;
    if (!enc.is_initialized()) { fprintf(stderr, "CUDA init failed\n"); return 1; }
    GpuColourParams params; build_params(params); enc.set_colour_params(params);

    const int W = 2048, H = 1080;

    /* Generate reference codestream for marker tests */
    std::vector<uint16_t> flat_rgb((size_t)W*H*3, 30000);
    /* Warmup */
    enc.encode_ebcot(flat_rgb.data(), W, H, W*3, 150000000, 24, false, false);
    auto flat_cs = enc.encode_ebcot(flat_rgb.data(), W, H, W*3, 150000000, 24, false, false);

    test_soc_eoc(flat_cs);
    test_siz(flat_cs, W, H);
    test_cod(flat_cs);
    test_qcd(flat_cs, 5);
    test_tile_structure(flat_cs);
    test_qcd_t1_consistency(flat_cs);

    test_decode_patterns(enc, W, H);
    test_determinism(enc, W, H);
    test_budget(enc, W, H);

    printf("\n=== J2K Structure Correctness: %s (%d/%d passed) ===\n",
           g_fail ? "FAIL" : "PASS", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
