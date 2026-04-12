/*
 * EBCOT Tier-2 Packet Assembly + J2K Codestream Builder
 * CPU-side code for assembling EBCOT T1-coded data into a valid J2K codestream.
 *
 * Called after GPU EBCOT T1 kernel produces coded bytes per code-block.
 * Assembles packets in CPRL progression order with single quality layer.
 */

#ifndef GPU_EBCOT_T2_H
#define GPU_EBCOT_T2_H

#include "gpu_ebcot.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>

/* J2K marker constants */
static constexpr uint16_t J2K_SOC_M = 0xFF4F;
static constexpr uint16_t J2K_SIZ_M = 0xFF51;
static constexpr uint16_t J2K_COD_M = 0xFF52;
static constexpr uint16_t J2K_QCD_M = 0xFF5C;
static constexpr uint16_t J2K_QCC_M = 0xFF5D;
static constexpr uint16_t J2K_TLM_M = 0xFF55;
static constexpr uint16_t J2K_SOT_M = 0xFF90;
static constexpr uint16_t J2K_SOD_M = 0xFF93;
static constexpr uint16_t J2K_EOC_M = 0xFFD9;


/* ===== Subband / Resolution / Code-block Geometry ===== */

struct SubbandGeom {
    int x0, y0;          /* top-left in DWT coefficient array */
    int width, height;   /* subband dimensions */
    int type;            /* SUBBAND_LL/HL/LH/HH */
    int level;           /* DWT level (0 = finest) */
    int res;             /* resolution level (num_levels - level for detail, num_levels for LL) */
    float step;          /* quantization step */
    int cb_x0;           /* first code-block column index */
    int cb_y0;           /* first code-block row index */
    int ncbx, ncby;      /* number of code-blocks in x and y */
    int cb_start_idx;    /* index of first code-block for this subband in the global CB array */
};


/* Build the code-block info table and subband geometry for a given image size.
 * Returns: vector of CodeBlockInfo entries (one per code-block), vector of SubbandGeom.
 * Code-blocks are ordered: for each subband (in resolution order), row-major within subband. */
inline void build_codeblock_table(
    int width, int height, int stride, int num_levels, float base_step, bool is_4k,
    std::vector<CodeBlockInfo>& cb_infos,
    std::vector<SubbandGeom>& subbands)
{
    cb_infos.clear();
    subbands.clear();

    /* Compute subband dimensions at each level */
    int w[8], h[8]; /* w[i], h[i] = dimensions at level i (0 = original) */
    w[0] = width;
    h[0] = height;
    for (int l = 1; l <= num_levels; l++) {
        w[l] = (w[l-1] + 1) / 2;
        h[l] = (h[l-1] + 1) / 2;
    }

    /* Perceptual step weights per level (matching V53 kernel_quantize_subband_ml) */
    /* LL5: 0.65, L5-AC: 0.85, L4: 0.95, L3: 1.05, L2: 1.12, L1: 1.20 */
    float level_weight[8] = {1.20f, 1.12f, 1.05f, 0.95f, 0.85f, 0.65f, 0.65f, 0.65f};

    /* LL subband (highest level) */
    {
        int ll_w = w[num_levels], ll_h = h[num_levels];
        float step = base_step * level_weight[num_levels];
        SubbandGeom sg;
        sg.x0 = 0; sg.y0 = 0;
        sg.width = ll_w; sg.height = ll_h;
        sg.type = SUBBAND_LL; sg.level = num_levels; sg.res = 0;
        sg.step = step;
        sg.ncbx = (ll_w + CB_DIM - 1) / CB_DIM;
        sg.ncby = (ll_h + CB_DIM - 1) / CB_DIM;
        sg.cb_start_idx = static_cast<int>(cb_infos.size());
        sg.cb_x0 = 0; sg.cb_y0 = 0;
        for (int cby = 0; cby < sg.ncby; cby++) {
            for (int cbx = 0; cbx < sg.ncbx; cbx++) {
                CodeBlockInfo cbi;
                cbi.x0 = static_cast<int16_t>(cbx * CB_DIM);
                cbi.y0 = static_cast<int16_t>(cby * CB_DIM);
                cbi.width  = static_cast<int16_t>(std::min(CB_DIM, ll_w - cbx * CB_DIM));
                cbi.height = static_cast<int16_t>(std::min(CB_DIM, ll_h - cby * CB_DIM));
                cbi.subband_type = SUBBAND_LL;
                cbi.level = static_cast<uint8_t>(num_levels);
                cbi.quant_step = step;
                cb_infos.push_back(cbi);
            }
        }
        subbands.push_back(sg);
    }

    /* Detail subbands, from highest level (coarsest) to level 1 (finest) */
    for (int l = num_levels; l >= 1; l--) {
        /* HL subband: rows [0..h[l]), cols [w[l]..w[l-1]) */
        /* LH subband: rows [h[l]..h[l-1]), cols [0..w[l]) */
        /* HH subband: rows [h[l]..h[l-1]), cols [w[l]..w[l-1]) */
        struct { int x0, y0, sw, sh, type; } defs[3] = {
            { w[l], 0,    w[l-1] - w[l], h[l],           SUBBAND_HL },
            { 0,    h[l], w[l],          h[l-1] - h[l],  SUBBAND_LH },
            { w[l], h[l], w[l-1] - w[l], h[l-1] - h[l],  SUBBAND_HH }
        };
        float weight = level_weight[std::min(l - 1, 5)];
        float step = base_step * weight;

        for (int s = 0; s < 3; s++) {
            SubbandGeom sg;
            sg.x0 = defs[s].x0; sg.y0 = defs[s].y0;
            sg.width = defs[s].sw; sg.height = defs[s].sh;
            sg.type = defs[s].type; sg.level = static_cast<uint8_t>(l - 1);
            sg.res = num_levels - l + 1;
            sg.step = step;
            sg.ncbx = (sg.width + CB_DIM - 1) / CB_DIM;
            sg.ncby = (sg.height + CB_DIM - 1) / CB_DIM;
            sg.cb_start_idx = static_cast<int>(cb_infos.size());
            sg.cb_x0 = 0; sg.cb_y0 = 0;
            for (int cby = 0; cby < sg.ncby; cby++) {
                for (int cbx = 0; cbx < sg.ncbx; cbx++) {
                    CodeBlockInfo cbi;
                    cbi.x0 = static_cast<int16_t>(sg.x0 + cbx * CB_DIM);
                    cbi.y0 = static_cast<int16_t>(sg.y0 + cby * CB_DIM);
                    cbi.width  = static_cast<int16_t>(std::min(CB_DIM, sg.width - cbx * CB_DIM));
                    cbi.height = static_cast<int16_t>(std::min(CB_DIM, sg.height - cby * CB_DIM));
                    cbi.subband_type = static_cast<uint8_t>(defs[s].type);
                    cbi.level = static_cast<uint8_t>(l - 1);
                    cbi.quant_step = step;
                    cb_infos.push_back(cbi);
                }
            }
            subbands.push_back(sg);
        }
    }
}


/* ===== Packet Header Encoding ===== */

/* Write a packet for a single subband's code-blocks.
 * For 1-layer CPRL with inclusion=1 for all, this is simplified:
 * - Empty-packet flag (1 bit = 1 for non-empty)
 * - For each code-block: inclusion (1 bit), zero bit-planes (tag tree), npasses, lengths
 */
struct BitWriter {
    std::vector<uint8_t>& buf;
    int bit_pos;
    uint8_t cur_byte;

    BitWriter(std::vector<uint8_t>& b) : buf(b), bit_pos(7), cur_byte(0) {}

    void write_bit(int b) {
        cur_byte |= ((b & 1) << bit_pos);
        bit_pos--;
        if (bit_pos < 0) {
            buf.push_back(cur_byte);
            cur_byte = 0;
            bit_pos = 7;
        }
    }

    void write_bits(int val, int nbits) {
        for (int i = nbits - 1; i >= 0; i--)
            write_bit((val >> i) & 1);
    }

    void flush() {
        if (bit_pos < 7)
            buf.push_back(cur_byte);
    }
};


/* Build a complete J2K codestream from EBCOT T1 coded data.
 * This is a simplified single-tile, single-layer, CPRL implementation. */
inline std::vector<uint8_t> build_ebcot_codestream(
    int width, int height, bool is_4k, bool is_3d,
    int num_levels, float base_step,
    const std::vector<SubbandGeom>& subbands,
    const uint8_t*  coded_data[3],   /* coded bytes per CB, per component */
    const uint16_t* coded_len[3],    /* actual length per CB */
    const uint8_t*  num_passes[3],   /* passes per CB */
    const uint16_t* pass_lengths[3], /* cumulative pass lengths per CB */
    int64_t target_bytes)
{
    std::vector<uint8_t> cs;
    cs.reserve(target_bytes > 0 ? target_bytes + 1024 : 1024*1024);

    auto w16 = [&](uint16_t v) { cs.push_back(v >> 8); cs.push_back(v & 0xFF); };
    auto w32 = [&](uint32_t v) { cs.push_back((v>>24)&0xFF); cs.push_back((v>>16)&0xFF);
                                  cs.push_back((v>>8)&0xFF);  cs.push_back(v&0xFF); };
    auto w8  = [&](uint8_t v)  { cs.push_back(v); };

    /* SOC */
    w16(J2K_SOC_M);

    /* SIZ */
    w16(J2K_SIZ_M);
    w16(2 + 2 + 32 + 2 + 3*3);
    w16(is_4k ? 0x0004 : 0x0003);
    w32(width); w32(height);
    w32(0); w32(0);
    w32(width); w32(height);
    w32(0); w32(0);
    w16(3);
    for (int c = 0; c < 3; c++) { w8(11); w8(1); w8(1); }

    /* COD */
    int num_precincts = num_levels + 1;
    w16(J2K_COD_M);
    w16(2 + 1 + 4 + 5 + num_precincts);
    w8(0x01); /* Scod: precinct partition */
    w8(0x04); /* CPRL progression */
    w16(1);   /* 1 quality layer */
    w8(1);    /* MCT=1 */
    w8(static_cast<uint8_t>(num_levels));
    w8(4);    /* code-block width exponent - 2 (2^(4+2) = 64? No, 2^(4+2)=64. DCI uses 32=2^5, so 5-2=3) */
    /* Actually DCI uses 32×32 code-blocks: xcb'=3, ycb'=3 (2^(3+2)=32) */
    cs.pop_back(); /* remove the 4 */
    w8(3);    /* xcb' = 3 → code-block width = 32 */
    w8(3);    /* ycb' = 3 → code-block height = 32 */
    w8(0x00); /* no bypass/reset/terminate */
    w8(0x00); /* filter = 0 (9/7 irreversible) */
    w8(0x77); /* precinct LL: 128×128 */
    for (int i = 1; i < num_precincts; i++) w8(0x88); /* precinct others: 256×256 */

    /* QCD — per-subband step sizes (matching T1 quantization) */
    {
        int nsb = 3 * num_levels + 1;
        w16(J2K_QCD_M);
        w16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        w8(0x22); /* scalar expounded, 1 guard bit */
        /* Encode step for each subband in standard order */
        /* LL, then for each level (coarsest→finest): HL, LH, HH */
        for (int i = 0; i < nsb; i++) {
            float step_val = (i < static_cast<int>(subbands.size())) ? subbands[i].step : base_step;
            /* Encode as (eps<<11)|man per ITU-T T.800 A.6.4 */
            int log2s = static_cast<int>(std::floor(std::log2(std::max(step_val, 0.001f))));
            int eps = 13 - log2s;
            float denom = std::ldexp(1.0f, 13 - eps);
            int man = static_cast<int>((step_val / denom - 1.0f) * 2048.0f);
            man = std::max(0, std::min(2047, man));
            w16(static_cast<uint16_t>((eps << 11) | man));
        }
    }

    /* QCC for components 0 and 2 (X/Z use 1.1× coarser step) */
    {
        int nsb = 3 * num_levels + 1;
        for (int comp : {0, 2}) {
            w16(J2K_QCC_M);
            w16(static_cast<uint16_t>(4 + 2 * nsb));
            w8(static_cast<uint8_t>(comp));
            w8(0x22);
            for (int i = 0; i < nsb; i++) {
                float step_val = ((i < static_cast<int>(subbands.size())) ? subbands[i].step : base_step) * 1.1f;
                int log2s = static_cast<int>(std::floor(std::log2(std::max(step_val, 0.001f))));
                int eps = 13 - log2s;
                float denom = std::ldexp(1.0f, 13 - eps);
                int man = static_cast<int>((step_val / denom - 1.0f) * 2048.0f);
                man = std::max(0, std::min(2047, man));
                w16(static_cast<uint16_t>((eps << 11) | man));
            }
        }
    }

    /* Compute tile-part data for all 3 components */
    /* For single-tile single-layer, all code-blocks are included in one tile-part. */
    /* Build SOD payload: packet headers + coded data */

    /* For CPRL order with 1 layer, packets are ordered:
     * component 0, res 0; component 0, res 1; ...; component 1, res 0; ... etc
     * But single-tile means one tile-part with all packets. */

    /* Actually for single tile-part, the order is simpler. Let's build the payload. */
    std::vector<uint8_t> sod_payload;

    for (int comp = 0; comp < 3; comp++) {
        for (int res = 0; res <= num_levels; res++) {
            /* Find subbands at this resolution level for this component */
            /* Resolution 0 = LL; resolution r>0 = HL_r + LH_r + HH_r */

            /* Packet header: non-empty flag (1 bit) */
            std::vector<uint8_t> pkt_header_buf;
            BitWriter bw(pkt_header_buf);
            bw.write_bit(1); /* non-empty packet */

            std::vector<uint8_t> pkt_body;

            for (size_t sb = 0; sb < subbands.size(); sb++) {
                if (subbands[sb].res != res) continue;

                int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
                int cb_start = subbands[sb].cb_start_idx;

                for (int cbi = 0; cbi < ncbs; cbi++) {
                    int cb_idx = cb_start + cbi;
                    uint16_t len = coded_len[comp][cb_idx];
                    uint8_t  np  = num_passes[comp][cb_idx];

                    if (np == 0 || len == 0) {
                        /* Code-block not included: tag tree inclusion = 0 */
                        bw.write_bit(0); /* not included in this layer */
                        continue;
                    }

                    /* Inclusion: for layer 0, first time = 1 */
                    bw.write_bit(1);

                    /* Zero bit-planes: encode as number of insignificant bit-planes.
                     * For simplicity, encode 0 (all bit-planes coded).
                     * The actual number is num_bp_max - num_bp_coded.
                     * Since we code all bit-planes, zero_bplanes = 0. */
                    bw.write_bit(0); /* zero bit-planes = 0 (tag tree with value 0) */

                    /* Number of coding passes */
                    if (np == 1)      { bw.write_bit(0); }
                    else if (np == 2) { bw.write_bits(2, 2); }
                    else if (np <= 5) { bw.write_bits(0xC | (np - 3), 4); }
                    else if (np <= 36){ bw.write_bits(0x1E0 | (np - 6), 9); }
                    else              { bw.write_bits(0xFF80 | (np - 37), 16); }

                    /* Code-block data length: LBLOCK encoding.
                     * Use LBLOCK = 3 (fixed) for simplicity.
                     * Length = total bytes for all passes. */
                    int lblock = 3;
                    /* Number of bits needed to represent length */
                    int len_bits = lblock + static_cast<int>(std::ceil(std::log2(std::max(1, static_cast<int>(np)))));
                    len_bits = std::max(len_bits, 1);
                    /* Increment lblock if length doesn't fit */
                    while ((1 << len_bits) <= len) {
                        bw.write_bit(1); /* increment LBLOCK */
                        lblock++;
                        len_bits++;
                    }
                    bw.write_bit(0); /* end of LBLOCK increment */
                    bw.write_bits(len, len_bits);

                    /* Append code-block data to packet body */
                    pkt_body.insert(pkt_body.end(),
                                    coded_data[comp] + (size_t)cb_idx * CB_BUF_SIZE,
                                    coded_data[comp] + (size_t)cb_idx * CB_BUF_SIZE + len);
                }
            }

            bw.flush();
            sod_payload.insert(sod_payload.end(), pkt_header_buf.begin(), pkt_header_buf.end());
            sod_payload.insert(sod_payload.end(), pkt_body.begin(), pkt_body.end());
        }
    }

    /* TLM — tile length marker */
    size_t tile_data_size = sod_payload.size();
    size_t tile_part_size = 12 + tile_data_size; /* SOT(12) + SOD(2) + data - actually SOT marker is 2+10=12, SOD marker is 2 */

    w16(J2K_TLM_M);
    w16(2 + 1 + 1 + 4);
    w8(0); w8(0x40);
    w32(static_cast<uint32_t>(tile_part_size + 2)); /* +2 for SOD marker */

    /* SOT */
    w16(J2K_SOT_M);
    w16(10);
    w16(0); /* tile index */
    w32(static_cast<uint32_t>(tile_part_size + 2)); /* tile-part length including SOT+SOD */
    w8(0);  /* tile-part index */
    w8(1);  /* number of tile-parts */

    /* SOD */
    w16(J2K_SOD_M);
    cs.insert(cs.end(), sod_payload.begin(), sod_payload.end());

    /* EOC */
    w16(J2K_EOC_M);

    /* Pad to minimum DCI size if needed */
    while (cs.size() < 16384) cs.push_back(0);

    return cs;
}


#endif /* GPU_EBCOT_T2_H */
