/*
 * EBCOT Tier-2 Packet Assembly + J2K Codestream Builder
 * CPU-side code for assembling EBCOT T1-coded data into a valid J2K codestream.
 *
 * Called after GPU EBCOT T1 kernel produces coded bytes per code-block.
 * V181: LRCP progression order (Layer-Resolution-Component-Position).
 * Packets interleaved as: for each res r, write packets for comp 0, 1, 2.
 * This avoids CPRL OpenJPEG quirk where all 18 packets were assigned to comp 0
 * (decoded comp 0 std=1337 vs expected 763; comp 1/2 nearly all-2048).
 */

#ifndef GPU_EBCOT_T2_H
#define GPU_EBCOT_T2_H

#include "gpu_ebcot.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <future>

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
            sg.step = step;  /* QCD writes this; T1 quantizes with step * inv_dwt_gain. */
            sg.ncbx = (sg.width + CB_DIM - 1) / CB_DIM;
            sg.ncby = (sg.height + CB_DIM - 1) / CB_DIM;
            sg.cb_start_idx = static_cast<int>(cb_infos.size());
            sg.cb_x0 = 0; sg.cb_y0 = 0;
            /* V185: irreversible 9/7 inverse DWT amplifies HL/LH by 2x and HH by 4x
             * (OPJ uses two_invK = 2/K instead of invK; see opj dwt.c "BUG_WEIRD_TWO_INVK").
             * To compensate, T1 must quantize with step × inv_dwt_gain so dequantized
             * coefficients are pre-scaled smaller; the inverse DWT then amplifies them
             * back to the right magnitude.  QCD writes the unscaled `step`. */
            float t1_step = step;
            if (defs[s].type == SUBBAND_HL || defs[s].type == SUBBAND_LH) t1_step = step * 2.0f;
            else if (defs[s].type == SUBBAND_HH) t1_step = step * 4.0f;
            for (int cby = 0; cby < sg.ncby; cby++) {
                for (int cbx = 0; cbx < sg.ncbx; cbx++) {
                    CodeBlockInfo cbi;
                    cbi.x0 = static_cast<int16_t>(sg.x0 + cbx * CB_DIM);
                    cbi.y0 = static_cast<int16_t>(sg.y0 + cby * CB_DIM);
                    cbi.width  = static_cast<int16_t>(std::min(CB_DIM, sg.width - cbx * CB_DIM));
                    cbi.height = static_cast<int16_t>(std::min(CB_DIM, sg.height - cby * CB_DIM));
                    cbi.subband_type = static_cast<uint8_t>(defs[s].type);
                    cbi.level = static_cast<uint8_t>(l - 1);
                    cbi.quant_step = t1_step;
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
/* V144: batched MSB-first bit writer.
 * J2K-compatible byte stuffing (T.800 B.10.1 / OpenJPEG bio approach):
 *   After emitting a byte equal to 0xFF, only 7 bits are available in the
 *   next byte (bit 7 is forced to 0 as a stuffing marker).  The decoder
 *   (OpenJPEG bio) checks whether the previous fetched byte was 0xFF and
 *   uses ct=7 instead of ct=8, reading only bits [6:0] from the next byte.
 *
 *   prev_ff tracks whether the last flushed byte was 0xFF.  When true,
 *   flush_byte() emits only 7 bits of acc (MSB first in bits [6:0]), leaving
 *   bit 7 = 0.  This matches OpenJPEG bio exactly and removes the need for
 *   post-pass sanitize (which was incorrectly discarding real data bits). */
struct BitWriter {
    std::vector<uint8_t>& buf;
    uint64_t acc;    /* pending bits, MSB = oldest */
    int      acc_n;  /* number of valid bits in acc */
    bool     prev_ff; /* was the last flushed byte 0xFF? */

    BitWriter(std::vector<uint8_t>& b) : buf(b), acc(0), acc_n(0), prev_ff(false) {}

    void flush_byte() {
        /* After 0xFF, only 7 bits fit (bit 7 must be 0). */
        int bits = prev_ff ? 7 : 8;
        acc_n -= bits;
        uint8_t byte = static_cast<uint8_t>((acc >> acc_n) & (prev_ff ? 0x7Fu : 0xFFu));
        buf.push_back(byte);
        prev_ff = (byte == 0xFF);
    }

    void write_bits(uint32_t val, int nbits) {
        if (nbits > 0 && nbits < 32) val &= (1u << nbits) - 1u;
        acc = (acc << nbits) | static_cast<uint64_t>(val);
        acc_n += nbits;
        while (acc_n >= (prev_ff ? 7 : 8)) flush_byte();
    }

    void write_bit(int b) { write_bits(static_cast<uint32_t>(b & 1), 1); }

    void flush() {
        if (acc_n > 0) {
            int bits = prev_ff ? 7 : 8;
            uint8_t byte = static_cast<uint8_t>((acc << (bits - acc_n)) & (prev_ff ? 0x7Fu : 0xFFu));
            buf.push_back(byte);
            prev_ff = (byte == 0xFF);
            acc_n = 0;
            acc = 0;
        }
    }
};


/* ===== Hierarchical Tag Tree (ITU-T T.800 B.10.2) ===== */
/*
 * J2K tag tree for encoding inclusion and zero-bit-planes (ZBP) per-precinct.
 * Leaf count = ncbx × ncby.  Internal nodes organised as a quadtree:
 *   level 0 = leaves (finest), level L = root (1×1).
 * Each node stores value = min(children), low = highest threshold
 * already emitted, known = final 1-bit has been sent.
 * encode() matches opj_tgt_encode: walks root→leaf, emits new bits only.
 */
struct TagTree {
    struct Node {
        int  value;   /* min value of subtree */
        int  low;     /* highest threshold already emitted */
        bool known;   /* final 1-bit sent; skip on re-entry */
        int  parent;  /* index of parent, -1 for root */
    };

    std::vector<Node> nodes;
    std::vector<int>  level_off; /* nodes[level_off[lv]..] = level lv */
    std::vector<int>  level_w;
    int               num_leaves = 0;

    void build(int ncbx, int ncby) {
        nodes.clear(); level_off.clear(); level_w.clear();
        int w = ncbx, h = ncby, total = 0;
        std::vector<std::pair<int,int>> dims;
        while (true) {
            dims.push_back({w, h});
            if (w == 1 && h == 1) break;
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        int nlv = (int)dims.size();
        for (int lv = 0; lv < nlv; lv++) {
            level_off.push_back(total); level_w.push_back(dims[lv].first);
            total += dims[lv].first * dims[lv].second;
        }
        nodes.resize(total);
        num_leaves = dims[0].first * dims[0].second;
        for (int lv = 0; lv < nlv; lv++) {
            int wlv = dims[lv].first, hlv = dims[lv].second, off = level_off[lv];
            for (int j = 0; j < hlv; j++) for (int i = 0; i < wlv; i++) {
                Node& n = nodes[off + j * wlv + i];
                n.value = 0x7FFFFFFF; n.low = 0; n.known = false;
                n.parent = (lv + 1 < nlv)
                    ? level_off[lv+1] + (j/2) * dims[lv+1].first + (i/2)
                    : -1;
            }
        }
    }

    /* Set leaf value and propagate minimum upward. */
    void set_leaf(int leaf_idx, int value) {
        nodes[leaf_idx].value = value;
        int idx = leaf_idx;
        while (nodes[idx].parent != -1) {
            int par = nodes[idx].parent;
            if (value < nodes[par].value) { nodes[par].value = value; idx = par; }
            else break;
        }
    }

    /* Encode leaf_idx with threshold (matches opj_tgt_encode). */
    void encode(BitWriter& bw, int leaf_idx, int threshold) {
        /* Collect path from leaf to root */
        int path[32], plen = 0;
        for (int idx = leaf_idx; idx != -1; idx = nodes[idx].parent)
            path[plen++] = idx;
        /* Process root → leaf */
        int low = 0;
        for (int pi = plen - 1; pi >= 0; pi--) {
            Node& node = nodes[path[pi]];
            if (low > node.low) node.low = low; else low = node.low;
            while (low < threshold) {
                if (low >= node.value) {
                    if (!node.known) { bw.write_bit(1); node.known = true; }
                    break;
                }
                bw.write_bit(0);
                ++low;
            }
            node.low = low;
        }
    }
};


/* Build a complete J2K codestream from EBCOT T1 coded data.
 * This is a simplified single-tile, single-layer, LRCP implementation. */
inline std::vector<uint8_t> build_ebcot_codestream(
    int width, int height, bool is_4k, bool is_3d,
    int num_levels, float base_step,
    const std::vector<SubbandGeom>& subbands,
    const uint8_t*  coded_data[3],   /* coded bytes per CB, per component */
    const uint16_t* coded_len[3],    /* actual length per CB */
    const uint8_t*  num_passes[3],   /* passes per CB */
    const uint16_t* pass_lengths[3], /* cumulative pass lengths per CB */
    const uint8_t*  cb_num_bp[3],    /* coded bit-planes per CB (from T1 kernel) */
    int64_t target_bytes,
    int cb_stride = CB_BUF_SIZE)     /* V137: row stride of coded_data (bytes per CB) */
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

    /* COD — V179: Scod=0x00 (no precinct partition, default 2^15×2^15 precincts).
     * Our build_tp writes one packet per resolution covering all codeblocks.
     * With Scod=0x01 (explicit precincts, 256×256), HL1 (1024×540) required
     * 4×3=12 packets per resolution but we wrote only 1, causing OpenJPEG to
     * read packet bodies as headers and report "segment too long" → ~12 dB PSNR.
     * With Scod=0x00, each resolution is one precinct → one packet per resolution. */
    w16(J2K_COD_M);
    w16(2 + 1 + 4 + 5);  /* no precinct partition bytes */
    w8(0x00); /* Scod=0: no precinct partition → default 2^15×2^15 per resolution */
    w8(0x00); /* LRCP progression (V181: was CPRL=0x04; CPRL assigned all packets to comp 0) */
    w16(1);   /* 1 quality layer */
    w8(0);    /* MCT=0: XYZ stored directly, no ICT component transform */
    w8(static_cast<uint8_t>(num_levels));
    w8(3);    /* xcb' = 3 → code-block width = 32 */
    w8(3);    /* ycb' = 3 → code-block height = 32 */
    w8(0x00); /* SPcod: BYPASS bit=0.  Re-enabling produced decode failures. */
    w8(0x00); /* filter = 0 (9/7 irreversible) */

    /* QCD — per-subband step sizes (matching T1 quantization) */
    {
        int nsb = 3 * num_levels + 1;
        w16(J2K_QCD_M);
        w16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        w8(0x42); /* scalar expounded, 2 guard bits (SMPTE 422M DCP profile) */
        /* Encode step for each subband in standard order */
        /* LL, then for each level (coarsest→finest): HL, LH, HH */
        for (int i = 0; i < nsb; i++) {
            float step_val = (i < static_cast<int>(subbands.size())) ? subbands[i].step : base_step;
            /* Encode as (eps<<11)|man per ITU-T T.800 A.6.4.
             * OPJ decoder (tcd.c): stepsize = (1+man/2048)*2^(Rb-eps), Rb=prec=12 (irreversible).
             * Set eps=12-log2s so stepsize = (1+man/2048)*2^log2s = step_val. */
            int log2s = static_cast<int>(std::floor(std::log2(std::max(step_val, 0.001f))));
            int eps = 12 - log2s;
            float denom = std::ldexp(1.0f, log2s);  /* = 2^log2s (Rb-eps = 12-(12-log2s) = log2s) */
            int man = static_cast<int>((step_val / denom - 1.0f) * 2048.0f);
            man = std::max(0, std::min(2047, man));
            w16(static_cast<uint16_t>((eps << 11) | man));
        }
    }

    /* V182: QCC removed — T1 uses base_step uniformly for all components.
     * Adding QCC with step×1.1 while T1 uses base_step causes a decoder amplitude
     * mismatch (decoder dequantizes with the wrong step for comp 0/2). */

    /* V182: Build per-component, per-resolution packet blobs.
     * LRCP order: for each resolution r, write packets for comp 0, 1, 2.
     * Simple rate control: limit each component to target_bytes/3. */
    /* pkt_by_res[comp][res] = packet bytes (header + body) for that (comp, res) pair */
    std::vector<std::vector<uint8_t>> pkt_by_res[3];
    for (int c = 0; c < 3; c++)
        pkt_by_res[c].resize(num_levels + 1);
    size_t per_comp_budget = (target_bytes > 0) ? static_cast<size_t>(target_bytes / 3) : SIZE_MAX;

    /* Compute per-subband p_max = band->numbps (OPJ tcd.c: eps+numgbits-1).
     * eps=12-log2s (to match OPJ Rb=12 stepsize formula), numgbits=2 (Sqcd=0x42):
     * band->numbps = (12-log2s) + 2 - 1 = 13-log2s = pmax.
     * ZBP z = pmax-nb → cblk->numbps = pmax-(pmax-nb) = nb. Correct. */
    auto sb_pmax_for_comp = [&](size_t sb, int /*comp*/) -> int {
        /* pmax = band->numbps = 13-log2(step). */
        float step = std::max(subbands[sb].step, 0.001f);
        int log2s = static_cast<int>(std::floor(std::log2f(step)));
        return 13 - log2s;
    };

    /* V189: per-component proportional truncation.  Compute total bytes for this
     * comp at full precision; if exceeds budget, scale every CB's len to fit by
     * truncating passes.  This gives smoother quality vs the previous all-or-
     * nothing CB inclusion (which dropped whole CBs once the budget filled). */
    float trunc_ratio[3] = {1.0f, 1.0f, 1.0f};
    if (target_bytes > 0) {
        for (int c = 0; c < 3; ++c) {
            size_t total = 0;
            int total_cbs = 0;
            for (size_t sb = 0; sb < subbands.size(); ++sb) {
                int cb_start = subbands[sb].cb_start_idx;
                int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
                for (int i = 0; i < ncbs; ++i) {
                    uint16_t len = coded_len[c][cb_start + i];
                    if (len > static_cast<uint16_t>(cb_stride - 1))
                        len = static_cast<uint16_t>(cb_stride - 1);
                    total += len;
                    if (num_passes[c][cb_start + i] > 0 && len > 0) ++total_cbs;
                }
            }
            size_t per_comp_target = target_bytes / 3;
            if (total > per_comp_target && total > 0)
                trunc_ratio[c] = static_cast<float>(per_comp_target) / static_cast<float>(total);
            (void)total_cbs;
        }
    }

    /* V181: build_tp fills pkt_by_res[comp][res] — one blob per (comp, res) pair.
     * Packets are then interleaved in LRCP order: for each res, comp 0, 1, 2.
     * sanitize is applied per-packet blob to ensure no 0xFF marker sequences. */
    auto build_tp = [&](int comp) {
        size_t comp_bytes = 0;
        const float ratio_c = trunc_ratio[comp];
        std::vector<uint8_t> pkt_header_buf;
        std::vector<uint8_t> pkt_body;
        pkt_header_buf.reserve(8192);
        pkt_body.reserve(512 * 1024);

        TagTree incl_tree, zbp_tree;

        for (int res = 0; res <= num_levels; res++) {
            pkt_header_buf.clear();
            pkt_body.clear();

            BitWriter bw(pkt_header_buf);
            bw.write_bit(1); /* non-empty packet */

            for (size_t sb = 0; sb < subbands.size(); sb++) {
                if (subbands[sb].res != res) continue;

                int ncbx  = subbands[sb].ncbx, ncby = subbands[sb].ncby;
                int ncbs  = ncbx * ncby;
                int cb_start = subbands[sb].cb_start_idx;
                int pmax  = sb_pmax_for_comp(sb, comp);
#ifdef GPU_J2K_DEBUG_NP
                if (comp == 0 && res <= 1) {
                    for (int cbi = 0; cbi < std::min(ncbs, 4); cbi++) {
                        int cb_idx = cb_start + cbi;
                        uint8_t np_v = num_passes[comp][cb_idx];
                        uint8_t nb_v = cb_num_bp ? cb_num_bp[comp][cb_idx] : 0;
                        uint16_t ln_v = coded_len[comp][cb_idx];
                        fprintf(stderr, "[T2_DBG] res=%d sb=%zu cbi=%d np=%d nb=%d len=%d pmax=%d z=%d\n",
                            res, sb, cbi, np_v, nb_v, ln_v, pmax, pmax - nb_v);
                    }
                }
#endif

                /* Build hierarchical tag trees for this subband.
                 * Flat 1-bit-per-CB encoding was wrong: for HL1 (32×17=544 CBs)
                 * the tree has 745 total nodes; the 201-bit discrepancy caused
                 * OpenJPEG to read ZBP/body bytes as inclusion bits, corrupting
                 * the entire AC packet stream and producing ~12 dB PSNR. */
                incl_tree.build(ncbx, ncby);
                zbp_tree.build(ncbx, ncby);

                /* Pre-scan: set leaf values before encoding any bits.
                 * Excluded CBs: incl=MAX (never included), zbp=pmax (all zero BPs).
                 * V189: when full inclusion would exceed budget, truncate the CB's
                 * passes to fit the remaining budget; keep at least 1 pass when
                 * possible (better than excluding the CB outright). */
                std::vector<bool> included(ncbs, false);
                std::vector<uint16_t> cb_len_use(ncbs, 0);
                std::vector<uint8_t>  cb_np_use(ncbs, 0);
                for (int cbi = 0; cbi < ncbs; cbi++) {
                    int cb_idx = cb_start + cbi;
                    uint16_t len = coded_len[comp][cb_idx];
                    uint8_t  np  = num_passes[comp][cb_idx];
                    if (len > static_cast<uint16_t>(cb_stride - 1))
                        len = static_cast<uint16_t>(cb_stride - 1);
                    if (np == 0 || len == 0) {
                        incl_tree.set_leaf(cbi, 0x7FFFFFFF);
                        zbp_tree.set_leaf(cbi, pmax);
                        continue;
                    }
                    int np_use = np;
                    uint16_t len_use = len;
                    /* If full CB fits remaining budget, include it as-is. */
                    if (comp_bytes + len > per_comp_budget) {
                        /* Truncate passes to fit remaining budget. */
                        size_t remaining = (per_comp_budget > comp_bytes)
                            ? per_comp_budget - comp_bytes : 0;
                        const uint16_t* pl = pass_lengths[comp]
                            + (size_t)cb_idx * MAX_PASSES;
                        np_use = 0; len_use = 0;
                        for (int p = 0; p < np; ++p) {
                            if ((size_t)pl[p] > remaining) break;
                            np_use = p + 1;
                            len_use = pl[p];
                        }
                    }
                    if (np_use == 0 || len_use == 0) {
                        incl_tree.set_leaf(cbi, 0x7FFFFFFF);
                        zbp_tree.set_leaf(cbi, pmax);
                        continue;
                    }
                    included[cbi] = true;
                    comp_bytes += len_use;
                    cb_len_use[cbi] = len_use;
                    cb_np_use[cbi]  = static_cast<uint8_t>(np_use);
                    incl_tree.set_leaf(cbi, 0);
                    int nb = (cb_num_bp != nullptr) ? static_cast<int>(cb_num_bp[comp][cb_idx]) : 0;
                    int z  = pmax - nb;
                    if (z < 0) z = 0;
                    zbp_tree.set_leaf(cbi, z);
                }
                (void)ratio_c;

                /* Encode packet header and body for each CB in order. */
                for (int cbi = 0; cbi < ncbs; cbi++) {
                    int cb_idx = cb_start + cbi;

                    /* Inclusion tag tree (threshold = layer+1 = 1 for 1st layer). */
                    incl_tree.encode(bw, cbi, 1);
                    if (!included[cbi]) continue;

                    /* ZBP tag tree (threshold = pmax encodes exact z value). */
                    zbp_tree.encode(bw, cbi, pmax);

                    /* V189: use truncated np / len from pre-scan. */
                    uint8_t  np  = cb_np_use[cbi];
                    uint16_t len = cb_len_use[cbi];

                    /* Number of coding passes (ITU-T T.800 Table B.4 / OpenJPEG t2_putnumpasses). */
                    if (np == 1)       bw.write_bit(0);
                    else if (np == 2)  bw.write_bits(2, 2);
                    else if (np <= 5)  bw.write_bits(0xC | (np - 3), 4);
                    else if (np <= 36) bw.write_bits(0x1E0 | (np - 6), 9);
                    else               bw.write_bits(0xFF80u | (unsigned(np) - 37u), 16);

                    /* Code-block length: Lblock + floor(log2(np)) bits. */
                    int lblock = 3;
                    int floor_log2_np = (np <= 1) ? 0 : (31 - __builtin_clz(static_cast<unsigned>(np)));
                    int len_bits = lblock + floor_log2_np;
                    if (len_bits < 1) len_bits = 1;
                    while ((1 << len_bits) <= len) { bw.write_bit(1); lblock++; len_bits++; }
                    bw.write_bit(0);
                    bw.write_bits(len, len_bits);

                    const uint8_t* src = coded_data[comp] + (size_t)cb_idx * cb_stride + 1;
                    pkt_body.insert(pkt_body.end(), src, src + len);
                }
            }

            bw.flush();
            auto& dst = pkt_by_res[comp][res];
            dst.insert(dst.end(), pkt_header_buf.begin(), pkt_header_buf.end());
            dst.insert(dst.end(), pkt_body.begin(), pkt_body.end());
        }
    }; /* end lambda build_tp */

    /* V183: BitWriter now implements J2K-compatible byte stuffing (7 bits after
     * 0xFF), so no post-pass sanitize is needed.  build_tp is called directly. */
    auto fut0 = std::async(std::launch::async, [&]() { build_tp(0); });
    auto fut1 = std::async(std::launch::async, [&]() { build_tp(1); });
    build_tp(2);
    fut0.wait();
    fut1.wait();

    /* V181: Interleave packets in LRCP order: for each resolution, write comp 0, 1, 2.
     * This replaces CPRL (comp outer) which caused OpenJPEG to read all 18 packets
     * as component 0 data — leaving comp 1 and comp 2 all-zero (decoded as 2048). */
#ifdef GPU_J2K_DEBUG_PACKETS
    for (int r = 0; r <= num_levels; r++)
        for (int c = 0; c < 3; c++)
            fprintf(stderr, "DEBUG pkt r=%d c=%d size=%zu\n", r, c, pkt_by_res[c][r].size());
#endif
    size_t total_pkt = 0;
    for (int r = 0; r <= num_levels; r++)
        for (int c = 0; c < 3; c++)
            total_pkt += pkt_by_res[c][r].size();

    uint32_t single_tp_size = static_cast<uint32_t>(12 + 2 + total_pkt);

    /* TLM — single tile-part for tile 0 */
    w16(J2K_TLM_M);
    w16(static_cast<uint16_t>(2 + 1 + 1 + 1 * (1 + 4))); /* Ltlm: 1 entry */
    w8(0);    /* Ztlm = 0 */
    w8(0x50); /* Stlm: ST=1 (1-byte Ttlm), SP=1 (4-byte Ptlm) */
    w8(0);              /* Ttlm = tile 0 */
    w32(single_tp_size);

    /* Single tile-part: 18 packets in LRCP order (r=0: c0,c1,c2; r=1: c0,c1,c2; ...) */
    w16(J2K_SOT_M);
    w16(10);
    w16(0);   /* tile index = 0 */
    w32(single_tp_size);
    w8(0);    /* tile-part index = 0 */
    w8(1);    /* number of tile-parts = 1 */
    w16(J2K_SOD_M);
    for (int r = 0; r <= num_levels; r++)
        for (int c = 0; c < 3; c++)
            cs.insert(cs.end(), pkt_by_res[c][r].begin(), pkt_by_res[c][r].end());

    /* EOC — final marker.  V139 FIX: NEVER pad after EOC.  Trailing zeros are
     * illegal J2K and cause dcp::verify_j2k to throw "missing marker start
     * byte" on low-content frames (e.g. black/near-uniform first frames). */
    w16(J2K_EOC_M);

    return cs;
}


#endif /* GPU_EBCOT_T2_H */
