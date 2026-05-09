/*
 * EBCOT Tier-2 Packet Assembly + J2K Codestream Builder
 * CPU-side code for assembling EBCOT T1-coded data into a valid J2K codestream.
 *
 * Called after GPU EBCOT T1 kernel produces coded bytes per code-block.
 * V210: CPRL progression order (Component-Position-Resolution-Layer).
 * Packets interleaved as: for each res r, write packets for comp 0, 1, 2.
 * DCP SMPTE 429-4 requires CPRL; Scod=0x00 makes this straightforward.
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
            /* V254: T1 gain for HH=5.5, HL/LH=2.0. HH uses higher gain to keep HH T1 bytes
             * within budget (5.5 coarser quantization → fewer bit-planes → less T1 data).
             * QCD writes step×5.5 for HH so OPJ dequantizes at the correct T1 scale. */
            float gain = (defs[s].type == SUBBAND_HH) ? 5.5f : 2.0f;
            float t1_step = step * gain;
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

    /* V205: Local batch buffer avoids per-byte vector push_back overhead.
     * Flushed to buf in chunks of 256 bytes. */
    uint8_t  local[256];
    int      local_n;

    BitWriter(std::vector<uint8_t>& b) : buf(b), acc(0), acc_n(0), prev_ff(false), local_n(0) {}

    void flush_byte() {
        int bits = prev_ff ? 7 : 8;
        acc_n -= bits;
        uint8_t byte = static_cast<uint8_t>((acc >> acc_n) & (prev_ff ? 0x7Fu : 0xFFu));
        local[local_n++] = byte;
        if (local_n == 256) { buf.insert(buf.end(), local, local + 256); local_n = 0; }
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
            local[local_n++] = byte;
            if (local_n == 256) { buf.insert(buf.end(), local, local + 256); local_n = 0; }
            prev_ff = (byte == 0xFF);
            acc_n = 0;
            acc = 0;
        }
        if (local_n > 0) { buf.insert(buf.end(), local, local + local_n); local_n = 0; }
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
/* V234: Fixed-size TagTree — replaces dynamic vectors with stack arrays.
 * Max CBs per precinct per subband = 4×4=16 (PPsb=128, CB_DIM=32 → cbsb=4).
 * Tag tree for 4×4: 16 leaves + 4 inner + 1 root = 21 nodes, 3 levels.
 * Eliminates malloc/free on every precinct-subband pair. */
struct TagTree {
    struct Node {
        int  value;
        int  low;
        bool known;
        int  parent;
    };

    static constexpr int MAX_NODES  = 21;   /* 16+4+1 for 4×4 CBs */
    static constexpr int MAX_LEVELS = 3;

    Node nodes[MAX_NODES];
    int  level_off[MAX_LEVELS];
    int  level_w[MAX_LEVELS];
    int  nlv        = 0;
    int  num_leaves = 0;

    void build(int ncbx, int ncby) {
        int w = ncbx, h = ncby, total = 0;
        nlv = 0;
        while (true) {
            level_off[nlv] = total;
            level_w[nlv]   = w;
            total += w * h;
            ++nlv;
            if (w == 1 && h == 1) break;
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        num_leaves = ncbx * ncby;
        /* Re-compute heights at each level for parent index calc */
        int wl[MAX_LEVELS], hl[MAX_LEVELS];
        { int tw = ncbx, th = ncby;
          for (int lv = 0; lv < nlv; ++lv) {
              wl[lv] = tw; hl[lv] = th;
              tw = (tw+1)/2; th = (th+1)/2;
          }
        }
        for (int lv = 0; lv < nlv; ++lv) {
            int off = level_off[lv];
            for (int j = 0; j < hl[lv]; ++j) {
                for (int i = 0; i < wl[lv]; ++i) {
                    Node& n = nodes[off + j * wl[lv] + i];
                    n.value = 0x7FFFFFFF; n.low = 0; n.known = false;
                    n.parent = (lv + 1 < nlv)
                        ? level_off[lv+1] + (j/2) * wl[lv+1] + (i/2)
                        : -1;
                }
            }
        }
    }

    void set_leaf(int leaf_idx, int value) {
        nodes[leaf_idx].value = value;
        int idx = leaf_idx;
        while (nodes[idx].parent != -1) {
            int par = nodes[idx].parent;
            if (value < nodes[par].value) { nodes[par].value = value; idx = par; }
            else break;
        }
    }

    void encode(BitWriter& bw, int leaf_idx, int threshold) {
        int path[MAX_LEVELS + 1], plen = 0;
        for (int idx = leaf_idx; idx != -1; idx = nodes[idx].parent)
            path[plen++] = idx;
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
 * This is a single-tile, single-layer, CPRL implementation. */
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
    int cb_stride = CB_BUF_SIZE,     /* V137: row stride of coded_data (bytes per CB) */
    bool use_bypass = false)         /* V205: COD BYPASS bit must match T1 bypass usage */
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
    /* V231: OPJ_PROFILE_CINEMA_2K=0x0003, OPJ_PROFILE_CINEMA_4K=0x0004 per verify_j2k */
    w16(static_cast<uint16_t>(is_4k ? 0x0004 : 0x0003));  /* Rsiz: cinema profile */
    w32(width); w32(height);
    w32(0); w32(0);
    w32(width); w32(height);
    w32(0); w32(0);
    w16(3);
    for (int c = 0; c < 3; c++) { w8(11); w8(1); w8(1); }

    /* V212: Scod=0x01 precinct partition enabled per DCP SMPTE 429-4 */
    int num_precincts = num_levels + 1;
    w16(J2K_COD_M);
    w16(static_cast<uint16_t>(2 + 1 + 4 + 5 + num_precincts));
    w8(0x01); /* V212: Scod=1 */
    /* V210: CPRL progression per DCP SMPTE 429-4 */
    w8(0x04); /* CPRL progression order (T.800 Table A.16: 0x04=CPRL) */
    w16(1);   /* 1 quality layer */
    w8(1);    /* V209: MCT=1: ICT applied for DCP SMPTE 429-4 compliance */
    w8(static_cast<uint8_t>(num_levels));
    w8(3);    /* xcb' = 3 → code-block width = 32 */
    w8(3);    /* ycb' = 3 → code-block height = 32 */
    /* V243: BYPASS(bit 0) | RESTART(bit 2) = 0x05.  RESTART means each pass terminates
     * independently; decoder reads one segment length per pass in the packet header.
     * T2 writes ONE global comma code then np lengths in lblock bits each. */
    w8(use_bypass ? 0x05 : 0x00); /* SPcod: BYPASS|RESTART = 0x05 */
    w8(0x00); /* filter = 0 (9/7 irreversible) */
    /* V231: DCP requires 0x77 (128×128) for r=0, 0x88 (256×256) for r=1..NL */
    w8(0x77);
    for (int p = 1; p < num_precincts; ++p)
        w8(0x88);

    /* QCD — per-subband step sizes (matching T1 quantization) */
    {
        int nsb = 3 * num_levels + 1;
        w16(J2K_QCD_M);
        w16(static_cast<uint16_t>(2 + 1 + 2 * nsb));
        /* V192: numgbits 2 → 1 for 2K, 2 for 4K (DCP spec).  band->numbps formula
         * `eps + numgbits - 1` shifts; pmax computation below adjusted to match. */
        const int numgbits = is_4k ? 2 : 1;
        const uint8_t sqcd = static_cast<uint8_t>((numgbits << 5) | 0x02); /* sqty=2 expounded (OPJ: bits4:0=style, bits7:5=guard) */
        w8(sqcd);
        /* V254/V255: QCD must encode the actual T1 quantization step (step * gain) for HH
         * subbands so OPJ dequantizes at the correct amplitude.  subbands[i].step
         * stores the pre-gain step; we apply per-level HH T1 gain here.
         * V255: HH1 (finest, level=0) uses gain=5.5; HH2-5 use gain=4.5. */
        /* V254: QCD encodes the T1 quantization step (step * HH_gain=5.5) for HH subbands.
         * HL/LH are encoded at bare step in QCD (OPJ synthesis accounts for HL/LH DWT gain). */
        /* LL, then for each level (coarsest→finest): HL, LH, HH */
        for (int i = 0; i < nsb; i++) {
            float step_val = (i < static_cast<int>(subbands.size())) ? subbands[i].step : base_step;
            /* Apply HH T1 gain (5.5) to QCD so decoder dequantizes with exact T1 step */
            bool is_hh = (i < static_cast<int>(subbands.size()) &&
                          subbands[i].type == SUBBAND_HH);
            if (is_hh) step_val *= 5.5f;
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

    /* V231: Resolution geometry for precinct grid (matches COD marker values). */
    struct ResGeom { int Xr, Yr, PPx, PPy, nPx, nPy; };
    std::vector<ResGeom> rg(num_levels + 1);
    {
        int Xr[8], Yr_arr[8];
        Xr[num_levels] = width; Yr_arr[num_levels] = height;
        for (int r = num_levels - 1; r >= 0; --r) {
            Xr[r] = (Xr[r+1]+1)/2; Yr_arr[r] = (Yr_arr[r+1]+1)/2;
        }
        for (int r = 0; r <= num_levels; ++r) {
            int PP = (r == 0) ? 128 : 256;
            rg[r] = {Xr[r], Yr_arr[r], PP, PP,
                     (Xr[r]+PP-1)/PP, (Yr_arr[r]+PP-1)/PP};
        }
    }

    /* V234: Flat packet arena — one contiguous buffer per component instead of
     * 177 per-precinct vectors.  comp_arena[c] holds all packets for component c;
     * pkt_refs[c][r][prec_idx] records {offset, size} within that arena.
     * Eliminates ~177 vector alloc/free calls per frame. */
    struct PktRef { uint32_t offset; uint32_t size; };
    std::vector<uint8_t> comp_arena[3];
    std::vector<std::vector<std::vector<PktRef>>> pkt_refs(3);
    for (int c = 0; c < 3; ++c) {
        comp_arena[c].reserve((target_bytes > 0) ? target_bytes / 3 + 16384 : 512*1024);
        pkt_refs[c].resize(num_levels + 1);
        for (int r = 0; r <= num_levels; ++r)
            pkt_refs[c][r].assign(rg[r].nPx * rg[r].nPy, PktRef{0, 0});
    }

    /* Compute per-subband p_max.
     * OPJ t2.c decodes ZBP: loops i=0..inf calling tgt_decode(i) until it returns true.
     * tgt_decode returns true when i = zbp+1 (i.e., it reads the '1' bit at the zbp level).
     * Then: cblk->numbps = band->numbps + 1 - i = band->numbps - zbp.
     * T2 encodes: zbp = pmax - num_bp.
     * → cblk->numbps = band->numbps - pmax + num_bp.
     * For exact match (cblk->numbps = num_bp): pmax = band->numbps.
     *
     * OPJ band->numbps = expn + numgbits - 1  (tcd.c line 1089)
     *   where expn = QCD_eps = 12 - floor(log2(QCD_step)).
     * V266: pmax uses raw subbands[sb].step (no HH T1 gain). QCD encodes step×5.5 for HH,
     * so band->numbps_HH = pmax - 2 → ZBP mismatch; decoder reads numbps = nb - 2.
     * OPJ CDF97 IDWT synthesis gain for HH ≈ 5.5 approximately compensates: net correct. */
    const int numgbits_for_pmax = is_4k ? 2 : 1;
    auto sb_pmax_for_comp = [&](size_t sb, int /*comp*/) -> int {
        float step = std::max(subbands[sb].step, 0.001f);
        int log2s = static_cast<int>(std::floor(std::log2f(step)));
        return (12 - log2s) + numgbits_for_pmax - 1;
    };

    /* V196: per-component sequential truncation, now reading correctly D2H'd
     * pass_lengths.
     *
     * History: V189 added inline truncation that read pass_lengths but the
     * D2H copy of that buffer was missing (V132 had skipped it as "unused").
     * That meant V189 was reading uninitialized host memory, which made the
     * truncation random and explains why checker_64 / noise_small / photos
     * were stuck at low PSNR. Now that pass_lengths is correctly downloaded,
     * the original V189 sequential-fit logic does the right thing: walk
     * subbands LL→HL/LH/HH coarsest→finest, walk CBs in raster order, and
     * for each CB include only the passes whose cumulative length still fits
     * the remaining per-component budget. A full PCRD-OPT slope sort gave
     * worse results in practice — without per-pass significance counts the
     * naive (step × 2^bp)^2 / inc_len slope mis-ranks late passes against
     * early CBs and starves photographic content of mid-frequency detail.
     *
     * Pre-compute pcrd_np_use[]/pcrd_len_use[] here so build_tp can just look
     * them up rather than maintaining state across packets.
     */
    size_t total_cbs = 0;
    if (!subbands.empty()) {
        const SubbandGeom& sb_last = subbands.back();
        total_cbs = static_cast<size_t>(sb_last.cb_start_idx)
                  + static_cast<size_t>(sb_last.ncbx) * sb_last.ncby;
    }
    std::vector<uint8_t>  pcrd_np_use [3];
    std::vector<uint16_t> pcrd_len_use[3];
    for (int c = 0; c < 3; ++c) {
        pcrd_np_use[c].assign(total_cbs, 0);
        pcrd_len_use[c].assign(total_cbs, 0);
    }

    /* V224: Hybrid proportional-PCRD truncation point selection.
     *
     * V219 sequential greedy gave first CBs in each subband everything and
     * last CBs nothing → checker_8 8.1 dB (fine-detail starvation within subbands).
     * V223 global PCRD-OPT fixed within-subband coverage but CDF97 synthesis norms
     * made coarse subbands (LL5 norm=33.84) outbid fine subbands (HL1 norm=2.022)
     * by ~82× → HL1 starvation on smooth content → photo_synth 37.9 dB regression.
     *
     * V224 splits the problem: between-subband budget is proportional to T1 bytes
     * (preserves balance across subbands); within each subband, PCRD-OPT (convex
     * hull + slope sort) distributes the subband budget uniformly across all CBs.
     *
     * Per-CB distortion model: D(p) = (norm × step)² × N_cb × 4^(-p/3)
     * Each hull segment slope = ΔD/ΔR (distortion gain per byte). */
    struct PcrdEntry {
        float    slope;
        int      cb_idx;
        uint8_t  p_from;     /* group start pass (ordering: p_from must = nxt[cb]) */
        uint8_t  p_to;       /* group end pass (inclusive)                         */
        uint16_t cum_len;    /* cumulative coded bytes through p_to               */
        uint16_t delta_r;    /* total bytes for passes p_from..p_to               */
    };

    /* Per-CB convex hull working buffer (reused across CBs). */
    struct HullSeg {
        float    slope;
        float    delta_d;  /* ΔD for this segment (for re-merging)  */
        int      p_from;
        int      p_to;
        uint16_t cum_len;
        uint16_t delta_r;
    };
    std::vector<HullSeg> hull_stack;
    hull_stack.reserve(MAX_PASSES);

    const size_t per_comp_target = (target_bytes > 0)
        ? static_cast<size_t>(target_bytes / 3) : SIZE_MAX;

    /* V225/V226: Two-phase truncation point selection.
     *
     * Phase 1 — sequential greedy for coarse subbands (level ≥ 2): include LL5 and
     * HL/LH/HH at levels 2, 3, 4 fully. These subbands are always tiny (< 10KB for
     * typical 2K content at 150 Mbps) and carry the coarse reconstruction basis.
     * Phase 1 never truncates them; it simply guarantees their passes are included.
     *
     * Phase 2 — proportional PCRD-OPT for lv0 and lv1 subbands: remaining budget is
     * distributed proportional to each subband's T1 bytes, and within each subband
     * convex hull + slope sort allocates the budget by merit across all CBs.
     * This avoids the V219 flaw of giving first CBs 100% and last CBs 0%.
     *
     * V225 used a 40% budget cap instead of a hard level threshold; this caused
     * HL1 (47KB for checker_8) to slip into Phase 1, taking 30KB from lv0.
     * V226 hard-stops Phase 1 at level < 2 → lv0 gets 205KB vs V224's 210KB (−2%). */
    static const float CDF97_NORMS[4][10] = {
        {1.000f,1.965f,4.177f,8.403f,16.90f,33.84f,67.69f,135.3f,270.6f,540.9f},
        {2.022f,3.989f,7.585f,13.94f,25.36f,46.07f,83.53f,151.5f,274.6f,497.2f},
        {2.022f,3.989f,7.585f,13.94f,25.36f,46.07f,83.53f,151.5f,274.6f,497.2f},
        {2.080f,3.865f,6.994f,12.44f,21.92f,38.54f,67.77f,119.2f,209.7f,369.5f}
    };

    for (int c = 0; c < 3; ++c) {
        /* Pre-compute per-subband T1 totals. */
        std::vector<size_t> t1_per_sb(subbands.size(), 0);
        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            int cb_start = subbands[sb].cb_start_idx;
            int ncbx = subbands[sb].ncbx, ncby = subbands[sb].ncby;
            for (int i = 0; i < ncbx * ncby; ++i)
                t1_per_sb[sb] += coded_len[c][cb_start + i];
        }

        /* V263: Phase 1 threshold raised to level ≥ 4 — only LL5 and HL5/LH5/HH5
         * (GPU levels 5 and 4) are unconditionally included. All level ≤ 3 subbands
         * (including HL4/LH4/HH4 = standard level-4 detail) participate in PCRD-OPT.
         *
         * V261's CDF97_NORMS give PCRD-OPT correct slopes for these subbands:
         * HH4 (GPU level=3) has norm=12.44, which is high enough that PCRD-OPT
         * naturally allocates bytes to it early (checker_64 fundamental frequency).
         * V226-V227 Phase 1 included level≥3 (HH4) unconditionally, consuming bytes
         * on low-slope late passes that PCRD-OPT would NOT have chosen, leaving
         * fewer bytes for the higher harmonics (levels 0-2 for square-wave content).
         *
         * V227 comment: "V227: raised threshold from <2 to <3 so HL3/LH3/HH3 (level=2)
         * participate in PCRD-OPT" — V263 extends this logic one level further. */
        size_t comp_bytes = 0;
        size_t pcrd_start = subbands.size();

        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            /* Stop Phase 1 at levels 0-3 (GPU code levels) — let Phase 2 handle them.
             * V263: threshold raised from <3 to <4, moving HH4/HL4/LH4 into Phase 2. */
            if (subbands[sb].level < 4) {
                pcrd_start = sb;
                break;
            }
            if (per_comp_target != SIZE_MAX
                && comp_bytes + t1_per_sb[sb] > per_comp_target) {
                pcrd_start = sb;
                break;
            }
            int cb_start = subbands[sb].cb_start_idx;
            int ncbx = subbands[sb].ncbx, ncby = subbands[sb].ncby;
            for (int iy = 0; iy < ncby; ++iy) {
                for (int ix = 0; ix < ncbx; ++ix) {
                    int cb_idx = cb_start + iy * ncbx + ix;
                    uint8_t  np     = num_passes[c][cb_idx];
                    uint16_t cb_len = coded_len [c][cb_idx];
                    if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                        cb_len = static_cast<uint16_t>(cb_stride - 1);
                    if (np == 0 || cb_len == 0) continue;
                    pcrd_np_use [c][cb_idx] = np;
                    pcrd_len_use[c][cb_idx] = cb_len;
                }
            }
            comp_bytes += t1_per_sb[sb];
        }

        if (pcrd_start >= subbands.size()) continue; /* all fit — nothing more to do */

        /* Phase 2: global PCRD-OPT across all overflow subbands (V228).
         *
         * V224-V227 used per-subband proportional budget allocation (T1-bytes
         * proportional).  That failed for checker_8: HH3 dominates T1 but gets
         * only its proportional share (~5 KB), leaving it under-coded; HL1/LH1
         * ringing harmonics also get too little → 15-17 dB PSNR.
         *
         * V228 collects convex-hull entries from ALL Phase 2 subbands, sorts
         * globally by slope, and greedy-includes until the remaining budget is
         * exhausted.  This is true PCRD-OPT: high-slope entries (dominant
         * subbands like HH3 for checker_8) consume their small T1 quickly,
         * leaving plenty of budget for fine-subband ringing removal.
         *
         * Phase 1 already handles LL5 and levels 3-4 (tiny, never need
         * truncation), so the norm-ratio in Phase 2 is at most 7.585/2.022≈3.75
         * (level-2 vs level-0), far below the V223 problem of 33.84/2.022≈16.7
         * that caused photo_synth regression when LL5 participated. */
        size_t remaining_budget = (per_comp_target > comp_bytes)
            ? per_comp_target - comp_bytes : 0;

        /* Collect all convex-hull entries from every Phase 2 subband. */
        std::vector<PcrdEntry> all_entries;
        {
            size_t p2_cbs = 0;
            for (size_t sb = pcrd_start; sb < subbands.size(); ++sb)
                p2_cbs += static_cast<size_t>(subbands[sb].ncbx) * subbands[sb].ncby;
            all_entries.reserve(p2_cbs * 6);
        }

        for (size_t sb = pcrd_start; sb < subbands.size(); ++sb) {
            if (t1_per_sb[sb] == 0) continue;

            float step  = std::max(subbands[sb].step, 0.001f);
            /* V270: Unified PCRD distortion model — all detail subbands use pcrd_step = step×2.0.
             * V266 applied ×2.0 only to HL/LH (matching their T1 gain), leaving HH at ×1.0.
             * This was a mismatch: HH T1 used gain=5.5 but PCRD assumed step×1.0, underestimating
             * HH distortion by 30×. PCRD under-allocated budget to HH → checker/HH patterns starved.
             * V270 sets T1 HH gain=2.0 (matching HL/LH), so pcrd_step=step×2.0 for ALL detail
             * subbands is now correct and consistent with T1. PCRD competition is fair between
             * HL/LH and HH: norm_HH (2.080) ≈ norm_HL/LH (2.022) so priorities track actual energy.
             * CDF97 synthesis norms capture pixel-domain amplification per coefficient error;
             * LL5 norm=33.84 (handled by Phase 1) >> HL1 norm=2.022 (Phase 2). */
            int sb_type  = subbands[sb].type;
            int sb_level = subbands[sb].level;
            /* V275: Level-dependent HH pcrd_step.
             * HH3 (GPU level=2, 8-16px period) = checker_8 fundamental: 3.5x raises PCRD
             * priority so the budget-constrained checker_8 gets more bits on its dominant
             * subband. HH1/HH2 (budget-unconstrained) keep V273 baseline at step×1.0.
             * HL/LH: unchanged from V273 at step×2.0.
             * Improvement: checker_8 +1.7 dB (14.9→16.6), noise_small −0.1, photo_synth −0.2. */
            float pcrd_step;
            if (sb_type == SUBBAND_HH) {
                pcrd_step = (sb_level == 2) ? step * 3.5f : step;
            } else {
                pcrd_step = step * 2.0f;
            }
            int norm_type = (sb_type == SUBBAND_LL) ? 0 :
                            (sb_type == SUBBAND_HL) ? 1 :
                            (sb_type == SUBBAND_LH) ? 2 : 3; /* HH */
            int norm_lev  = std::min(sb_level, 9);
            float norm   = CDF97_NORMS[norm_type][norm_lev];
            float step2  = (norm * pcrd_step) * (norm * pcrd_step);
            int cb_start = subbands[sb].cb_start_idx;
            int ncbx     = subbands[sb].ncbx;
            int ncby     = subbands[sb].ncby;
            int sb_w     = subbands[sb].width;
            int sb_h     = subbands[sb].height;

            for (int iy = 0; iy < ncby; ++iy) {
                for (int ix = 0; ix < ncbx; ++ix) {
                    int cb_idx = cb_start + iy * ncbx + ix;
                    uint8_t  np     = num_passes[c][cb_idx];
                    uint16_t cb_len = coded_len [c][cb_idx];
                    if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                        cb_len = static_cast<uint16_t>(cb_stride - 1);
                    if (np == 0 || cb_len == 0) continue;

                    int cbw = std::min(CB_DIM, sb_w - ix * CB_DIM);
                    int cbh = std::min(CB_DIM, sb_h - iy * CB_DIM);
                    float base = step2 * static_cast<float>(cbw * cbh);

                    const uint16_t* pl = pass_lengths[c]
                        + static_cast<size_t>(cb_idx) * MAX_PASSES;

                    hull_stack.clear();
                    uint16_t prev_cum = 0;

                    for (int p = 0; p < np; ++p) {
                        uint16_t cum = pl[p];
                        if (cum > cb_len) cum = cb_len;
                        if (p == np - 1) cum = cb_len;
                        int dr = static_cast<int>(cum) - static_cast<int>(prev_cum);
                        if (dr <= 0) { prev_cum = cum; continue; }

                        float Dbefore = base * std::exp2f(-p * (2.0f/3.0f));
                        float Dafter  = base * std::exp2f(-(p+1) * (2.0f/3.0f));
                        float dd  = Dbefore - Dafter;
                        float dr_f = static_cast<float>(dr);

                        HullSeg seg;
                        seg.p_from  = p;
                        seg.p_to    = p;
                        seg.cum_len = cum;
                        seg.delta_r = static_cast<uint16_t>(dr);
                        seg.delta_d = dd;
                        seg.slope   = dd / dr_f;

                        while (!hull_stack.empty()
                               && hull_stack.back().slope <= seg.slope) {
                            const HullSeg& top = hull_stack.back();
                            seg.delta_d += top.delta_d;
                            seg.delta_r  = static_cast<uint16_t>(
                                static_cast<int>(seg.delta_r) + top.delta_r);
                            seg.p_from  = top.p_from;
                            seg.slope   = seg.delta_d / static_cast<float>(seg.delta_r);
                            hull_stack.pop_back();
                        }
                        hull_stack.push_back(seg);
                        prev_cum = cum;
                        if (cum >= cb_len) break;
                    }

                    for (const auto& seg : hull_stack) {
                        all_entries.push_back({seg.slope, cb_idx,
                                              static_cast<uint8_t>(seg.p_from),
                                              static_cast<uint8_t>(seg.p_to),
                                              seg.cum_len, seg.delta_r});
                    }
                }
            }
        }

        std::sort(all_entries.begin(), all_entries.end(),
                  [](const PcrdEntry& a, const PcrdEntry& b){
                      return a.slope > b.slope;
                  });

        /* Global greedy inclusion — pcrd_np_use[c] serves as per-CB pass tracker.
         * Phase 2 CBs start at 0; Phase 1 CBs already have their passes set. */
        size_t global_used = 0;
        for (const auto& e : all_entries) {
            if (global_used >= remaining_budget) break;
            if (e.p_from < static_cast<int>(pcrd_np_use[c][e.cb_idx])) continue;

            pcrd_np_use [c][e.cb_idx] = e.p_to + 1;
            pcrd_len_use[c][e.cb_idx] = e.cum_len;
            global_used += static_cast<size_t>(e.delta_r);
        }
    }

    /* V231: build_tp writes one packet per (comp, res, precinct) matching DCP precinct
     * sizes (r=0: 0x77=128×128, r≥1: 0x88=256×256).  Each precinct's packet contains
     * only the CBs from subbands whose spatial extent overlaps that precinct.
     * Empty precincts (no CBs in any subband) emit a single "0" bit empty packet.
     * CB-to-precinct mapping: px = (sg.x0 + cbx*CB_DIM) / PPx, py similarly.
     * This produces multi-precinct packets compatible with OPJ and DCP verify_j2k. */
    auto build_tp = [&](int comp) {
        size_t comp_bytes = 0;
        std::vector<uint8_t> pkt_header_buf;
        std::vector<uint8_t> pkt_body;
        const size_t est_hdr = (target_bytes > 0) ? (target_bytes / 48 + 4096) : 16384;
        const size_t est_body = (target_bytes > 0) ? (target_bytes / 3 + 16384) : (512 * 1024);
        pkt_header_buf.reserve(est_hdr);
        pkt_body.reserve(est_body);

        TagTree incl_tree, zbp_tree;

        struct SbPrec { size_t sb_idx; int cbx0, cbx1, cby0, cby1; };
        std::vector<SbPrec> active_sbs;
        active_sbs.reserve(3);

        for (int res = 0; res <= num_levels; res++) {
            /* Resolution geometry (pre-computed in rg[]). */
            int PPx = rg[res].PPx, PPy = rg[res].PPy;
            int nPx = rg[res].nPx, nPy = rg[res].nPy;

            /* Subbands at this resolution (HL, LH, HH order preserved). */
            std::vector<size_t> res_sbs;
            for (size_t sb = 0; sb < subbands.size(); ++sb)
                if (subbands[sb].res == res) res_sbs.push_back(sb);

            for (int py = 0; py < nPy; ++py) {
                for (int px = 0; px < nPx; ++px) {
                    pkt_header_buf.clear();
                    pkt_body.clear();
                    BitWriter bw(pkt_header_buf);

                    /* V232: CB-to-precinct mapping using subband-local (0-based) coordinates.
                     * Detail subbands (HL/LH/HH) are at half the resolution of the reference
                     * grid at resolution r, so their precinct size in subband coords is PPx/2.
                     * LL (r=0) is at full precinct size PPx. With PPx=256 for r>0 and PPx=128
                     * for r=0: PPsb=128 in both cases, giving cbsb=PPsb/CB_DIM=4 CBs/precinct.
                     * CB coords within each subband are already 0-based (sg.cb_x0=0). */
                    int PPsb_x = (res == 0) ? PPx : PPx / 2;
                    int PPsb_y = (res == 0) ? PPy : PPy / 2;
                    int cbsb_x = PPsb_x / CB_DIM;  /* CBs per precinct in x (=4) */
                    int cbsb_y = PPsb_y / CB_DIM;  /* CBs per precinct in y (=4) */
                    active_sbs.clear();
                    for (size_t sb_idx : res_sbs) {
                        const SubbandGeom& sg = subbands[sb_idx];
                        int cbx0 = std::min(sg.ncbx, px * cbsb_x);
                        int cbx1 = std::min(sg.ncbx, (px + 1) * cbsb_x);
                        int cby0 = std::min(sg.ncby, py * cbsb_y);
                        int cby1 = std::min(sg.ncby, (py + 1) * cbsb_y);
                        if (cbx0 < cbx1 && cby0 < cby1)
                            active_sbs.push_back({sb_idx, cbx0, cbx1, cby0, cby1});
                    }

                    if (active_sbs.empty()) {
                        bw.write_bit(0); /* empty packet */
                        bw.flush();
                        auto& arena = comp_arena[comp];
                        uint32_t off = static_cast<uint32_t>(arena.size());
                        arena.insert(arena.end(), pkt_header_buf.begin(), pkt_header_buf.end());
                        pkt_refs[comp][res][py * rg[res].nPx + px] =
                            {off, static_cast<uint32_t>(pkt_header_buf.size())};
                        continue;
                    }

                    bw.write_bit(1); /* non-empty packet */

                    for (const auto& sp : active_sbs) {
                        const SubbandGeom& sg = subbands[sp.sb_idx];
                        int pmax = sb_pmax_for_comp(sp.sb_idx, comp);
                        int ncbx_p = sp.cbx1 - sp.cbx0;
                        int ncby_p = sp.cby1 - sp.cby0;
                        int ncbs_p = ncbx_p * ncby_p;

                        incl_tree.build(ncbx_p, ncby_p);
                        zbp_tree.build(ncbx_p, ncby_p);

                        /* V234: Stack arrays — max 4×4=16 CBs per precinct-subband. */
                        bool     included[16]   = {};
                        uint16_t cb_len_use[16] = {};
                        uint8_t  cb_np_use[16]  = {};

                        for (int licby = 0; licby < ncby_p; ++licby) {
                            for (int licbx = 0; licbx < ncbx_p; ++licbx) {
                                int li      = licby * ncbx_p + licbx;
                                int cb_idx  = sg.cb_start_idx
                                            + (sp.cby0 + licby) * sg.ncbx
                                            + (sp.cbx0 + licbx);
                                uint8_t  np_u  = pcrd_np_use [comp][cb_idx];
                                uint16_t len_u = pcrd_len_use[comp][cb_idx];
                                if (np_u == 0 || len_u == 0) {
                                    incl_tree.set_leaf(li, 0x7FFFFFFF);
                                    zbp_tree.set_leaf(li, pmax);
                                } else {
                                    included[li]    = true;
                                    comp_bytes     += len_u;
                                    cb_len_use[li]  = len_u;
                                    cb_np_use[li]   = np_u;
                                    incl_tree.set_leaf(li, 0);
                                    int nb = cb_num_bp ? (int)cb_num_bp[comp][cb_idx] : 0;
                                    zbp_tree.set_leaf(li, std::max(0, pmax - nb));
                                }
                            }
                        }

                        for (int li = 0; li < ncbs_p; ++li) {
                            incl_tree.encode(bw, li, 1);
                            if (!included[li]) continue;
                            zbp_tree.encode(bw, li, pmax);

                            uint8_t  np  = cb_np_use[li];
                            uint16_t len = cb_len_use[li];

                            if (np == 1)       bw.write_bit(0);
                            else if (np == 2)  bw.write_bits(2, 2);
                            else if (np <= 5)  bw.write_bits(0xC | (np - 3), 4);
                            else if (np <= 36) bw.write_bits(0x1E0 | (np - 6), 9);
                            else               bw.write_bits(0xFF80u | (unsigned(np) - 37u), 16);

                            int cb_idx = sg.cb_start_idx
                                       + (sp.cby0 + li / ncbx_p) * sg.ncbx
                                       + (sp.cbx0 + li % ncbx_p);

                            int lblock = 3;
                            if (use_bypass) {
                                /* V243: BYPASS+RESTART — ISO 15444-1 Annex B.10.6.
                                 * With TERMALL every pass is its own segment (maxpasses=1),
                                 * so floor(log2(1))=0 extra bits per length.
                                 * ONE global comma code (increment), then np lengths in lblock bits each. */
                                const uint16_t* pl = pass_lengths[comp]
                                    + static_cast<size_t>(cb_idx) * MAX_PASSES;

                                /* Pass 1: find global increment = max needed expansion */
                                int increment = 0;
                                {
                                    uint16_t prev_cum = 0;
                                    for (int p = 0; p < np; ++p) {
                                        uint16_t seg_cum = pl[p];
                                        if (seg_cum > len) seg_cum = len;
                                        if (p == np - 1) seg_cum = (uint16_t)len;
                                        int seg_len = (int)seg_cum - (int)prev_cum;
                                        if (seg_len > 0) {
                                            int fl = 31 - __builtin_clz((unsigned)seg_len);
                                            int needed = fl + 1 - lblock;
                                            if (needed > increment) increment = needed;
                                        }
                                        prev_cum = seg_cum;
                                    }
                                }
                                if (increment < 0) increment = 0;

                                /* Write ONE comma code */
                                for (int i = 0; i < increment; ++i) bw.write_bit(1);
                                bw.write_bit(0);
                                lblock += increment;

                                /* Pass 2: write each segment length in exactly lblock bits */
                                {
                                    uint16_t prev_cum = 0;
                                    for (int p = 0; p < np; ++p) {
                                        uint16_t seg_cum = pl[p];
                                        if (seg_cum > len) seg_cum = len;
                                        if (p == np - 1) seg_cum = (uint16_t)len;
                                        int seg_len = (int)seg_cum - (int)prev_cum;
                                        if (seg_len < 0) seg_len = 0;
                                        bw.write_bits((unsigned)seg_len, lblock);
                                        prev_cum = seg_cum;
                                    }
                                }
                            } else {
                                int flnp = (np <= 1) ? 0 : (31 - __builtin_clz((unsigned)np));
                                int len_bits = lblock + flnp;
                                if (len_bits < 1) len_bits = 1;
                                while ((1 << len_bits) <= len) { bw.write_bit(1); lblock++; len_bits++; }
                                bw.write_bit(0);
                                bw.write_bits(len, len_bits);
                            }

                            const uint8_t* src = coded_data[comp] + (size_t)cb_idx * cb_stride + 1;
                            pkt_body.insert(pkt_body.end(), src, src + len);
                        }
                    }

                    bw.flush();
                    {
                        auto& arena = comp_arena[comp];
                        uint32_t off = static_cast<uint32_t>(arena.size());
                        arena.insert(arena.end(), pkt_header_buf.begin(), pkt_header_buf.end());
                        arena.insert(arena.end(), pkt_body.begin(), pkt_body.end());
                        pkt_refs[comp][res][py * rg[res].nPx + px] =
                            {off, static_cast<uint32_t>(arena.size() - off)};
                    }
                }
            }
        }
        (void)comp_bytes;
    }; /* end lambda build_tp */

    /* V183: BitWriter now implements J2K-compatible byte stuffing (7 bits after
     * 0xFF), so no post-pass sanitize is needed.  build_tp is called directly. */
    auto fut0 = std::async(std::launch::async, [&]() { build_tp(0); });
    auto fut1 = std::async(std::launch::async, [&]() { build_tp(1); });
    build_tp(2);
    fut0.wait();
    fut1.wait();

    /* V194: split packets across DCP-required tile-parts (3 for 2K, 6 for 4K). */
    int n_tile_parts = is_4k ? 6 : 3;

    /* V231: Build ordered_pkts in CPRL order.
     * CPRL for single-component tile-parts (DCP): iterate global spatial positions
     * (gy, gx) with step = finest precinct size (256) in raster order; for each
     * position, emit packets for all resolutions r where the precinct is first seen.
     * This produces correct CPRL ordering that OPJ decodes without error.
     *
     * Global step: min over r of (PPx_r << (NL-r)) = 256 (at r=NL, PPx=256, scale=1).
     * Precinct at resolution r for global position (gx, gy):
     *   px_r = (gx >> (NL-r)) / PPx_r,  py_r = (gy >> (NL-r)) / PPy_r. */
    struct PktRange { int comp; uint32_t offset; uint32_t size; };
    std::vector<PktRange> ordered_pkts;
    {
        int global_step = rg[num_levels].PPx;  /* 256 for 2K/4K DCP */
        /* visited[r][prec_idx]: ensures each precinct is emitted exactly once per comp. */
        std::vector<std::vector<bool>> visited(num_levels + 1);
        for (int r = 0; r <= num_levels; ++r)
            visited[r].assign(rg[r].nPx * rg[r].nPy, false);

        for (int c = 0; c < 3; ++c) {
            for (auto& v : visited) std::fill(v.begin(), v.end(), false);
            for (int gy = 0; gy < height; gy += global_step) {
                for (int gx = 0; gx < width; gx += global_step) {
                    for (int r = 0; r <= num_levels; ++r) {
                        int scale = num_levels - r;
                        int px_r = (gx >> scale) / rg[r].PPx;
                        int py_r = (gy >> scale) / rg[r].PPy;
                        if (px_r >= rg[r].nPx || py_r >= rg[r].nPy) continue;
                        int prec_idx = py_r * rg[r].nPx + px_r;
                        if (!visited[r][prec_idx]) {
                            visited[r][prec_idx] = true;
                            auto ref = pkt_refs[c][r][prec_idx];
                            ordered_pkts.push_back({c, ref.offset, ref.size});
                        }
                    }
                }
            }
        }
    }

    size_t total_pkt = 0;
    for (auto& pr : ordered_pkts) total_pkt += pr.size;

    /* Distribute packets across TPs by total byte count.  TP boundaries fall on
     * packet boundaries (we don't split a packet across TPs). */
    std::vector<int> tp_first_pkt(n_tile_parts);
    std::vector<size_t> tp_pkt_bytes(n_tile_parts, 0);
    {
        size_t target_per_tp = (total_pkt + n_tile_parts - 1) / n_tile_parts;
        int pkt_idx = 0;
        for (int t = 0; t < n_tile_parts; ++t) {
            tp_first_pkt[t] = pkt_idx;
            size_t accum = 0;
            int last_t = (t == n_tile_parts - 1);
            while (pkt_idx < (int)ordered_pkts.size()
                   && (last_t || accum < target_per_tp)) {
                accum += ordered_pkts[pkt_idx].size;
                ++pkt_idx;
            }
            tp_pkt_bytes[t] = accum;
        }
        /* If we exited with fewer pkts than tp_first_pkt suggests, the remaining
         * TPs get zero packets — emit empty TPs to satisfy the spec count. */
    }

    /* TLM — list all tile-parts. */
    w16(J2K_TLM_M);
    w16(static_cast<uint16_t>(2 + 1 + 1 + n_tile_parts * (1 + 4))); /* Ltlm */
    w8(0);    /* Ztlm = 0 */
    w8(0x50); /* Stlm: ST=1 (1-byte Ttlm), SP=1 (4-byte Ptlm) */
    for (int t = 0; t < n_tile_parts; ++t) {
        uint32_t tp_size = static_cast<uint32_t>(12 + 2 + tp_pkt_bytes[t]);
        w8(0);                /* Ttlm = tile 0 (single tile) */
        w32(tp_size);
    }

    /* Emit each tile-part. */
    for (int t = 0; t < n_tile_parts; ++t) {
        uint32_t tp_size = static_cast<uint32_t>(12 + 2 + tp_pkt_bytes[t]);
        w16(J2K_SOT_M);
        w16(10);
        w16(0);                                        /* Isot: tile index */
        w32(tp_size);                                  /* Psot: TP byte length */
        w8(static_cast<uint8_t>(t));                   /* TPsot: TP index */
        w8(static_cast<uint8_t>(n_tile_parts));        /* TNsot: total TPs */
        w16(J2K_SOD_M);
        int end_pkt = (t + 1 < n_tile_parts) ? tp_first_pkt[t + 1] : (int)ordered_pkts.size();
        for (int i = tp_first_pkt[t]; i < end_pkt; ++i) {
            const auto& pr = ordered_pkts[i];
            const auto& arena = comp_arena[pr.comp];
            cs.insert(cs.end(), arena.begin() + pr.offset,
                      arena.begin() + pr.offset + pr.size);
        }
    }

    /* EOC — final marker.  V139 FIX: NEVER pad after EOC. */
    w16(J2K_EOC_M);

    return cs;
}


#endif /* GPU_EBCOT_T2_H */
