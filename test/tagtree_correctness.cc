/*
 * TagTree Correctness Test
 * Verifies the GPU TagTree implementation (gpu_ebcot_t2.h) against a CPU reference.
 *
 * Build:
 *   g++ -std=c++17 -O2 \
 *       -include test/gpu_ebcot_preinclude.h \
 *       -I/home/amit/dcp-o-matic-gpu/src \
 *       -I/home/amit/dcp-o-matic-gpu/src/lib \
 *       -o test/tagtree_correctness test/tagtree_correctness.cc
 *
 * The -include flag pre-defines GPU_EBCOT_H and provides forward declarations
 * so that the CUDA-dependent gpu_ebcot.h is skipped. TagTree and BitWriter
 * are pure C++ and do not depend on any CUDA functionality.
 */

#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <string>

#include "lib/gpu_ebcot_t2.h"

/* ========================================================================= */
/* CPU Reference Tag Tree (matches ITU-T T.800 B.10.2 exactly)               */
/* ========================================================================= */
struct RefTagTree {
    struct Node {
        int  value;   /* min of subtree */
        int  low;     /* highest threshold emitted */
        bool known;   /* final 1-bit sent */
        int  parent;  /* parent index, -1 = root */
    };

    std::vector<Node> nodes;
    std::vector<int>  level_off;
    int num_leaves = 0;

    void build(int ncbx, int ncby) {
        nodes.clear();  level_off.clear();

        std::vector<std::pair<int,int>> dims;
        {
            int w = ncbx, h = ncby;
            while (true) {
                dims.push_back({w, h});
                if (w == 1 && h == 1) break;
                w = (w + 1) / 2;
                h = (h + 1) / 2;
            }
        }

        int nlv = (int)dims.size();
        int total = 0;
        for (int lv = 0; lv < nlv; lv++) {
            level_off.push_back(total);
            total += dims[lv].first * dims[lv].second;
        }

        nodes.resize(total);
        num_leaves = dims[0].first * dims[0].second;

        for (int lv = 0; lv < nlv; lv++) {
            int w = dims[lv].first, h = dims[lv].second;
            int off = level_off[lv];
            for (int j = 0; j < h; j++) {
                for (int i = 0; i < w; i++) {
                    Node& n = nodes[off + j * w + i];
                    n.value = INT_MAX;
                    n.low   = 0;
                    n.known = false;
                    n.parent = (lv + 1 < nlv)
                        ? level_off[lv+1] + (j/2) * dims[lv+1].first + (i/2)
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
            if (value < nodes[par].value) {
                nodes[par].value = value;
                idx = par;
            } else break;
        }
    }

    /* Encode leaf via root→leaf walk, writing bits using BitWriter.
     * Same algorithm as TagTree::encode(). */
    void encode(BitWriter& bw, int leaf_idx, int threshold) {
        int path[32], plen = 0;
        for (int idx = leaf_idx; idx != -1; idx = nodes[idx].parent)
            path[plen++] = idx;

        int low = 0;
        for (int pi = plen - 1; pi >= 0; pi--) {
            Node& node = nodes[path[pi]];
            if (low > node.low) node.low = low; else low = node.low;

            while (low < threshold) {
                if (low >= node.value) {
                    if (!node.known) {
                        bw.write_bit(1);
                        node.known = true;
                    }
                    goto next_level;
                }
                bw.write_bit(0);
                ++low;
            }
        next_level:
            node.low = low;
        }
    }
};

/* ========================================================================= */
/* Helpers                                                                    */
/* ========================================================================= */

/* Convert a byte buffer to a bit-string for diagnostics. */
static std::string buf_to_bit_string(const std::vector<uint8_t>& buf) {
    std::string s;
    for (uint8_t byte : buf) {
        for (int b = 7; b >= 0; b--) {
            s += ((byte >> b) & 1) ? '1' : '0';
        }
    }
    return s;
}

static int passed = 0, failed = 0;

static void report(const char* name, bool ok,
                   const std::vector<uint8_t>& gpu_buf,
                   const std::vector<uint8_t>& ref_buf)
{
    if (ok) {
        printf("PASS: %s\n", name);
        passed++;
    } else {
        printf("FAIL: %s\n", name);
        printf("  GPU  bytes (%zu): %s\n", gpu_buf.size(),
               buf_to_bit_string(gpu_buf).c_str());
        printf("  Ref  bytes (%zu): %s\n", ref_buf.size(),
               buf_to_bit_string(ref_buf).c_str());
        failed++;
    }
}

/* ========================================================================= */
/* Test Cases                                                                 */
/* ========================================================================= */

/* Test A: Single leaf value=5.
 * With threshold=6 (enough to reach value), expect 000001 (0 bits until value
 * reached, then 1). This matches JPEG2000 tag tree encoding for value 5. */
static void test_single_leaf() {
    /* --- GPU --- */
    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(1, 1);
        tt.set_leaf(0, 5);
        tt.encode(bw, 0, 6);
        bw.flush();
    }

    /* --- Ref --- */
    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(1, 1);
        ref.set_leaf(0, 5);
        ref.encode(bw, 0, 6);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("Single leaf (1x1), value=5, threshold=6", ok, gpu_buf, ref_buf);
}

/* Test B: 2×2 uniform (all=3), threshold=4.
 * Root gets min=3. Each leaf encoding goes through internal nodes. */
static void test_2x2_uniform() {
    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(2, 2);
        for (int i = 0; i < 4; i++) tt.set_leaf(i, 3);
        for (int i = 0; i < 4; i++) tt.encode(bw, i, 4);
        bw.flush();
    }

    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(2, 2);
        for (int i = 0; i < 4; i++) ref.set_leaf(i, 3);
        for (int i = 0; i < 4; i++) ref.encode(bw, i, 4);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("2x2 uniform (all=3, threshold=4)", ok, gpu_buf, ref_buf);
}

/* Test C: 2×2 mixed {0, MAXINT, 5, 2}, threshold=large. */
static void test_2x2_mixed() {
    std::vector<int> leaf_vals = {0, INT_MAX, 5, 2};
    const int threshold = 256; /* large enough to fully encode all values */

    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(2, 2);
        for (int i = 0; i < 4; i++) tt.set_leaf(i, leaf_vals[i]);
        for (int i = 0; i < 4; i++) tt.encode(bw, i, threshold);
        bw.flush();
    }

    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(2, 2);
        for (int i = 0; i < 4; i++) ref.set_leaf(i, leaf_vals[i]);
        for (int i = 0; i < 4; i++) ref.encode(bw, i, threshold);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("2x2 mixed {0, MAXINT, 5, 2}", ok, gpu_buf, ref_buf);
}

/* Test D: 4×4 uniform (all=7), threshold=8. */
static void test_4x4_uniform() {
    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(4, 4);
        for (int i = 0; i < 16; i++) tt.set_leaf(i, 7);
        for (int i = 0; i < 16; i++) tt.encode(bw, i, 8);
        bw.flush();
    }

    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(4, 4);
        for (int i = 0; i < 16; i++) ref.set_leaf(i, 7);
        for (int i = 0; i < 16; i++) ref.encode(bw, i, 8);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("4x4 uniform (all=7, threshold=8)", ok, gpu_buf, ref_buf);
}

/* Test E: Threshold limits encoding.
 * value=7, threshold=3 → should emit 3 zeros, no terminal 1
 * because low reaches threshold before reaching value. */
static void test_threshold() {
    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(1, 1);
        tt.set_leaf(0, 7);
        tt.encode(bw, 0, 3);
        bw.flush();
    }

    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(1, 1);
        ref.set_leaf(0, 7);
        ref.encode(bw, 0, 3);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("Threshold: value=7, threshold=3", ok, gpu_buf, ref_buf);
}

/* Test F: Re-entry test.
 * First call emits full encoding for value=5.
 * Second call with same threshold emits nothing (already known). */
static void test_reentry() {
    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(1, 1);
        tt.set_leaf(0, 5);
        tt.encode(bw, 0, 6);  /* first */
        tt.encode(bw, 0, 6);  /* second: should emit nothing */
        bw.flush();
    }

    /* Build reference with a single call to get expected bits */
    std::vector<uint8_t> ref_one;
    {
        BitWriter bw(ref_one);
        RefTagTree ref;
        ref.build(1, 1);
        ref.set_leaf(0, 5);
        ref.encode(bw, 0, 6);
        bw.flush();
    }

    /* Ref two calls should equal ref_one (second call adds nothing) */
    std::vector<uint8_t> ref_two;
    {
        BitWriter bw(ref_two);
        RefTagTree ref;
        ref.build(1, 1);
        ref.set_leaf(0, 5);
        ref.encode(bw, 0, 6);
        ref.encode(bw, 0, 6);
        bw.flush();
    }

    bool ref_reentry_correct = (ref_two == ref_one);
    bool ok = (gpu_buf == ref_one) && ref_reentry_correct;
    printf("  Ref re-entry same as single: %s\n",
           ref_reentry_correct ? "yes" : "NO");
    report("Re-entry: encode same leaf twice", ok, gpu_buf, ref_one);
}

/* Test G: 1×4 strip.
 * Verifies linear quadtree (1-wide at all levels). */
static void test_1x4_strip() {
    std::vector<int> vals = {1, 3, 2, 4};

    std::vector<uint8_t> gpu_buf;
    {
        BitWriter bw(gpu_buf);
        TagTree tt;
        tt.build(1, 4);
        for (int i = 0; i < 4; i++) tt.set_leaf(i, vals[i]);
        for (int i = 0; i < 4; i++) tt.encode(bw, i, 5);
        bw.flush();
    }

    std::vector<uint8_t> ref_buf;
    {
        BitWriter bw(ref_buf);
        RefTagTree ref;
        ref.build(1, 4);
        for (int i = 0; i < 4; i++) ref.set_leaf(i, vals[i]);
        for (int i = 0; i < 4; i++) ref.encode(bw, i, 5);
        bw.flush();
    }

    bool ok = (gpu_buf == ref_buf);
    report("1x4 strip {1,3,2,4}, threshold=5", ok, gpu_buf, ref_buf);
}

/* ========================================================================= */
/* Main                                                                       */
/* ========================================================================= */
int main() {
    printf("TagTree Correctness Test\n");
    printf("========================\n\n");

    test_single_leaf();
    test_2x2_uniform();
    test_2x2_mixed();
    test_4x4_uniform();
    test_threshold();
    test_reentry();
    test_1x4_strip();

    printf("\n========================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
