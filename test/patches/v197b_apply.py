#!/usr/bin/env python3
"""Apply V197b: leftover-budget redistribution for the PCRD pre-pass.

Inserts comp_bytes_used[3] tracking and a second-pass loop that admits
extra passes for truncated CBs out of the global byte leftover.  No
existing pcrd_*_use entries are reduced — only added to — so this can
only improve PSNR, never regress it.

Idempotent: refuses to apply if the marker comment is already present.
"""
import sys

PATH = "src/lib/gpu_ebcot_t2.h"
MARKER = "/* V197b: second-pass leftover redistribution."

with open(PATH, "r") as f:
    text = f.read()

if MARKER in text:
    print("V197b marker already present — refusing to re-apply", file=sys.stderr)
    sys.exit(1)

OLD = """    const size_t per_comp_target = (target_bytes > 0)
        ? static_cast<size_t>(target_bytes / 3) : SIZE_MAX;
    for (int c = 0; c < 3; ++c) {
        size_t comp_bytes = 0;
        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            int cb_start = subbands[sb].cb_start_idx;
            int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
            for (int i = 0; i < ncbs; ++i) {
                int cb_idx = cb_start + i;
                uint8_t  np = num_passes[c][cb_idx];
                uint16_t cb_len = coded_len[c][cb_idx];
                if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                    cb_len = static_cast<uint16_t>(cb_stride - 1);
                if (np == 0 || cb_len == 0) continue;

                uint8_t  np_use  = np;
                uint16_t len_use = cb_len;
                if (comp_bytes + cb_len > per_comp_target) {
                    /* Walk passes; admit prefix that fits remaining budget. */
                    size_t remaining = (per_comp_target > comp_bytes)
                        ? per_comp_target - comp_bytes : 0;
                    const uint16_t* pl = pass_lengths[c]
                        + static_cast<size_t>(cb_idx) * MAX_PASSES;
                    np_use = 0; len_use = 0;
                    for (int p = 0; p < np; ++p) {
                        uint16_t cum = pl[p];
                        if (cum > cb_len) cum = cb_len;
                        if (static_cast<size_t>(cum) > remaining) break;
                        np_use = static_cast<uint8_t>(p + 1);
                        len_use = cum;
                    }
                }
                if (np_use == 0 || len_use == 0) continue;
                pcrd_np_use [c][cb_idx] = np_use;
                pcrd_len_use[c][cb_idx] = len_use;
                comp_bytes += len_use;
            }
        }
    }
"""

NEW = """    const size_t per_comp_target = (target_bytes > 0)
        ? static_cast<size_t>(target_bytes / 3) : SIZE_MAX;
    size_t comp_bytes_used[3] = {0, 0, 0};
    for (int c = 0; c < 3; ++c) {
        size_t comp_bytes = 0;
        for (size_t sb = 0; sb < subbands.size(); ++sb) {
            int cb_start = subbands[sb].cb_start_idx;
            int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
            for (int i = 0; i < ncbs; ++i) {
                int cb_idx = cb_start + i;
                uint8_t  np = num_passes[c][cb_idx];
                uint16_t cb_len = coded_len[c][cb_idx];
                if (cb_len > static_cast<uint16_t>(cb_stride - 1))
                    cb_len = static_cast<uint16_t>(cb_stride - 1);
                if (np == 0 || cb_len == 0) continue;

                uint8_t  np_use  = np;
                uint16_t len_use = cb_len;
                if (comp_bytes + cb_len > per_comp_target) {
                    /* Walk passes; admit prefix that fits remaining budget. */
                    size_t remaining = (per_comp_target > comp_bytes)
                        ? per_comp_target - comp_bytes : 0;
                    const uint16_t* pl = pass_lengths[c]
                        + static_cast<size_t>(cb_idx) * MAX_PASSES;
                    np_use = 0; len_use = 0;
                    for (int p = 0; p < np; ++p) {
                        uint16_t cum = pl[p];
                        if (cum > cb_len) cum = cb_len;
                        if (static_cast<size_t>(cum) > remaining) break;
                        np_use = static_cast<uint8_t>(p + 1);
                        len_use = cum;
                    }
                }
                if (np_use == 0 || len_use == 0) continue;
                pcrd_np_use [c][cb_idx] = np_use;
                pcrd_len_use[c][cb_idx] = len_use;
                comp_bytes += len_use;
            }
        }
        comp_bytes_used[c] = comp_bytes;
    }

    /* V197b: second-pass leftover redistribution.  After the first pass each
     * component is bounded by per_comp_target = target/3; for patterns like
     * checker_64 the total used was well under target (727 of 781 KB).  This
     * second pass redistributes the global leftover across components in a
     * single forward sweep, admitting EXTRA passes for CBs that were
     * truncated.  Because we only ADD to existing pcrd_*_use entries, a CB
     * that was fully admitted stays admitted — only truncated CBs gain.  No
     * regression risk: PSNR cannot drop. */
    if (target_bytes > 0) {
        size_t total_target = static_cast<size_t>(target_bytes);
        size_t total_used = comp_bytes_used[0] + comp_bytes_used[1] + comp_bytes_used[2];
        if (total_used < total_target) {
            size_t leftover = total_target - total_used;
            for (int c = 0; c < 3 && leftover > 0; ++c) {
                for (size_t sb = 0; sb < subbands.size() && leftover > 0; ++sb) {
                    int cb_start = subbands[sb].cb_start_idx;
                    int ncbs = subbands[sb].ncbx * subbands[sb].ncby;
                    for (int i = 0; i < ncbs && leftover > 0; ++i) {
                        int cb_idx = cb_start + i;
                        uint8_t  np_full = num_passes[c][cb_idx];
                        uint8_t  np_used = pcrd_np_use[c][cb_idx];
                        if (np_used >= np_full) continue;
                        uint16_t cb_len_full = coded_len[c][cb_idx];
                        if (cb_len_full > static_cast<uint16_t>(cb_stride - 1))
                            cb_len_full = static_cast<uint16_t>(cb_stride - 1);
                        uint16_t len_used = pcrd_len_use[c][cb_idx];
                        const uint16_t* pl = pass_lengths[c]
                            + static_cast<size_t>(cb_idx) * MAX_PASSES;
                        for (int p = np_used; p < np_full; ++p) {
                            uint16_t cum = pl[p];
                            if (cum > cb_len_full) cum = cb_len_full;
                            uint16_t inc = (cum > len_used) ? (cum - len_used) : 0;
                            if (inc == 0) continue;
                            if (inc > leftover) break;
                            np_used = static_cast<uint8_t>(p + 1);
                            len_used = cum;
                            leftover -= inc;
                        }
                        pcrd_np_use [c][cb_idx] = np_used;
                        pcrd_len_use[c][cb_idx] = len_used;
                    }
                }
            }
        }
    }
"""

if OLD not in text:
    print(f"V197b: OLD block not found in {PATH}; aborting", file=sys.stderr)
    sys.exit(1)

with open(PATH, "w") as f:
    f.write(text.replace(OLD, NEW, 1))

print("V197b applied")
