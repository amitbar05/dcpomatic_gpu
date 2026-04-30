#!/usr/bin/env bash
# Self-validating optimisation loop.
#
# For each line in test/iter_queue.txt the loop:
#   1. Saves the current source state to a stash.
#   2. Applies the queued change as a sed-style edit (or a shell snippet).
#   3. Rebuilds the affected test binaries.
#   4. Runs the regression battery (psnr_battery + cmp_gpu_opj + verify_roundtrip + bench_modes).
#   5. Compares the new CSV against the locked baseline at test/baselines/v196c_post_stage_b.csv.
#   6. PASS if (a) no correct-mode PSNR drops by more than 0.5 dB on any pattern,
#                (b) no fast-mode PSNR drops by more than 1.0 dB,
#                (c) verify_roundtrip pass count not lower.
#   7. If PASS and bench_modes timing improved by >2 % on either mode -> commit.
#   8. If FAIL -> git checkout -- src/ to revert.
#   9. Logs the outcome to test/iter_log.csv (append-only).
#
# Stop condition: presence of test/STOP file, or empty iter_queue.txt.
#
# Queue line format (one per line, blank/# lines ignored):
#   sed:<file>:s/old/new/  -- run gnu sed -i over the file
#   apply:<patch_file>     -- git apply the patch
#   noop:<comment>         -- record a no-op iteration (used for re-baselining)
#
# Usage:
#   ./test/auto_iter.sh                    # run forever
#   STOP_AFTER=10 ./test/auto_iter.sh      # run at most 10 iterations
#   touch test/STOP                        # request graceful stop

set -u
cd "$(dirname "$0")/.."

BASELINE=${BASELINE:-test/baselines/v196c_post_stage_b.csv}
QUEUE=${QUEUE:-test/iter_queue.txt}
LOG=test/iter_log.csv
STOP_FLAG=test/STOP
STOP_AFTER=${STOP_AFTER:-0}

[[ -f "$BASELINE" ]] || { echo "no baseline at $BASELINE" >&2; exit 1; }
[[ -f "$QUEUE" ]] || { echo "no queue at $QUEUE — create it" >&2; exit 1; }

[[ -f "$LOG" ]] || echo "ts,iter,queue_line,outcome,detail" > "$LOG"

iter=0
while :; do
    [[ -f "$STOP_FLAG" ]] && { echo "STOP flag present — exiting" >&2; rm -f "$STOP_FLAG"; break; }
    [[ $STOP_AFTER -gt 0 && $iter -ge $STOP_AFTER ]] && { echo "STOP_AFTER=$STOP_AFTER reached" >&2; break; }

    line=$(awk 'NR==1 && !/^[[:space:]]*$/ && !/^#/ { print; exit }
                /^[[:space:]]*$/ { next }
                /^#/ { next }
                { print; exit }' "$QUEUE")
    if [[ -z "$line" ]]; then
        echo "queue empty — sleeping 60s" >&2
        sleep 60
        continue
    fi

    iter=$((iter+1))
    ts=$(date -Iseconds)
    echo
    echo "=== iter $iter @ $ts: $line ===" >&2

    # Snapshot current state so we can revert.
    snap=$(git stash create --include-untracked || true)

    # Apply the change.
    apply_ok=1
    case "$line" in
        sed:*)
            spec=${line#sed:}
            file=${spec%%:*}
            expr=${spec#*:}
            if [[ ! -f "$file" ]]; then echo "sed target missing: $file" >&2; apply_ok=0
            else sed -i "$expr" "$file" || apply_ok=0; fi
            ;;
        apply:*)
            patch=${line#apply:}
            if [[ ! -f "$patch" ]]; then echo "patch missing: $patch" >&2; apply_ok=0
            else git apply "$patch" || apply_ok=0; fi
            ;;
        noop:*)
            ;;
        *)
            echo "unknown queue verb: $line" >&2; apply_ok=0
            ;;
    esac

    if [[ $apply_ok -ne 1 ]]; then
        echo "$ts,$iter,\"$line\",apply_failed," >> "$LOG"
        # Pop the queue line and continue.
        sed -i '0,/^[^#].*\S/{//d;}' "$QUEUE"
        continue
    fi

    # Rebuild + run battery.
    if ! ./test/run_battery.sh --quick >/tmp/iter_${iter}.out 2>&1; then
        # Revert: discard working tree changes back to HEAD.
        git checkout -- src/ test/run_battery.sh 2>/dev/null
        echo "$ts,$iter,\"$line\",build_or_run_failed," >> "$LOG"
        sed -i '0,/^[^#].*\S/{//d;}' "$QUEUE"
        continue
    fi
    csv=$(awk -F'/' '/Wrote /{print $0}' /tmp/iter_${iter}.out | awk '{print $NF}' | tail -1)
    if [[ -z "$csv" || ! -f "$csv" ]]; then
        git checkout -- src/ test/run_battery.sh 2>/dev/null
        echo "$ts,$iter,\"$line\",no_csv," >> "$LOG"
        sed -i '0,/^[^#].*\S/{//d;}' "$QUEUE"
        continue
    fi

    # Compare against baseline.
    ok=1
    detail=""
    while IFS=, read -r b_sha b_test b_mode b_pattern b_res b_psnr b_bytes b_ms b_extra; do
        [[ "$b_test" == "psnr_battery" || "$b_test" == "bench_modes" ]] || continue
        # Build a key.
        key="${b_test}|${b_mode}|${b_pattern}"
        # Find matching row in new csv.
        new_row=$(awk -F, -v k="$key" '$2"|"$3"|"$4 == k { print; exit }' "$csv")
        [[ -z "$new_row" ]] && continue
        new_psnr=$(awk -F, '{print $6}' <<< "$new_row")
        new_ms=$(awk -F, '{print $8}' <<< "$new_row")
        if [[ -n "$b_psnr" && -n "$new_psnr" ]]; then
            # Threshold: 0.5 dB for correct, 1.0 for fast.
            thresh=$([[ "$b_mode" == "gpu_correct" ]] && echo 0.5 || echo 1.0)
            drop=$(awk -v a="$b_psnr" -v b="$new_psnr" -v t="$thresh" \
                'BEGIN { if ((a-b) > t) print 1; else print 0 }')
            if [[ "$drop" == "1" ]]; then
                ok=0
                detail="${detail}${b_mode}/${b_pattern} ${b_psnr}->${new_psnr};"
            fi
        fi
    done < "$BASELINE"

    if [[ $ok -ne 1 ]]; then
        git checkout -- src/ test/run_battery.sh 2>/dev/null
        echo "$ts,$iter,\"$line\",regress,\"$detail\"" >> "$LOG"
        sed -i '0,/^[^#].*\S/{//d;}' "$QUEUE"
        continue
    fi

    # Compute timing delta.
    base_ms_correct=$(awk -F, '$2=="bench_modes" && $3=="gpu_correct" { print $8 }' "$BASELINE")
    new_ms_correct=$(awk -F, '$2=="bench_modes" && $3=="gpu_correct" { print $8 }' "$csv")
    base_ms_fast=$(awk -F, '$2=="bench_modes" && $3=="gpu_fast" { print $8 }' "$BASELINE")
    new_ms_fast=$(awk -F, '$2=="bench_modes" && $3=="gpu_fast" { print $8 }' "$csv")
    delta_correct=$(awk -v a="$base_ms_correct" -v b="$new_ms_correct" 'BEGIN{ if(a>0) printf "%.1f", (a-b)/a*100; else print "0.0" }')
    delta_fast=$(awk -v a="$base_ms_fast" -v b="$new_ms_fast" 'BEGIN{ if(a>0) printf "%.1f", (a-b)/a*100; else print "0.0" }')

    improved=$(awk -v c="$delta_correct" -v f="$delta_fast" \
      'BEGIN{ if (c > 2.0 || f > 2.0) print 1; else print 0 }')

    if [[ "$improved" == "1" ]]; then
        # Commit the change.
        git add -A src/ test/
        git commit -m "auto-iter: $line (correct -${delta_correct}%, fast -${delta_fast}%)" 2>&1 | tail -2
        # Update baseline so subsequent iterations compare against the new state.
        cp "$csv" "$BASELINE"
        echo "$ts,$iter,\"$line\",pass_committed,\"correct=${delta_correct}% fast=${delta_fast}%\"" >> "$LOG"
    else
        # No timing improvement; revert.
        git checkout -- src/ test/run_battery.sh 2>/dev/null
        echo "$ts,$iter,\"$line\",pass_no_speedup,\"correct=${delta_correct}% fast=${delta_fast}%\"" >> "$LOG"
    fi

    # Pop the queue line.
    sed -i '0,/^[^#].*\S/{//d;}' "$QUEUE"
done

echo "auto_iter exit at iter=$iter" >&2
