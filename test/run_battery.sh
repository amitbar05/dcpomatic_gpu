#!/usr/bin/env bash
# Regression battery for the CUDA J2K encoder.
#
# Rebuilds the standalone tests against current src/lib, runs them, parses
# PSNR/bytes/timing, and writes a single CSV under test/baselines/.
# Used both for one-off baselining and as the regression engine for the
# Stage D auto-iteration loop.
#
# Usage:
#   ./test/run_battery.sh                # standard run; writes test/baselines/<sha>_<ts>.csv
#   ./test/run_battery.sh --quick        # skip slow tests (verify_correctness_v2)
#   ./test/run_battery.sh --no-rebuild   # use existing binaries (faster iteration)

set -u
cd "$(dirname "$0")/.."

QUICK=0
REBUILD=1
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        --no-rebuild) REBUILD=0 ;;
        *) echo "unknown flag: $arg"; exit 2 ;;
    esac
done

SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
TS=$(date +%Y%m%d_%H%M%S)
OUT=test/baselines/${SHA}_${TS}.csv
LOG=test/baselines/${SHA}_${TS}.log

mkdir -p test/baselines

NVCC_FLAGS="-O3 --use_fast_math -arch=sm_61 -std=c++17 -I src -I src/lib -I/usr/include/openjpeg-2.5"
NVCC_LIBS="-lcudart -lopenjp2"

# Build cuda_j2k_encoder.o ONCE per battery run, then link each test against it.
# Cuts the per-iter wall time roughly in half: 5 nvcc compiles of the 5500-line
# encoder dominated the iter cycle.
ENCODER_OBJ="test/cuda_j2k_encoder.o"
ENCODER_SRC="src/lib/cuda_j2k_encoder.cu"
if [[ $REBUILD -eq 1 || ! -f "$ENCODER_OBJ" || "$ENCODER_SRC" -nt "$ENCODER_OBJ" \
      || src/lib/gpu_ebcot.h -nt "$ENCODER_OBJ" \
      || src/lib/gpu_ebcot_t2.h -nt "$ENCODER_OBJ" ]]; then
    echo "  build: $ENCODER_OBJ (shared encoder TU)" >&2
    if ! nvcc $NVCC_FLAGS -c -o "$ENCODER_OBJ" "$ENCODER_SRC" >>"$LOG" 2>&1; then
        echo "  ENCODER BUILD FAILED (see $LOG)" >&2
        exit 1
    fi
fi

build_one() {
    local out=$1 src=$2 extra=${3:-}
    if [[ $REBUILD -eq 1 || ! -x "$out" || "$src" -nt "$out" || "$ENCODER_OBJ" -nt "$out" ]]; then
        echo "  build: $out" >&2
        # Link the test source against the shared encoder .o.
        if ! nvcc $NVCC_FLAGS $extra -o "$out" "$src" "$ENCODER_OBJ" $NVCC_LIBS >>"$LOG" 2>&1; then
            echo "  BUILD FAILED: $out (see $LOG)" >&2
            return 1
        fi
    fi
    return 0
}

echo "# CUDA J2K regression battery — sha=$SHA ts=$TS" >&2
echo "git_sha,test,mode,pattern,res,psnr_db,bytes,ms_per_frame,extra" > "$OUT"

# ---- 1. cmp_gpu_opj: head-to-head GPU vs OPJ on 7 patterns -------------
build_one test/cmp_gpu_opj test/cmp_gpu_opj.cc || exit 1
echo "  run: cmp_gpu_opj" >&2
./test/cmp_gpu_opj 2>>"$LOG" | awk -v sha="$SHA" '
    /^[a-z_][a-z0-9_]* +\|/ {
        gsub(/[|]/, " ");
        gsub(/dB/, "");
        gsub(/B/, "");
        # fields: pattern, gpu_psnr, gpu_bytes, opj_psnr, opj_bytes, delta
        printf "%s,cmp_gpu_opj,gpu_correct,%s,2K,%s,%s,,opj_psnr=%s opj_bytes=%s\n",
               sha, $1, $2, $3, $4, $5
    }' >> "$OUT"

# ---- 2. psnr_battery: 13 patterns, correct mode (FAST=0) ---------------
build_one test/psnr_battery test/psnr_battery.cc || exit 1
echo "  run: psnr_battery (correct, 2K)" >&2
./test/psnr_battery 2>>"$LOG" | awk -v sha="$SHA" -v mode=gpu_correct '
    /PSNR_Y/ {
        # line: "  pattern  cs=NNN  B  PSNR_Y =  XX.X dB"
        n = split($0, f, /[ \t]+/);
        pat=f[2]; bytes=""; psnr="";
        for (i=1; i<=n; i++) {
            if (f[i] ~ /^cs=/) { sub(/^cs=/,"",f[i]); bytes=f[i] }
            if (f[i-1] == "=" && f[i] ~ /^[0-9.]+$/) psnr=f[i];
        }
        printf "%s,psnr_battery,%s,%s,2K,%s,%s,,\n", sha, mode, pat, psnr, bytes
    }' >> "$OUT"

echo "  run: psnr_battery (fast, 2K)" >&2
USE_FAST=1 ./test/psnr_battery 2>>"$LOG" | awk -v sha="$SHA" -v mode=gpu_fast '
    /PSNR_Y/ {
        n = split($0, f, /[ \t]+/);
        pat=f[2]; bytes=""; psnr="";
        for (i=1; i<=n; i++) {
            if (f[i] ~ /^cs=/) { sub(/^cs=/,"",f[i]); bytes=f[i] }
            if (f[i-1] == "=" && f[i] ~ /^[0-9.]+$/) psnr=f[i];
        }
        printf "%s,psnr_battery,%s,%s,2K,%s,%s,,\n", sha, mode, pat, psnr, bytes
    }' >> "$OUT"

# ---- 3. verify_roundtrip: aggregated pass/fail counts -------------------
build_one test/verify_roundtrip test/verify_roundtrip.cc || exit 1
echo "  run: verify_roundtrip" >&2
./test/verify_roundtrip 2>>"$LOG" | tee -a "$LOG" | awk -v sha="$SHA" '
    /=== Summary:/ {
        # "=== Summary: NN PASS, MM FAIL ==="
        for (i=1;i<=NF;i++) {
            if ($i ~ /^[0-9]+$/) {
                if ($(i+1) ~ /PASS/) pass=$i;
                if ($(i+1) ~ /FAIL/) fail=$i;
            }
        }
        printf "%s,verify_roundtrip,gpu,summary,2K,,,, pass=%s fail=%s\n", sha, pass, fail
    }' >> "$OUT"

# ---- 4. bench_modes: timing -------------------------------------------
build_one test/bench_modes test/bench_modes.cc || exit 1
echo "  run: bench_modes" >&2
./test/bench_modes 2>>"$LOG" | tee -a "$LOG" | awk -v sha="$SHA" '
    /CORRECT|FAST/ {
        # "CORRECT: 81.66 ms/frame  size=785222"
        ms="";
        for (i=1;i<=NF;i++) if ($i ~ /^[0-9.]+$/ && ms == "") ms=$i;
        mode = ($0 ~ /CORRECT/) ? "gpu_correct" : "gpu_fast";
        printf "%s,bench_modes,%s,timing,2K,,,%s,\n", sha, mode, ms
    }' >> "$OUT"

# ---- 5. cmp_dwt: DWT-domain stats -------------------------------------
build_one test/cmp_dwt test/cmp_dwt.cc || exit 1
echo "  run: cmp_dwt" >&2
./test/cmp_dwt 2>>"$LOG" | tee -a "$LOG" | tail -5 | awk -v sha="$SHA" '
    { gsub(/,/, ";"); printf "%s,cmp_dwt,gpu,dwt_stats,2K,,,, %s\n", sha, $0 }' >> "$OUT"

# ---- 6. verify_correctness_v2 (slow, optional) ------------------------
if [[ $QUICK -eq 0 ]]; then
    if [[ -x test/verify_correctness_v2 ]]; then
        echo "  run: verify_correctness_v2 (slow)" >&2
        ./test/verify_correctness_v2 2>>"$LOG" | tee -a "$LOG" | awk -v sha="$SHA" '
            /[0-9]+ PASS, *[0-9]+ FAIL/ {
                for (i=1;i<=NF;i++) {
                    if ($i ~ /^[0-9]+$/ && $(i+1) ~ /PASS/) pass=$i;
                    if ($i ~ /^[0-9]+$/ && $(i+1) ~ /FAIL/) fail=$i;
                }
                printf "%s,verify_correctness_v2,gpu,summary,2K,,,, pass=%s fail=%s\n", sha, pass, fail
            }' >> "$OUT"
    else
        echo "  skip: verify_correctness_v2 binary not present" >&2
    fi
fi

# ---- digest ------------------------------------------------------------
echo >&2
echo "Wrote $OUT" >&2
MIN_CORRECT=$(awk -F, '$3=="gpu_correct" && $6 != "" {print $6}' "$OUT" | sort -n | head -1)
MIN_FAST=$(awk -F, '$3=="gpu_fast" && $6 != "" {print $6}' "$OUT" | sort -n | head -1)
echo "min(gpu_correct PSNR) = ${MIN_CORRECT:-n/a}" >&2
echo "min(gpu_fast    PSNR) = ${MIN_FAST:-n/a}" >&2

echo "$OUT"
