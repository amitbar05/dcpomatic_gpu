#!/bin/bash
# Automated CUDA/Slang J2K encoder optimization loop
# Continuously tests correctness, benchmarks, and iterates

set -e
cd "$(dirname "$0")/.."

ITER=0
BEST_CORRECT=999
BEST_FAST=999
LOG="test/optimize_log.txt"

echo "=== DCP-o-matic GPU J2K Encoder Optimization Loop ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

while true; do
    ITER=$((ITER + 1))
    echo "--- Iteration $ITER --- $(date) ---" | tee -a "$LOG"

    # 1. Build CUDA encoder
    echo "Building CUDA encoder..." | tee -a "$LOG"
    nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
        -I src -I src/lib -I/usr/include/openjpeg-2.5 \
        -o test/psnr_battery test/psnr_battery.cc \
        src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2 2>&1 | tail -1 | tee -a "$LOG"

    # 2. PSNR battery
    echo "PSNR battery:" | tee -a "$LOG"
    ./test/psnr_battery 2>&1 | tee -a "$LOG"

    # 3. Roundtrip verification
    echo "Building roundtrip test..." | tee -a "$LOG"
    nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
        -I src -I src/lib -I/usr/include/openjpeg-2.5 \
        -o test/verify_roundtrip test/verify_roundtrip.cc \
        src/lib/cuda_j2k_encoder.cu -lcudart -lopenjp2 2>&1 | tail -1 | tee -a "$LOG"

    echo "Roundtrip:" | tee -a "$LOG"
    FAILS=$(./test/verify_roundtrip 2>&1 | grep -c "FAIL" || true)
    echo "Failures: $FAILS" | tee -a "$LOG"

    # 4. Benchmark correct mode
    echo "Building bench_phases..." | tee -a "$LOG"
    nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
        -I src -I src/lib \
        -o test/bench_phases test/bench_phases.cc \
        src/lib/cuda_j2k_encoder.cu -lcudart 2>&1 | tail -1 | tee -a "$LOG"

    CORRECT_MS=$(DCP_GPU_BENCH=1 ./test/bench_phases 2>&1 | grep "mode=correct" | awk '{print $2}')
    FAST_MS=$(DCP_GPU_BENCH=1 ./test/bench_phases fast 2>&1 | grep "mode=fast" | awk '{print $2}')

    echo "Correct: ${CORRECT_MS}ms/frame  Fast: ${FAST_MS}ms/frame" | tee -a "$LOG"

    # Track bests
    if [ -n "$CORRECT_MS" ]; then
        CORRECT_VAL=$(echo "$CORRECT_MS" | cut -d' ' -f1)
        if [ "$(echo "$CORRECT_VAL < $BEST_CORRECT" | bc -l 2>/dev/null || true)" = "1" ]; then
            BEST_CORRECT=$CORRECT_VAL
            echo "NEW BEST CORRECT: ${BEST_CORRECT}ms" | tee -a "$LOG"
        fi
    fi

    echo "Best so far: correct=${BEST_CORRECT}ms fast=${BEST_FAST}ms" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # Wait a moment before next iteration
    sleep 2
done
