#!/bin/bash
# Comprehensive test runner for DCP-o-matic GPU J2K encoder

cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILS=0
PASSES=0

run_test() {
    local name="$1"
    local binary="$2"
    local timeout="${3:-120}"
    echo -n "  ${name}... "
    if [ ! -x "$binary" ]; then
        echo -e "${YELLOW}SKIP (binary not found)${NC}"
        return
    fi
    local fails=0 passes=0
    if timeout "$timeout" "$binary" > /tmp/test_out_$$.txt 2>&1; then
        :
    else
        local rc=$?
        if [ "$rc" -ne 124 ]; then
            echo -e "${RED}CRASH (exit code $rc)${NC}"
            head -5 /tmp/test_out_$$.txt
            FAILS=$((FAILS + 1))
            rm -f /tmp/test_out_$$.txt
            return
        fi
    fi
    fails=$(grep -c "FAIL:" /tmp/test_out_$$.txt 2>/dev/null; true)
    passes=$(grep -c "PASS:" /tmp/test_out_$$.txt 2>/dev/null; true)
    fails=${fails:-0}
    passes=${passes:-0}
    if [ "$fails" != "0" ] && [ "$fails" != "" ]; then
        echo -e "${RED}FAIL ($fails failures, $passes passes)${NC}"
        grep "FAIL:" /tmp/test_out_$$.txt | head -5
        FAILS=$((FAILS + fails))
    elif [ "$passes" != "0" ] && [ "$passes" != "" ]; then
        echo -e "${GREEN}PASS ($passes tests)${NC}"
    else
        echo -e "${GREEN}OK${NC}"
    fi
    PASSES=$((PASSES + passes))
    rm -f /tmp/test_out_$$.txt
}

echo "============================================"
echo " DCP-o-matic GPU J2K Encoder Test Suite"
echo "============================================"
echo ""

# Build all tests
echo "Building tests..."
nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
    -I src -I src/lib -I/usr/include/openjpeg-2.5 \
    -o test/psnr_battery test/psnr_battery.cc src/lib/cuda_j2k_encoder.cu \
    -lcudart -lopenjp2 2>&1 | grep -v "warning" || true

nvcc -O3 --use_fast_math -arch=sm_61 -std=c++17 \
    -I src -I src/lib -I/usr/include/openjpeg-2.5 \
    -o test/verify_roundtrip test/verify_roundtrip.cc src/lib/cuda_j2k_encoder.cu \
    -lcudart -lopenjp2 2>&1 | grep -v "warning" || true

echo ""

# Run tests
echo "--- PSNR Battery (2K, 150 Mbps) ---"
run_test "psnr_battery" "test/psnr_battery" 180

echo ""
echo "--- Round-Trip Verification ---"
run_test "verify_roundtrip" "test/verify_roundtrip" 180

echo ""
echo "--- Benchmark (if available) ---"
if [ -x "test/gpu_benchmark" ]; then
    timeout 60 test/gpu_benchmark 2>&1 | tail -20
fi

echo ""
echo "============================================"
echo " Summary: $((PASSES + FAILS)) total tests"
echo -e "  ${GREEN}$PASSES PASS${NC}, ${RED}$FAILS FAIL${NC}"
echo "============================================"

exit $FAILS
