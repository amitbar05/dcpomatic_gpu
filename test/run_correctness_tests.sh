#!/bin/bash
# Runs all component correctness tests and summarizes results.
# Build all tests first: see individual build commands in each file.

cd "$(dirname "$0")/.."

PASS=0; FAIL=0; SKIP=0

run_test() {
    local name="$1"; local bin="$2"
    if [ ! -x "$bin" ]; then
        echo "SKIP $name (binary not found: $bin)"
        SKIP=$((SKIP+1)); return
    fi
    local out; out=$("$bin" 2>&1)
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name (exit $rc)"
        echo "$out" | tail -5 | sed 's/^/  /'
        FAIL=$((FAIL+1))
    fi
}

echo "=== GPU J2K Encoder Component Correctness Test Suite ==="
echo "Running on: $(date)"
echo ""

run_test "BitWriter"            test/bitwriter_correctness
run_test "TagTree"              test/tagtree_correctness
run_test "MQ-Coder"             test/mq_correctness
run_test "Bypass-Mode"          test/bypass_correctness
run_test "DWT"                  test/dwt_correctness
run_test "ICT"                  test/ict_correctness
run_test "Edge-Cases"           test/edge_case_tests
run_test "Quantization/QCD"     test/quantization_correctness
run_test "Roundtrip"            test/verify_roundtrip

echo ""
echo "=== Summary: $PASS PASS, $FAIL FAIL, $SKIP SKIP ==="
[ $FAIL -eq 0 ] && echo "ALL TESTS PASSED" || echo "SOME TESTS FAILED"
exit $FAIL
