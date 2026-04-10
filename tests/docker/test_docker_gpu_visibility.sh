#!/bin/bash
# test_docker_gpu_visibility.sh — Verify Docker sees remote GPUs
#
# This test runs INSIDE a Docker container started with --gpus all.
# It checks that nvidia-smi and CUDA see the expected number of GPUs.
#
# Expected environment:
#   EXPECTED_GPU_COUNT — total GPUs (local + remote) expected

set -euo pipefail

echo "=== Docker GPU Visibility Test ==="

EXPECTED=${EXPECTED_GPU_COUNT:-2}
pass=0
fail=0

check() {
    local name="$1"
    local result="$2"
    if [ "$result" -eq 0 ]; then
        echo "  PASS: $name"
        pass=$((pass + 1))
    else
        echo "  FAIL: $name"
        fail=$((fail + 1))
    fi
}

# 1. nvidia-smi must be available
nvidia-smi > /dev/null 2>&1
check "nvidia-smi runs" $?

# 2. nvidia-smi must show expected GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  INFO: nvidia-smi reports $GPU_COUNT GPU(s)"
if [ "$GPU_COUNT" -eq "$EXPECTED" ]; then
    echo "  PASS: GPU count matches expected ($EXPECTED)"
    pass=$((pass + 1))
else
    echo "  FAIL: GPU count $GPU_COUNT != expected $EXPECTED"
    fail=$((fail + 1))
fi

# 3. List all GPU names
echo "  INFO: GPUs found:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
    echo "    $line"
done

# 4. Check CUDA device count
if command -v python3 > /dev/null 2>&1; then
    CUDA_COUNT=$(python3 -c "
import ctypes
libcuda = ctypes.CDLL('libcuda.so.1')
libcuda.cuInit(0)
count = ctypes.c_int(0)
libcuda.cuDeviceGetCount(ctypes.byref(count))
print(count.value)
" 2>/dev/null || echo "0")
    echo "  INFO: CUDA reports $CUDA_COUNT device(s)"
    if [ "$CUDA_COUNT" -eq "$EXPECTED" ]; then
        echo "  PASS: CUDA device count matches expected ($EXPECTED)"
        pass=$((pass + 1))
    else
        echo "  FAIL: CUDA device count $CUDA_COUNT != expected $EXPECTED"
        fail=$((fail + 1))
    fi
fi

echo "=== Docker GPU Visibility: $pass passed, $fail failed ==="
exit $((fail > 0 ? 1 : 0))
