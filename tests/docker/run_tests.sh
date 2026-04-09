#!/bin/bash
set -euo pipefail

echo "========================================"
echo " RGPU Docker Integration Test Suite"
echo "========================================"
echo ""

# Generate rgpu.toml from environment variables
mkdir -p /etc/rgpu
cat > /etc/rgpu/rgpu.toml <<EOF
[client]
include_local_gpus = false
create_virtual_devices = false

[[client.servers]]
address = "${RGPU_SERVER}"
token = "${RGPU_TOKEN}"
EOF

echo "Config: server=${RGPU_SERVER}"
echo ""

# Start client daemon in background
echo "Starting RGPU client daemon..."
rgpu client --config /etc/rgpu/rgpu.toml &
DAEMON_PID=$!

# Wait for daemon IPC socket (Unix socket at /tmp/rgpu.sock)
echo "Waiting for daemon IPC socket..."
for i in $(seq 1 30); do
    if [ -S /tmp/rgpu.sock ]; then
        echo "Daemon ready (PID ${DAEMON_PID})."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Daemon did not start in time (30s timeout)"
        kill $DAEMON_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

echo ""

# Run tests
PASS=0
FAIL=0
TOTAL=0

run_test() {
    local name="$1"
    local cmd="$2"
    local env_setup="${3:-}"

    TOTAL=$((TOTAL + 1))
    echo "========================================"
    echo " Test: ${name}"
    echo "========================================"

    if eval "${env_setup} ${cmd}"; then
        echo ""
        echo ">>> ${name}: PASS"
        PASS=$((PASS + 1))
    else
        echo ""
        echo ">>> ${name}: FAIL"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# CUDA test
run_test "CUDA Interpose" \
    "/usr/local/bin/test_cuda" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_cuda_interpose.so"

# Vulkan test
run_test "Vulkan ICD" \
    "/usr/local/bin/test_vulkan" \
    "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/rgpu_icd.json VK_LOADER_DEBUG=error"

# NVML test
run_test "NVML Interpose" \
    "/usr/local/bin/test_nvml" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvml_interpose.so"

# NVENC test
run_test "NVENC Interpose" \
    "/usr/local/bin/test_nvenc" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvenc_interpose.so"

# NVDEC test
run_test "NVDEC Interpose" \
    "/usr/local/bin/test_nvdec" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvdec_interpose.so"

# Cleanup
echo "Stopping daemon..."
kill $DAEMON_PID 2>/dev/null || true
wait $DAEMON_PID 2>/dev/null || true

# Summary
echo ""
echo "========================================"
echo " RESULTS: ${PASS}/${TOTAL} passed, ${FAIL} failed"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    exit 1
fi
exit 0
