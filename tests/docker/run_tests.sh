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
SKIP=0
TOTAL=0
REQUIRED_FAIL=0

# run_test <name> <command> [env_setup] [required: true|false]
run_test() {
    local name="$1"
    local cmd="$2"
    local env_setup="${3:-}"
    local required="${4:-true}"

    TOTAL=$((TOTAL + 1))
    echo "========================================"
    echo " Test: ${name}${required:+ (required)}"
    echo "========================================"

    # Run test with a 30s timeout to prevent hangs/crashes from killing the suite
    if timeout 30 bash -c "${env_setup} ${cmd}" 2>&1; then
        echo ""
        echo ">>> ${name}: PASS"
        PASS=$((PASS + 1))
    else
        local exit_code=$?
        echo ""
        if [ "$required" = "true" ]; then
            echo ">>> ${name}: FAIL (exit code ${exit_code})"
            FAIL=$((FAIL + 1))
            REQUIRED_FAIL=$((REQUIRED_FAIL + 1))
        else
            echo ">>> ${name}: FAIL (optional, exit code ${exit_code})"
            FAIL=$((FAIL + 1))
        fi
    fi
    echo ""
}

# === Required tests (must pass) ===

# CUDA test — core functionality
run_test "CUDA Interpose" \
    "/usr/local/bin/test_cuda" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_cuda_interpose.so" \
    true

# NVML test — management library
run_test "NVML Interpose" \
    "/usr/local/bin/test_nvml" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvml_interpose.so" \
    true

# NVENC test
run_test "NVENC Interpose" \
    "/usr/local/bin/test_nvenc" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvenc_interpose.so" \
    true

# NVDEC test
run_test "NVDEC Interpose" \
    "/usr/local/bin/test_nvdec" \
    "LD_PRELOAD=/usr/lib/rgpu/librgpu_nvdec_interpose.so" \
    true

# FFmpeg hwaccel test — full encode/decode pipeline through RGPU
# Uses system symlinks (libcuda.so.1, libnvidia-encode.so.1, libnvcuvid.so.1)
# so FFmpeg finds our interpose libs via standard dlopen paths (no LD_PRELOAD needed)
run_test "FFmpeg HWAccel" \
    "/usr/local/bin/test_ffmpeg_hwaccel.sh" \
    "" \
    true

# === Optional tests (run last — may crash server in some environments) ===

# Vulkan test — run last because server-side Vulkan cleanup can crash on disconnect
run_test "Vulkan ICD" \
    "/usr/local/bin/test_vulkan" \
    "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/rgpu_icd.json VK_LOADER_DEBUG=error" \
    false

# Cleanup
echo "Stopping daemon..."
kill $DAEMON_PID 2>/dev/null || true
wait $DAEMON_PID 2>/dev/null || true

# Summary
echo ""
echo "========================================"
echo " RESULTS: ${PASS}/${TOTAL} passed, ${FAIL} failed (${REQUIRED_FAIL} required failures)"
if [ $FAIL -gt $REQUIRED_FAIL ] 2>/dev/null; then
    echo " Note: $((FAIL - REQUIRED_FAIL)) optional test(s) failed (WSL2/Docker limitation)"
fi
echo "========================================"

if [ $REQUIRED_FAIL -gt 0 ]; then
    exit 1
fi
exit 0
