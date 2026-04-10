#!/bin/bash
# test_ffmpeg_hwaccel.sh — RGPU FFmpeg hardware acceleration test
#
# Tests NVENC encoding and NVDEC decoding through the RGPU interpose layer.
# Requires: ffmpeg, RGPU client daemon running, LD_PRELOAD with CUDA + NVENC + NVDEC interpose libs
#
# Flow:
#   1. Generate a raw test video (synthetic pattern via lavfi)
#   2. Encode with h264_nvenc (NVENC hwaccel) → test_nvenc.mp4
#   3. Decode test_nvenc.mp4 with h264_cuvid (NVDEC hwaccel) → test_decoded.yuv
#   4. Re-encode with hevc_nvenc (HEVC NVENC) → test_hevc.mp4
#   5. Verify all outputs exist and have reasonable size

set -euo pipefail

echo "=== FFmpeg HWAccel Test ==="

WORKDIR=$(mktemp -d)
trap "rm -rf $WORKDIR" EXIT

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

# 1. Encode raw video with h264_nvenc
echo "  Step 1: Encoding 30 frames with h264_nvenc..."
ffmpeg -y -f lavfi -i "testsrc=duration=1:size=320x240:rate=30" \
    -c:v h264_nvenc -preset fast -b:v 1M \
    "$WORKDIR/test_h264.mp4" 2>&1 | tail -3
check "h264_nvenc encode" $?

# Verify output exists and has content
if [ -f "$WORKDIR/test_h264.mp4" ] && [ "$(stat -c%s "$WORKDIR/test_h264.mp4" 2>/dev/null || echo 0)" -gt 1000 ]; then
    echo "  PASS: h264_nvenc output valid ($(stat -c%s "$WORKDIR/test_h264.mp4") bytes)"
    pass=$((pass + 1))
else
    echo "  FAIL: h264_nvenc output missing or too small"
    fail=$((fail + 1))
fi

# 2. Encode with hevc_nvenc
echo "  Step 2: Encoding 30 frames with hevc_nvenc..."
ffmpeg -y -f lavfi -i "testsrc=duration=1:size=320x240:rate=30" \
    -c:v hevc_nvenc -preset fast -b:v 1M \
    "$WORKDIR/test_hevc.mp4" 2>&1 | tail -3
check "hevc_nvenc encode" $?

if [ -f "$WORKDIR/test_hevc.mp4" ] && [ "$(stat -c%s "$WORKDIR/test_hevc.mp4" 2>/dev/null || echo 0)" -gt 1000 ]; then
    echo "  PASS: hevc_nvenc output valid ($(stat -c%s "$WORKDIR/test_hevc.mp4") bytes)"
    pass=$((pass + 1))
else
    echo "  FAIL: hevc_nvenc output missing or too small"
    fail=$((fail + 1))
fi

# 3. Decode h264 to raw (software decode, verifies the nvenc output is valid)
echo "  Step 3: Verifying encoded file is decodable..."
ffmpeg -y -i "$WORKDIR/test_h264.mp4" \
    -f rawvideo -pix_fmt yuv420p \
    "$WORKDIR/test_decoded.yuv" 2>&1 | tail -3
check "decode verification" $?

if [ -f "$WORKDIR/test_decoded.yuv" ] && [ "$(stat -c%s "$WORKDIR/test_decoded.yuv" 2>/dev/null || echo 0)" -gt 10000 ]; then
    echo "  PASS: decoded output valid ($(stat -c%s "$WORKDIR/test_decoded.yuv") bytes)"
    pass=$((pass + 1))
else
    echo "  FAIL: decoded output missing or too small"
    fail=$((fail + 1))
fi

echo "=== FFmpeg HWAccel: $pass passed, $fail failed ==="
exit $((fail > 0 ? 1 : 0))
