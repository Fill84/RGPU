#!/bin/bash
# Build RGPU Linux .so interpose libraries for Docker container integration.
#
# This script uses Docker to build the libraries inside a Linux container,
# then extracts them to packaging/linux/container-libs/.
#
# Prerequisites: Docker must be running.
#
# Usage:
#   ./packaging/linux/build-container-libs.sh
#
# Output:
#   packaging/linux/container-libs/
#     libcuda.so.1          (CUDA compute interpose)
#     libnvidia-ml.so.1     (NVML GPU discovery interpose)
#     libnvidia-encode.so.1 (NVENC video encoding interpose)
#     libnvcuvid.so.1       (NVDEC video decoding interpose)
#     librgpu_vk_icd.so     (Vulkan ICD)
#     + symlinks without version suffix

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/container-libs"

echo "=== Building RGPU container libraries ==="
echo "Project root: $PROJECT_ROOT"
echo "Output dir:   $OUTPUT_DIR"

# Build inside Docker
echo ""
echo "--- Building Linux .so files in Docker ---"
docker build -f "$SCRIPT_DIR/Dockerfile.builder" -t rgpu-builder "$PROJECT_ROOT"

# Extract artifacts
echo ""
echo "--- Extracting libraries ---"
mkdir -p "$OUTPUT_DIR"
docker rm -f rgpu-extract 2>/dev/null || true
docker create --name rgpu-extract rgpu-builder
docker cp rgpu-extract:/output/. "$OUTPUT_DIR/"
docker rm rgpu-extract

echo ""
echo "--- Built libraries ---"
ls -la "$OUTPUT_DIR/"

echo ""
echo "=== Done! ==="
echo ""
echo "To use in Docker containers, add to your docker-compose.yml:"
echo ""
echo "  services:"
echo "    my-app:"
echo "      volumes:"
echo "        - $OUTPUT_DIR:/usr/lib/rgpu:ro"
echo "      environment:"
echo "        - RGPU_IPC_ADDRESS=host.docker.internal:9877"
echo "        - LD_LIBRARY_PATH=/usr/lib/rgpu"
