#!/usr/bin/env bash
set -euo pipefail

# RGPU Linux Package Build Script
#
# Usage:
#   ./build-linux.sh           # Build with default version
#   ./build-linux.sh 0.2.0     # Build with specific version
#   ./build-linux.sh --skip-build  # Skip cargo build, just package
#
# Prerequisites:
#   - Rust toolchain (rustup)
#   - cargo-deb: cargo install cargo-deb
#   - cargo-generate-rpm: cargo install cargo-generate-rpm

VERSION="${1:-0.1.0}"
SKIP_BUILD=false

if [ "${1:-}" = "--skip-build" ]; then
    SKIP_BUILD=true
    VERSION="${2:-0.1.0}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== RGPU Linux Package Build ==="
echo "Version: ${VERSION}"
echo "Project: ${PROJECT_ROOT}"
echo ""

# Step 1: Build release
if [ "$SKIP_BUILD" = false ]; then
    echo "[1/4] Building release..."
    cd "$PROJECT_ROOT"
    cargo build --release
    echo "  Build successful."
else
    echo "[1/4] Skipping build (--skip-build)"
fi

# Step 2: Verify artifacts
echo "[2/4] Verifying build artifacts..."
MISSING=false
for artifact in rgpu librgpu_cuda_interpose.so librgpu_vk_icd.so; do
    if [ -f "$PROJECT_ROOT/target/release/${artifact}" ]; then
        SIZE=$(du -h "$PROJECT_ROOT/target/release/${artifact}" | cut -f1)
        echo "  Found: ${artifact} (${SIZE})"
    else
        echo "  MISSING: ${artifact}"
        MISSING=true
    fi
done

if [ "$MISSING" = true ]; then
    echo "ERROR: Missing build artifacts. Run 'cargo build --release' first."
    exit 1
fi

# Step 3: Build .deb package
echo "[3/4] Building .deb package..."
if command -v cargo-deb >/dev/null 2>&1 || cargo deb --version >/dev/null 2>&1; then
    cd "$PROJECT_ROOT"
    cargo deb -p rgpu-cli --no-build
    echo "  .deb package:"
    ls -lh target/debian/rgpu_*.deb 2>/dev/null || echo "  (not found)"
else
    echo "  SKIP: cargo-deb not installed"
    echo "  Install with: cargo install cargo-deb"
fi

# Step 4: Build .rpm package
echo "[4/4] Building .rpm package..."
if command -v cargo-generate-rpm >/dev/null 2>&1 || cargo generate-rpm --version >/dev/null 2>&1; then
    cd "$PROJECT_ROOT"
    cargo generate-rpm -p crates/rgpu-cli
    echo "  .rpm package:"
    ls -lh target/generate-rpm/rgpu-*.rpm 2>/dev/null || echo "  (not found)"
else
    echo "  SKIP: cargo-generate-rpm not installed"
    echo "  Install with: cargo install cargo-generate-rpm"
fi

echo ""
echo "=== Linux packaging complete ==="
