#!/usr/bin/env bash
set -euo pipefail

# RGPU macOS Package Build Script
#
# Usage:
#   ./build-macos.sh           # Build with default version
#   ./build-macos.sh 0.2.0     # Build with specific version
#   ./build-macos.sh --skip-build  # Skip cargo build, just package
#
# Prerequisites:
#   - macOS with Xcode command-line tools (pkgbuild, productbuild)
#   - Rust toolchain (rustup)

VERSION="${1:-0.1.0}"
SKIP_BUILD=false

if [ "${1:-}" = "--skip-build" ]; then
    SKIP_BUILD=true
    VERSION="${2:-0.1.0}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PKG_DIR="$SCRIPT_DIR/pkg"
STAGING="$PROJECT_ROOT/target/macos-pkg-staging"
IDENTIFIER="com.rgpu"
OUTPUT_DIR="$PROJECT_ROOT/target"

echo "=== RGPU macOS Package Build ==="
echo "Version: ${VERSION}"
echo "Project: ${PROJECT_ROOT}"
echo ""

# Step 1: Build release
if [ "$SKIP_BUILD" = false ]; then
    echo "[1/5] Building release..."
    cd "$PROJECT_ROOT"
    cargo build --release
    echo "  Build successful."
else
    echo "[1/5] Skipping build (--skip-build)"
fi

# Step 2: Verify artifacts
echo "[2/5] Verifying build artifacts..."
MISSING=false
for artifact in rgpu librgpu_cuda_interpose.dylib librgpu_vk_icd.dylib; do
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

# Step 3: Create staging directory
echo "[3/5] Staging files..."
rm -rf "$STAGING"
mkdir -p "$STAGING/usr/local/bin"
mkdir -p "$STAGING/usr/local/lib/rgpu"
mkdir -p "$STAGING/usr/local/share/vulkan/icd.d"
mkdir -p "$STAGING/usr/local/share/rgpu"

# Copy artifacts
cp "$PROJECT_ROOT/target/release/rgpu" "$STAGING/usr/local/bin/"
cp "$PROJECT_ROOT/target/release/librgpu_cuda_interpose.dylib" "$STAGING/usr/local/lib/rgpu/"
cp "$PROJECT_ROOT/target/release/librgpu_vk_icd.dylib" "$STAGING/usr/local/lib/rgpu/"
cp "$PROJECT_ROOT/packaging/config/rgpu_icd_macos.json" "$STAGING/usr/local/share/vulkan/icd.d/rgpu_icd.json"
cp "$PROJECT_ROOT/packaging/config/rgpu.toml.template" "$STAGING/usr/local/share/rgpu/"
cp "$SCRIPT_DIR/launchd/"*.plist "$STAGING/usr/local/share/rgpu/"

# Set permissions
chmod 755 "$STAGING/usr/local/bin/rgpu"
chmod 644 "$STAGING/usr/local/lib/rgpu/"*
chmod 644 "$STAGING/usr/local/share/vulkan/icd.d/rgpu_icd.json"

echo "  Staged to: $STAGING"

# Step 4: Build component package
echo "[4/5] Building component package..."
chmod +x "$PKG_DIR/scripts/postinstall"

pkgbuild \
    --root "$STAGING" \
    --identifier "$IDENTIFIER" \
    --version "$VERSION" \
    --scripts "$PKG_DIR/scripts" \
    --install-location "/" \
    "$OUTPUT_DIR/rgpu-component.pkg"

echo "  Component package: $OUTPUT_DIR/rgpu-component.pkg"

# Step 5: Build product package (final installer with UI)
echo "[5/5] Building product package..."

# Update version in distribution.xml
sed "s/version=\"0.1.0\"/version=\"${VERSION}\"/g" "$PKG_DIR/distribution.xml" > "$OUTPUT_DIR/distribution.xml"

productbuild \
    --distribution "$OUTPUT_DIR/distribution.xml" \
    --package-path "$OUTPUT_DIR" \
    --resources "$PKG_DIR/resources" \
    "$OUTPUT_DIR/rgpu-${VERSION}-macos-x64.pkg"

# Cleanup
rm -f "$OUTPUT_DIR/rgpu-component.pkg"
rm -f "$OUTPUT_DIR/distribution.xml"
rm -rf "$STAGING"

FINAL_PKG="$OUTPUT_DIR/rgpu-${VERSION}-macos-x64.pkg"
if [ -f "$FINAL_PKG" ]; then
    SIZE=$(du -h "$FINAL_PKG" | cut -f1)
    echo ""
    echo "=== Build Complete ==="
    echo "  Package: ${FINAL_PKG} (${SIZE})"
    echo ""
    echo "  Install with: sudo installer -pkg ${FINAL_PKG} -target /"
    echo ""
    echo "  Note: The package is unsigned. Users may need to right-click"
    echo "  and select 'Open' to bypass Gatekeeper on first run."
else
    echo "ERROR: Package not found at ${FINAL_PKG}"
    exit 1
fi
