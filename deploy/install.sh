#!/usr/bin/env bash
set -euo pipefail

# RGPU Install Script
# Installs the RGPU binary, config, and systemd services.

BINARY_NAME="rgpu"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/rgpu"
SYSTEMD_DIR="/etc/systemd/system"

# Check for root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: this script must be run as root (or with sudo)"
    exit 1
fi

# Find the binary
BINARY_PATH=""
if [ -f "./target/release/${BINARY_NAME}" ]; then
    BINARY_PATH="./target/release/${BINARY_NAME}"
elif [ -f "./target/debug/${BINARY_NAME}" ]; then
    BINARY_PATH="./target/debug/${BINARY_NAME}"
else
    echo "Error: ${BINARY_NAME} binary not found. Run 'cargo build --release' first."
    exit 1
fi

echo "=== RGPU Installer ==="
echo ""

# Install binary
echo "Installing binary to ${INSTALL_DIR}/${BINARY_NAME}..."
install -m 755 "${BINARY_PATH}" "${INSTALL_DIR}/${BINARY_NAME}"

# Create config directory
echo "Creating config directory ${CONFIG_DIR}..."
mkdir -p "${CONFIG_DIR}"

# Install default config if none exists
if [ ! -f "${CONFIG_DIR}/rgpu.toml" ]; then
    echo "Installing default config to ${CONFIG_DIR}/rgpu.toml..."
    cat > "${CONFIG_DIR}/rgpu.toml" << 'TOML'
# RGPU Configuration
# See documentation for all options.

[server]
bind = "0.0.0.0"
port = 9876
server_id = 1
max_clients = 16
# cert_path = "/etc/rgpu/cert.pem"
# key_path = "/etc/rgpu/key.pem"
# transport = "tcp"  # or "quic"

[client]
# gpu_ordering = "LocalFirst"  # LocalFirst, RemoteFirst, ByCapability

# [[client.servers]]
# address = "gpu-server.local:9876"
# token = ""
# transport = "tcp"

[security]
# [[security.tokens]]
# token = "your-token-here"
# name = "client-1"
TOML
else
    echo "Config already exists at ${CONFIG_DIR}/rgpu.toml, skipping."
fi

# Install systemd services
if [ -d "${SYSTEMD_DIR}" ]; then
    echo "Installing systemd services..."

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    install -m 644 "${SCRIPT_DIR}/rgpu-server.service" "${SYSTEMD_DIR}/"
    install -m 644 "${SCRIPT_DIR}/rgpu-client.service" "${SYSTEMD_DIR}/"

    systemctl daemon-reload
    echo "Systemd services installed. Enable with:"
    echo "  sudo systemctl enable --now rgpu-server"
    echo "  sudo systemctl enable --now rgpu-client"
else
    echo "Systemd not found, skipping service installation."
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Quick start:"
echo "  Server: rgpu server --config /etc/rgpu/rgpu.toml"
echo "  Client: rgpu client --server <host>:9876 --config /etc/rgpu/rgpu.toml"
echo "  Token:  rgpu token --name my-client"
echo "  Info:   rgpu info --server <host>:9876"
