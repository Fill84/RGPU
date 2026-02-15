#!/bin/sh
set -e

# Update shared library cache
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
fi

# Reload systemd if available
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload
fi

echo ""
echo "=== RGPU installed successfully ==="
echo ""
echo "To start the server:  sudo systemctl start rgpu-server"
echo "To start the client:  sudo systemctl start rgpu-client"
echo "To enable on boot:    sudo systemctl enable rgpu-server"
echo ""
echo "Config file: /etc/rgpu/rgpu.toml"
echo "Launch UI:   rgpu ui"
echo ""
