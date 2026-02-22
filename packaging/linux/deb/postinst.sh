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

# Update icon cache
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t /usr/share/icons/hicolor 2>/dev/null || true
fi

# Update desktop database
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database /usr/share/applications 2>/dev/null || true
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
