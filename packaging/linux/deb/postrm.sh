#!/bin/sh
set -e

# Update shared library cache
if command -v ldconfig >/dev/null 2>&1; then
    ldconfig
fi

# Reload systemd
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload
fi
