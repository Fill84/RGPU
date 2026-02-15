#!/bin/sh
ldconfig
systemctl daemon-reload 2>/dev/null || true
