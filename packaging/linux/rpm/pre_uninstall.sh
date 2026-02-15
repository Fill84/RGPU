#!/bin/sh
systemctl stop rgpu-server 2>/dev/null || true
systemctl stop rgpu-client 2>/dev/null || true
systemctl disable rgpu-server 2>/dev/null || true
systemctl disable rgpu-client 2>/dev/null || true
