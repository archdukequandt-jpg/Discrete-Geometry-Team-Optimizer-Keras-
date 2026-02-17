#!/bin/bash
# macOS launcher (permission-safe)
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# If the file is blocked by Gatekeeper, right-click → Open once.
# Ensure python3 exists
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

"$PY" launcher.py
