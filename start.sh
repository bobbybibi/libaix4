#!/usr/bin/env bash
# start.sh — One-click launcher for libaix (Linux / macOS).
#
# Usage:
#   ./start.sh              # Install deps, train if needed, start server
#   ./start.sh --port 8080  # Custom port
#
# Drop the libaix folder anywhere and double-click or run this script.

set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════╗"
echo "║         libaix — AI launcher         ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Find Python 3
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        major=$(echo "$version" | cut -d. -f1)
        if [ "$major" -ge 3 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3 not found. Install it from https://www.python.org"
    exit 1
fi

echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

exec "$PYTHON" start.py "$@"
