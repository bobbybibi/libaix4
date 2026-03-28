#!/usr/bin/env bash
# assist.sh — Project assistant for libaix.
# Usage: ./assist.sh <command>
#
# Automates common project tasks so you can do minimal work.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; exit 1; }

# ── Commands ────────────────────────────────────────────────────────

cmd_setup() {
    info "Installing dependencies…"
    pip install -r requirements.txt --quiet
    ok "Dependencies installed."
}

cmd_train() {
    info "Training the XOR neural network…"
    python train.py
    ok "Training complete."
}

cmd_test() {
    info "Running test suite…"
    python -m pytest tests/ -v
    ok "All tests passed."
}

cmd_lint() {
    info "Linting code with ruff…"
    python -m ruff check .
    ok "No lint issues found."
}

cmd_format() {
    info "Formatting code with ruff…"
    python -m ruff format .
    ok "Code formatted."
}

cmd_check() {
    info "Running all checks (lint + tests)…"
    cmd_lint
    cmd_test
    ok "All checks passed."
}

cmd_clean() {
    info "Cleaning build artifacts…"
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
    find . -name '*.pyc' -delete 2>/dev/null || true
    ok "Clean."
}

cmd_all() {
    cmd_setup
    echo
    cmd_lint
    echo
    cmd_test
    echo
    cmd_train
    echo
    ok "All tasks completed successfully."
}

cmd_help() {
    cat <<EOF

${CYAN}libaix project assistant${NC}

Usage: ./assist.sh <command>

Commands:
  ${GREEN}setup${NC}    Install all dependencies
  ${GREEN}train${NC}    Train the XOR neural network
  ${GREEN}test${NC}     Run the pytest test suite
  ${GREEN}lint${NC}     Lint code with ruff
  ${GREEN}format${NC}   Auto-format code with ruff
  ${GREEN}check${NC}    Run lint + tests
  ${GREEN}clean${NC}    Remove build artifacts and caches
  ${GREEN}all${NC}      Setup → lint → test → train (full pipeline)
  ${GREEN}help${NC}     Show this message

Examples:
  ./assist.sh setup          # first-time setup
  ./assist.sh all            # run everything
  ./assist.sh check          # quick validation before committing

EOF
}

# ── Main dispatcher ─────────────────────────────────────────────────

case "${1:-help}" in
    setup)  cmd_setup  ;;
    train)  cmd_train  ;;
    test)   cmd_test   ;;
    lint)   cmd_lint   ;;
    format) cmd_format ;;
    check)  cmd_check  ;;
    clean)  cmd_clean  ;;
    all)    cmd_all    ;;
    help)   cmd_help   ;;
    *)
        fail "Unknown command: $1"
        cmd_help
        ;;
esac
