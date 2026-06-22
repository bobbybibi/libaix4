#!/usr/bin/env python3
"""
start.py — Zero-config self-deploying launcher for libaix.

Drop the whole libaix folder onto any PC or server, then run:

    python start.py

It will:
  1. Auto-install missing dependencies (numpy, flask, etc.)
  2. Train the knowledge AI model if no trained model exists
  3. Start the web server on http://0.0.0.0:5000

Options:
    python start.py                  # Default: install + train + serve
    python start.py --port 8080      # Custom port
    python start.py --host 127.0.0.1 # Bind to localhost only
    python start.py --skip-train     # Skip training even if model missing
    python start.py --retrain        # Force retrain the knowledge model
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

# Always run from the directory containing this script
os.chdir(Path(__file__).resolve().parent)

# ── Step 1: Auto-install dependencies ────────────────────────────────

REQUIRED = {
    "numpy": "numpy>=1.24",
    "flask": "flask>=3.0",
    "pypdf": "pypdf>=4.0",
    "werkzeug": "werkzeug>=3.0",
}


def _check_and_install() -> None:
    """Install any missing packages from requirements.txt."""
    missing: list[str] = []
    for mod, spec in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(spec)

    if not missing:
        return

    print(f"Installing missing packages: {', '.join(missing)}")
    req_file = Path("requirements.txt")
    if req_file.exists():
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"],
        )
    else:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "--quiet"],
        )
    print("Dependencies installed ✓\n")


# ── Step 2: Train knowledge model if needed ──────────────────────────

MODEL_DIR = Path("models")
REQUIRED_MODEL_FILES = ["knowledge.npz", "vectorizer.json", "answer_map.json"]


def _model_exists() -> bool:
    return all((MODEL_DIR / f).exists() for f in REQUIRED_MODEL_FILES)


def _train_model() -> None:
    print("Training knowledge AI model (first-time setup)…")
    subprocess.check_call([sys.executable, "train_knowledge.py"])
    print("Knowledge model trained ✓\n")


# ── Step 3: Launch ───────────────────────────────────────────────────

def _resolve_port(host: str, requested: int, strict: bool = False) -> int:
    """Return a port that is safe to bind, never killing whatever holds one.

    libaix only *probes* ports (a quick bind test) — it never terminates or
    signals the process already using a port. If ``requested`` is free, it is
    returned as-is. If it is busy:

      • ``strict=True``  → print guidance and exit (leave the other app alone).
      • ``strict=False`` → step aside onto the next free port and report it.
    """
    from net_utils import find_available_port, is_port_available

    if not (0 < requested <= 65535):
        print(f"Error: port {requested} is out of range (must be 1–65535).")
        sys.exit(2)

    if is_port_available(host, requested):
        return requested

    if strict:
        print(f"Error: port {requested} is already in use by another app.")
        print("  libaix will NOT terminate it. Pick another port with --port,")
        print("  or drop --strict-port to auto-select the next free port.")
        sys.exit(1)

    fallback = find_available_port(host, requested + 1)
    if fallback is None:
        print(f"Error: port {requested} is in use and no free port was found nearby.")
        print("  No other processes were touched. Try a different --port.")
        sys.exit(1)

    print(f"Note: port {requested} is in use by another app — leaving it untouched.")
    print(f"  Stepping aside to the next free port: {fallback}\n")
    return fallback


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="libaix self-deploying launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--retrain", action="store_true", help="Force retrain knowledge model")
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if --port is busy instead of auto-selecting the next free port "
        "(libaix never kills the app already using a port)",
    )
    args = parser.parse_args()

    # Resolve a safe port BEFORE doing any heavy work (install/train).
    port = _resolve_port(args.host, args.port, strict=args.strict_port)

    # Install deps
    _check_and_install()

    # Train if needed
    if args.retrain or (not args.skip_train and not _model_exists()):
        _train_model()
    elif not _model_exists():
        print("Warning: No trained model found. Chat will be unavailable.")
        print("  Run: python start.py --retrain\n")

    # Start the app
    display_host = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"Starting libaix on http://{args.host}:{port}")
    print(f"  Chat UI:  http://{display_host}:{port}/")
    print(f"  Admin:    http://{display_host}:{port}/admin")
    if args.host in ("0.0.0.0", "::"):
        print("  (also accessible via your machine's IP address on your network)")
    print("  Press Ctrl+C to stop.\n")

    from app import app
    app.run(host=args.host, port=port, debug=False)


if __name__ == "__main__":
    main()
