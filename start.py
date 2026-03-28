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

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="libaix self-deploying launcher")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--retrain", action="store_true", help="Force retrain knowledge model")
    args = parser.parse_args()

    # Install deps
    _check_and_install()

    # Train if needed
    if args.retrain or (not args.skip_train and not _model_exists()):
        _train_model()
    elif not _model_exists():
        print("Warning: No trained model found. Chat will be unavailable.")
        print("  Run: python start.py --retrain\n")

    # Start the app
    print(f"Starting libaix on http://{args.host}:{args.port}")
    print(f"  Chat UI:  http://localhost:{args.port}/")
    print(f"  Admin:    http://localhost:{args.port}/admin")
    print("  Press Ctrl+C to stop.\n")

    from app import app
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
