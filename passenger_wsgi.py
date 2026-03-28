"""
passenger_wsgi.py — WSGI entry point for shared hosting (cPanel / Passenger).

Deploy:
  1. Upload the libaix folder to your hosting account (e.g. ~/libaix/)
  2. In cPanel → Setup Python App, set:
       - Application root: /home/<user>/libaix
       - Application startup file: passenger_wsgi.py
       - Application Entry point: application
  3. pip install -r requirements.txt  (via SSH or cPanel terminal)
  4. python train_knowledge.py        (first time only)
  5. Restart the Python app in cPanel
"""

from __future__ import annotations

import os
import sys

# Ensure the app directory is on the Python path
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

os.chdir(app_dir)

from app import app as application  # noqa: E402, F401
