#!/usr/bin/env bash
# Mac / Linux launcher. First run installs everything (a few minutes), later runs start instantly.
cd "$(dirname "$0")"
if [ ! -f venv/installed.ok ]; then
  echo "First run: setting up. This takes a few minutes, please wait..."
  python3 -m venv venv || { echo "Python 3.10 or newer is required: https://www.python.org/downloads/"; exit 1; }
  source venv/bin/activate
  python -m pip install --upgrade pip
  pip install gradio -r requirements.txt && touch venv/installed.ok || { echo "Setup failed. Check your internet connection and try again."; exit 1; }
else
  source venv/bin/activate
fi
echo "Starting the app. Your browser will open automatically (or go to http://127.0.0.1:7860)."
python lowlight_app.py
