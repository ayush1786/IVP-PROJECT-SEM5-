@echo off
rem Windows launcher. First run installs everything (a few minutes), later runs start instantly.
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python was not found.
  echo Install Python 3.10 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during installation. Then run this file again.
  pause
  exit /b 1
)

if not exist venv\installed.ok (
  echo First run: setting up. This takes a few minutes, please wait...
  %PY% -m venv venv
  call venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install gradio -r requirements.txt
  if errorlevel 1 (
    echo Setup failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
  echo ok> venv\installed.ok
) else (
  call venv\Scripts\activate.bat
)

echo Starting the app. Your browser will open automatically (or go to http://127.0.0.1:7860).
python lowlight_app.py
pause
