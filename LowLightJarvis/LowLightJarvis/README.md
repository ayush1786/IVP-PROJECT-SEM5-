# Low-Light Image and Video Enhancement with Jarvis

A program that measures how dark an image or video is, picks an enhancement method automatically
(gamma correction, histogram equalization, CLAHE, Retinex, or smoothing + Retinex), and lets you
talk to an assistant called Jarvis. It has a web-style screen with three tabs: Jarvis (image),
Compare all methods, and Video.

## Files
| File | Purpose |
|---|---|
| `lowlight_app.py` | The whole program: analysis, enhancement methods, video, Jarvis, and the screen |
| `app.py` | Entry point for the website (Hugging Face Spaces) |
| `requirements.txt`, `packages.txt` | Libraries the website needs |
| `run_app.bat` | Double-click launcher for Windows |
| `run_app.sh` | Launcher for Mac / Linux |

---------------------------------------------------------------------

## A. Run it on your desktop

You need Python 3.10 or newer (free): https://www.python.org/downloads/
On Windows, tick **"Add python.exe to PATH"** in the installer.

**Windows**
1. Unzip the folder.
2. Double-click `run_app.bat`.
3. First run only: wait a few minutes while it installs everything.
4. Your browser opens at http://127.0.0.1:7860. Keep the black window open while using the app.
5. To close the app, close the black window.

**Mac / Linux**
1. Unzip the folder, open Terminal, and go into it, for example: `cd ~/Downloads/LowLightJarvis`
2. Run: `bash run_app.sh`
3. Your browser opens at http://127.0.0.1:7860. Press Ctrl+C in Terminal to stop.

## B. Run it as a website

**Option 1: quick link with Google Colab (about 5 minutes, temporary)**
1. Go to https://colab.research.google.com and create a new notebook.
2. Click the folder icon on the left, then the upload icon, and upload `lowlight_app.py`.
3. Run this in a cell:
```
!pip install -q gradio
import lowlight_app
lowlight_app.build_ui().launch(share=True, debug=True)
```
4. Click the link ending in `gradio.live`. It works while the Colab tab stays open.

**Option 2: permanent website with Hugging Face Spaces (free)**
1. Create a free account at https://huggingface.co
2. Click your profile picture, then **New Space**.
3. Give it a name, choose **Gradio** as the SDK, keep **Public**, and click **Create Space**.
4. Click **Files**, then **Add file**, then **Upload files**, and upload these four files:
   `app.py`, `lowlight_app.py`, `requirements.txt`, `packages.txt`
5. Click **Commit changes**. The Space builds for a few minutes (see the **Logs** tab).
6. Your website is at `https://huggingface.co/spaces/YOUR-USERNAME/YOUR-SPACE-NAME`

---------------------------------------------------------------------

## C. Using the app (a one-minute demo)
1. **Jarvis tab:** upload a dark photo. Jarvis scans it, says how dark it is, and enhances it automatically.
2. Check the before/after table under the enhanced image.
3. Type in the chat box: `use CLAHE`, `try retinex ssr`, `make it brighter`, `undo`, `compare all methods`,
   `what is retinex?`, `what time is it`, `12 * 5`.
4. **Compare all methods tab:** click the button to see every method side by side.
5. **Video tab:** upload a short dark video (the first 10 seconds are processed), choose Auto, and click
   **Enhance video**. The report shows brightness, flicker before and after, and processing speed.

Optional: to let Jarvis chat about any topic, set the environment variable `ANTHROPIC_API_KEY`
before starting the app. Without it, Jarvis works offline for everything in this project.

## D. If something goes wrong
- **"Python was not found" (Windows):** install Python and tick "Add python.exe to PATH", then run `run_app.bat` again.
  If a Microsoft Store window opens instead, install Python from python.org.
- **Setup failed:** check your internet connection and run the launcher again.
- **Page does not open:** go to http://127.0.0.1:7860 in your browser yourself.
- **Port already in use:** close the old black window first, or run `python lowlight_app.py --port 7861`.
- **Video will not play in the browser:** install ffmpeg (the website installs it automatically).
- **Slow video:** use a short clip. Only the first 10 seconds are processed, at up to 640 pixels wide.
