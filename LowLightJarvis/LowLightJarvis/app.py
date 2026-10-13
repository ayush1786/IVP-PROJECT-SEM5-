# Entry point used by Hugging Face Spaces (and handy for `python app.py`)
from lowlight_app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.launch()
