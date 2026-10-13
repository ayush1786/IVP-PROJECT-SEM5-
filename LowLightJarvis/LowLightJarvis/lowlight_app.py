"""
Low-Light Image and Video Enhancement with Jarvis
=================================================

One file that contains:
  1. Image analysis      - measures brightness, contrast, noise, detail (entropy)
  2. Enhancement methods - gamma, histogram equalization, CLAHE, Retinex (SSR/MSR),
                           and a "smooth first" step that removes blocky artifacts
  3. Smart auto mode     - picks a method from how dark the image is
  4. Video enhancement   - frame by frame, with temporal smoothing to stop flicker
  5. Jarvis              - a chat assistant that scans, enhances, compares and explains
  6. A Gradio web app    - upload / webcam, before-after, chat, compare, video

Run on your laptop:   python lowlight_app.py
Run in Google Colab:  !pip install -q gradio
                      import lowlight_app
                      lowlight_app.build_ui().launch(share=True, debug=True)

Optional: set the environment variable ANTHROPIC_API_KEY to let Jarvis chat about
any topic (general conversation). Without it, Jarvis works fully offline for everything
related to this project plus small questions (time, date, maths, jokes).
"""
import ast
import datetime
import json
import operator
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import cv2
import numpy as np


# =============================================================================
# 1. IMAGE ANALYSIS
# =============================================================================
def to_gray_f(img):
    """Grey version of a BGR image as float32 (0-255 scale)."""
    f = img.astype(np.float32)
    return f[:, :, 0] * 0.114 + f[:, :, 1] * 0.587 + f[:, :, 2] * 0.299


def get_brightness(img):
    return float(to_gray_f(img).mean() / 255.0)


def get_contrast(img):
    return float(to_gray_f(img).std() / 255.0)


def get_noise(img):
    """Noise estimate (Immerkaer's method)."""
    gray = to_gray_f(img)
    h, w = gray.shape
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    filtered = cv2.filter2D(gray, -1, kernel)
    return float(np.sum(np.abs(filtered)) * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2)))


def get_entropy(img):
    gray = np.clip(to_gray_f(img), 0, 255).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def darkness_level(brightness):
    if brightness >= 0.40:
        return "Normal"
    if brightness >= 0.25:
        return "Dim"
    if brightness >= 0.12:
        return "Dark"
    return "Very dark"


def analyze(img):
    b = get_brightness(img)
    return {
        "brightness": b,
        "contrast": get_contrast(img),
        "noise": get_noise(img),
        "entropy": get_entropy(img),
        "dark_ratio": float((to_gray_f(img) < 0.15 * 255).mean()),
        "level": darkness_level(b),
    }


# =============================================================================
# 2. ENHANCEMENT METHODS  (every function: BGR uint8 in, BGR uint8 out)
# =============================================================================
def auto_gamma_value(img):
    mean = float(np.clip(get_brightness(img), 0.01, 0.99))
    return float(np.clip(np.log(0.45) / np.log(mean), 0.25, 1.0))


def gamma_correction(img, gamma=None):
    """out = in ** gamma on decimal pixel values (gamma < 1 brightens)."""
    if gamma is None:
        gamma = auto_gamma_value(img)
    out = np.power(np.clip(img.astype(np.float32), 0, 255) / 255.0, gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def histogram_equalization(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhance(img, clip_limit=3.0, tile_size=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    engine = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = engine.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def fast_blur(channel, sigma):
    """Gaussian blur. For large sigma we blur a shrunken copy (same result, much faster)."""
    factor = int(max(1, sigma // 16))
    if factor == 1:
        return cv2.GaussianBlur(channel, (0, 0), sigma)
    h, w = channel.shape
    small = cv2.resize(channel, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / factor)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def retinex_raw(img, sigmas):
    """log(image) - log(blurred image), averaged over the given blur sizes."""
    f = img.astype(np.float32) + 1.0
    out = np.zeros_like(f)
    for c in range(3):
        ch = f[:, :, c]
        total = 0
        for s in sigmas:
            total = total + (np.log10(ch) - np.log10(fast_blur(ch, s) + 1e-6))
        out[:, :, c] = total / len(sigmas)
    return out


def stretch(x, low, high):
    return (np.clip((x - low) / max(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)


def retinex(img, sigmas):
    raw = retinex_raw(img, sigmas)
    low, high = np.percentile(raw, [1, 99])
    return stretch(raw, low, high)


def retinex_ssr(img, sigma=80):
    return retinex(img, (sigma,))


def retinex_msr(img, sigmas=(15, 80, 250)):
    return retinex(img, sigmas)


def smooth_dark_image(img, strength=1.0):
    """Remove blocky artifacts and noise BEFORE enhancing. Returns float32 (not rounded)."""
    scale = max(img.shape[:2]) / 600.0
    luma_sigma = 1.5 * strength * scale        # light blur on brightness (keeps detail)
    chroma_sigma = 8.0 * strength * scale      # strong blur on colour (hides colour blocks)
    ycrcb = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.GaussianBlur(ycrcb[:, :, 0], (0, 0), luma_sigma)
    ycrcb[:, :, 1] = cv2.GaussianBlur(ycrcb[:, :, 1], (0, 0), chroma_sigma)
    ycrcb[:, :, 2] = cv2.GaussianBlur(ycrcb[:, :, 2], (0, 0), chroma_sigma)
    out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR) * 255.0
    return np.clip(out, 0, 255).astype(np.float32)


NATURAL = "Natural (smooth night)"


def natural_core(img, gamma=None, target=0.30):
    """Smooth, natural-looking enhancement that also works on almost-black photos.

    Why it looks better than Retinex on very dark images: a very dark photo only has a handful of
    brightness levels (0, 1, 2 ...). Brightening exaggerates those rounding steps into blocky contours.
    So we (1) average them away with a heavy blur wherever the picture is dark, but keep a light blur
    where there is real detail, (2) brighten with a gentle gamma curve, and (3) take the colour from a
    heavily blurred copy so no colour blocks appear. Returns (image, gamma that suits this frame).
    """
    f = img.astype(np.float32) / 255.0
    s = max(img.shape[:2]) / 1000.0                       # blur sizes scale with the picture size
    y = f[:, :, 0] * 0.114 + f[:, :, 1] * 0.587 + f[:, :, 2] * 0.299
    y_light = cv2.GaussianBlur(y, (0, 0), 1.0 * s)
    # the darker the photo, the fewer real brightness levels it has, so the stronger the averaging we need
    t = float(np.clip((float(y.mean()) - 0.02) / 0.08, 0.0, 1.0))
    y_heavy = fast_blur(y, (9.0 * (1 - t) + 2.5 * t) * s)
    w = np.clip((y_heavy - 0.01) / 0.07, 0.0, 1.0)         # 0 = dark area, 1 = bright area with real detail
    w = w * w * (3 - 2 * w)
    ys = w * y_light + (1 - w) * y_heavy
    auto_gamma = float(np.clip(np.log(target) / np.log(np.clip(ys.mean(), 0.002, 0.9)), 0.3, 1.0))
    g = auto_gamma if gamma is None else gamma
    yt = np.power(np.clip(ys, 0, 1), g)
    eps = 0.004
    cs = np.stack([fast_blur(f[:, :, c], 6.0 * s) for c in range(3)], axis=2)
    ycs = cs[:, :, 0] * 0.114 + cs[:, :, 1] * 0.587 + cs[:, :, 2] * 0.299
    ratio = np.power((cs + eps) / (ycs[:, :, None] + eps), 0.85)
    out = np.clip(yt[:, :, None] * ratio, 0, 1)
    return (out * 255).astype(np.uint8), auto_gamma


def natural_enhance(img):
    return natural_core(img)[0]


METHODS = {
    "Gamma": gamma_correction,
    "Histogram equalization": histogram_equalization,
    "CLAHE": clahe_enhance,
    "Retinex SSR": retinex_ssr,
    "Retinex MSR": retinex_msr,
    "Smooth + Gamma": lambda im: gamma_correction(smooth_dark_image(im, 1.0)),
    "Smooth + Retinex MSR": lambda im: retinex_msr(smooth_dark_image(im, 1.0)),
    NATURAL: natural_enhance,
}
AUTO_LIGHT = "Retinex MSR (light smoothing)"
ALL_METHODS = dict(METHODS)
ALL_METHODS[AUTO_LIGHT] = lambda im: retinex_msr(smooth_dark_image(im, 0.4))


# =============================================================================
# 3. SMART AUTO MODE
# =============================================================================
def plan_for_level(level):
    """Which method suits which darkness level, and why."""
    if level == "Normal":
        return "None", "The image already has good brightness, so no enhancement is needed."
    if level == "Dim":
        return "CLAHE", "The image is only slightly dim, so adaptive contrast enhancement (CLAHE) is enough."
    if level == "Dark":
        return NATURAL, ("The image is dark, so I used my natural night enhancement: it brightens smoothly "
                         "and keeps noise and colour blocks under control.")
    return NATURAL, ("The image is very dark, so I averaged away its rounding noise first and then brightened it "
                     "with a gentle curve, which keeps the result smooth and natural.")


def resize_max(img, max_side=1280):
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def apply_method(img, method):
    if method == "None":
        return img.copy()
    return ALL_METHODS[method](img)


def auto_enhance(img):
    before = analyze(img)
    method, reason = plan_for_level(before["level"])
    out = apply_method(img, method)
    return {"image": out, "method": method, "reason": reason, "before": before, "after": analyze(out)}


def manual_enhance(img, method):
    before = analyze(img)
    out = apply_method(img, method)
    return {"image": out, "method": method, "reason": f"You asked for {method}.",
            "before": before, "after": analyze(out)}


def compare_all(img):
    """Run every method. Returns a list of (name, image, stats)."""
    rows = [("Original", img, analyze(img))]
    for name in METHODS:
        out = METHODS[name](img)
        rows.append((name, out, analyze(out)))
    return rows


def to_bgr(arr):
    """Gradio gives RGB arrays (maybe grey or RGBA). Convert to a BGR uint8 image."""
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return resize_max(arr)


def to_rgb(img):
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stats_markdown(result):
    b, a = result["before"], result["after"]
    return (f"**Darkness level:** {b['level']}  |  **Method:** {result['method']}\n\n"
            "| Measure | Before | After |\n|---|---|---|\n"
            f"| Brightness (0-1) | {b['brightness']:.3f} | {a['brightness']:.3f} |\n"
            f"| Contrast (0-1) | {b['contrast']:.3f} | {a['contrast']:.3f} |\n"
            f"| Entropy (detail) | {b['entropy']:.2f} | {a['entropy']:.2f} |\n"
            f"| Noise estimate | {b['noise']:.2f} | {a['noise']:.2f} |\n")


# =============================================================================
# 4. VIDEO ENHANCEMENT WITH FLICKER REDUCTION
# =============================================================================
class TemporalEnhancer:
    """Enhances frames one by one while keeping brightness steady over time.

    Enhancing each frame alone makes brightness jump between frames (flicker).
    We stabilise it by smoothing, over time, the numbers that decide the brightness
    (the gamma value, or the Retinex stretch range) and by lightly blending each
    output with the previous one.
    """

    def __init__(self, method, alpha=0.9, blend=0.1):
        self.method, self.alpha, self.blend = method, alpha, blend
        self.gamma = None
        self.bounds = None
        self.prev = None

    def _ema(self, old, new):
        return new if old is None else self.alpha * old + (1 - self.alpha) * new

    def process(self, frame):
        m = self.method
        if m in ("Gamma", "Smooth + Gamma"):
            src = smooth_dark_image(frame, 1.0) if m.startswith("Smooth") else frame
            self.gamma = self._ema(self.gamma, auto_gamma_value(src))
            out = gamma_correction(src, self.gamma)
        elif m == NATURAL:
            out, auto_g = natural_core(frame, gamma=self.gamma)
            self.gamma = auto_g if self.gamma is None else self.alpha * self.gamma + (1 - self.alpha) * auto_g
        elif "Retinex" in m:
            strength = {"Smooth + Retinex MSR": 1.0, AUTO_LIGHT: 0.4}.get(m, 0.0)
            src = smooth_dark_image(frame, strength) if strength else frame
            raw = retinex_raw(src, (80,) if m == "Retinex SSR" else (15, 80, 250))
            low, high = np.percentile(raw, [1, 99])
            self.bounds = (self._ema(None if self.bounds is None else self.bounds[0], low),
                           self._ema(None if self.bounds is None else self.bounds[1], high))
            out = stretch(raw, *self.bounds)
        else:
            out = ALL_METHODS[m](frame)
        if self.prev is not None and self.prev.shape == out.shape:
            out = cv2.addWeighted(out, 1 - self.blend, self.prev, self.blend, 0)
        self.prev = out
        return out


def relative_flicker(values):
    """Average frame-to-frame brightness change, relative to the average brightness (in %)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or v.mean() == 0:
        return 0.0
    return float(np.abs(np.diff(v)).mean() / v.mean() * 100.0)


def enhance_video(path, method="Auto", max_seconds=10, max_side=640):
    """Enhance a video file. Returns (output_path, report_markdown)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "Could not open that video file."
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if method == "Auto":
        samples = []
        positions = np.linspace(0, max(total - 1, 0), 10).astype(int) if total else [0]
        for p in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
            ok, fr = cap.read()
            if ok:
                samples.append(get_brightness(fr))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not samples:
            return None, "Could not read frames from that video."
        level = darkness_level(float(np.mean(samples)))
        method, reason = plan_for_level(level)
        if method == "None":
            cap.release()
            return path, f"**Darkness level:** {level}. {reason}"
    else:
        level, reason = "n/a", f"You asked for {method}."

    max_frames = int(fps * max_seconds)
    enhancer = TemporalEnhancer(method)
    tmp_dir = tempfile.mkdtemp()
    raw_path = os.path.join(tmp_dir, "raw.mp4")
    writer, before, after, n = None, [], [], 0
    t0 = time.time()
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_max(frame, max_side)
        out = enhancer.process(frame)
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)
        before.append(get_brightness(frame))
        after.append(get_brightness(out))
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    if n == 0:
        return None, "Could not read frames from that video."
    elapsed = max(time.time() - t0, 1e-6)

    final_path = os.path.join(tmp_dir, "enhanced.mp4")
    if shutil.which("ffmpeg"):       # re-encode so every browser can play it
        done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
                               "-vcodec", "libx264", "-pix_fmt", "yuv420p", final_path])
        if done.returncode != 0 or not os.path.exists(final_path):
            final_path = raw_path
    else:
        final_path = raw_path

    report = (f"**Darkness level:** {level}  |  **Method:** {method}\n\n{reason}\n\n"
              "| Measure | Original | Enhanced |\n|---|---|---|\n"
              f"| Average brightness (0-1) | {np.mean(before):.3f} | {np.mean(after):.3f} |\n"
              f"| Flicker (relative brightness change per frame) | {relative_flicker(before):.2f}% | "
              f"{relative_flicker(after):.2f}% |\n\n"
              f"Processed {n} frames at {n / elapsed:.1f} frames per second "
              f"(first {max_seconds} seconds of the video, resized to at most {max_side}px).")
    return final_path, report


# =============================================================================
# 5. JARVIS (the assistant)
# =============================================================================
CONCEPTS = {
    "gamma": "Gamma correction replaces every pixel value (scaled 0-1) by value ** gamma. With gamma below 1, "
             "dark pixels are lifted much more than bright ones. It is fast, but it can amplify noise and wash out colours.",
    "histogram": "Histogram equalization spreads the pixel values so they use the whole range from black to white. "
                 "It boosts overall contrast quickly, but the result can look harsh. I apply it to brightness only so colours stay natural.",
    "clahe": "CLAHE equalizes small tiles of the image separately, with a clip limit that stops any tile being boosted too much. "
             "It improves local contrast with less harshness, but it barely helps when an image is almost completely black.",
    "retinex": "Retinex assumes an image is the scene multiplied by the lighting. It estimates the lighting by blurring, then removes it "
               "using log(image) - log(blurred image). SSR uses one blur size, MSR averages several. It recovers shadow detail well, "
               "but can look greyish or noisy.",
    "psnr": "PSNR (Peak Signal-to-Noise Ratio) measures pixel error against a well-lit reference photo, in decibels. Higher is better.",
    "ssim": "SSIM (Structural Similarity) compares structure, contrast and brightness patterns with a reference photo. "
            "It ranges from 0 to 1, and closer to 1 is better.",
    "entropy": "Entropy measures how much information (detail) an image holds. Higher usually means more visible detail, "
               "but noise also raises it, so read it together with the picture.",
    "noise": "Brightening a dark photo also amplifies its noise and compression blocks. That is why I smooth very dark images first, "
             "in decimal numbers, and then enhance them.",
    "flicker": "In video, enhancing each frame alone makes brightness jump between frames. I reduce this by smoothing the brightness "
               "settings over time and blending each frame lightly with the previous one.",
    "low light": "Low-light enhancement turns dark, noisy images into clearer, brighter ones. The hard part is brightening shadows "
                 "without amplifying noise or losing natural colours.",
    "smooth": "Smoothing before enhancing removes noise and compression blocks from a very dark image. I blur colour more than brightness, "
              "because colour blocks are the ugliest artifacts, and I keep decimal numbers so no new rounding steps appear.",
}
JOKES = [
    "Why did the photo go to therapy? It had too many unresolved shadows.",
    "I told my camera a joke in the dark. It said it didn't get it, and I should add more light.",
    "Retinex walks into a bar. The bartender says: why so dark? Retinex says: not for long.",
]
HELP = ("Here is what I can do:\n"
        "- **Scan an image:** upload a photo and I will measure how dark it is and enhance it automatically.\n"
        "- **Enhance on request:** say things like *enhance this*, *use CLAHE*, *try retinex*, *make it brighter*, or *undo*.\n"
        "- **Compare methods:** say *compare all methods*.\n"
        "- **Explain:** ask *what is retinex?*, *explain PSNR*, *what is flicker?*\n"
        "- **Small questions:** the time, the date, quick maths, or a joke.\n"
        "- **Real conversation:** open *Jarvis brain settings* and connect an AI model to chat about anything.\n"
        "- **Voice:** press the microphone button to talk to me, and keep *Jarvis speaks* ticked to hear my replies.\n"
        "- **Video:** use the Video tab to enhance a dark video.")
PERSONA = (
    "You are Jarvis, the voice assistant of a student's low-light image and video enhancement project, built with Python and OpenCV. "
    "Talk like a warm, natural, curious person in a real conversation, not like a manual. "
    "Your replies are read out loud, so use short, plain sentences. Never use markdown, bullet points, tables, or emojis. "
    "Keep most replies to two to four sentences unless the user asks for detail. "
    "You can chat about anything: general knowledge, study help, advice, maths, jokes. "
    "Be honest, and say so when you do not know something. "
    "You cannot see what is in the photo. You only know the measurements your tools give you, so never describe the picture's content. "
    "When the user wants something done to their image (scan it, check how dark it is, enhance it, make it brighter or darker, "
    "undo, compare methods), use your tools, and then tell them what you did in plain words with the key numbers, "
    "for example the brightness before and after. "
    "If no image is loaded, ask them to upload one or use the webcam. "
    "Video enhancement is on the Video tab, so point them there. "
    "If asked how things work (gamma correction, histogram equalization, CLAHE, Retinex, PSNR, SSIM, entropy, flicker), "
    "explain simply with an everyday example. "
    "Reply in English unless the user asks for another language."
)


def safe_math(text):
    """Evaluate simple arithmetic safely. Returns a string or None."""
    expr = text.lower()
    for word in ["what is", "what's", "calculate", "compute", "whats", "solve", "="]:
        expr = expr.replace(word, " ")
    expr = expr.replace("?", " ").replace("^", "**").replace("x", "*").replace(",", "").strip()
    if not re.fullmatch(r"[\d\s\.\+\-\*/\(\)%]+", expr) or not re.search(r"\d", expr) \
            or not re.search(r"[\+\-\*/%]", expr.lstrip("-")):
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -ev(node.operand) if isinstance(node.op, ast.USub) else ev(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return ops[type(node.op)](left, right)
        raise ValueError("unsupported")

    try:
        value = ev(ast.parse(expr, mode="eval").body)
    except Exception:
        return None
    return str(round(value, 6)) if isinstance(value, float) else str(value)


def content_text(content):
    """Chat messages are plain text in some Gradio versions and a list of text blocks in others."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


# -----------------------------------------------------------------------------
# THE AI BRAIN
# Lets Jarvis use a real language model (Gemini, Groq, Claude or any OpenAI-compatible
# service). The model talks naturally AND decides by itself when to call the image tools
# below (scan, enhance, brighten, compare, undo). No extra libraries are needed.
# -----------------------------------------------------------------------------
OFFLINE = "Offline (built-in rules, no AI model)"
CUSTOM = "OpenAI-compatible (custom)"
PROVIDERS = {
    OFFLINE: {"kind": "offline"},
    "Google Gemini (free key)": {
        "kind": "openai", "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": "GEMINI_API_KEY", "default_model": "gemini-flash-latest",
        "prefer": ["gemini-flash-latest", "flash"], "key_url": "https://aistudio.google.com/apikey"},
    "Groq (free key)": {
        "kind": "openai", "base": "https://api.groq.com/openai/v1",
        "env": "GROQ_API_KEY", "default_model": "llama-3.3-70b-versatile",
        "prefer": ["llama-3.3-70b-versatile", "gpt-oss-120b", "llama-3.1-8b-instant", "llama"],
        "key_url": "https://console.groq.com/keys"},
    "Anthropic Claude": {
        "kind": "anthropic", "base": "https://api.anthropic.com/v1",
        "env": "ANTHROPIC_API_KEY", "default_model": "claude-haiku-5-5",
        "prefer": ["haiku", "sonnet"], "newest_first": True, "key_url": "https://console.anthropic.com"},
    CUSTOM: {
        "kind": "openai", "base": "https://api.openai.com/v1", "env": "OPENAI_API_KEY",
        "default_model": "", "prefer": ["gpt-4o-mini", "gpt"], "key_url": "your provider's website"},
}
MAX_TOOL_ROUNDS = 4
BAD_MODEL_WORDS = ("embed", "tts", "whisper", "guard", "image", "audio", "live", "transcribe", "moderation",
                   "imagen", "veo", "aqa", "robotics", "computer-use", "learnlm", "gemma", "orpheus",
                   "safeguard", "realtime", "dall", "davinci", "babbage")

# The actions Jarvis can take on the user's image. The AI model chooses when to call them.
TOOL_SPECS = [
    {"name": "analyze_image",
     "description": "Measure the user's current image (brightness, contrast, noise, detail) and report how dark it is. "
                    "Does not change the picture.",
     "properties": {}},
    {"name": "enhance_image",
     "description": "Enhance the user's dark image. Use method 'auto' unless the user asks for a specific method.",
     "properties": {"method": {"type": "string", "enum": ["auto"] + list(METHODS),
                               "description": "auto, or one specific method"}}},
    {"name": "adjust_brightness",
     "description": "Make the picture currently on screen brighter or darker.",
     "properties": {"direction": {"type": "string", "enum": ["brighter", "darker"]}},
     "required": ["direction"]},
    {"name": "compare_methods",
     "description": "Run every enhancement method on the original image and report brightness, contrast and detail for each.",
     "properties": {}},
    {"name": "reset_image",
     "description": "Show the user's original image again, undoing all enhancement.",
     "properties": {}},
]


class LLMError(Exception):
    """A problem talking to the AI service, worded so Jarvis can say it out loud."""


def default_provider():
    """Choose the AI provider automatically when a key is already set in the environment."""
    for name, cfg in PROVIDERS.items():
        if cfg.get("env") and os.environ.get(cfg["env"], "").strip():
            return name
    return OFFLINE


def friendly_http_error(code, body):
    if code in (401, 403):
        return "The AI service rejected the API key. Check that you pasted the whole key for the provider you selected."
    if code == 404:
        return "The AI service could not find that model or address. Clear the Model box so I can choose a model myself."
    if code == 429:
        return "The AI service says I am asking too fast, or the free quota is used up. Wait a minute and try again."
    if code == 400:
        return "The AI service refused the request (" + body.strip()[:150] + ")."
    return "The AI service had a problem (error " + str(code) + ")."


def http_json(url, payload=None, headers=None, timeout=60):
    """Send a JSON request with the standard library only. GET when payload is None."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "User-Agent": "LowLightJarvis/1.0"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=hdrs, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise LLMError(friendly_http_error(err.code, body)) from None
    except urllib.error.URLError as err:
        raise LLMError("I could not reach the AI service (" + str(err.reason) + "). Check the internet connection.") from None
    except (TimeoutError, OSError):
        raise LLMError("The AI service took too long to answer. Please try again.") from None
    except ValueError:
        raise LLMError("The AI service sent a reply I could not read.") from None


def check_base_url(url):
    url = url.strip().rstrip("/")
    local = re.match(r"^http://(localhost|127\.0\.0\.1)(:\d+)?(/|$)", url)
    if not (url.startswith("https://") or local):
        raise LLMError("The base URL must start with https:// (or http://localhost for a model on this computer).")
    return url


def make_brain(provider, key, model, base_url):
    """Turn the screen settings into a brain description. Returns None for offline mode."""
    cfg = PROVIDERS.get(provider or OFFLINE) or PROVIDERS[OFFLINE]
    if cfg["kind"] == "offline":
        return None
    key = (key or "").strip() or os.environ.get(cfg.get("env", ""), "").strip()
    base = cfg["base"]
    if provider == CUSTOM and (base_url or "").strip():
        base = check_base_url(base_url)
    return {"provider": provider, "key": key, "model": (model or "").strip(), "base": base.rstrip("/")}


# ---- choosing a model automatically (so an out-of-date model name can never break Jarvis)
_MODEL_CACHE = {}


def list_model_ids(cfg, key, base):
    if cfg["kind"] == "anthropic":
        data = http_json(base + "/models?limit=100", None, {"x-api-key": key, "anthropic-version": "2023-06-01"})
    else:
        data = http_json(base + "/models", None, {"Authorization": "Bearer " + key})
    ids = []
    for item in (data.get("data") or data.get("models") or []):
        mid = item.get("id") or item.get("name") if isinstance(item, dict) else None
        if mid:
            mid = str(mid)
            ids.append(mid[len("models/"):] if mid.startswith("models/") else mid)
    return ids


def version_key(model_id):
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", model_id)] or [0.0]


def pick_model(ids, cfg):
    good = [i for i in ids if not any(w in i.lower() for w in BAD_MODEL_WORDS)]
    for pref in cfg.get("prefer", []):
        found = [i for i in good if pref in i]
        if not found:
            continue
        stable = [i for i in found if not re.search(r"preview|exp|lite|thinking", i)] or found
        return stable[0] if cfg.get("newest_first") else max(stable, key=version_key)
    return good[0] if good else None


def resolve_model(brain):
    cfg = PROVIDERS[brain["provider"]]
    if brain["model"]:
        return brain["model"]
    cache_key = (brain["provider"], brain["base"], brain["key"][-6:])
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    try:
        chosen = pick_model(list_model_ids(cfg, brain["key"], brain["base"]), cfg)
    except LLMError:
        chosen = None                       # fall back to the default; a bad key is reported by the chat call itself
    if chosen:
        _MODEL_CACHE[cache_key] = chosen
        return chosen
    if cfg.get("default_model"):
        return cfg["default_model"]
    raise LLMError("I could not choose a model automatically. Type a model name in the Model box.")


# ---- the tools
def tool_stats(a):
    return {"darkness_level": a["level"], "brightness": round(a["brightness"], 3),
            "contrast": round(a["contrast"], 3), "detail_entropy": round(a["entropy"], 2),
            "noise": round(a["noise"], 2)}


def run_tool(name, args, state):
    """Carry out one tool call from the AI model. Returns text for the model to read."""
    original = state["original"]
    if original is None:
        return "No image is loaded yet. Ask the user to upload a photo or use the webcam."
    args = args if isinstance(args, dict) else {}
    try:
        if name == "analyze_image":
            return json.dumps(tool_stats(analyze(original)))
        if name == "enhance_image":
            method = str(args.get("method") or "auto")
            if method.lower() == "auto":
                result = auto_enhance(original)
            elif method in METHODS:
                result = manual_enhance(original, method)
            else:
                return "Unknown method '" + method + "'. Valid methods: auto, " + ", ".join(METHODS)
            record(state, result)
            return json.dumps({"method_used": result["method"], "reason": result["reason"],
                               "before": tool_stats(result["before"]), "after": tool_stats(result["after"])})
        if name == "adjust_brightness":
            brighter = str(args.get("direction", "brighter")).lower() != "darker"
            base = state["current"] if state["current"] is not None else original
            out = gamma_correction(base, 0.75 if brighter else 1.3)
            record(state, {"image": out, "method": "Extra brightness" if brighter else "Reduced brightness",
                           "reason": "Made it brighter." if brighter else "Made it darker.",
                           "before": analyze(original), "after": analyze(out)})
            return json.dumps({"now": tool_stats(analyze(out))})
        if name == "compare_methods":
            rows = [{"method": n, "brightness": round(s["brightness"], 3), "contrast": round(s["contrast"], 3),
                     "detail_entropy": round(s["entropy"], 2)} for n, _, s in compare_all(original)]
            return json.dumps(rows) + " The pictures are in the 'Compare all methods' tab."
        if name == "reset_image":
            record(state, {"image": original.copy(), "method": "None", "reason": "Back to the original image.",
                           "before": analyze(original), "after": analyze(original)})
            return "The original image is shown again."
    except Exception as err:                # a tool problem must never crash the conversation
        return "The tool failed: " + str(err)
    return "Unknown tool: " + str(name)


def tools_for(kind):
    """The tool list in the format each kind of service expects."""
    out = []
    for t in TOOL_SPECS:
        schema = {"type": "object", "properties": t["properties"]}
        if t.get("required"):
            schema["required"] = t["required"]
        if kind == "anthropic":
            out.append({"name": t["name"], "description": t["description"], "input_schema": schema})
        else:
            fn = {"name": t["name"], "description": t["description"]}
            if t["properties"]:
                fn["parameters"] = schema
            out.append({"type": "function", "function": fn})
    return out


# ---- talking to the AI service
def build_system_prompt(state):
    now = datetime.datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    if state["original"] is None:
        context = "No image is loaded right now."
    else:
        a = analyze(state["original"])
        context = ("The user has loaded an image. The original is " + a["level"].lower() + ", with brightness "
                   + format(a["brightness"], ".2f") + " and contrast " + format(a["contrast"], ".2f") + ".")
        r = state["result"]
        if r:
            context += (" The picture on screen now used " + r["method"] + " and has brightness "
                        + format(r["after"]["brightness"], ".2f") + ".")
    return PERSONA + "\n\nDate and time on the computer running this app: " + now + ".\n" + context


def clean_history(history, message):
    """Chat history as plain alternating user/assistant text, ending with the new message."""
    msgs = []
    for h in (history or [])[-14:]:
        if isinstance(h, dict) and h.get("role") in ("user", "assistant"):
            text = content_text(h.get("content")).strip()
            if text:
                msgs.append({"role": h["role"], "content": text[:700]})
    msgs.append({"role": "user", "content": message})
    merged = []
    for m in msgs:                          # services want turns to alternate
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    return merged


def chat_anthropic(brain, system, messages, state, use_tools, max_tokens):
    headers = {"x-api-key": brain["key"], "anthropic-version": "2023-06-01"}
    msgs = list(messages)
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {"model": brain["model"], "max_tokens": max_tokens, "system": system, "messages": msgs}
        if use_tools and state is not None:
            payload["tools"] = tools_for("anthropic")
        resp = http_json(brain["base"] + "/messages", payload, headers)
        blocks = resp.get("content") or []
        calls = [b for b in blocks if b.get("type") == "tool_use"]
        if calls and resp.get("stop_reason") == "tool_use":
            echo = []
            for b in blocks:
                if b.get("type") == "text":
                    echo.append({"type": "text", "text": b.get("text", "")})
                elif b.get("type") == "tool_use":
                    echo.append({"type": "tool_use", "id": b["id"], "name": b.get("name"), "input": b.get("input") or {}})
            msgs.append({"role": "assistant", "content": echo})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c["id"], "content": run_tool(c.get("name"), c.get("input"), state)}
                for c in calls]})
            continue
        text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        if text:
            return text
        raise LLMError("The AI service sent an empty reply.")
    raise LLMError("I got stuck in a loop while using my tools.")


def chat_openai(brain, system, messages, state, use_tools, max_tokens):
    headers = {"Authorization": "Bearer " + brain["key"]}
    msgs = [{"role": "system", "content": system}] + list(messages)
    tools_on = bool(use_tools and state is not None)
    for _ in range(MAX_TOOL_ROUNDS):
        payload = {"model": brain["model"], "messages": msgs, "max_tokens": max_tokens, "temperature": 0.7}
        if tools_on:
            payload["tools"] = tools_for("openai")
            payload["tool_choice"] = "auto"
        try:
            resp = http_json(brain["base"] + "/chat/completions", payload, headers)
        except LLMError as err:
            if tools_on and "refused the request" in str(err):
                tools_on = False            # this model cannot use tools: carry on as a plain conversation
                continue
            raise
        choices = resp.get("choices") or []
        if not choices:
            raise LLMError("The AI service sent an empty reply.")
        msg = choices[0].get("message") or {}
        calls = msg.get("tool_calls") or []
        if calls:
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                msgs.append({"role": "tool", "tool_call_id": call.get("id", ""),
                             "content": run_tool(fn.get("name"), args, state)})
            continue
        text = (msg.get("content") or "").strip()
        if text:
            return text
        raise LLMError("The AI service sent an empty reply.")
    raise LLMError("I got stuck in a loop while using my tools.")


def llm_chat(brain, system, messages, state=None, use_tools=True, max_tokens=600):
    cfg = PROVIDERS[brain["provider"]]
    fn = chat_anthropic if cfg["kind"] == "anthropic" else chat_openai
    return fn(brain, system, messages, state, use_tools, max_tokens)


def llm_converse(message, state, history, brain):
    """One conversation turn with the AI model. May run image tools through `state`."""
    cfg = PROVIDERS[brain["provider"]]
    if not brain["key"] and not brain["base"].startswith("http://"):
        raise LLMError("I need an API key first. Paste one in Jarvis brain settings. You can get a key at "
                       + cfg.get("key_url", "your provider") + ".")
    brain = dict(brain, model=resolve_model(brain))
    return llm_chat(brain, build_system_prompt(state), clean_history(history, message), state)


def test_brain(provider, key, model, base_url):
    """The 'Test connection' button."""
    try:
        brain = make_brain(provider, key, model, base_url)
        if brain is None:
            return "Offline mode is selected, so no AI model is used. Choose a provider above to turn on the AI brain."
        cfg = PROVIDERS[provider]
        if not brain["key"] and not brain["base"].startswith("http://"):
            return "Paste your API key first. You can get one at " + cfg.get("key_url", "your provider") + "."
        brain["model"] = resolve_model(brain)
        reply = llm_chat(brain, "You are Jarvis. Reply with one short, friendly sentence.",
                         [{"role": "user", "content": "Say hello and tell me you are ready."}],
                         use_tools=False, max_tokens=60)
        return "**Connected.** Model in use: **" + brain["model"] + "**\n\nJarvis says: " + reply
    except LLMError as err:
        return "**Not connected.** " + str(err)
    except Exception as err:
        return "**Not connected.** Something unexpected went wrong: " + str(err)


def describe(stats):
    return (f"Brightness is {stats['brightness']:.2f} (0 is black, 1 is white), contrast {stats['contrast']:.2f}, "
            f"detail (entropy) {stats['entropy']:.2f}. I would call it **{stats['level'].lower()}**.")


def new_state():
    return {"original": None, "current": None, "result": None}


def set_image(state, bgr):
    state["original"] = bgr
    state["current"] = None
    state["result"] = None


def record(state, result):
    state["current"] = result["image"]
    state["result"] = result


def scan_message(result):
    return (f"I scanned your image. {describe(result['before'])}\n\n{result['reason']} "
            f"Brightness went from {result['before']['brightness']:.2f} to {result['after']['brightness']:.2f}.")


def offline_reply(message, state, history):
    """Built-in rule-based Jarvis (no AI model). Image results are stored in `state`."""
    text = (message or "").strip()
    low = text.lower()
    if not text:
        return "I'm listening. Ask me something, or type *help*."

    if re.search(r"\b(hi|hello|hey|namaste|good (morning|evening|afternoon))\b", low) and len(low.split()) <= 4:
        return "Hello! I'm Jarvis. Upload a dark photo and I'll scan and enhance it, or type *help* to see what I can do."
    if re.search(r"\b(thanks|thank you|thx)\b", low):
        return "You're welcome! Anything else you'd like to try?"
    if re.search(r"\b(bye|goodbye|see you)\b", low):
        return "Goodbye! Come back when it gets dark."
    if re.search(r"who are you|your name|what are you", low):
        return ("I'm Jarvis, the assistant of this low-light enhancement project. I can scan an image, tell you how dark it is, "
                "enhance it, compare methods and explain how they work.")
    if re.search(r"\b(help|what can you do|capabilities)\b", low):
        return HELP

    answer = safe_math(text)
    if answer is not None:
        return f"{text.strip(' ?')} = **{answer}**"
    if re.search(r"\b(time|date|day|today)\b", low) and not re.search(r"image|photo|picture", low):
        now = datetime.datetime.now()
        return f"It's {now.strftime('%I:%M %p')} on {now.strftime('%A, %d %B %Y')} (by this computer's clock)."
    if "joke" in low:
        return random.choice(JOKES)

    # Explanations of concepts
    if re.search(r"what is|what's|what are|explain|define|tell me about|how does|how do|meaning|why", low):
        for key in ["gamma", "histogram", "clahe", "retinex", "psnr", "ssim", "entropy", "flicker", "smooth", "noise", "low light", "low-light"]:
            if key in low:
                return CONCEPTS["low light" if key == "low-light" else key]

    # Image actions
    wants_image = re.search(r"enhance|brighten|brighter|darker|compare|analy|scan|how dark|undo|original|reset|"
                            r"gamma|clahe|histogram|equaliz|retinex|ssr|msr|smooth|improve|fix|noise|brightness|detect|check", low)
    if wants_image:
        if state["original"] is None:
            return "Please upload an image first (or use the webcam), and then I'll get to work."
        original = state["original"]
        if re.search(r"undo|original|reset", low):
            record(state, {"image": original.copy(), "method": "None", "reason": "Back to your original image.",
                           "before": analyze(original), "after": analyze(original)})
            return "Done. Here is your original image again."
        if "compare" in low:
            rows = compare_all(original)
            lines = ["| Method | Brightness | Contrast | Entropy |", "|---|---|---|---|"]
            lines += [f"| {n} | {s['brightness']:.2f} | {s['contrast']:.2f} | {s['entropy']:.2f} |" for n, _, s in rows]
            return ("I ran every method on your image:\n\n" + "\n".join(lines) +
                    "\n\nOpen the **Compare all methods** tab to see the pictures side by side.")
        if re.search(r"brighter|brighten more|increase brightness|more light", low):
            base = state["current"] if state["current"] is not None else original
            out = gamma_correction(base, 0.75)
            record(state, {"image": out, "method": "Extra gamma 0.75", "reason": "Made it brighter.",
                           "before": analyze(original), "after": analyze(out)})
            return f"Done, I made it brighter. Brightness is now {get_brightness(out):.2f}."
        if re.search(r"darker|too bright|reduce brightness|less bright", low):
            base = state["current"] if state["current"] is not None else original
            out = gamma_correction(base, 1.3)
            record(state, {"image": out, "method": "Gamma 1.3 (darken)", "reason": "Made it darker.",
                           "before": analyze(original), "after": analyze(out)})
            return f"Done, I made it a little darker. Brightness is now {get_brightness(out):.2f}."

        chosen = None
        if "smooth" in low and "gamma" in low:
            chosen = "Smooth + Gamma"
        elif "smooth" in low and ("retinex" in low or "msr" in low):
            chosen = "Smooth + Retinex MSR"
        elif "clahe" in low:
            chosen = "CLAHE"
        elif "histogram" in low or "equaliz" in low:
            chosen = "Histogram equalization"
        elif "ssr" in low:
            chosen = "Retinex SSR"
        elif "retinex" in low or "msr" in low:
            chosen = "Retinex MSR"
        elif "gamma" in low:
            chosen = "Gamma"
        if chosen:
            result = manual_enhance(original, chosen)
            record(state, result)
            return (f"I applied **{chosen}**. Brightness went from {result['before']['brightness']:.2f} "
                    f"to {result['after']['brightness']:.2f}.")
        if re.search(r"analy|scan|how dark|detect|check|noise|brightness", low) and not re.search(r"enhance|brighten|improve|fix", low):
            return "Here is what I measured. " + describe(analyze(original))
        result = auto_enhance(original)
        record(state, result)
        return scan_message(result)

    return ("I'm in offline mode, so I can only talk about this project and handle small things like maths or the time. "
            "Open *Jarvis brain settings* and connect an AI model, and I can chat about anything. Type *help* to see what I can do.")


def jarvis_reply(message, state, history, brain_settings=None):
    """Main entry point. brain_settings = (provider, key, model, base_url), or None for offline mode.

    With an AI model connected, the model does the talking and calls the image tools itself.
    If the AI service fails for any reason, Jarvis says why and answers in offline mode instead.
    """
    if brain_settings:
        try:
            brain = make_brain(*brain_settings)
            if brain is not None:
                return llm_converse((message or "").strip(), state, history, brain)
        except LLMError as err:
            return ("(My AI brain is not working: " + str(err) + " I'll use offline mode for now.)\n\n"
                    + offline_reply(message, state, history))
        except Exception as err:
            return ("(My AI brain had an unexpected problem: " + str(err) + " I'll use offline mode for now.)\n\n"
                    + offline_reply(message, state, history))
    return offline_reply(message, state, history)


# =============================================================================
# 6. GRADIO WEB APP
# =============================================================================
MIC_LABEL = "Talk to Jarvis (microphone)"

# Runs in the browser after each Jarvis reply: reads it out loud with the best English voice the browser has.
# With "conversation mode" on, it presses the microphone button again when Jarvis finishes speaking.
SPEAK_JS = r"""
(history, enabled, converse) => {
  if (!history || !history.length) return;
  if (window.__jarvisSpokenLen === history.length) return;      // nothing new was said
  window.__jarvisSpokenLen = history.length;
  if (!enabled || !('speechSynthesis' in window)) return;
  const last = history[history.length - 1];
  if (!last || last.role !== 'assistant') return;
  let text = last.content;
  if (Array.isArray(text)) {
    text = text.map(b => (b && typeof b.text === 'string') ? b.text : '').join(' ');
  }
  if (typeof text !== 'string') return;
  const tableAt = text.indexOf('|');                            // do not read tables out loud
  if (tableAt > 0) text = text.slice(0, tableAt);
  text = text.replace(/[*_`#>|]/g, ' ').replace(/\s+/g, ' ').trim();
  if (!text) return;
  if (text.length > 500) text = text.slice(0, 500);
  const voices = (window.speechSynthesis.getVoices && window.speechSynthesis.getVoices()) || [];
  const score = (v) => {
    let s = 0;
    if (/natural|neural|online/i.test(v.name)) s += 10;
    if (/google/i.test(v.name)) s += 6;
    if (/^en[-_]IN/i.test(v.lang)) s += 3;
    if (/^en[-_](US|GB)/i.test(v.lang)) s += 2;
    return s;
  };
  const english = voices.filter(v => /^en[-_]/i.test(v.lang));
  english.sort((a, b) => score(b) - score(a));
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'en-US';
  if (english.length) { utterance.voice = english[0]; utterance.lang = english[0].lang; }
  utterance.onend = () => {
    if (converse) {
      const mic = document.getElementById('mic-btn');
      if (mic) setTimeout(() => mic.click(), 400);
    }
  };
  window.__jarvisUtterance = utterance;                         // keep a reference so the browser does not drop it
  window.speechSynthesis.speak(utterance);
}
"""

# Runs in the browser when the microphone button is pressed: listens once and returns the words heard.
MIC_JS = r"""
async () => {
  const btn = document.getElementById('mic-btn');
  const label = btn ? btn.textContent : '';
  const setLabel = (t) => { try { if (btn) btn.textContent = t; } catch (e) {} };
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Voice input needs Google Chrome or Microsoft Edge.'); return ''; }
  try { window.speechSynthesis.cancel(); } catch (e) {}
  setLabel('Listening... speak now');
  const heard = await new Promise((resolve) => {
    let finished = false;
    const done = (t) => { if (!finished) { finished = true; resolve(t || ''); } };
    try {
      const rec = new SR();
      rec.lang = 'en-IN';
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      rec.onresult = (e) => done(e.results[0][0].transcript);
      rec.onerror = (e) => {
        if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
          alert('Please allow microphone access for this page, then press the button again.');
        }
        done('');
      };
      rec.onend = () => done('');
      rec.start();
    } catch (err) { done(''); }
  });
  setLabel(label);
  return heard;
}
"""


def build_ui():
    import gradio as gr

    def make_chatbot():
        try:
            return gr.Chatbot(type="messages", height=330, label="Jarvis")
        except TypeError:           # newer Gradio versions only use the messages format
            return gr.Chatbot(height=330, label="Jarvis")

    def outputs(history, state):
        result = state["result"]
        info = stats_markdown(result) if result else ""
        return history, to_rgb(state["current"]), info, state

    def on_image_change(image, history, state):
        history = list(history or [])
        if image is None:
            return outputs(history, new_state())
        state = new_state()
        set_image(state, to_bgr(image))
        result = auto_enhance(state["original"])
        record(state, result)
        history.append({"role": "assistant", "content": scan_message(result)})
        return outputs(history, state)

    def on_enhance(image, method, history, state):
        history = list(history or [])
        if state["original"] is None and image is not None:
            set_image(state, to_bgr(image))
        if state["original"] is None:
            history.append({"role": "assistant", "content": "Please upload an image first."})
            return outputs(history, state)
        if method == "Auto":
            result = auto_enhance(state["original"])
            message = scan_message(result)
        else:
            result = manual_enhance(state["original"], method)
            message = (f"I applied **{method}**. Brightness went from {result['before']['brightness']:.2f} "
                       f"to {result['after']['brightness']:.2f}.")
        record(state, result)
        history.append({"role": "assistant", "content": message})
        return outputs(history, state)

    def on_chat(message, history, image, state, provider, api_key, model_name, base_url):
        history = list(history or [])
        if state["original"] is None and image is not None:
            set_image(state, to_bgr(image))
        if not (message or "").strip():                # nothing was said (empty box or a silent microphone)
            history_out, out_image, info, state = outputs(history, state)
            return history_out, out_image, info, "", state
        reply = jarvis_reply(message, state, history, (provider, api_key, model_name, base_url))
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        history_out, out_image, info, state = outputs(history, state)
        return history_out, out_image, info, "", state

    def on_compare(image, state):
        if state["original"] is None and image is not None:
            set_image(state, to_bgr(image))
        if state["original"] is None:
            return [], "Please upload an image on the Jarvis tab first."
        rows = compare_all(state["original"])
        gallery = [(to_rgb(img), f"{name} | brightness {s['brightness']:.2f} | entropy {s['entropy']:.2f}")
                   for name, img, s in rows]
        table = ["| Method | Brightness | Contrast | Entropy | Noise |", "|---|---|---|---|---|"]
        table += [f"| {n} | {s['brightness']:.3f} | {s['contrast']:.3f} | {s['entropy']:.2f} | {s['noise']:.2f} |"
                  for n, _, s in rows]
        return gallery, "\n".join(table)

    def on_video(path, method):
        if not path:
            return None, "Please upload a video first."
        return enhance_video(path, method)

    with gr.Blocks(title="Low-Light Enhancement with Jarvis") as demo:
        gr.Markdown("# Low-Light Image and Video Enhancement with Jarvis\n"
                    "Upload a dark photo and Jarvis will measure it, pick a method and enhance it. "
                    "Then chat with Jarvis, compare methods, or enhance a video.")
        state = gr.State(new_state())

        with gr.Tabs():
            with gr.Tab("Jarvis"):
                with gr.Row():
                    with gr.Column():
                        image_in = gr.Image(label="Dark image (upload or webcam)", type="numpy",
                                            sources=["upload", "webcam"])
                        method = gr.Dropdown(["Auto"] + list(METHODS), value="Auto", label="Method")
                        enhance_btn = gr.Button("Enhance", variant="primary")
                    with gr.Column():
                        image_out = gr.Image(label="Enhanced image", type="numpy", interactive=False)
                        info = gr.Markdown()
                with gr.Accordion("Jarvis brain settings: connect an AI model for real conversation", open=False):
                    gr.Markdown("Without an AI model, Jarvis uses built-in rules and can only talk about this project. "
                                "Connect one and he can chat about anything, and decide by himself when to scan or enhance "
                                "your image. Free keys are available from [Google AI Studio](https://aistudio.google.com/apikey) "
                                "(Gemini) and [Groq](https://console.groq.com/keys); free-tier limits can change. "
                                "Your key is used only while this page is open and is not saved to any file.")
                    provider = gr.Dropdown(list(PROVIDERS), value=default_provider(), label="AI model provider")
                    api_key = gr.Textbox(label="API key", type="password",
                                         placeholder="Paste your key here (or leave empty if it is set on this computer)")
                    model_box = gr.Textbox(label="Model (leave empty to choose automatically)")
                    base_box = gr.Textbox(label="Base URL (only for the custom provider)")
                    test_btn = gr.Button("Test connection")
                    test_out = gr.Markdown()
                chat = make_chatbot()
                with gr.Row():
                    box = gr.Textbox(label="Talk to Jarvis", scale=4,
                                     placeholder="Type here, or press the microphone button and speak")
                    mic_btn = gr.Button(MIC_LABEL, elem_id="mic-btn", scale=1)
                with gr.Row():
                    speak = gr.Checkbox(label="Jarvis speaks his replies (uses your browser's voice)", value=True)
                    converse = gr.Checkbox(label="Conversation mode: Jarvis listens again after he speaks (experimental)",
                                           value=False)

            with gr.Tab("Compare all methods"):
                compare_btn = gr.Button("Compare all methods on my image", variant="primary")
                gallery = gr.Gallery(label="Results", columns=4, height=520)
                table = gr.Markdown()

            with gr.Tab("Video"):
                video_in = gr.Video(label="Dark video")
                video_method = gr.Dropdown(["Auto"] + list(METHODS), value="Auto", label="Method")
                video_btn = gr.Button("Enhance video", variant="primary")
                video_out = gr.Video(label="Enhanced video")
                video_info = gr.Markdown()

        chat_outputs = [chat, image_out, info, box, state]
        brain_inputs = [provider, api_key, model_box, base_box]
        say_it = dict(fn=None, inputs=[chat, speak, converse], outputs=None, js=SPEAK_JS)   # speak the newest reply
        image_in.change(on_image_change, [image_in, chat, state], [chat, image_out, info, state]).then(**say_it)
        enhance_btn.click(on_enhance, [image_in, method, chat, state], [chat, image_out, info, state]).then(**say_it)
        box.submit(on_chat, [box, chat, image_in, state] + brain_inputs, chat_outputs).then(**say_it)
        mic_btn.click(fn=None, inputs=None, outputs=[box], js=MIC_JS).then(
            on_chat, [box, chat, image_in, state] + brain_inputs, chat_outputs).then(**say_it)
        test_btn.click(test_brain, brain_inputs, test_out)
        compare_btn.click(on_compare, [image_in, state], [gallery, table])
        video_btn.click(on_video, [video_in, video_method], [video_out, video_info])
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Low-light enhancement app with Jarvis")
    parser.add_argument("--share", action="store_true", help="create a public link (needed in Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    in_colab = os.path.isdir("/content") and "COLAB_RELEASE_TAG" in os.environ
    # On a laptop the browser opens by itself; in Colab we create a public link instead
    build_ui().launch(share=args.share or in_colab, server_port=args.port, inbrowser=not in_colab)
