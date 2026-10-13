Low-Light Image and Video Enhancement with Jarvis

A Python-based project that enhances dark images and videos using classical image-processing techniques. It analyses input brightness, contrast, noise, and entropy, then selects an enhancement approach. The project also includes Jarvis, an assistant that can help scan, enhance, compare, and explain results through a Gradio interface.

Table of Contents
Overview
Features
Tech Stack
Enhancement Methods
Project Workflow
Requirements
Installation
Running the Application
Using the Application
Evaluation Metrics
Reported Results
Limitations
Future Improvements
References

Overview
Images and videos captured in poor lighting can appear dark, low in contrast, and noisy. This project aims to make those inputs easier to view by measuring image characteristics and applying suitable enhancement methods.
The system supports still images and short video clips. Its proposed Natural (smooth night) method is designed to reduce noise and blocky colour artefacts in very dark images by smoothing dark regions, applying adaptive gamma correction, and recovering colour from a blurred copy. For videos, temporal smoothing helps reduce visible flicker between frames.
The project report describes a Gradio-based desktop/web interface and Jarvis assistant. Jarvis can operate with built-in rules or, when configured, an external AI API key.

Features
Analyse brightness, contrast, noise, and entropy.
Classify images as Normal, Dim, Dark, or Very dark.
Enhance images using several classical techniques.
Compare methods and view before/after quality measurements.
Process short videos frame by frame.
Smooth video settings across frames to help reduce flicker.
Use a Gradio interface with Jarvis, image-comparison, and video workflows.
Run on a CPU; a GPU is not required for the reported implementation.
Tech Stack
Technology	Purpose
Python 3.10+	Main programming language
OpenCV	Image and video processing
NumPy	Array operations and numerical calculations
scikit-image	Test images and image-quality metrics
Matplotlib	Visualisations and plots
Gradio	Web/desktop user interface
FFmpeg	Re-encoding enhanced video to H.264
Enhancement Methods
Gamma correction — adjusts brightness using a power-law transformation. It is fast but can also amplify noise.
Histogram equalization — spreads intensity values to increase contrast, but may make noise more visible.
CLAHE — applies contrast-limited adaptive histogram equalization to local image regions. In this project, it is applied to the L channel in LAB colour space.
Retinex SSR — estimates illumination using a Gaussian-blurred image and enhances reflectance.
Retinex MSR — combines Retinex estimates at multiple scales for stronger shadow recovery, though noise may increase.
Natural (smooth night) — the proposed method. It smooths dark regions before brightening, automatically chooses gamma, and uses a blurred copy to help preserve natural-looking colour.
The report also describes automatic selection: Normal images are left unchanged, Dim images use CLAHE, and Dark/Very dark images use the Natural method.

Project Workflow
Load an image or video.
Measure image brightness, contrast, noise, and entropy.
Classify the input by darkness level.
Select an enhancement method automatically or compare methods.
Enhance the image, or process video frame by frame.
Apply temporal smoothing to video settings and blend consecutive output frames to reduce flicker.
Display the output and report measurements before and after enhancement.

Requirements
Python 3.10 or newer
The project's application file or notebook
The Python libraries listed in the project's dependency file or installation instructions
FFmpeg for video re-encoding

> **Note:** The report identifies the application file as `lowlight_app.py`, but does not provide a complete dependency list or repository URL. Confirm the filenames in your project folder before following the commands below.
Installation
Clone or download this project repository.
Open a terminal in the project directory.
Create and activate a virtual environment (recommended).
Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install the dependencies. If the repository contains a `requirements.txt` file:
```bash
pip install -r requirements.txt
```
If no dependency file is included, install the required packages according to the imports in the source code. The report lists OpenCV, NumPy, scikit-image, Matplotlib, and Gradio. FFmpeg must also be available for video re-encoding.
Running the Application
If your project folder contains `lowlight_app.py`, launch it with:
```bash
python lowlight_app.py
```
Open the local Gradio URL printed in the terminal.
If the project is supplied as a Google Colab notebook, open the notebook and run its cells in order. The report also describes deployment through Hugging Face Spaces. Exact commands may differ depending on the files and configuration included in your repository.
Using the Application
Images
Open the Jarvis or image-processing tab.
Upload an image.
Scan or analyse it to view the darkness classification and measurements.
Run automatic enhancement or select a method.
Compare the input and output and review the before/after metrics.
Comparing methods
Use the comparison tab to compare supported methods on the same input. Look at visual quality as well as the numerical metrics; a brighter image is not necessarily a cleaner or more faithful image.
Videos
Open the video tab and upload a supported clip.
Choose automatic enhancement or a supported method.
Run processing and review the enhanced video.
Compare brightness and flicker measurements when available.
According to the report, video processing is limited to the first 10 seconds and resized to at most 640 pixels wide in the web version.
Jarvis assistant
Jarvis can use built-in rules without an API key. Conversational AI features require a supported external API key configured according to the source code's instructions. Do not commit API keys or other secrets to Git. Use environment variables or a local, untracked configuration file.
Evaluation Metrics
Brightness: mean grayscale/luminance value; indicates overall lightness.
Contrast: standard deviation of grayscale/luminance values.
Entropy: describes the distribution of intensity values; higher entropy does not always mean better detail because noise can increase it.
Noise estimate: estimates noise using a Laplacian-based method.
PSNR (Peak Signal-to-Noise Ratio): compares an enhanced result with the known reference image; higher is generally better.
SSIM (Structural Similarity Index): measures structural similarity to the reference; higher is generally better.
Video flicker: measures relative frame-to-frame brightness change; lower values indicate less brightness variation under the report's definition.
Reported Results
The project report evaluates four sample photographs, each darkened at two levels, for eight dark inputs. It also evaluates a three-second test video at 480 × 270 resolution and 30 frames per second.
Key results reported for the Natural method:
Moderate darkness: SSIM 0.755, estimated noise 0.95, processing time about 15 ms per image.
Severe darkness: SSIM 0.652, estimated noise 0.81, processing time about 12 ms per image.
Test video: brightness increased from 0.092 to 0.250; flicker decreased from 15.81% to 6.73%; reported speed was about 99 frames/second at the tested resolution.
These are results from the report's small, synthetically darkened sample set. They should not be treated as a guarantee for all real-world images or videos. The report also notes that plain gamma correction achieved higher average PSNR on that test set, while the Natural method prioritised cleaner-looking output and stronger structural similarity in severe darkness.
Limitations
Details that were never captured by the camera cannot be fully recovered.
Extremely dark images may become smoother but remain soft, and slight colour tints or halos can occur.
Retinex and histogram methods may amplify noise in very dark regions.
The reported quantitative evaluation used four synthetically darkened sample images rather than a large set of real low-light photographs with paired ground truth.
The web video workflow is designed for short, resized clips, not long high-resolution real-time processing.
Jarvis's conversational AI capabilities depend on the configured external service and its usage limits.
Future Improvements
Evaluate on a larger real-world low-light dataset, such as LOL, with suitable reference images.
Test on more cameras, scenes, and lighting conditions.
Improve preservation of fine details while controlling noise and colour shifts.
Evaluate temporal consistency on longer videos and more varied motion.
Add automated tests and document supported file formats, size limits, and API configuration.
References
The project report cites the following foundational sources:
Gonzalez, R. C., and Woods, R. E. Digital Image Processing, 4th ed., Pearson, 2018.
Land, E. H., and McCann, J. J. “Lightness and Retinex theory.” Journal of the Optical Society of America, 61(1), 1971.
Jobson, D. J., Rahman, Z., and Woodell, G. A. “A multiscale Retinex for bridging the gap between color images and the human observation of scenes.” IEEE Transactions on Image Processing, 6(7), 1997.
Pizer, S. M., et al. “Adaptive histogram equalization and its variations.” Computer Vision, Graphics, and Image Processing, 39(3), 1987.
Zuiderveld, K. “Contrast limited adaptive histogram equalization.” In Graphics Gems IV, 1994.
Immerkaer, J. “Fast noise variance estimation.” Computer Vision and Image Understanding, 64(2), 1996.
Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. “Image quality assessment: from error visibility to structural similarity.” IEEE Transactions on Image Processing, 13(4), 2004.
Bradski, G. “The OpenCV Library.” Dr. Dobb's Journal of Software Tools, 2000.
van der Walt, S., et al. “scikit-image: image processing in Python.” PeerJ, 2, e453, 2014.
Abid, A., et al. “Gradio: Hassle-free sharing and testing of ML models in the wild.” arXiv:1906.02569, 2019.
---
Author: Ayush Mittal , Anshul Gathoria
Course: Image and Video Processing  
Institute: SVKM's NMIMS, MPSTME, Indore  
Academic Year: 2026–2027
