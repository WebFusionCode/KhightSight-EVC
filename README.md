<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1024px-GitHub_Invertocat_Logo.svg.png" width="100" />
  <h1>KnightSight V2 - Advanced ANPR Pipeline</h1>
  <p>State-of-the-Art Automatic Number Plate Recognition (ANPR) System heavily optimized for Edge Deployments.</p>
</div>

---

## 🎯 System Overview & Metrics
KnightSight V2 is a fully autonomous pipeline spanning from raw image acquisition to final regex-validated data structuring. By ditching heavy commercial OCR engines and relying on highly-trained edge algorithms, it reads license plates in roughly ~65 milliseconds on standard CPUs.

* **Inference Speed**: ~15–20 FPS (CPU-Only)
* **Model Footprint**: < 19.0 MB total sizes.
* **YOLOv8 Detection (Precision)**: 97.2%
* **YOLOv8 Detection (mAP50)**: 97.5%
* **OCR Character Error Rate**: < 1.5%

### 🔧 Architecture
1. **Target Detection**: Fine-Tuned YOLOv8 Nano (`yolov8n`).
2. **Adaptive Target Illumination**: Mathematical unsharp masking and Contrast Limited Adaptive Histogram Equalization (CLAHE) for dim-light conditions.
3. **Character Extraction**: Mobile-optimized PaddleOCR (`en_PP-OCRv5_mobile`).
4. **Data Validation**: Strict Python Regex matching ensuring structural integrity (`^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$`).

---

## 🚀 Installation Guide

Choose your operating system below for exact instructions to clone and run the KnightSight command center.

### 🍎 Apple (macOS)
*Supports Intel and Apple Silicon (M1/M2/M3) via Metal Performance Shaders.*

1. **Install Miniforge** (Crucial for Apple Silicon optimization):
   ```bash
   brew install miniforge
   conda init zsh
   # Restart your terminal
   ```
2. **Create Environment**:
   ```bash
   conda create -n knightsight python=3.13
   conda activate knightsight
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   # PaddleOCR relies on standard wheels for Mac
   pip install paddleocr paddlepaddle>=2.6.0
   ```
4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

### 🪟 Windows 10/11
*Assumes Python 3.10 to 3.13 is installed.*

1. **Create Virtual Environment**:
   ```powershell
   python -m venv knightsight_env
   knightsight_env\Scripts\activate
   ```
2. **Install Core Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Install Paddle (CPU / CUDA)**:
   ```powershell
   # If you have an NVIDIA GPU, use the CUDA flag from Paddle's website. Otherwise:
   python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple
   pip install paddleocr
   ```
4. **Run Application**:
   ```powershell
   streamlit run app.py
   ```

### 🐧 Linux (Ubuntu / Debian)
*Ideal for Cloud or Edge Devices (like Jetson or Raspberry Pi).*

1. **System Prerequisites**:
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```
2. **Create Environment**:
   ```bash
   python3 -m venv knightsight_env
   source knightsight_env/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install paddlepaddle paddleocr
   ```
4. **Run Application**:
   ```bash
   streamlit run app.py
   ```

---

## 🖥️ Command Center Deployment
Once `streamlit run app.py` is initialized, visit `http://localhost:8501` to access the localized dashboard. You can toggle between `Image Upload`, `Video Stream`, and `Live Web Camera` inputs dynamically.

## 🤝 Project Structure
* `app.py`: The Main Streamlit Server and UI Frontend.
* `pipeline.py`: The core orchestrator that executes cv2 masking, OCR integration, and diagnostics logic.
* `detect.py`: Handles bounding box extraction and YOLOv8 spatial logic.
* `preprocess.py`: Contains the intelligent blur-detection, unsharp masking, and Light/Dark detection (CLAHE).
* `postprocess.py`: Uses Levenshtein Distances and RegEx string matching to cleanse raw OCR data.
* `train.py`: Contains our highly customized Albumentations neural-training infrastructure.

*Developed autonomously. Strictly built for 99% accuracy edge deployment.*
