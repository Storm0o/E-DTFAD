# E-DTFAD: Enhanced Discrete Tensor-based Fractional Anisotropic Diffusion

**Authors:** Aman Yadav & Aayush Kumar

> An enhanced implementation of the DTFAD framework for speckle noise removal in medical and SAR images, with four novel algorithmic improvements and a web-based interactive demo.

---

## 📋 Overview

This project extends the original **DTFAD (Discrete Tensor-based Fractional Anisotropic Diffusion)** algorithm by Wei & Liao with four targeted enhancements (Tasks A–D) that significantly improve denoising quality, automation, and computational efficiency.

| Metric | Original DTFAD | **E-DTFAD (Ours)** | Improvement |
|--------|:--------------:|:-------------------:|:-----------:|
| PSNR (L=1) | 28.24 dB | **31.87 dB** | +3.63 dB |
| SSIM (L=1) | 0.823 | **0.891** | +8.3% |
| PSNR (L=4) | 29.81 dB | **33.42 dB** | +3.61 dB |
| SSIM (L=4) | 0.847 | **0.908** | +7.2% |
| PSNR (L=10) | 30.95 dB | **34.18 dB** | +3.23 dB |
| SSIM (L=10) | 0.862 | **0.919** | +6.6% |

---

## 🔬 Enhancements (Tasks A–D)

### Task A — Adaptive α(x,y) via Structure Tensor
- Spatially varying fractional order derived from local coherence
- Preserves edges while aggressively smoothing flat regions

### Task B — MAD-Based Automated Thresholding
- Replaces manual threshold tuning with Median Absolute Deviation
- Robust, fully automatic, no hyperparameter search required

### Task C — Bilateral GLI Pre-Smoothing
- Edge-aware smoothing of gradient-like images before diffusion
- Reduces ringing artifacts common at sharp transitions

### Task D — Dynamic GL Truncation
- Adapts Grünwald-Letnikov series truncation to local noise level
- Cuts computation by ~30% while maintaining accuracy

---

## 🗂 Project Structure

```
Vision/
├── dtfad/                        # Core E-DTFAD Python package
│   ├── __init__.py
│   ├── adaptive_alpha.py         # Task A: Structure-tensor α(x,y)
│   ├── mad_threshold.py          # Task B: MAD-based thresholding
│   ├── gli_bilateral.py          # Task C: Bilateral pre-smoothing
│   ├── gl_derivative.py          # Task D: Dynamic GL truncation
│   ├── diffusion_engine.py       # Main diffusion loop
│   └── utils.py                  # PSNR/SSIM metrics, helpers
│
├── webapp/                       # Flask web application (demo)
│   ├── app.py                    # Main Flask server
│   ├── static/
│   │   └── app.js
│   └── templates/
│       └── index.html
│
├── latex_report/                 # IEEE-format LaTeX report
│   ├── DTFAD_Comparison_Report.tex
│   └── figures/                  # 47 comparison + chart figures
│
├── report_figures/               # Result images & comparison grids
│   ├── *_comparison_grid.png     # Side-by-side method comparison
│   ├── *_psnr_bars_*.png         # PSNR bar charts
│   └── *_ssim_bars_*.png         # SSIM bar charts
│
├── benchmark_results_all.json    # Full benchmark data (10 imgs × 3 levels × 6 methods)
├── benchmark_dtfad.py            # Benchmark runner
├── benchmark_original_vs_upgraded.py
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Demo
```bash
python webapp/app.py
```
Then open **http://localhost:5000** in your browser.

Upload any grayscale image, choose a speckle noise level (L = 1, 4, or 10 looks), and compare all 6 denoising methods side-by-side with live PSNR/SSIM metrics.

### 3. Run Benchmarks
```bash
python benchmark_dtfad.py
python benchmark_original_vs_upgraded.py
```

---

## 📊 Compared Methods

| Method | Description |
|--------|-------------|
| Gaussian Filter | Standard spatial averaging |
| Median Filter | Edge-preserving rank filter |
| NL-Means | Non-local patch-based denoising |
| Bilateral Filter | Edge-aware spatial filter |
| DTFAD (Original) | Wei & Liao 2023 baseline |
| **E-DTFAD (Ours)** | **This work — all 4 enhancements** |

---

## 📄 Report

The full IEEE-format comparison report is in `latex_report/DTFAD_Comparison_Report.tex`.

**To compile on Overleaf:**
1. Download or clone this repo
2. Upload the `latex_report/` folder to [Overleaf](https://www.overleaf.com)
3. Set the main file to `DTFAD_Comparison_Report.tex`
4. Click Compile

A pre-compiled PDF (`DTFAD_IEEE_Transactions_Report.pdf`) is also included.

---

## 🌐 Deploy the Webapp

### Render (Free Tier)
1. Fork this repo on GitHub
2. Go to [render.com](https://render.com) → New Web Service → connect your fork
3. Build command: `pip install -r requirements.txt`
4. Start command: `python webapp/app.py`

### Railway
1. Connect this repo at [railway.app](https://railway.app)
2. It auto-detects Python; set start command to `python webapp/app.py`

---

## 📦 Requirements

```
numpy
opencv-python
scipy
flask
scikit-image
matplotlib
```

---

## 📚 Reference

> Wei, J., & Liao, C. (2023). *Discrete Tensor-Based Fractional Anisotropic Diffusion for Speckle Noise Removal*. IEEE Transactions on Image Processing.

---

## 👥 Authors

- **Aman Yadav**
- **Aayush Kumar**

---

*This project was developed as an academic research enhancement of the DTFAD framework.*
