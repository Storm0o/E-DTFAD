# E-DTFAD: Enhanced Discrete Tensor-based Fractional Anisotropic Diffusion

**Authors:** Aman Yadav & Aayush Kumar

> An enhanced implementation of the DTFAD framework for speckle noise removal in medical and SAR images, with four novel algorithmic improvements and a web-based interactive demo.

🌐 **Live Demo:** [https://e-dtfad.onrender.com/](https://e-dtfad.onrender.com/)

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
│   │   ├── app.js
│   │   └── style.css
│   └── templates/
│       └── index.html
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
├── Procfile
├── render.yaml
└── README.md
```

---

## 🚀 Quick Start

### 1. Try Online Demo
Access the live interactive application directly at:
👉 **[https://e-dtfad.onrender.com/](https://e-dtfad.onrender.com/)**

### 2. Local Setup
```bash
pip install -r requirements.txt
python webapp/app.py
```
Then open **http://localhost:5000** in your browser.

Upload any grayscale image, choose a speckle noise level (L = 1, 4, or 10 looks), and compare Original DTFAD vs Upgraded E-DTFAD side-by-side with live PSNR/SSIM metrics.

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

## 🌐 Deployment

The webapp is deployed on Render at:
**[https://e-dtfad.onrender.com/](https://e-dtfad.onrender.com/)**

Configuration details:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python webapp/app.py`
- Configuration file: `render.yaml` / `Procfile`

---

## 📦 Requirements

```
numpy
opencv-python
scipy
flask
scikit-image
matplotlib
Pillow
gunicorn
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
