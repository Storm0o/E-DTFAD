"""
DTFAD Web Application — Interactive Speckle Denoising Demo.

Upload an image, apply synthetic speckle noise at a chosen level,
then compare Original DTFAD vs Upgraded DTFAD side-by-side with
full metrics (PSNR, SSIM, noise estimation, α range, etc.).
"""

import os
import sys
import time
import base64

try:
    # pyrefly: ignore [missing-import]
    import numpy as np
    # pyrefly: ignore [missing-import]
    import cv2
    # pyrefly: ignore [missing-import]
    from flask import Flask, render_template, request, jsonify
except ModuleNotFoundError as e:
    print(f"\n  ERROR: {e}")
    print("  Please install the required packages with:")
    print("    pip install numpy opencv-python flask scipy\n")
    sys.exit(1)

# Add project root so dtfad can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dtfad import load_image, compute_psnr, compute_ssim
from dtfad.adaptive_alpha import compute_gradient_magnitude, map_alpha
from dtfad.mad_threshold import estimate_noise_variance_mad, compute_dynamic_thresholds
from dtfad.gli_bilateral import compute_gli_bilateral
from dtfad.gl_derivative import gl_derivative_2d, compute_max_terms_map
from dtfad.diffusion_engine import dtfad_denoise, _diffusion_coefficient

WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(WEBAPP_DIR, 'templates'),
            static_folder=os.path.join(WEBAPP_DIR, 'static'))

UPLOAD_DIR = os.path.join(WEBAPP_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Noise generators ─────────────────────────────────────────────────
def add_gaussian_speckle(image, variance=0.04, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, np.sqrt(variance), image.shape)
    return np.clip(image + image * noise, 0.0, 1.0)


def add_gamma_speckle(image, L=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.gamma(shape=L, scale=1.0 / L, size=image.shape)
    return np.clip(image * noise, 0.0, 1.0)


# ── Original (baseline) DTFAD ────────────────────────────────────────
def dtfad_original_denoise(image, n_iter=15, dt=0.1, alpha=1.1,
                           max_terms=15, k=0.5):
    u = image.copy()
    alpha_matrix = np.full(u.shape, alpha, dtype=np.float64)
    terms_map = np.full(u.shape, max_terms, dtype=np.int32)

    for _ in range(n_iter):
        g = compute_gradient_magnitude(u)
        c = np.exp(-((g / k) ** 2))
        derivs = gl_derivative_2d(u, alpha_matrix, terms_map, max_terms)
        update = np.zeros_like(u)
        for d in ('north', 'south', 'east', 'west'):
            update -= c * derivs[d]
        u = u + dt * update
        u = np.clip(u, 0.0, 1.0)
    return u


def img_to_base64(img_array):
    """Convert a [0,1] float image to a base64-encoded PNG string."""
    img_uint8 = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img_uint8)
    return base64.b64encode(buffer).decode('utf-8')


def compute_all_metrics(clean, noisy, denoised, label=""):
    """Compute comprehensive metrics for a denoised result."""
    psnr_noisy = compute_psnr(clean, noisy)
    psnr_denoised = compute_psnr(clean, denoised)
    ssim_noisy = compute_ssim(clean, noisy)
    ssim_denoised = compute_ssim(clean, denoised)

    # Residual noise analysis
    residual = np.abs(denoised - clean)
    mse = np.mean((denoised - clean) ** 2)

    return {
        "label": label,
        "psnr_noisy": round(psnr_noisy, 2),
        "psnr_denoised": round(psnr_denoised, 2),
        "psnr_gain": round(psnr_denoised - psnr_noisy, 2),
        "ssim_noisy": round(ssim_noisy, 4),
        "ssim_denoised": round(ssim_denoised, 4),
        "ssim_gain": round(ssim_denoised - ssim_noisy, 4),
        "mse": round(mse, 6),
        "residual_mean": round(np.mean(residual), 6),
        "residual_max": round(np.max(residual), 4),
    }


# ── Routes ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Read parameters
    noise_type = request.form.get('noise_type', 'gaussian')
    noise_level = float(request.form.get('noise_level', '0.04'))
    n_iter = int(request.form.get('n_iter', '15'))
    dt = float(request.form.get('dt', '0.1'))

    # Read uploaded image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        return jsonify({"error": "Cannot decode image"}), 400

    # Convert to grayscale float [0,1]
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr
    clean = img_gray.astype(np.float64) / 255.0

    # Resize if too large (keep processing fast for web)
    max_dim = 256
    if max(clean.shape) > max_dim:
        scale = max_dim / max(clean.shape)
        new_h = int(clean.shape[0] * scale)
        new_w = int(clean.shape[1] * scale)
        clean = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_AREA)

    rng = np.random.default_rng(seed=42)

    # Add noise
    if noise_type == 'gamma':
        L = max(1, int(1.0 / noise_level)) if noise_level > 0 else 10
        noisy = add_gamma_speckle(clean, L=L, rng=rng)
        noise_desc = f"Gamma speckle (L={L})"
    else:
        noisy = add_gaussian_speckle(clean, variance=noise_level, rng=rng)
        noise_desc = f"Gaussian speckle (var={noise_level})"

    # ── Noise estimation (MAD) ────────────────────────────────────
    sigma_hat = estimate_noise_variance_mad(noisy)
    k1, k2 = compute_dynamic_thresholds(sigma_hat)

    # ── Adaptive alpha stats ──────────────────────────────────────
    grad_mag = compute_gradient_magnitude(noisy)
    alpha_matrix = map_alpha(grad_mag, 1.0, 1.05)
    terms_map = compute_max_terms_map(grad_mag, 5, 15)

    params = {
        "noise_type": noise_desc,
        "noise_level": noise_level,
        "estimated_sigma": round(sigma_hat, 6),
        "k1": round(k1, 6),
        "k2": round(k2, 6),
        "alpha_min": round(float(alpha_matrix.min()), 4),
        "alpha_max": round(float(alpha_matrix.max()), 4),
        "alpha_mean": round(float(alpha_matrix.mean()), 4),
        "terms_min": int(terms_map.min()),
        "terms_max": int(terms_map.max()),
        "terms_mean": round(float(terms_map.mean()), 1),
        "image_size": f"{clean.shape[1]}x{clean.shape[0]}",
        "n_iter": n_iter,
        "dt": dt,
    }

    # ── Run Original DTFAD ────────────────────────────────────────
    t0 = time.time()
    denoised_orig = dtfad_original_denoise(noisy, n_iter=n_iter, dt=dt)
    time_orig = round(time.time() - t0, 2)

    # ── Run Upgraded DTFAD ────────────────────────────────────────
    t1 = time.time()
    denoised_upgraded = dtfad_denoise(noisy, n_iter=n_iter, dt=dt)
    time_upgraded = round(time.time() - t1, 2)

    # ── Compute metrics ──────────────────────────────────────────
    metrics_orig = compute_all_metrics(clean, noisy, denoised_orig, "Original DTFAD")
    metrics_upgraded = compute_all_metrics(clean, noisy, denoised_upgraded, "Upgraded DTFAD")

    metrics_orig["time"] = time_orig
    metrics_upgraded["time"] = time_upgraded

    # ── Encode images to base64 ──────────────────────────────────
    images = {
        "clean": img_to_base64(clean),
        "noisy": img_to_base64(noisy),
        "denoised_original": img_to_base64(denoised_orig),
        "denoised_upgraded": img_to_base64(denoised_upgraded),
    }

    return jsonify({
        "images": images,
        "params": params,
        "metrics_original": metrics_orig,
        "metrics_upgraded": metrics_upgraded,
    })


if __name__ == '__main__':
    print("\n  DTFAD Web Demo")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
