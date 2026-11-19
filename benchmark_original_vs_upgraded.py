import os
import sys
import time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dtfad import (
    load_image,
    save_image,
    compute_psnr,
    compute_ssim,
)
from dtfad.adaptive_alpha import compute_gradient_magnitude
from dtfad.gl_derivative import gl_derivative_2d
from dtfad.diffusion_engine import _diffusion_coefficient, dtfad_denoise
from benchmark_dtfad import make_side_by_side

def dtfad_original_denoise(image: np.ndarray,
                           n_iter: int = 20,
                           dt: float = 0.1,
                           alpha: float = 1.5,
                           max_terms: int = 15,
                           k: float = 0.05) -> np.ndarray:
    """
    A baseline implementation of the ORIGINAL DTFAD (Fractional Anisotropic Diffusion).
    Lacks the 4 upgrades from your project:
      - NO Adaptive Alpha (uses constant alpha = 1.5)
      - NO MAD automatic thresholding (uses constant k1=k2=k)
      - NO Bilateral GLI pre-smoothing (no GLI modulation at all)
      - NO Dynamic Truncation (always uses max_terms)
    """
    u = image.copy()
    
    # Constant alpha matrix
    alpha_matrix = np.full(u.shape, alpha, dtype=np.float64)
    # Constant terms map
    terms_map = np.full(u.shape, max_terms, dtype=np.int32)
    
    for t in range(n_iter):
        g = compute_gradient_magnitude(u)
        
        # Original usually uses a single k for Perona-Malik
        c = np.exp(-((g / k) ** 2))
        
        derivs = gl_derivative_2d(u, alpha_matrix, terms_map, max_terms)
        
        update = np.zeros_like(u)
        for direction in ('north', 'south', 'east', 'west'):
            update -= c * derivs[direction]
            
        u = u + dt * update
        u = np.clip(u, 0.0, 1.0)
        
    return u

def run_comparison():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SET12_DIR = os.path.join(BASE_DIR, "Dataset", "Set12")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "Original_vs_Upgraded")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Pick a few standard images
    test_images = ["01.png", "04.png", "08.png", "12.png"]
    rng = np.random.default_rng(seed=42)
    
    print("============================================================")
    print("  Comparison: ORIGINAL DTFAD vs UPGRADED DTFAD (Yours)")
    print("  Noise: Gaussian Speckle (var = 0.04)")
    print("============================================================")
    
    psnr_orig_list, psnr_up_list = [], []
    ssim_orig_list, ssim_up_list = [], []
    
    for img_name in test_images:
        img_path = os.path.join(SET12_DIR, img_name)
        if not os.path.exists(img_path):
            continue
            
        clean = load_image(img_path)
        
        # Add Gaussian speckle (var=0.04)
        noise = rng.normal(0, np.sqrt(0.04), clean.shape)
        noisy = np.clip(clean + clean * noise, 0.0, 1.0)
        
        # Run Original DTFAD
        t0 = time.time()
        # Tune k empirically for best PSNR on this noise level to be fair to original
        denoised_orig = dtfad_original_denoise(noisy, n_iter=15, dt=0.1, alpha=1.5, max_terms=15, k=0.1)
        t_orig = time.time() - t0
        
        # Run Upgraded DTFAD
        t1 = time.time()
        denoised_upgraded = dtfad_denoise(noisy, n_iter=15, dt=0.1)
        t_up = time.time() - t1
        
        p_n = compute_psnr(clean, noisy)
        p_orig = compute_psnr(clean, denoised_orig)
        p_up = compute_psnr(clean, denoised_upgraded)
        
        s_orig = compute_ssim(clean, denoised_orig)
        s_up = compute_ssim(clean, denoised_upgraded)
        
        psnr_orig_list.append(p_orig)
        psnr_up_list.append(p_up)
        ssim_orig_list.append(s_orig)
        ssim_up_list.append(s_up)
        
        print(f"Image: {img_name} (Noisy PSNR: {p_n:.2f}dB)")
        print(f"  [Original] PSNR: {p_orig:.2f}dB | SSIM: {s_orig:.4f} | Time: {t_orig:.1f}s")
        print(f"  [Upgraded] PSNR: {p_up:.2f}dB | SSIM: {s_up:.4f} | Time: {t_up:.1f}s")
        print(f"  -> Improvement: +{p_up - p_orig:.2f}dB\n")
        
        # Save side-by-side
        # Clean | Original | Upgraded
        c = np.clip(clean * 255, 0, 255).astype(np.uint8)
        o = np.clip(denoised_orig * 255, 0, 255).astype(np.uint8)
        u = np.clip(denoised_upgraded * 255, 0, 255).astype(np.uint8)
        
        h, w = c.shape[:2]
        pad = np.ones((30, w), dtype=np.uint8) * 255
        
        def add_label(strip, text):
            out = strip.copy()
            cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,), 1, cv2.LINE_AA)
            return out
            
        c_labeled = np.vstack([add_label(pad, "Clean"), c])
        o_labeled = np.vstack([add_label(pad, "Original DTFAD"), o])
        u_labeled = np.vstack([add_label(pad, "Upgraded DTFAD"), u])
        
        sep = np.ones((h + 30, 2), dtype=np.uint8) * 128
        comparison = np.hstack([c_labeled, sep, o_labeled, sep, u_labeled])
        
        cv2.imwrite(os.path.join(RESULTS_DIR, f"{img_name.split('.')[0]}_comparison.png"), comparison)

    if psnr_orig_list:
        print("============================================================")
        print("  AVERAGE RESULTS")
        print("============================================================")
        print(f"  [Original] PSNR: {np.mean(psnr_orig_list):.2f}dB | SSIM: {np.mean(ssim_orig_list):.4f}")
        print(f"  [Upgraded] PSNR: {np.mean(psnr_up_list):.2f}dB | SSIM: {np.mean(ssim_up_list):.4f}")
        print(f"  -> Average Improvement: +{np.mean(psnr_up_list) - np.mean(psnr_orig_list):.2f}dB")

if __name__ == '__main__':
    run_comparison()
