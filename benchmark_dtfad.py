"""
Benchmark: DTFAD Denoising vs Speckle2Void
==========================================

Runs the DTFAD pipeline on available datasets, saves result images
(clean / noisy / denoised side-by-side), and computes PSNR & SSIM
at multiple speckle-noise variance levels.

Speckle2Void reference PSNR values (from arXiv 2007.02075, Table I,
L=1 single-look synthetic speckle on BSD test images):
    - Speckle2Void (local):    ~23-26 dB (image-dependent)
    - Speckle2Void (non-local): ~23-27 dB
    - SAR-BM3D:                 ~22-24 dB
    - PPB:                      ~20-22 dB

Note: Speckle2Void uses *multiplicative* Gamma-distributed speckle
(yi = ni * xi, ni ~ Gamma(L, L)), which is different from Gaussian
multiplicative noise. We test with both models for fair comparison.

Usage:
    python benchmark_dtfad.py
"""

import os
import sys
import time
import glob
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2

# Add the project root to path so dtfad can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dtfad import (
    dtfad_denoise,
    load_image,
    save_image,
    add_speckle_noise,
    compute_psnr,
    compute_ssim,
)


# ── Noise Models ──────────────────────────────────────────────────────
def add_gamma_speckle(image, L=1, rng=None):
    """Add multiplicative Gamma speckle noise (SAR model).

    yi = xi * ni,  ni ~ Gamma(L, 1/L)  (mean=1, var=1/L)

    This is the noise model used in Speckle2Void (arXiv 2007.02075).
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.gamma(shape=L, scale=1.0 / L, size=image.shape)
    noisy = image * noise
    return np.clip(noisy, 0.0, 1.0)


# ── Dataset Discovery ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASETS = {
    "Set12": os.path.join(DATASET_DIR, "Set12"),
    "CSet12": os.path.join(DATASET_DIR, "CSet12"),
    "BM3D_test": os.path.join(DATASET_DIR, "BM3D test imgae"),
    "Images": os.path.join(DATASET_DIR, "Images"),
}

# Image file extensions to search for
IMAGE_EXTS = ("*.png", "*.bmp", "*.jpg", "*.jpeg", "*.tif", "*.tiff")


def discover_clean_images(dataset_name, dataset_path):
    """Find clean (original) images in a dataset directory.

    For Set12, skip noisy variants (files with _15, _35, _50 suffix).
    """
    all_files = []
    for ext in IMAGE_EXTS:
        all_files.extend(glob.glob(os.path.join(dataset_path, ext)))

    # Filter out noisy variants and label images
    clean = []
    for f in sorted(all_files):
        basename = os.path.splitext(os.path.basename(f))[0]
        # Skip noisy variants (e.g., 01_15, butterfly_35) and labels
        if any(basename.endswith(s) for s in ("_15", "_35", "_50", "_label", "_edge")):
            continue
        # Skip colour images (we need grayscale)
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert colour to grayscale
            pass  # We'll handle in load_image
        clean.append(f)

    return clean


def make_side_by_side(clean, noisy, denoised):
    """Create a horizontal side-by-side comparison image."""
    # Convert from [0,1] float to uint8
    c = np.clip(clean * 255, 0, 255).astype(np.uint8)
    n = np.clip(noisy * 255, 0, 255).astype(np.uint8)
    d = np.clip(denoised * 255, 0, 255).astype(np.uint8)

    # Add labels
    h, w = c.shape[:2]
    label_h = 30
    pad = np.ones((label_h, w), dtype=np.uint8) * 255

    def add_label(strip, text):
        out = strip.copy()
        cv2.putText(out, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,), 1, cv2.LINE_AA)
        return out

    c_labeled = np.vstack([add_label(pad, "Clean"), c])
    n_labeled = np.vstack([add_label(pad, "Noisy"), n])
    d_labeled = np.vstack([add_label(pad, "DTFAD"), d])

    separator = np.ones((h + label_h, 2), dtype=np.uint8) * 128
    return np.hstack([c_labeled, separator, n_labeled, separator, d_labeled])


# ── Speckle2Void Reference Results ───────────────────────────────────
# From Table I in arXiv 2007.02075 (L=1, single-look)
# These are approximate PSNR values on standard test images
SPECKLE2VOID_REF = {
    "description": "Speckle2Void (arXiv 2007.02075) — L=1 synthetic Gamma speckle",
    "note": "Values from Table I. Exact per-image values depend on the test set used in the paper (BSD test images).",
    "methods": {
        "PPB":                   {"avg_psnr_approx": "~20-22 dB"},
        "SAR-BM3D":              {"avg_psnr_approx": "~22-24 dB"},
        "CNN baseline (supervised)": {"avg_psnr_approx": "~24-26 dB"},
        "ID-CNN (supervised)":   {"avg_psnr_approx": "~24-26 dB"},
        "Speckle2Void (local)":  {"avg_psnr_approx": "~23-26 dB"},
        "Speckle2Void (NL)":     {"avg_psnr_approx": "~23-27 dB"},
    },
}


# ── Main Benchmark ───────────────────────────────────────────────────
def run_benchmark():
    """Run the full DTFAD benchmark across all datasets and noise levels."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Noise configurations to test
    # Gaussian multiplicative speckle (as used in DTFAD utils)
    gaussian_variances = [0.01, 0.04, 0.1]
    # Gamma speckle (SAR model, as used in Speckle2Void)
    gamma_looks = [1, 4, 10]

    rng = np.random.default_rng(seed=42)

    all_results = []

    print("=" * 80)
    print("  DTFAD Benchmark — PSNR/SSIM Evaluation")
    print("  Comparing against Speckle2Void reference values")
    print("=" * 80)

    for ds_name, ds_path in DATASETS.items():
        if not os.path.isdir(ds_path):
            print(f"\n⚠  Dataset '{ds_name}' not found at {ds_path}, skipping.")
            continue

        images = discover_clean_images(ds_name, ds_path)
        if not images:
            print(f"\n⚠  No clean images found in '{ds_name}', skipping.")
            continue

        print(f"\n{'─' * 80}")
        print(f"  Dataset: {ds_name}  ({len(images)} images)")
        print(f"{'─' * 80}")

        ds_result_dir = os.path.join(RESULTS_DIR, ds_name)
        os.makedirs(ds_result_dir, exist_ok=True)

        # ── Test 1: Gaussian multiplicative speckle ───────────────
        for var in gaussian_variances:
            print(f"\n  ▸ Gaussian speckle (σ² = {var})")
            print(f"    {'Image':<25} {'PSNR_noisy':>12} {'PSNR_dtfad':>12} {'SSIM_noisy':>12} {'SSIM_dtfad':>12}")
            print(f"    {'─' * 73}")

            psnr_noisy_list = []
            psnr_denoised_list = []
            ssim_noisy_list = []
            ssim_denoised_list = []

            for img_path in images:
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                try:
                    clean = load_image(img_path)
                except Exception as e:
                    print(f"    [SKIP] {img_name}: {e}")
                    continue

                # Resize large images to keep benchmark fast
                if max(clean.shape) > 512:
                    scale = 512 / max(clean.shape)
                    new_h = int(clean.shape[0] * scale)
                    new_w = int(clean.shape[1] * scale)
                    clean = cv2.resize(clean, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

                noisy = add_speckle_noise(clean, variance=var, rng=rng)
                t0 = time.time()
                denoised = dtfad_denoise(noisy, n_iter=15, dt=0.1)
                elapsed = time.time() - t0

                psnr_n = compute_psnr(clean, noisy)
                psnr_d = compute_psnr(clean, denoised)
                ssim_n = compute_ssim(clean, noisy)
                ssim_d = compute_ssim(clean, denoised)

                psnr_noisy_list.append(psnr_n)
                psnr_denoised_list.append(psnr_d)
                ssim_noisy_list.append(ssim_n)
                ssim_denoised_list.append(ssim_d)

                print(f"    {img_name:<25} {psnr_n:>10.2f}dB {psnr_d:>10.2f}dB "
                      f"{ssim_n:>10.4f}   {ssim_d:>10.4f}   ({elapsed:.1f}s)")

                # Save side-by-side result image
                comparison = make_side_by_side(clean, noisy, denoised)
                out_path = os.path.join(ds_result_dir,
                                        f"{img_name}_gaussian_var{var:.2f}.png")
                cv2.imwrite(out_path, comparison)

                # Save individual denoised result
                denoised_path = os.path.join(ds_result_dir,
                                             f"{img_name}_denoised_var{var:.2f}.png")
                save_image(denoised_path, denoised)

            if psnr_denoised_list:
                avg_psnr_n = np.mean(psnr_noisy_list)
                avg_psnr_d = np.mean(psnr_denoised_list)
                avg_ssim_n = np.mean(ssim_noisy_list)
                avg_ssim_d = np.mean(ssim_denoised_list)
                gain = avg_psnr_d - avg_psnr_n

                print(f"    {'─' * 73}")
                print(f"    {'AVERAGE':<25} {avg_psnr_n:>10.2f}dB {avg_psnr_d:>10.2f}dB "
                      f"{avg_ssim_n:>10.4f}   {avg_ssim_d:>10.4f}")
                print(f"    PSNR gain: +{gain:.2f} dB")

                all_results.append({
                    "dataset": ds_name,
                    "noise_type": "gaussian_speckle",
                    "noise_param": f"var={var}",
                    "avg_psnr_noisy": avg_psnr_n,
                    "avg_psnr_dtfad": avg_psnr_d,
                    "psnr_gain": gain,
                    "avg_ssim_noisy": avg_ssim_n,
                    "avg_ssim_dtfad": avg_ssim_d,
                })

        # ── Test 2: Gamma speckle (SAR model — Speckle2Void comparison) ─
        for L in gamma_looks:
            print(f"\n  ▸ Gamma speckle (L = {L} look{'s' if L > 1 else ''})")
            print(f"    {'Image':<25} {'PSNR_noisy':>12} {'PSNR_dtfad':>12} {'SSIM_noisy':>12} {'SSIM_dtfad':>12}")
            print(f"    {'─' * 73}")

            psnr_noisy_list = []
            psnr_denoised_list = []
            ssim_noisy_list = []
            ssim_denoised_list = []

            for img_path in images:
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                try:
                    clean = load_image(img_path)
                except Exception as e:
                    print(f"    [SKIP] {img_name}: {e}")
                    continue

                if max(clean.shape) > 512:
                    scale = 512 / max(clean.shape)
                    new_h = int(clean.shape[0] * scale)
                    new_w = int(clean.shape[1] * scale)
                    clean = cv2.resize(clean, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

                noisy = add_gamma_speckle(clean, L=L, rng=rng)
                t0 = time.time()
                denoised = dtfad_denoise(noisy, n_iter=15, dt=0.1)
                elapsed = time.time() - t0

                psnr_n = compute_psnr(clean, noisy)
                psnr_d = compute_psnr(clean, denoised)
                ssim_n = compute_ssim(clean, noisy)
                ssim_d = compute_ssim(clean, denoised)

                psnr_noisy_list.append(psnr_n)
                psnr_denoised_list.append(psnr_d)
                ssim_noisy_list.append(ssim_n)
                ssim_denoised_list.append(ssim_d)

                print(f"    {img_name:<25} {psnr_n:>10.2f}dB {psnr_d:>10.2f}dB "
                      f"{ssim_n:>10.4f}   {ssim_d:>10.4f}   ({elapsed:.1f}s)")

                comparison = make_side_by_side(clean, noisy, denoised)
                out_path = os.path.join(ds_result_dir,
                                        f"{img_name}_gamma_L{L}.png")
                cv2.imwrite(out_path, comparison)

                denoised_path = os.path.join(ds_result_dir,
                                             f"{img_name}_denoised_gamma_L{L}.png")
                save_image(denoised_path, denoised)

            if psnr_denoised_list:
                avg_psnr_n = np.mean(psnr_noisy_list)
                avg_psnr_d = np.mean(psnr_denoised_list)
                avg_ssim_n = np.mean(ssim_noisy_list)
                avg_ssim_d = np.mean(ssim_denoised_list)
                gain = avg_psnr_d - avg_psnr_n

                print(f"    {'─' * 73}")
                print(f"    {'AVERAGE':<25} {avg_psnr_n:>10.2f}dB {avg_psnr_d:>10.2f}dB "
                      f"{avg_ssim_n:>10.4f}   {avg_ssim_d:>10.4f}")
                print(f"    PSNR gain: +{gain:.2f} dB")

                all_results.append({
                    "dataset": ds_name,
                    "noise_type": "gamma_speckle",
                    "noise_param": f"L={L}",
                    "avg_psnr_noisy": avg_psnr_n,
                    "avg_psnr_dtfad": avg_psnr_d,
                    "psnr_gain": gain,
                    "avg_ssim_noisy": avg_ssim_n,
                    "avg_ssim_dtfad": avg_ssim_d,
                })

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  SUMMARY — DTFAD vs Speckle2Void Comparison")
    print(f"{'=' * 80}")

    print(f"\n  {'Dataset':<12} {'Noise Model':<20} {'Param':<10} "
          f"{'PSNR_noisy':>12} {'PSNR_DTFAD':>12} {'Gain':>8} {'SSIM_DTFAD':>12}")
    print(f"  {'─' * 88}")

    for r in all_results:
        print(f"  {r['dataset']:<12} {r['noise_type']:<20} {r['noise_param']:<10} "
              f"{r['avg_psnr_noisy']:>10.2f}dB {r['avg_psnr_dtfad']:>10.2f}dB "
              f"{r['psnr_gain']:>+6.2f}dB {r['avg_ssim_dtfad']:>10.4f}")

    print(f"\n  Speckle2Void reference (L=1 Gamma speckle, BSD test images):")
    for method, vals in SPECKLE2VOID_REF["methods"].items():
        print(f"    {method:<35} PSNR {vals['avg_psnr_approx']}")

    print(f"\n  📁 Result images saved to: {RESULTS_DIR}")
    print(f"     Side-by-side comparisons: <dataset>/<image>_<noise>.png")
    print(f"     Individual denoised:      <dataset>/<image>_denoised_<noise>.png")

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,noise_type,noise_param,avg_psnr_noisy,avg_psnr_dtfad,psnr_gain,avg_ssim_noisy,avg_ssim_dtfad\n")
        for r in all_results:
            f.write(f"{r['dataset']},{r['noise_type']},{r['noise_param']},"
                    f"{r['avg_psnr_noisy']:.4f},{r['avg_psnr_dtfad']:.4f},"
                    f"{r['psnr_gain']:.4f},{r['avg_ssim_noisy']:.4f},"
                    f"{r['avg_ssim_dtfad']:.4f}\n")
    print(f"     CSV summary:              {csv_path}")


if __name__ == "__main__":
    run_benchmark()
