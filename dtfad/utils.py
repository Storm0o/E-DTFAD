"""
Utility helpers for the DTFAD pipeline.

Provides image I/O (grayscale), synthetic speckle noise generation,
and quality metrics (PSNR, SSIM).
"""

import numpy as np
import cv2


def load_image(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array normalised to [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    return img.astype(np.float64) / 255.0


def save_image(path: str, img: np.ndarray) -> None:
    """Save a [0, 1] float image as an 8-bit grayscale file."""
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


def add_speckle_noise(image: np.ndarray, variance: float = 0.04,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Add multiplicative speckle noise: J = I + I * n, n ~ N(0, variance)."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, np.sqrt(variance), image.shape)
    noisy = image + image * noise
    return np.clip(noisy, 0.0, 1.0)


def compute_psnr(original: np.ndarray, denoised: np.ndarray,
                 data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim(original: np.ndarray, denoised: np.ndarray,
                 data_range: float = 1.0, win_size: int = 7) -> float:
    """Structural Similarity Index (mean over the image)."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    sigma = 1.5
    half = win_size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = np.outer(g, g)
    kernel /= kernel.sum()

    mu1 = cv2.filter2D(original, -1, kernel)
    mu2 = cv2.filter2D(denoised, -1, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.filter2D(original ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(denoised ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(original * denoised, -1, kernel) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(np.mean(ssim_map))
