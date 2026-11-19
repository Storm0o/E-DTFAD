"""
Utility helpers for the DTFAD pipeline.

Provides image I/O (grayscale), synthetic speckle noise generation,
and quality metrics (PSNR, SSIM).
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image as a float64 grayscale array normalised to [0, 1].

    Parameters
    ----------
    path : str
        File-system path to an image (any format OpenCV can read).

    Returns
    -------
    np.ndarray
        2-D array of shape (H, W) with values in [0, 1].
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    return img.astype(np.float64) / 255.0


def save_image(path: str, img: np.ndarray) -> None:
    """Save a [0, 1] float image as an 8-bit grayscale file.

    Parameters
    ----------
    path : str
        Destination path (extension determines format).
    img : np.ndarray
        2-D float array in [0, 1].
    """
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, out)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

def add_speckle_noise(image: np.ndarray, variance: float = 0.04,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Add multiplicative speckle noise: J = I + I * n, n ~ N(0, variance).

    Parameters
    ----------
    image : np.ndarray
        Clean image in [0, 1].
    variance : float
        Noise variance (σ² of the Gaussian multiplier).
    rng : np.random.Generator, optional
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy image clipped to [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, np.sqrt(variance), image.shape)
    noisy = image + image * noise
    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_psnr(original: np.ndarray, denoised: np.ndarray,
                 data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB).

    Parameters
    ----------
    original, denoised : np.ndarray
        Images of identical shape, values in [0, data_range].
    data_range : float
        Maximum possible pixel value.

    Returns
    -------
    float
        PSNR in decibels.
    """
    mse = np.mean((original - denoised) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim(original: np.ndarray, denoised: np.ndarray,
                 data_range: float = 1.0, win_size: int = 7) -> float:
    """Structural Similarity Index (mean over the image).

    Uses a Gaussian-weighted window of size `win_size`.

    Parameters
    ----------
    original, denoised : np.ndarray
        Grayscale images of identical shape, in [0, data_range].
    data_range : float
        Dynamic range of the images.
    win_size : int
        Side length of the Gaussian window (must be odd).

    Returns
    -------
    float
        Mean SSIM value in [-1, 1].
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Gaussian kernel
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
