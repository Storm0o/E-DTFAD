"""
Task B — Automated MAD Thresholding.

Removes manual ``k₁`` and ``k₂`` parameters from the dynamic threshold
function.  Instead, the noise level is estimated automatically via the
Median Absolute Deviation (MAD) of the finest-scale diagonal wavelet
coefficients (Donoho & Johnstone, 1994), and the two bounds are derived
from that estimate.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.ndimage import convolve


def _hh_subband(image: np.ndarray) -> np.ndarray:
    """Extract the finest-scale diagonal (HH₁) wavelet detail coefficients.

    Uses a simple 2-D Haar-like high-pass filter to approximate the HH
    sub-band without requiring a full wavelet transform library.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), float64.

    Returns
    -------
    np.ndarray
        HH detail coefficients (same shape, zero-padded borders).
    """
    # Diagonal detail ≈ second-order mixed partial (Laplacian-of-diagonal)
    # Using the standard 3×3 kernel for HH approximation:
    #   [[ 1, -1],
    #    [-1,  1]]  applied at stride 1 via convolution
    kernel = np.array([[ 1.0, -1.0],
                       [-1.0,  1.0]], dtype=np.float64)
    # Pad kernel to 3×3 for scipy (centre it)
    k3 = np.zeros((3, 3), dtype=np.float64)
    k3[0:2, 0:2] = kernel
    return convolve(image, k3, mode='reflect')


def estimate_noise_variance_mad(image: np.ndarray) -> float:
    """Estimate the noise standard deviation using the MAD estimator.

    σ̂ = median(|HH₁|) / 0.6745

    where HH₁ are the finest-scale diagonal wavelet coefficients and
    0.6745 is the 75th percentile of the standard normal distribution
    (consistency factor for Gaussian noise).

    Parameters
    ----------
    image : np.ndarray
        Noisy grayscale image (H, W), float64, values in [0, 1].

    Returns
    -------
    float
        Estimated noise standard deviation (σ̂ ≥ 0).
    """
    hh = _hh_subband(image)
    mad = np.median(np.abs(hh))
    sigma_hat = mad / 0.6745
    return float(max(sigma_hat, 0.0))


def compute_dynamic_thresholds(sigma_hat: float,
                               k1_factor: float = 0.5,
                               k2_factor: float = 3.0) -> tuple[float, float]:
    """Derive the dynamic threshold bounds from the estimated noise level.

    k₁ = k1_factor × σ̂   (edge-protection lower bound)
    k₂ = k2_factor × σ̂   (smoothing upper bound)

    The invariant k₁ < k₂ is guaranteed as long as k1_factor < k2_factor.

    Parameters
    ----------
    sigma_hat : float
        Estimated noise standard deviation (from ``estimate_noise_variance_mad``).
    k1_factor : float
        Multiplier for the lower bound (default 0.5).
    k2_factor : float
        Multiplier for the upper bound (default 3.0).

    Returns
    -------
    tuple[float, float]
        (k₁, k₂) with k₁ < k₂.

    Raises
    ------
    ValueError
        If k1_factor ≥ k2_factor.
    """
    if k1_factor >= k2_factor:
        raise ValueError(
            f"k1_factor must be < k2_factor, got {k1_factor} and {k2_factor}"
        )
    k1 = k1_factor * sigma_hat
    k2 = k2_factor * sigma_hat
    return k1, k2
