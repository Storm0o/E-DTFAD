"""
Task B - Automated MAD Thresholding.

Removes manual k1 and k2 parameters. The noise level is estimated
automatically via the Median Absolute Deviation (MAD) of the finest-scale
diagonal wavelet coefficients (Donoho & Johnstone, 1994), and the two
threshold bounds are derived from that estimate.
"""

import numpy as np
from scipy.ndimage import convolve


def _hh_subband(image: np.ndarray) -> np.ndarray:
    """Extract the finest-scale diagonal (HH1) wavelet detail coefficients."""
    kernel = np.array([[ 1.0, -1.0],
                       [-1.0,  1.0]], dtype=np.float64)
    k3 = np.zeros((3, 3), dtype=np.float64)
    k3[0:2, 0:2] = kernel
    return convolve(image, k3, mode='reflect')


def estimate_noise_variance_mad(image: np.ndarray) -> float:
    """Estimate the noise standard deviation using the MAD estimator.

    sigma_hat = median(|HH1|) / 0.6745
    """
    hh = _hh_subband(image)
    mad = np.median(np.abs(hh))
    sigma_hat = mad / 0.6745
    return float(max(sigma_hat, 0.0))


def compute_dynamic_thresholds(sigma_hat: float,
                               k1_factor: float = 0.5,
                               k2_factor: float = 3.0) -> tuple[float, float]:
    """Derive the dynamic threshold bounds from the estimated noise level.

    k1 = k1_factor * sigma_hat   (edge-protection lower bound)
    k2 = k2_factor * sigma_hat   (smoothing upper bound)
    """
    if k1_factor >= k2_factor:
        raise ValueError(
            f"k1_factor must be < k2_factor, got {k1_factor} and {k2_factor}"
        )
    k1 = k1_factor * sigma_hat
    k2 = k2_factor * sigma_hat
    return k1, k2
