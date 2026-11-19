"""
Task A — Spatially Adaptive Fractional Order (α).

Replaces a hardcoded ``alpha = 1.1`` with a per-pixel α matrix derived
from local gradient magnitudes.  Sharp edges receive lower orders (≈ 1.0)
to preserve structure, while flat / homogeneous regions receive higher
orders (up to 1.5) for stronger smoothing.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from scipy.ndimage import gaussian_filter


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute the gradient magnitude of a 2-D image using Sobel operators.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W), float64, values in [0, 1].

    Returns
    -------
    np.ndarray
        Gradient magnitude array of the same shape, ≥ 0.
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_structure_tensor_trace(image: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """Compute the trace of the smoothed Structure Tensor.

    The structure tensor J = G_rho * (∇I ∇I^T).
    The trace of J is μ₁ + μ₂, which serves as a robust, noise-resistant
    edge and texture indicator. Smoothing the squared gradient avoids
    over-smoothing textured regions where raw gradients fluctuate wildly.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W), float64.
    rho : float
        Integration scale (sigma) for the Gaussian smoothing.

    Returns
    -------
    np.ndarray
        Robust feature map (sqrt of the trace), same scale as gradient mag.
    """
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    grad_sq = gx ** 2 + gy ** 2
    
    if rho > 0:
        trace = gaussian_filter(grad_sq, sigma=rho)
    else:
        trace = grad_sq
        
    return np.sqrt(np.maximum(trace, 0.0))


def map_alpha(feature_map: np.ndarray,
              alpha_min: float = 1.0,
              alpha_max: float = 1.05) -> np.ndarray:
    """Map the robust feature map to spatially varying fractional orders.

    The mapping is *inversely* proportional to feature strength:

        α(x, y) = α_max − (α_max − α_min) · F(x,y) / max(F)

    This ensures:
      • Edges/Textures (high energy)  → α ≈ α_min (near 1.0, mild diff)
      • Flat regions (low energy) → α ≈ α_max (up to 1.5, strong smooth)

    Parameters
    ----------
    feature_map : np.ndarray
        Robust non-negative feature map (H, W) from Structure Tensor.
    alpha_min : float
        Minimum fractional order (applied at strongest edges/textures).
        Must satisfy 1.0 ≤ alpha_min < alpha_max.
    alpha_max : float
        Maximum fractional order (applied in flat regions).
        Must satisfy alpha_max ≤ 2.0.

    Returns
    -------
    np.ndarray
        α matrix of same shape, values in [alpha_min, alpha_max].
    """
    if not (1.0 <= alpha_min < alpha_max <= 2.0):
        raise ValueError(
            f"Require 1.0 ≤ alpha_min < alpha_max ≤ 2.0, "
            f"got alpha_min={alpha_min}, alpha_max={alpha_max}"
        )

    f_max = feature_map.max()
    if f_max < 1e-12:
        return np.full_like(feature_map, alpha_max)

    normalised = feature_map / f_max  # [0, 1]
    alpha_matrix = alpha_max - (alpha_max - alpha_min) * normalised

    return alpha_matrix
