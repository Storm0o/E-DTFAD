"""
Task A - Spatially Adaptive Fractional Order (alpha).

Replaces a hardcoded alpha = 1.1 with a per-pixel alpha matrix derived
from local gradient magnitudes. Sharp edges receive lower orders (near 1.0)
to preserve structure, while flat regions receive higher orders for
stronger smoothing.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute the gradient magnitude of a 2-D image using Sobel operators."""
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)


def compute_structure_tensor_trace(image: np.ndarray, rho: float = 1.0) -> np.ndarray:
    """Compute the trace of the smoothed Structure Tensor.

    The structure tensor J = G_rho * (nabla_I nabla_I^T).
    The trace (mu1 + mu2) serves as a robust, noise-resistant edge indicator.
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

    alpha(x, y) = alpha_max - (alpha_max - alpha_min) * F(x,y) / max(F)

    Edges/Textures (high energy) -> alpha near alpha_min (mild diffusion)
    Flat regions (low energy)    -> alpha near alpha_max (strong smoothing)
    """
    if not (1.0 <= alpha_min < alpha_max <= 2.0):
        raise ValueError(
            f"Require 1.0 <= alpha_min < alpha_max <= 2.0, "
            f"got alpha_min={alpha_min}, alpha_max={alpha_max}"
        )

    f_max = feature_map.max()
    if f_max < 1e-12:
        return np.full_like(feature_map, alpha_max)

    normalised = feature_map / f_max
    alpha_matrix = alpha_max - (alpha_max - alpha_min) * normalised

    return alpha_matrix
