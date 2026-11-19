"""
Task C - Pre-Smoothing Kernel Upgrade (Bilateral Filter).

Replaces the basic Gaussian convolution used to compute the Gray Level
Indicator (GLI) with an edge-preserving bilateral filter. The colour
(range) sigma is set to the image's own standard deviation so that
smoothing adapts to the image content automatically.
"""

import numpy as np
import cv2


def compute_gli_bilateral(image: np.ndarray,
                          sigma_spatial: float = 2.0,
                          sigma_color: float | None = None,
                          diameter: int = 0) -> np.ndarray:
    """Compute the Gray Level Indicator (GLI) via bilateral pre-smoothing.

    GLI(x, y) = BilateralFilter(I)(x, y) / (I(x, y) + eps)

    Values close to 1.0 indicate locally smooth regions; deviations
    signal edges or texture.
    """
    if sigma_color is None:
        sigma_color = float(np.std(image))
        sigma_color = max(sigma_color, 1e-6)

    if diameter <= 0:
        diameter = max(int(np.ceil(2 * sigma_spatial)) * 2 + 1, 5)

    img32 = image.astype(np.float32)
    smoothed = cv2.bilateralFilter(
        img32,
        d=diameter,
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_spatial),
    )
    smoothed = smoothed.astype(np.float64)

    eps = 1e-10
    gli = smoothed / (image + eps)

    return gli
