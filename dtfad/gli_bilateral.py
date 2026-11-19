"""
Task C — Pre-Smoothing Kernel Upgrade (Bilateral Filter).

Replaces the basic Gaussian convolution used to compute the Gray Level
Indicator (GLI) with an edge-preserving **bilateral filter**.  The colour
(range) sigma is set to the image's own standard deviation so that
smoothing adapts to the image content automatically.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2


def compute_gli_bilateral(image: np.ndarray,
                          sigma_spatial: float = 2.0,
                          sigma_color: float | None = None,
                          diameter: int = 0) -> np.ndarray:
    """Compute the Gray Level Indicator (GLI) via bilateral pre-smoothing.

    GLI(x, y)  =  BilateralFilter(I)(x, y)  /  (I(x, y) + ε)

    Values close to 1.0 indicate locally smooth regions; deviations
    signal edges or texture.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W), float64, values in [0, 1].
    sigma_spatial : float
        Spatial sigma (pixel neighbourhood radius) for the bilateral filter.
    sigma_color : float or None
        Range (colour/intensity) sigma.  When ``None``, the standard
        deviation of ``image`` is used, coupling the filter to the
        image's inherent intensity spread.
    diameter : int
        Diameter of each pixel neighbourhood.  When 0, it is computed
        automatically from ``sigma_spatial``.

    Returns
    -------
    np.ndarray
        GLI map of shape (H, W), values > 0.
    """
    if sigma_color is None:
        sigma_color = float(np.std(image))
        # Clamp to a sensible minimum to avoid degenerate filtering
        sigma_color = max(sigma_color, 1e-6)

    if diameter <= 0:
        diameter = max(int(np.ceil(2 * sigma_spatial)) * 2 + 1, 5)

    # OpenCV's bilateral filter expects uint8 or float32
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
