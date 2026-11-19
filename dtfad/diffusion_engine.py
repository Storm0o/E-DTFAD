"""
DTFAD Diffusion Engine - Core Orchestrator.

Ties together all four upgrade tasks into a single iterative diffusion
pipeline:
  1. Bilateral pre-smoothing -> GLI  (Task C)
  2. MAD noise estimation -> k1, k2  (Task B)
  3. Gradient -> adaptive alpha matrix (Task A)
  4. Gradient -> dynamic truncation    (Task D)
  5. Iterative fractional diffusion

The diffusion update follows:
    I^{t+1} = I^t + dt * sum_d c_d * D^alpha_d I^t

where c_d is the Perona-Malik diffusion coefficient gated by the
dynamic thresholds k1 and k2.
"""

import numpy as np

from .adaptive_alpha import (
    compute_gradient_magnitude,
    compute_structure_tensor_trace,
    map_alpha,
)
from .mad_threshold import estimate_noise_variance_mad, compute_dynamic_thresholds
from .gli_bilateral import compute_gli_bilateral
from .gl_derivative import gl_derivative_2d, compute_max_terms_map


def _diffusion_coefficient(feature_map: np.ndarray,
                           k1: float,
                           k2: float) -> np.ndarray:
    """Compute the Perona-Malik diffusion coefficient: c(F) = exp(-(F/k2)^2)."""
    if k2 < 1e-12:
        return np.zeros_like(feature_map)
    return np.exp(-((feature_map / k2) ** 2))


def dtfad_denoise(image: np.ndarray,
                  n_iter: int = 20,
                  dt: float = 0.1,
                  alpha_min: float = 1.0,
                  alpha_max: float = 1.05,
                  min_terms: int = 5,
                  max_terms: int = 15,
                  sigma_spatial: float = 2.0,
                  k1_factor: float = 0.5,
                  k2_factor: float = 3.0,
                  verbose: bool = False) -> np.ndarray:
    """Run the full E-DTFAD speckle-denoising pipeline.

    Pipeline flow:
      Step 1: Bilateral pre-smooth -> GLI (Task C)
      Step 2: MAD noise estimate -> k1, k2 (Task B)
      Step 3: GLI edge indicator -> smooth-region weight
      Step 4: Iterative diffusion loop:
              - Sobel gradient magnitude for diffusion coefficient
              - Structure tensor trace for adaptive alpha (Task A) + truncation map (Task D)
              - Perona-Malik coefficient on gradient
              - GL fractional derivatives in 4 directions
              - Accumulate update modulated by GLI smooth weight
    """
    gli = compute_gli_bilateral(image, sigma_spatial=sigma_spatial)

    sigma_hat = estimate_noise_variance_mad(image)
    k1, k2 = compute_dynamic_thresholds(sigma_hat, k1_factor, k2_factor)

    if verbose:
        print(f"MAD sigma_hat = {sigma_hat:.6f},  k1 = {k1:.6f},  k2 = {k2:.6f}")

    gli_dev = np.abs(1.0 - gli)
    gli_dev_max = gli_dev.max()
    if gli_dev_max > 1e-12:
        gli_edge_norm = gli_dev / gli_dev_max
    else:
        gli_edge_norm = np.zeros_like(gli)
    gli_smooth_weight = 1.0 - gli_edge_norm

    u = image.copy()

    for t in range(n_iter):
        g = compute_gradient_magnitude(u)

        f = compute_structure_tensor_trace(u)
        alpha_matrix = map_alpha(f, alpha_min, alpha_max)
        terms_map = compute_max_terms_map(f, min_terms, max_terms)

        c = _diffusion_coefficient(g, k1, k2)

        derivs = gl_derivative_2d(u, alpha_matrix, terms_map, max_terms)

        update = np.zeros_like(u)
        for direction in ('north', 'south', 'east', 'west'):
            update -= c * derivs[direction]

        update *= gli_smooth_weight

        u = u + dt * update
        u = np.clip(u, 0.0, 1.0)

        if verbose:
            print(f"  iter {t + 1:3d}/{n_iter}  "
                  f"max|update|={np.max(np.abs(dt * update)):.6f}")

    return u
