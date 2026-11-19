"""
DTFAD Diffusion Engine — Core Orchestrator.

Ties together all four upgrade tasks into a single iterative diffusion
pipeline:
  1. Bilateral pre-smoothing → GLI  (Task C)
  2. MAD noise estimation → k₁, k₂  (Task B)
  3. Gradient → adaptive α matrix   (Task A)
  4. Gradient → dynamic truncation   (Task D)
  5. Iterative fractional diffusion

The diffusion update follows:

    I^{t+1} = I^t  +  dt × Σ_d  c_d · D^α_d I^t

where c_d is the Perona–Malik diffusion coefficient gated by the
dynamic thresholds k₁ and k₂.
"""

# pyrefly: ignore [missing-import]
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
    """Compute the Perona–Malik diffusion coefficient.

    c(F) = exp(−(F / k₂)²)

    This gives a smooth, monotonically decreasing response from 1 (flat
    regions) towards 0 (strong edges), without the hard cut-off that
    the Tukey biweight introduces.

    Parameters
    ----------
    feature_map : np.ndarray
        Robust feature map (H, W).
    k1 : float
        Lower threshold (reserved for future two-threshold schemes).
    k2 : float
        Smoothing scale parameter (MAD derived).

    Returns
    -------
    np.ndarray
        Diffusion coefficient image in (0, 1].
    """
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
    """Run the full DTFAD speckle-denoising pipeline.

    Parameters
    ----------
    image : np.ndarray
        Noisy grayscale image (H, W), float64, values in [0, 1].
    n_iter : int
        Number of diffusion iterations.
    dt : float
        Time-step size (stability requires dt ≤ 0.25).
    alpha_min, alpha_max : float
        Bounds for the spatially adaptive fractional order.
    min_terms, max_terms : int
        Bounds for dynamic GL series truncation.
    sigma_spatial : float
        Bilateral filter spatial sigma (for GLI).
    k1_factor, k2_factor : float
        Multipliers for MAD-based threshold derivation.
    verbose : bool
        If True, print iteration progress.

    Returns
    -------
    np.ndarray
        Denoised image (H, W), float64, values clipped to [0, 1].
    """
    # --- Step 1: Pre-smooth with bilateral filter → GLI (Task C) ----------
    gli = compute_gli_bilateral(image, sigma_spatial=sigma_spatial)

    # --- Step 2: Estimate noise → derive k₁, k₂ (Task B) -----------------
    sigma_hat = estimate_noise_variance_mad(image)
    k1, k2 = compute_dynamic_thresholds(sigma_hat, k1_factor, k2_factor)

    if verbose:
        print(f"MAD σ̂ = {sigma_hat:.6f},  k₁ = {k1:.6f},  k₂ = {k2:.6f}")

    # --- Step 3: Precompute GLI edge indicator (Task C) --------------------
    #   GLI ≈ 1.0 in flat regions, deviates in edges/texture.
    #   Use |1 − GLI| as an edge strength, then invert to get a
    #   smooth-region weight: high where flat, low at edges.
    gli_dev = np.abs(1.0 - gli)
    gli_dev_max = gli_dev.max()
    if gli_dev_max > 1e-12:
        gli_edge_norm = gli_dev / gli_dev_max          # [0, 1]
    else:
        gli_edge_norm = np.zeros_like(gli)
    gli_smooth_weight = 1.0 - gli_edge_norm            # 1 = flat, 0 = edge

    # --- Step 4: Iterative diffusion loop ---------------------------------
    u = image.copy()

    for t in range(n_iter):
        # Gradient magnitude for diffusion coefficient (Sobel-based)
        g = compute_gradient_magnitude(u)

        # Structure tensor trace for adaptive α and truncation (noise-robust)
        f = compute_structure_tensor_trace(u)
        alpha_matrix = map_alpha(f, alpha_min, alpha_max)
        terms_map = compute_max_terms_map(f, min_terms, max_terms)

        # Perona–Malik diffusion coefficient on gradient magnitude
        c = _diffusion_coefficient(g, k1, k2)

        # Fractional derivatives in four directions
        derivs = gl_derivative_2d(u, alpha_matrix, terms_map, max_terms)

        # Accumulate diffusion update
        update = np.zeros_like(u)
        for direction in ('north', 'south', 'east', 'west'):
            update -= c * derivs[direction]

        # Modulate by GLI smooth-region weight
        update *= gli_smooth_weight

        u = u + dt * update
        u = np.clip(u, 0.0, 1.0)

        if verbose:
            print(f"  iter {t + 1:3d}/{n_iter}  "
                  f"max|update|={np.max(np.abs(dt * update)):.6f}")

    return u
