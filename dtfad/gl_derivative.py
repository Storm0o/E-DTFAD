"""
Tasks A + D - Grunwald-Letnikov (GL) Fractional Derivative with
Spatially Adaptive alpha and Dynamic Truncation.

The GL derivative of fractional order alpha is defined by the series:

    D^alpha f(x) = sum_{j=0}^{N-1} w_j(alpha) * f(x - j*h) / h^alpha

where the binomial-like weights satisfy the recurrence:

    w_0 = 1,   w_j = w_{j-1} * (1 - (alpha + 1) / j)

This module accepts a per-pixel alpha matrix (Task A) and dynamically
truncates the series (Task D): flat regions stop at min_terms,
textured regions run up to max_terms.
"""

import numpy as np


def gl_coefficients(alpha: float, n_terms: int) -> np.ndarray:
    """Compute the first n_terms Grunwald-Letnikov weights for order alpha."""
    w = np.zeros(n_terms, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, n_terms):
        w[j] = w[j - 1] * (1.0 - (alpha + 1.0) / j)
    return w


def compute_max_terms_map(gradient_mag: np.ndarray,
                          min_terms: int = 5,
                          max_terms: int = 15) -> np.ndarray:
    """Map gradient magnitudes to per-pixel truncation depths.

    Low gradient (flat)   -> min_terms (save compute)
    High gradient (edges) -> max_terms (higher accuracy)
    """
    g_max = gradient_mag.max()
    if g_max < 1e-12:
        return np.full(gradient_mag.shape, min_terms, dtype=np.int32)

    normalised = gradient_mag / g_max
    terms = min_terms + normalised * (max_terms - min_terms)
    return np.clip(np.round(terms), min_terms, max_terms).astype(np.int32)


def _gl_directional(image: np.ndarray,
                    alpha_matrix: np.ndarray,
                    terms_map: np.ndarray,
                    direction: str,
                    global_max_terms: int = 15) -> np.ndarray:
    """Compute the GL fractional derivative in one cardinal direction."""
    H, W = image.shape
    result = np.zeros_like(image)

    shifted = np.zeros((global_max_terms, H, W), dtype=np.float64)
    for j in range(global_max_terms):
        if direction == 'north':
            if j == 0:
                shifted[j] = image
            else:
                shifted[j, j:, :] = image[:-j, :] if j < H else 0
        elif direction == 'south':
            if j == 0:
                shifted[j] = image
            else:
                shifted[j, :-j, :] = image[j:, :] if j < H else 0
        elif direction == 'west':
            if j == 0:
                shifted[j] = image
            else:
                shifted[j, :, j:] = image[:, :-j] if j < W else 0
        elif direction == 'east':
            if j == 0:
                shifted[j] = image
            else:
                shifted[j, :, :-j] = image[:, j:] if j < W else 0
        else:
            raise ValueError(f"Unknown direction: {direction}")

    alpha_q = np.round(alpha_matrix, 2)
    unique_alphas = np.unique(alpha_q)

    for a_val in unique_alphas:
        alpha_mask = alpha_q == a_val
        w = gl_coefficients(a_val, global_max_terms)

        depth_vals = np.unique(terms_map[alpha_mask])
        for n in depth_vals:
            cell_mask = alpha_mask & (terms_map == n)
            accum = np.zeros((H, W), dtype=np.float64)
            for j in range(int(n)):
                accum += w[j] * shifted[j]
            result[cell_mask] = accum[cell_mask]

    return result


def gl_derivative_2d(image: np.ndarray,
                     alpha_matrix: np.ndarray,
                     terms_map: np.ndarray | None = None,
                     global_max_terms: int = 15
                     ) -> dict[str, np.ndarray]:
    """Compute GL fractional derivatives in all four cardinal directions."""
    if terms_map is None:
        terms_map = np.full(image.shape, global_max_terms, dtype=np.int32)

    derivs = {}
    for direction in ('north', 'south', 'east', 'west'):
        derivs[direction] = _gl_directional(
            image, alpha_matrix, terms_map, direction, global_max_terms
        )
    return derivs
