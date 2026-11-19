"""
Tasks A + D — Grünwald-Letnikov (GL) Fractional Derivative with
Spatially Adaptive α and Dynamic Truncation.

The GL derivative of fractional order α is defined by the series:

    D^α f(x) ≈ Σ_{j=0}^{N-1}  w_j(α) · f(x − j·h)   /  h^α

where the binomial-like weights satisfy the recurrence:

    w_0 = 1,   w_j = w_{j−1} · (1 − (α + 1) / j)

This module:
  • Accepts a *per-pixel* α matrix (Task A).
  • Dynamically truncates the series (Task D): flat regions stop at
    ``min_terms`` (default 5), textured regions run up to ``max_terms``
    (default 15).
"""

# pyrefly: ignore [missing-import]
import numpy as np


# ---------------------------------------------------------------------------
# GL coefficient helpers
# ---------------------------------------------------------------------------

def gl_coefficients(alpha: float, n_terms: int) -> np.ndarray:
    """Compute the first ``n_terms`` Grünwald-Letnikov weights for order α.

    Parameters
    ----------
    alpha : float
        Fractional differentiation order (typically in [1.0, 2.0]).
    n_terms : int
        Number of weights to compute (≥ 1).

    Returns
    -------
    np.ndarray
        1-D array of shape (n_terms,) with the GL weights.
    """
    w = np.zeros(n_terms, dtype=np.float64)
    w[0] = 1.0
    for j in range(1, n_terms):
        w[j] = w[j - 1] * (1.0 - (alpha + 1.0) / j)
    return w


# ---------------------------------------------------------------------------
# Dynamic truncation map
# ---------------------------------------------------------------------------

def compute_max_terms_map(gradient_mag: np.ndarray,
                          min_terms: int = 5,
                          max_terms: int = 15) -> np.ndarray:
    """Map gradient magnitudes to per-pixel truncation depths.

    Linear interpolation:
      • Low gradient (flat)   → ``min_terms``  (save compute)
      • High gradient (edges) → ``max_terms``  (higher accuracy)

    Parameters
    ----------
    gradient_mag : np.ndarray
        Non-negative gradient magnitude (H, W).
    min_terms : int
        Minimum number of GL series terms.
    max_terms : int
        Maximum number of GL series terms.

    Returns
    -------
    np.ndarray
        Integer array of shape (H, W) with values in [min_terms, max_terms].
    """
    g_max = gradient_mag.max()
    if g_max < 1e-12:
        return np.full(gradient_mag.shape, min_terms, dtype=np.int32)

    normalised = gradient_mag / g_max
    terms = min_terms + normalised * (max_terms - min_terms)
    return np.clip(np.round(terms), min_terms, max_terms).astype(np.int32)


# ---------------------------------------------------------------------------
# GL derivative in 2-D (four cardinal directions)
# ---------------------------------------------------------------------------

def _gl_directional(image: np.ndarray,
                    alpha_matrix: np.ndarray,
                    terms_map: np.ndarray,
                    direction: str,
                    global_max_terms: int = 15) -> np.ndarray:
    """Compute the GL fractional derivative in one cardinal direction.

    The image is shifted in the specified direction and weighted by the
    GL coefficients.  Boundary pixels use zero-padding (Dirichlet BC).

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W), float64.
    alpha_matrix : np.ndarray
        Per-pixel fractional orders (H, W).
    terms_map : np.ndarray
        Per-pixel truncation depth (H, W), int.
    direction : str
        One of ``'north'``, ``'south'``, ``'east'``, ``'west'``.
    global_max_terms : int
        Upper bound for the series (used to pre-compute all shifts).

    Returns
    -------
    np.ndarray
        Fractional derivative image (H, W).
    """
    H, W = image.shape
    result = np.zeros_like(image)

    # Pre-compute shifted images for all offsets up to global_max_terms
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

    # Vectorised computation grouped by unique (alpha, n_terms) pairs
    # to avoid a per-pixel Python loop.
    # Quantise alpha to 2 decimal places for grouping efficiency.
    alpha_q = np.round(alpha_matrix, 2)
    unique_alphas = np.unique(alpha_q)

    for a_val in unique_alphas:
        alpha_mask = alpha_q == a_val
        w = gl_coefficients(a_val, global_max_terms)

        # For each possible truncation depth within this alpha group
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
    """Compute GL fractional derivatives in all four cardinal directions.

    Parameters
    ----------
    image : np.ndarray
        Input image (H, W), float64.
    alpha_matrix : np.ndarray
        Per-pixel fractional order (H, W).
    terms_map : np.ndarray or None
        Per-pixel truncation depth.  If ``None``, all pixels use
        ``global_max_terms``.
    global_max_terms : int
        Upper bound on the number of GL series terms.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys ``'north'``, ``'south'``, ``'east'``,
        ``'west'``, each mapping to an (H, W) derivative image.
    """
    if terms_map is None:
        terms_map = np.full(image.shape, global_max_terms, dtype=np.int32)

    derivs = {}
    for direction in ('north', 'south', 'east', 'west'):
        derivs[direction] = _gl_directional(
            image, alpha_matrix, terms_map, direction, global_max_terms
        )
    return derivs
