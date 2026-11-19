"""
Unit tests for Task D — Dynamic GL Derivative Truncation.

Validates:
  1. Flat regions receive ``min_terms`` truncation.
  2. Edge/texture regions receive ``max_terms`` truncation.
  3. All truncation depths are in [min_terms, max_terms].
  4. GL coefficients are finite and bounded.
  5. GL derivative output is finite (no NaN / Inf).
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pytest

from dtfad.adaptive_alpha import compute_gradient_magnitude
from dtfad.gl_derivative import gl_coefficients, gl_derivative_2d, compute_max_terms_map


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def step_edge_image():
    """64×64 image with a vertical step edge at col 32."""
    img = np.zeros((64, 64), dtype=np.float64)
    img[:, 32:] = 1.0
    return img


@pytest.fixture
def flat_image():
    """64×64 constant image."""
    return np.full((64, 64), 0.5, dtype=np.float64)


# -----------------------------------------------------------------------
# Tests — gl_coefficients
# -----------------------------------------------------------------------

class TestGLCoefficients:
    def test_first_weight_is_one(self):
        w = gl_coefficients(1.1, 10)
        assert abs(w[0] - 1.0) < 1e-15

    def test_returns_requested_length(self):
        for n in [1, 5, 15, 20]:
            w = gl_coefficients(1.3, n)
            assert len(w) == n

    def test_finite_values(self):
        w = gl_coefficients(1.5, 15)
        assert np.all(np.isfinite(w))

    def test_weights_bounded(self):
        """Weights should not diverge for typical fractional orders."""
        w = gl_coefficients(1.2, 15)
        assert np.all(np.abs(w) < 100), f"Weights too large: {w}"

    def test_alpha_1_recovers_first_difference(self):
        """For α = 1, weights should approximate [1, -1, 0, 0, ...]."""
        w = gl_coefficients(1.0, 5)
        expected = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(w, expected, atol=1e-12)


# -----------------------------------------------------------------------
# Tests — compute_max_terms_map
# -----------------------------------------------------------------------

class TestMaxTermsMap:
    def test_flat_region_min_terms(self, flat_image):
        g = compute_gradient_magnitude(flat_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        assert np.all(tmap == 5)

    def test_edge_gets_max_terms(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        # At the edge itself (col ≈32) the gradient is maximal
        edge_terms = tmap[:, 30:35].max()
        assert edge_terms == 15

    def test_all_in_range(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        assert tmap.min() >= 5
        assert tmap.max() <= 15

    def test_shape_matches_input(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        assert tmap.shape == step_edge_image.shape

    def test_integer_dtype(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        assert np.issubdtype(tmap.dtype, np.integer)


# -----------------------------------------------------------------------
# Tests — gl_derivative_2d
# -----------------------------------------------------------------------

class TestGLDerivative2D:
    def test_output_is_finite(self, step_edge_image):
        alpha_matrix = np.full(step_edge_image.shape, 1.2)
        derivs = gl_derivative_2d(step_edge_image, alpha_matrix,
                                  global_max_terms=10)
        for direction, d in derivs.items():
            assert np.all(np.isfinite(d)), (
                f"NaN/Inf found in {direction} derivative"
            )

    def test_flat_image_near_zero_derivative_integer_order(self, flat_image):
        """For integer order α=1, the derivative of a constant IS zero."""
        alpha_matrix = np.full(flat_image.shape, 1.0)
        derivs = gl_derivative_2d(flat_image, alpha_matrix,
                                  global_max_terms=10)
        margin = 10
        for direction, d in derivs.items():
            interior = d[margin:-margin, margin:-margin]
            assert np.max(np.abs(interior)) < 1e-6, (
                f"Integer-order flat derivative should be ~0 in {direction}: "
                f"max = {np.max(np.abs(interior))}"
            )

    def test_flat_image_uniform_derivative_fractional(self, flat_image):
        """For fractional α, the GL derivative of a constant is non-zero
        but should be *uniform* across all interior pixels."""
        n_terms = 10
        alpha_matrix = np.full(flat_image.shape, 1.3)
        derivs = gl_derivative_2d(flat_image, alpha_matrix,
                                  global_max_terms=n_terms)
        margin = n_terms
        for direction, d in derivs.items():
            interior = d[margin:-margin, margin:-margin]
            # All interior values should be identical (constant input)
            assert np.std(interior) < 1e-10, (
                f"Fractional derivative of constant should be uniform in "
                f"{direction}: std = {np.std(interior)}"
            )

    def test_returns_four_directions(self, step_edge_image):
        alpha_matrix = np.full(step_edge_image.shape, 1.1)
        derivs = gl_derivative_2d(step_edge_image, alpha_matrix,
                                  global_max_terms=5)
        assert set(derivs.keys()) == {'north', 'south', 'east', 'west'}

    def test_shape_preserved(self, step_edge_image):
        alpha_matrix = np.full(step_edge_image.shape, 1.1)
        derivs = gl_derivative_2d(step_edge_image, alpha_matrix,
                                  global_max_terms=5)
        for d in derivs.values():
            assert d.shape == step_edge_image.shape

    def test_dynamic_truncation_accepted(self, step_edge_image):
        """Verify that per-pixel terms_map is respected without error."""
        g = compute_gradient_magnitude(step_edge_image)
        tmap = compute_max_terms_map(g, min_terms=5, max_terms=15)
        alpha_matrix = np.full(step_edge_image.shape, 1.2)
        derivs = gl_derivative_2d(step_edge_image, alpha_matrix,
                                  terms_map=tmap, global_max_terms=15)
        for d in derivs.values():
            assert np.all(np.isfinite(d))
