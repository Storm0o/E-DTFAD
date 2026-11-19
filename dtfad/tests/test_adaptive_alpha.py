"""
Unit tests for Task A — Spatially Adaptive Fractional Order (α).

Validates:
  1. α values are strictly within [alpha_min, alpha_max] for all pixels.
  2. Edge pixels (high gradient) receive lower α than flat pixels.
  3. Uniform (flat) images produce α ≈ alpha_max everywhere.
  4. Invalid parameter ranges raise ValueError.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pytest

from dtfad.adaptive_alpha import compute_gradient_magnitude, map_alpha


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def step_edge_image():
    """128×128 synthetic image with a sharp vertical step edge at col 64."""
    img = np.zeros((128, 128), dtype=np.float64)
    img[:, 64:] = 1.0
    return img


@pytest.fixture
def flat_image():
    """128×128 constant-intensity image."""
    return np.full((128, 128), 0.5, dtype=np.float64)


@pytest.fixture
def gradient_image():
    """128×128 horizontal gradient from 0 to 1."""
    return np.tile(np.linspace(0, 1, 128), (128, 1))


# -----------------------------------------------------------------------
# Tests — compute_gradient_magnitude
# -----------------------------------------------------------------------

class TestGradientMagnitude:
    def test_flat_image_zero_gradient(self, flat_image):
        g = compute_gradient_magnitude(flat_image)
        assert g.shape == flat_image.shape
        assert np.allclose(g, 0.0, atol=1e-10)

    def test_step_edge_peaks_at_edge(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        # Maximum gradient should be near column 64 (the edge)
        col_max = np.argmax(g.mean(axis=0))
        assert 62 <= col_max <= 66, f"Expected edge near col 64, got {col_max}"

    def test_non_negative(self, gradient_image):
        g = compute_gradient_magnitude(gradient_image)
        assert np.all(g >= 0)


# -----------------------------------------------------------------------
# Tests — map_alpha
# -----------------------------------------------------------------------

class TestMapAlpha:
    def test_values_in_range(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        alpha = map_alpha(g, alpha_min=1.0, alpha_max=1.5)
        assert alpha.min() >= 1.0 - 1e-10
        assert alpha.max() <= 1.5 + 1e-10

    def test_edges_get_lower_alpha(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        alpha = map_alpha(g, alpha_min=1.0, alpha_max=1.5)
        # Edge region: columns 62-66, Flat region: columns 0-30
        edge_alpha = alpha[:, 62:67].mean()
        flat_alpha = alpha[:, 0:30].mean()
        assert edge_alpha < flat_alpha, (
            f"Edge α ({edge_alpha:.4f}) should be < flat α ({flat_alpha:.4f})"
        )

    def test_flat_image_gets_max_alpha(self, flat_image):
        g = compute_gradient_magnitude(flat_image)
        alpha = map_alpha(g, alpha_min=1.0, alpha_max=1.5)
        assert np.allclose(alpha, 1.5, atol=1e-6)

    def test_shape_preserved(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        alpha = map_alpha(g)
        assert alpha.shape == step_edge_image.shape

    def test_custom_bounds(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        alpha = map_alpha(g, alpha_min=1.1, alpha_max=1.8)
        assert alpha.min() >= 1.1 - 1e-10
        assert alpha.max() <= 1.8 + 1e-10

    def test_invalid_bounds_raises(self, step_edge_image):
        g = compute_gradient_magnitude(step_edge_image)
        with pytest.raises(ValueError):
            map_alpha(g, alpha_min=1.5, alpha_max=1.0)  # min > max
        with pytest.raises(ValueError):
            map_alpha(g, alpha_min=0.5, alpha_max=1.5)  # min < 1.0
        with pytest.raises(ValueError):
            map_alpha(g, alpha_min=1.0, alpha_max=2.5)  # max > 2.0
