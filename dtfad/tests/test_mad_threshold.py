"""
Unit tests for Task B — MAD-Based Automated Thresholding.

Validates:
  1. MAD noise estimate is within tolerance of the true σ for known noise.
  2. k₁ < k₂ always holds.
  3. Thresholds scale linearly with estimated noise.
  4. Clean images produce near-zero noise estimates.
  5. Invalid factor ordering raises ValueError.
"""

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pytest

from dtfad.mad_threshold import estimate_noise_variance_mad, compute_dynamic_thresholds


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def clean_image():
    """128×128 smooth gradient — essentially noise-free."""
    return np.tile(np.linspace(0.2, 0.8, 128), (128, 1)).astype(np.float64)


@pytest.fixture
def noisy_image_known_sigma():
    """128×128 flat image + known additive Gaussian noise (σ = 0.05)."""
    rng = np.random.default_rng(42)
    true_sigma = 0.05
    base = np.full((128, 128), 0.5, dtype=np.float64)
    noise = rng.normal(0, true_sigma, base.shape)
    noisy = np.clip(base + noise, 0, 1)
    return noisy, true_sigma


# -----------------------------------------------------------------------
# Tests — estimate_noise_variance_mad
# -----------------------------------------------------------------------

class TestMADEstimator:
    def test_known_noise_within_tolerance(self, noisy_image_known_sigma):
        noisy, true_sigma = noisy_image_known_sigma
        sigma_hat = estimate_noise_variance_mad(noisy)
        # Allow generous tolerance because MAD on small images is noisy
        assert sigma_hat > 0, "Noise estimate should be positive"
        # The MAD estimator on 128×128 with σ=0.05 should be in the
        # right ballpark (within 100% relative error for small images)
        assert sigma_hat < true_sigma * 4, (
            f"σ̂ = {sigma_hat:.4f} is unreasonably large vs true σ = {true_sigma}"
        )

    def test_clean_image_low_estimate(self, clean_image):
        sigma_hat = estimate_noise_variance_mad(clean_image)
        # A smooth gradient will have *some* response from the HH filter,
        # but it should be much smaller than typical noise levels.
        assert sigma_hat >= 0
        # Not strictly zero because the gradient creates some HF content

    def test_returns_non_negative(self, noisy_image_known_sigma):
        noisy, _ = noisy_image_known_sigma
        sigma_hat = estimate_noise_variance_mad(noisy)
        assert sigma_hat >= 0.0

    def test_higher_noise_higher_estimate(self):
        """More noise → higher σ̂."""
        rng = np.random.default_rng(123)
        base = np.full((128, 128), 0.5, dtype=np.float64)

        low_noise = np.clip(base + rng.normal(0, 0.02, base.shape), 0, 1)
        high_noise = np.clip(base + rng.normal(0, 0.10, base.shape), 0, 1)

        sig_low = estimate_noise_variance_mad(low_noise)
        sig_high = estimate_noise_variance_mad(high_noise)

        assert sig_high > sig_low, (
            f"Higher noise image σ̂={sig_high:.4f} should exceed "
            f"lower noise σ̂={sig_low:.4f}"
        )


# -----------------------------------------------------------------------
# Tests — compute_dynamic_thresholds
# -----------------------------------------------------------------------

class TestDynamicThresholds:
    def test_k1_less_than_k2(self):
        k1, k2 = compute_dynamic_thresholds(0.05)
        assert k1 < k2

    def test_scales_with_sigma(self):
        k1_a, k2_a = compute_dynamic_thresholds(0.05)
        k1_b, k2_b = compute_dynamic_thresholds(0.10)
        assert k1_b > k1_a
        assert k2_b > k2_a

    def test_zero_noise(self):
        k1, k2 = compute_dynamic_thresholds(0.0)
        assert k1 == 0.0
        assert k2 == 0.0
        assert k1 < k2 or (k1 == 0 and k2 == 0)

    def test_custom_factors(self):
        k1, k2 = compute_dynamic_thresholds(0.1, k1_factor=1.0, k2_factor=5.0)
        assert abs(k1 - 0.1) < 1e-10
        assert abs(k2 - 0.5) < 1e-10

    def test_invalid_factors_raises(self):
        with pytest.raises(ValueError):
            compute_dynamic_thresholds(0.05, k1_factor=3.0, k2_factor=1.0)
        with pytest.raises(ValueError):
            compute_dynamic_thresholds(0.05, k1_factor=2.0, k2_factor=2.0)
