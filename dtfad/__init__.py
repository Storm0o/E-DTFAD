"""
DTFAD — Dynamical Threshold-Based Fractional Anisotropic Diffusion
for speckle noise removal.

Upgraded implementation featuring:
  - Spatially adaptive fractional order α (Task A)
  - MAD-based automated thresholding (Task B)
  - Bilateral filter pre-smoothing for GLI (Task C)
  - Dynamic GL derivative truncation (Task D)
"""

from .adaptive_alpha import compute_gradient_magnitude, map_alpha
from .mad_threshold import estimate_noise_variance_mad, compute_dynamic_thresholds
from .gli_bilateral import compute_gli_bilateral
from .gl_derivative import gl_coefficients, gl_derivative_2d, compute_max_terms_map
from .diffusion_engine import dtfad_denoise
from .utils import load_image, save_image, add_speckle_noise, compute_psnr, compute_ssim

__all__ = [
    "compute_gradient_magnitude",
    "map_alpha",
    "estimate_noise_variance_mad",
    "compute_dynamic_thresholds",
    "compute_gli_bilateral",
    "gl_coefficients",
    "gl_derivative_2d",
    "compute_max_terms_map",
    "dtfad_denoise",
    "load_image",
    "save_image",
    "add_speckle_noise",
    "compute_psnr",
    "compute_ssim",
]
