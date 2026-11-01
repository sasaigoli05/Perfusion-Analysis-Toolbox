"""
Processing settings for the perfusion calculator.

The calculator used to take an `argparse.Namespace`, which tied the library to
the command line: anything calling it from another pipeline had to fabricate a
Namespace and know which attributes were read. `PerfusionSettings` is the same
information as a plain dataclass with defaults, so library callers construct it
directly and the CLI merely fills it in.

Defaults are the configuration validated on the 79-study cohort, so
`PerfusionSettings()` reproduces the published behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class PerfusionSettings:
    # --- reference images -------------------------------------------------
    # Only used to inherit geometry and timing for the written outputs. A
    # caller working purely in memory can leave both None and pass `geometry`
    # to the calculator instead.
    image_path: str | None = None
    mask_path: str | None = None

    # --- timing -----------------------------------------------------------
    tr: float | None = None
    """Frame interval in seconds. None measures it from the image header."""

    # --- motion -----------------------------------------------------------
    motion_correction: bool = True

    # --- arterial input function -----------------------------------------
    aif_mode: str = "voxel"
    """'voxel' scores individual voxels; 'region' requires connected components."""

    aif_mask: Any = None
    """
    Optional arterial mask constraining where the AIF is taken from -- a path,
    or an array in (Z, Y, X). Supplying an anatomical vessel segmentation here
    improved agreement with the reference by +0.068 Dice on the cohort.
    """

    # --- conditioning before deconvolution --------------------------------
    denoise_method: str = "gaussian"
    """'gaussian' | 'temporal' | 'curvature' | 'tips' | 'none'; combine with '+'."""

    smooth_fwhm: float = 4.0
    tips_radius: int = 3
    tips_h: float = 10.0
    temporal_sigma: float = 1.0
    curvature_iterations: int = 5

    # --- deconvolution ----------------------------------------------------
    deconv_method: str = "svd"
    """'svd' | 'osvd' | 'osvd_oi' | 'bayes'."""

    svd_frac: float = 0.20
    max_delay_s: float = 16.0

    tmax_reference: bool = True
    """
    Re-reference Tmax so normal tissue reads ~0 before writing it.

    Raw Tmax inherits the AIF's own delay, so the fraction of brain past the
    clinical 6s threshold moved from 9% to 90% across seven cohort scans
    purely with AIF timing. Referencing narrowed that to 9-23%. Turn off only
    to reproduce an unreferenced run.
    """

    # --- Bayesian estimator ----------------------------------------------
    bayes_spatial: float = 0.0
    bayes_prior_weight: float = 1.0

    # --- physical constants ----------------------------------------------
    rho: float = 1.0
    """Brain tissue density, g/mL."""
    kh: float = 1.0
    """Haematocrit correction factor."""

    @classmethod
    def from_namespace(cls, ns) -> "PerfusionSettings":
        """Build from an argparse.Namespace, ignoring unrelated attributes."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in vars(ns).items() if k in known})

    def replace(self, **kw) -> "PerfusionSettings":
        """A copy with some fields changed -- settings stay immutable in use."""
        known = {f.name for f in fields(self)}
        unknown = set(kw) - known
        if unknown:
            raise TypeError(f"unknown setting(s): {sorted(unknown)}")
        return PerfusionSettings(**{**vars(self), **kw})
