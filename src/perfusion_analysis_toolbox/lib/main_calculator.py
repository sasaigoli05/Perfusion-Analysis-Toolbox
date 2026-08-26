#!/usr/bin/env python3

import argparse
import logging
import os

import itk
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt
from .ParamsCalculator import register as register


# In-plane smoothing applied to the CTC before deconvolution. Axis 0 (Z) is
# left untouched: slices are thick (~4mm) and cross-slice blur would wash out
# ischemic contrast.
SMOOTH_FWHM_MM = 4.0

# Fallback in-plane voxel size (mm) used only when the reference image reports
# no usable spacing.
DEFAULT_VOXEL_SIZE_MM = 0.391

# Minimum overlap required before the atlas mask replaces the vessel mask.
MIN_ARTERIAL_VOXELS = 5


class MainCalculator:
    """
    Main perfusion calculator.

    Assumes:
    - raw_perf is a 4D numpy array (Z, Y, X, T)
    - mask and vessels are 3D numpy arrays (Z, Y, X)

    Image geometry (spacing, origin, direction) is read from the reference
    images at config.image_path / config.mask_path rather than passed in, so
    the outputs always inherit the input headers.
    """

    def __init__(
        self,
        raw_perf: npt.ArrayLike,
        mask: npt.ArrayLike,
        vessels: npt.ArrayLike,
        config: argparse.Namespace,
        save_path: str,
        logger: logging.Logger = None,
    ):
        if raw_perf.ndim != 4:
            raise ValueError("Expected raw_perf of shape (Z, Y, X, T)")

        self.logger = (
            logger
            if logger
            else utils.get_logger("MainCalculator", level=logging.INFO)
        )

        # Store arrays
        self.raw_perf = raw_perf.astype(np.float32)
        self.mask = mask.astype(bool)
        self.vessels = vessels.astype(bool)

        # Shapes
        self.nS, self.nR, self.nC, self.nT = self.raw_perf.shape

        # Config + paths
        self.config = config
        self.savepath = save_path

        # Store reference file paths (CRITICAL)
        self.image_path = config.image_path
        self.mask_path = config.mask_path

        os.makedirs(self.savepath, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helper: safe write using metadata copy
    # ------------------------------------------------------------------
    def _write_like(self, arr: np.ndarray, ref_img: itk.Image, out_path: str):
        """
        Write numpy array as NIfTI, copying metadata from ref_img.
        This avoids all ITK Python type issues.
        """
        img = itk.image_view_from_array(arr.astype(np.float32))
        img.CopyInformation(ref_img)
        itk.imwrite(img, out_path)

    # ------------------------------------------------------------------
    # Internal helper: in-plane voxel size from image metadata
    # ------------------------------------------------------------------
    def _in_plane_voxel_size_mm(self, ref_img) -> float:
        """
        Mean in-plane voxel size (mm) from the reference image spacing.

        ITK spacing is (x, y, z), so the in-plane components are the first
        two. Falls back to DEFAULT_VOXEL_SIZE_MM when the header carries no
        real spacing (e.g. a synthetic image left at unit spacing).
        """
        sp = list(ref_img.GetSpacing())
        in_plane = [float(v) for v in sp[:2] if float(v) > 0]
        if len(in_plane) != 2:
            print(
                f"[MAIN] No usable in-plane spacing in header ({sp}); "
                f"falling back to {DEFAULT_VOXEL_SIZE_MM} mm"
            )
            return DEFAULT_VOXEL_SIZE_MM
        return sum(in_plane) / 2.0

    # ------------------------------------------------------------------
    # Internal helper: atlas-constrained arterial mask (optional)
    # ------------------------------------------------------------------
    def _arterial_mask(self, ref_3d):
        """
        Narrow the AIF search region using the atlas ICA mask, if configured.

        Returns the vessel mask to use. When PERFUSION_ATLAS_PATH is unset, or
        registration fails, or the warped mask does not overlap the existing
        vessel mask, the unmodified vessel mask is returned so the default
        pipeline behaviour is unchanged.
        """
        if register.resolve_atlas_path() is None:
            return self.vessels

        # Registration needs a 3D fixed image; use the pre-bolus baseline frame.
        firstvolume = os.path.join(self.savepath, "firstvolume.nii.gz")
        self._write_like(self.raw_perf[..., 0], ref_3d, firstvolume)

        # Neutral cache id: the transform is cached inside this run's output
        # directory, so no patient identifier belongs in the filename.
        ica = register.get_arterial_mask(
            firstvolume, cache_dir=self.savepath, patient_id="atlas"
        )
        if ica is None:
            return self.vessels
        if ica.shape != self.vessels.shape:
            print(
                f"[MAIN] Atlas mask shape {ica.shape} != vessel mask "
                f"{self.vessels.shape}; keeping vessel mask"
            )
            return self.vessels

        combined = self.vessels & ica
        n = int(combined.sum())
        if n < MIN_ARTERIAL_VOXELS:
            print(
                f"[MAIN] Atlas ∩ vessel mask too small ({n} voxels); "
                f"keeping vessel mask"
            )
            return self.vessels

        print(
            f"[MAIN] Atlas-constrained arterial mask: {n} voxels "
            f"(from {int(self.vessels.sum())} vessel voxels)"
        )
        return combined

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self):
        self.main_cal()

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    def main_cal(self):
        print("\n[MAIN] Starting perfusion pipeline...")

        # Load reference images
        ref_3d = itk.imread(self.mask_path)
        ref_4d = itk.imread(self.image_path)

        # Step 1: CTC from raw signal
        print("[MAIN] Computing CTC / bolus timing...")
        CTC, _bolus = ctc.DSC_conc(self.raw_perf, self.mask.copy())

        # Strip CTA artifact by truncating the time series at the onset frame.
        # Hard-zeroing the CTA frames creates a discontinuity that wraps around
        # inside the circulant deconvolution matrix, producing spurious late
        # residue peaks (Tmax ~55s) for every voxel. Truncating eliminates
        # those zero frames entirely so the signal remains continuous.
        cta_start = ctc.detect_cta_artifact(CTC, self.mask)
        if cta_start < CTC.shape[-1]:
            CTC = CTC[..., :cta_start]
            print(
                f"[MAIN] CTA artifact: truncated to {cta_start} frames "
                f"(dropped {self.nT - cta_start})"
            )

        # Step 2: AIF from UNSMOOTHED CTC
        print("[MAIN] Computing AIF...")
        vessels = self._arterial_mask(ref_3d)
        AIF = aif.DSC_aif(self.mask.copy(), CTC, vessels, self.savepath)
        print("[MAIN] AIF complete.")

        # Step 3: Smooth CTC for deconvolution (AFTER AIF).
        # Suppresses Poisson noise enough for stable deconvolution at fine
        # in-plane resolution. Applied only to axes 1 and 2 (see SMOOTH_FWHM_MM).
        voxel_size_mm = self._in_plane_voxel_size_mm(ref_3d)
        sigma = (SMOOTH_FWHM_MM / voxel_size_mm) / 2.355
        print(
            f"[MAIN] Smoothing CTC: {SMOOTH_FWHM_MM}mm FWHM at "
            f"{voxel_size_mm:.4f}mm/px -> sigma={sigma:.2f}px (in-plane only)"
        )
        CTC_smooth = np.zeros_like(CTC)
        for t in range(CTC.shape[-1]):
            vol = CTC[..., t].copy().astype(float)
            vol[~self.mask] = 0.0
            smoothed = gaussian_filter(vol, sigma=[0, sigma, sigma])
            smoothed[~self.mask] = 0.0
            CTC_smooth[..., t] = smoothed
        CTC = CTC_smooth

        # Step 4: Compute maps from smoothed CTC
        print("[MAIN] Computing CBV...")
        CBV = cbv.cal(self.mask.copy(), CTC, AIF, self.config)

        print("[MAIN] Computing CBF / Tmax...")
        CBF, TMAX = cbf.cal(
            CTC,
            AIF,
            self.mask.copy(),
            svd_frac=float(self.config.svd_frac),
        )

        print("[MAIN] Computing MTT...")
        MTT = mtt.cal(CBV, CBF, self.mask.copy())

        # Step 5: Write outputs
        print("[MAIN] Writing NIfTI outputs to:", self.savepath)
        self._write_like(CBF, ref_3d, f"{self.savepath}/cbf.nii.gz")
        self._write_like(CBV, ref_3d, f"{self.savepath}/cbv.nii.gz")
        self._write_like(MTT, ref_3d, f"{self.savepath}/mtt.nii.gz")
        # TMAX from oSVD is already in seconds; no further TR multiplication needed
        self._write_like(
            TMAX.astype(np.float32), ref_3d, f"{self.savepath}/tmax.nii.gz"
        )
        # CTC may be shorter than ref_4d after CTA truncation — write without metadata
        try:
            self._write_like(CTC, ref_4d, f"{self.savepath}/ctc.nii.gz")
        except Exception:
            ctc_img = itk.image_view_from_array(CTC.astype(np.float32))
            itk.imwrite(ctc_img, f"{self.savepath}/ctc.nii.gz")

        print("[MAIN] Pipeline complete.\n")
