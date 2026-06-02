#!/usr/bin/env python3

import argparse
import logging
import os

import itk
import numpy as np
import numpy.typing as npt
import torch

from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt


class MainCalculator:
    """
    Main perfusion calculator.

    This class assumes:
    - raw_perf is a 4D numpy array (Z, Y, X, T)
    - mask and vessels are 3D numpy arrays (Z, Y, X)
    - Metadata is copied from reference images at write-time
    """

    def __init__(
        self,
        raw_perf: npt.ArrayLike,
        mask: npt.ArrayLike,
        vessels: npt.ArrayLike,
        origin: itk.Point,
        spacing: itk.Vector,
        direction: itk.Matrix,
        config: argparse.Namespace,
        save_path: str,
        device: torch.device,
        logger: logging.Logger = None,
    ):
        if raw_perf.ndim != 4:
            raise ValueError(
                "Expected raw_perf of shape (Z, Y, X, T)"
            )

        self.logger = (
            logger
            if logger
            else utils.get_logger("MainCalculator", level=logging.INFO)
        )

        self.logger.info(f"Sending raw perfusion image to '{device}'")

        # Store arrays
        self.raw_perf = raw_perf.astype(np.float32)
        self.mask = mask.astype(bool)
        self.vessels = vessels.astype(bool)

        # Shapes
        self.nS, self.nR, self.nC, self.nT = self.raw_perf.shape
        self.size = [self.nS, self.nR, self.nC, self.nT]

        # Config + paths
        self.config = config
        self.savepath = save_path
        self.device = device

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
    # Entry point
    # ------------------------------------------------------------------
    def run(self):
        self.main_cal()

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    @staticmethod
    def _motion_correct(raw_perf: np.ndarray) -> np.ndarray:
        """
        Rigid motion correction: register each 3D volume to volume 0.
        Uses SimpleITK Euler3D (6-DOF rigid) with mean-squares metric.
        Fast single-resolution registration — CTP motion is typically <3mm.
        """
        import SimpleITK as sitk

        nZ, nY, nX, nT = raw_perf.shape
        fixed_np   = raw_perf[..., 0].astype(np.float32)
        fixed_sitk = sitk.GetImageFromArray(fixed_np)
        corrected  = raw_perf.copy()

        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.15)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsGradientDescentLineSearch(
            learningRate=1.0, numberOfIterations=50,
            convergenceMinimumValue=1e-5, convergenceWindowSize=5,
        )
        reg.SetOptimizerScalesFromPhysicalShift()

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

        n_moved = 0
        for t in range(1, nT):
            moving_sitk = sitk.GetImageFromArray(raw_perf[..., t].astype(np.float32))
            init_tx = sitk.CenteredTransformInitializer(
                fixed_sitk, moving_sitk, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            reg.SetInitialTransform(init_tx, inPlace=False)
            tx = reg.Execute(fixed_sitk, moving_sitk)
            # Skip if transform is essentially identity (saves write time)
            params = np.array(tx.GetParameters())
            if np.abs(params).max() > 1e-4:
                resampler.SetTransform(tx)
                corrected[..., t] = sitk.GetArrayFromImage(
                    resampler.Execute(moving_sitk)
                )
                n_moved += 1

        print(f"[MOCO] {n_moved}/{nT-1} volumes had non-trivial motion")
        return corrected

    def main_cal(self):
        print("\n[MAIN] Starting perfusion pipeline...")

        # Load reference images
        ref_3d = itk.imread(self.mask_path)
        ref_4d = itk.imread(self.image_path)

        # Step 0: Motion correction (PyPeT-style, EXPERIMENT).
        # To revert: remove the next two lines.
        print("[MAIN] Motion correction...")
        self.raw_perf = self._motion_correct(self.raw_perf)

        # Step 1: CTC from raw signal
        print("[MAIN] Computing CTC / bolus timing...")
        CTC, BOLUS = ctc.DSC_conc(self.raw_perf, self.mask.copy())

        # Strip CTA artifact by truncating the time series at the onset frame.
        # Hard-zeroing the CTA frames creates a discontinuity that wraps around
        # inside the circulant deconvolution matrix, producing spurious late
        # residue peaks (Tmax ~55s) for every voxel. Truncating eliminates
        # those zero frames entirely so the signal remains continuous.
        cta_start = ctc.detect_cta_artifact(CTC, self.mask)
        if cta_start < CTC.shape[-1]:
            CTC   = CTC[..., :cta_start]
            BOLUS = np.minimum(BOLUS, cta_start - 1)
            print(f"[MAIN] CTA artifact: truncated to {cta_start} frames "
                  f"(dropped {self.nT - cta_start})")

        # Step 2: AIF from UNSMOOTHED CTC
        print("[MAIN] Computing AIF...")
        AIF = aif.DSC_aif(
            self.mask.copy(), CTC, BOLUS,
            self.size, self.config, self.device,
            self.vessels, self.savepath,
        )
        print("[MAIN] AIF complete.")

        # Step 3: Smooth CTC for deconvolution (AFTER AIF).
        # Apply only in-plane (axes 1 and 2); axis 0 is Z (4mm slices) where
        # cross-slice blur would destroy ischemic contrast.
        # 4mm FWHM at 0.391mm/px ≈ sigma=4.3px — necessary at this fine resolution
        # to suppress Poisson noise enough for stable oSVD deconvolution.
        from scipy.ndimage import gaussian_filter
        voxel_size_mm = 0.391
        fwhm_mm = 4.0
        sigma = (fwhm_mm / voxel_size_mm) / 2.355
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
        CBV = cbv.cal(self.mask.copy(), CTC, AIF, self.config, self.device)

        print("[MAIN] Computing CBF / Tmax...")
        CBF, TMAX = cbf.cal(
            CTC, AIF, self.mask.copy(),
            svd_frac=float(self.config.svd_frac),
            max_delay_s=float(self.config.max_delay_s),
        )

        print("[MAIN] Computing MTT...")
        MTT = mtt.cal(CBV, CBF, self.mask.copy())

        # Step 5: Write outputs
        print("[MAIN] Writing NIfTI outputs to:", self.savepath)
        self._write_like(CBF,  ref_3d, f"{self.savepath}/cbf.nii.gz")
        self._write_like(CBV,  ref_3d, f"{self.savepath}/cbv.nii.gz")
        self._write_like(MTT,  ref_3d, f"{self.savepath}/mtt.nii.gz")
        # TMAX from oSVD is already in seconds; no further TR multiplication needed
        self._write_like(TMAX.astype(np.float32), ref_3d, f"{self.savepath}/tmax.nii.gz")
        # CTC may be shorter than ref_4d after CTA truncation — write without metadata
        try:
            self._write_like(CTC, ref_4d, f"{self.savepath}/ctc.nii.gz")
        except Exception:
            ctc_img = itk.image_view_from_array(CTC.astype(np.float32))
            itk.imwrite(ctc_img, f"{self.savepath}/ctc.nii.gz")

        print("[MAIN] Pipeline complete.\n")
