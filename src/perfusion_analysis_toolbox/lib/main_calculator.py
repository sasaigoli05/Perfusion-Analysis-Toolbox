#!/usr/bin/env python3

import argparse
import logging
import os

import itk
import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import gaussian_filter

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
    def main_cal(self):
        print("\n[MAIN] Starting perfusion pipeline...")

        # Load reference images
        ref_3d = itk.imread(self.mask_path)
        ref_4d = itk.imread(self.image_path)

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

        # Step 2: 1mm spatial pre-smooth BEFORE AIF extraction (PyPeT approach).
        # PyPeT smooths the full volume first so that AIF and tissue CTC are
        # drawn from the same signal.  Previous approach smoothed after AIF.
        # EXPERIMENT (experiment/pypet-aif): swap these two blocks to revert.
        voxel_size_mm = 0.391
        pre_sigma = (1.0 / voxel_size_mm) / 2.355
        CTC_smooth = np.zeros_like(CTC)
        for t in range(CTC.shape[-1]):
            vol = CTC[..., t].copy().astype(float)
            vol[~self.mask] = 0.0
            smoothed = gaussian_filter(vol, sigma=[0, pre_sigma, pre_sigma])
            smoothed[~self.mask] = 0.0
            CTC_smooth[..., t] = smoothed
        CTC = CTC_smooth

        # Step 3: AIF from smoothed CTC
        print("[MAIN] Computing AIF...")
        AIF = aif.DSC_aif(
            self.mask.copy(), CTC, BOLUS,
            self.size, self.config, self.device,
            self.vessels, self.savepath,
        )
        print("[MAIN] AIF complete.")

        # No-signal mask from smoothed CTC peak.
        aif_peak = float(np.max(AIF["gv"]))
        ctc_peak_map = CTC.max(axis=-1)
        threshold = float(np.percentile(ctc_peak_map[self.mask], 15))
        no_signal = self.mask & (ctc_peak_map < threshold)
        print(f"[MAIN] No-signal mask: AIF_peak={aif_peak:.1f}  "
              f"p15_threshold={threshold:.1f}  "
              f"{no_signal.sum()} voxels zeroed "
              f"({no_signal[self.mask].mean()*100:.1f}% of brain)")

        # Step 5: Deconvolution on lightly-smoothed CTC.
        print("[MAIN] Computing CBV...")
        CBV = cbv.cal(self.mask.copy(), CTC, AIF, self.config, self.device)

        print("[MAIN] Computing CBF / Tmax...")
        CBF, TMAX = cbf.cal(
            CTC, AIF, self.mask.copy(),
            svd_frac=float(self.config.svd_frac),
            max_delay_s=float(self.config.max_delay_s),
        )

        # Zero no-signal (core) voxels.
        CBF[no_signal]  = 0.0
        TMAX[no_signal] = 0.0

        print("[MAIN] Computing MTT...")
        MTT = mtt.cal(CBV, CBF, self.mask.copy())

        # Step 6: Write outputs
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
