#!/usr/bin/env python3

import argparse
import logging

import itk
import numpy as np
import numpy.typing as npt
import torch
import os


from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt


class MainCalculator:
    """Main calculator."""

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
                "Expected input numpy array or torch tensor"
                "of size (slice, row, column, time)"
            )

        self.logger = (
            logger
            if logger
            else utils.get_logger("MainCalculator", level=logging.DEBUG)
        )
        self.logger.info(f"Sending the raw perfusion image to '{device}'")

        # Use float32 (fixes ITK int wrapping issues and is better numerically)
        self.raw_perf = raw_perf.astype(np.float32)

        self.spacing = spacing
        self.config = config
        self.itkinfo = [origin, spacing, direction, save_path]
        self.device = device

        self.nS = self.raw_perf.shape[0]
        self.nR = self.raw_perf.shape[1]
        self.nC = self.raw_perf.shape[2]
        self.nT = self.raw_perf.shape[3]
        self.size = [self.nS, self.nR, self.nC, self.nT]

        self.mask = mask
        self.savepath = save_path
        self.vessels = vessels

        # ---------- NEW: build CTA proxy + attach registration info ----------
        # Use temporal mean of perfusion as a pseudo-CTA volume (S,R,C)
        cta_np = self.raw_perf.mean(axis=3)  # (z, y, x)

        # Only set these if they weren't already set on config
        if not hasattr(self.config, "cta_np"):
            self.config.cta_np = cta_np

        if not hasattr(self.config, "fixed_image_path"):
            self.config.fixed_image_path = (
                "/home/sasaigoli/Documents/Data/reoriented_ccw.nii.gz"
            )
        # ----------------------------------------------------------------------


    def run(self):
        self.main_cal()

    def main_cal(self):
        print("\n[MAIN] Starting perfusion pipeline...")
        os.makedirs(self.savepath, exist_ok=True)

        mask = self.mask
        vessels = self.vessels

        # ============================
        # CUDA CHECK BEFORE PIPELINE
        # ============================
        import torch

        def check_cuda_or_warn():
            if torch.cuda.is_available():
                print("[SYSTEM] CUDA detected — GPU acceleration ENABLED.")
                return True
            else:
                print("\n[WARNING] CUDA GPU NOT AVAILABLE.")
                print("The pipeline will run on CPU, which may be significantly slower.")
                answer = input("Proceed with CPU? (y/n): ").strip().lower()
                if answer != "y":
                    raise RuntimeError("Execution cancelled — no CUDA device.")
                print("[SYSTEM] Running in CPU mode...")
                return False
       
        check_cuda_or_warn()

        print("[MAIN] Computing CTC/BOLUS...")
        CTC, BOLUS = ctc.DSC_conc(self.raw_perf, mask.copy())

        print("[MAIN] Computing AIF...")
        AIF = aif.DSC_aif(
            mask.copy(),
            CTC,
            BOLUS,
            self.size,
            self.config,
            self.device,
            vessels,
            self.savepath,
            cta_np=getattr(self.config, "cta_np", None),
            fixed_image_path=getattr(self.config, "fixed_image_path", None),
        )
        print("[MAIN] AIF done.")

        print("[MAIN] Computing CBV...")
        CBV = cbv.cal(mask.copy(), CTC, AIF, self.config, self.device)

        print("[MAIN] Computing CBF/TMAX...")
        CBF, TMAX = cbf.cal(CTC, AIF, mask.copy())

        print("[MAIN] Computing MTT...")
        MTT = mtt.cal(CBV, CBF, mask.copy())

        print("[MAIN] Writing NIFTI outputs to:", self.savepath)
        itk.imwrite(itk.image_view_from_array(MTT), f"{self.savepath}/mtt.nii.gz")
        itk.imwrite(itk.image_view_from_array(CBV), f"{self.savepath}/cbv.nii.gz")
        itk.imwrite(itk.image_view_from_array(CBF), f"{self.savepath}/cbf.nii.gz")
        itk.imwrite(itk.image_view_from_array(CTC), f"{self.savepath}/ctc.nii.gz")

        print("[MAIN] Pipeline complete.\n")
