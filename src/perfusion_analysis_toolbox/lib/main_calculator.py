# #!/usr/bin/env python3

# import argparse
# import logging

# import itk
# import numpy.typing as npt
# import torch

# from . import utils
# from .ParamsCalculator import ctc as ctc
# from .ParamsCalculator import cbv as cbv
# from .ParamsCalculator import cbf as cbf
# from .ParamsCalculator import aif as aif
# from .ParamsCalculator import mtt as mtt


# class MainCalculator:
#     """Main calculator."""

#     def __init__(
#         self,
#         raw_perf: npt.ArrayLike,
#         mask: npt.ArrayLike,
#         vessels: npt.ArrayLike,
#         origin: itk.Point,
#         spacing: itk.Vector,
#         direction: itk.Matrix,
#         config: argparse.Namespace,
#         save_path: str,
#         device: torch.device,
#         logger: logging.Logger = None,
#     ):
#         if raw_perf.ndim != 4:
#             raise ValueError(
#                 "Expected input numpy array or torch tensor"
#                 "of size (slice, row, column, time)"
#             )

#         self.logger = (
#             logger
#             if logger
#             else utils.get_logger("MainCalculator", level=logging.DEBUG)
#         )
#         self.logger.info(f"Sending the raw perfusion image to '{device}'")

#         self.raw_perf = raw_perf.astype(int)
#         self.spacing = spacing
#         self.config = config
#         self.itkinfo = [origin, spacing, direction, save_path]
#         self.device = device
#         self.nS = self.raw_perf.shape[0]
#         self.nR = self.raw_perf.shape[1]
#         self.nC = self.raw_perf.shape[2]
#         self.nT = self.raw_perf.shape[3]
#         self.size = [self.nS, self.nR, self.nC, self.nT]
#         self.mask = mask
#         self.savepath = save_path
#         self.vessels = vessels

#     def run(self):
#         self.main_cal()

#     def main_cal(self):
#         mask = self.mask
#         temp_mask = mask.copy()
#         temp_mask1 = mask.copy()
#         temp_mask2 = mask.copy()
#         temp_mask3 = mask.copy()
#         temp_mask4 = mask.copy()
#         vessels = self.vessels
#         CTC, BOLUS = ctc.DSC_conc(self.raw_perf, temp_mask)
#         AIF = aif.DSC_aif(self,
#             temp_mask1,
#             CTC,
#             BOLUS,
#             self.size,
#             self.config,
#             self.device,
#             vessels,
#         )
#         CBV = cbv.cal(temp_mask2, CTC, AIF, self.config, self.device)
#         CBF, TMAX = cbf.cal(CTC, AIF, temp_mask3)
#         MTT = mtt.cal(CBV, CBF, temp_mask4)

#         mtt_img = itk.image_view_from_array(MTT)
#         itk.imwrite(mtt_img, f"{self.savepath}/mtt.nii.gz")
#         cbv_img = itk.image_view_from_array(CBV)
#         itk.imwrite(cbv_img, f"{self.savepath}/cbv.nii.gz")
#         cbf_img = itk.image_view_from_array(CBF)
#         itk.imwrite(cbf_img, f"{self.savepath}/cbf.nii.gz")
#         ctc_img = itk.image_view_from_array(CTC)
#         itk.imwrite(ctc_img, f"{self.savepath}/ctc.nii.gz")

#!/usr/bin/env python3

import argparse
import logging
import numpy as np

import itk
import numpy.typing as npt
import torch

from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt


class MainCalculator:
    """Main calculator with safe dtype + NaN handling."""

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
                "Expected input numpy array or torch tensor of size (slice, row, column, time)"
            )

        self.logger = logger if logger else utils.get_logger("MainCalculator", level=logging.DEBUG)
        self.logger.info(f"Sending the raw perfusion image to '{device}'")

        # --- Safe mode knob (default ON) ---
        self.safe_mode = bool(getattr(config, "safe_mode", True))

        # --- Store geometry/IO info ---
        self.origin = origin
        self.spacing = spacing
        self.direction = direction
        self.config = config
        self.device = device
        self.savepath = save_path

        # --- Ingest & sanitize ---
        self.raw_perf = self._clean_vol(raw_perf)        # (S,R,C,T) float32, no NaNs
        self.mask = self._prepare_mask(mask, self.raw_perf.shape)          # 3D bool
        self.vessels = self._prepare_mask(vessels, self.raw_perf.shape)    # 3D bool

        # --- Sizes ---
        self.nS, self.nR, self.nC, self.nT = self.raw_perf.shape
        self.size = [self.nS, self.nR, self.nC, self.nT]

    # -------------------- safety helpers --------------------

    def _clean_vol(self, x: npt.ArrayLike) -> np.ndarray:
        """cast→float32, scrub NaN/Inf, ensure C-contiguous."""
        a = np.asarray(x, dtype=np.float32)
        if self.safe_mode:
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(a)

    def _prepare_mask(self, m: npt.ArrayLike, vol_shape) -> np.ndarray:
        """ensure boolean 3D mask matching (S,R,C); collapse 4D masks over time axis if needed."""
        a = np.asarray(m)
        if a.ndim == 4 and a.shape[3] == vol_shape[3]:
            # collapse time dimension → any voxel marked at any time counts
            a = np.any(a, axis=3)
            self.logger.debug("Collapsed 4D mask to 3D via time-any()")
        if a.ndim != 3:
            raise ValueError(f"Mask must be 3D (S,R,C); got shape {a.shape}")
        if a.shape != vol_shape[:3]:
            raise ValueError(f"Mask shape {a.shape} does not match volume spatial shape {vol_shape[:3]}")
        return (a > 0)

    def _clean_arr(self, arr: npt.ArrayLike) -> np.ndarray:
        """final scrub before writing: float32, no NaN/Inf, contiguous."""
        a = np.asarray(arr, dtype=np.float32)
        if self.safe_mode:
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(a)

    def _to_itk_image(self, arr: npt.ArrayLike):
        """Safe NumPy → ITK (keeps metadata handling minimal to avoid dimension mismatches)."""
        a = self._clean_arr(arr)
        img = itk.image_view_from_array(a)
        # If you need geometry, add guarded setters here for matching dims.
        return img

    # -------------------- public API --------------------

    def run(self):
        self.main_cal()

    def main_cal(self):
        # copy masks for each stage to preserve originals
        temp_mask  = self.mask.copy()
        temp_mask1 = self.mask.copy()
        temp_mask2 = self.mask.copy()
        temp_mask3 = self.mask.copy()
        temp_mask4 = self.mask.copy()
        vessels    = self.vessels

        # --- Compute steps ---
        CTC, BOLUS = ctc.DSC_conc(self.raw_perf, temp_mask)

        AIF = aif.DSC_aif(
            self,
            temp_mask1,
            CTC,
            BOLUS,
            self.size,
            self.config,
            self.device,
            vessels,
        )

        CBV = cbv.cal(temp_mask2, CTC, AIF, self.config, self.device)
        CBF, TMAX = cbf.cal(CTC, AIF, temp_mask3)
        MTT = mtt.cal(CBV, CBF, temp_mask4)

        # --- Clean outputs (removes NaNs from any upstream division), cast to float32 ---
        MTT = self._clean_arr(MTT)
        CBV = self._clean_arr(CBV)
        CBF = self._clean_arr(CBF)
        CTC = self._clean_arr(CTC)

        # --- Write outputs via safe converter (avoids itk.SL, ensures wrapped type) ---
        itk.imwrite(self._to_itk_image(MTT), f"{self.savepath}/mtt.nii.gz")
        itk.imwrite(self._to_itk_image(CBV), f"{self.savepath}/cbv.nii.gz")
        itk.imwrite(self._to_itk_image(CBF), f"{self.savepath}/cbf.nii.gz")
        itk.imwrite(self._to_itk_image(CTC), f"{self.savepath}/ctc.nii.gz")
