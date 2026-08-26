#!/usr/bin/env python3
from typing import Tuple

import numpy as np
import numpy.typing as npt
import itk
from scipy.ndimage import zoom


def read_signal(
    image_path: str,
    mask_path: str,
    vessel_path: str,
) -> Tuple[
    npt.NDArray, npt.NDArray, npt.NDArray, itk.Point, itk.Vector, itk.Matrix
]:
    """
    Load the 4D perfusion image + 3D mask + 3D vessel mask, and make sure
    mask and vessels are in the same (z, y, x) grid as the perfusion.

    Returns
    -------
    RawPerfImg : (Z, Y, X, T) float32
    mask       : (Z, Y, X) bool
    vessels    : (Z, Y, X) bool
    origin     : ITK origin of perfusion image
    spacing    : ITK spacing of perfusion image
    direction  : ITK direction of perfusion image
    """

    print(image_path, mask_path, vessel_path)
    print(image_path)

    # -------------------------------------------------
    # 1) Read the 4D perfusion image
    # -------------------------------------------------
    perf_img = itk.imread(image_path, itk.F)  # 4D float
    perf_arr = itk.array_view_from_image(perf_img)  # usually (T, Z, Y, X)

    if perf_arr.ndim != 4:
        raise ValueError(
            f"Expected 4D perfusion image, got shape {perf_arr.shape}"
        )

    # Transpose to (Z, Y, X, T) for MainCalculator
    # ITK (x,y,z,t) -> numpy (t,z,y,x) -> we want (z,y,x,t)
    RawPerfImg = np.transpose(perf_arr, (1, 2, 3, 0)).astype(np.float32)
    nS, nR, nC, nT = RawPerfImg.shape
    print(f"Raw perfusion shape (Z, Y, X, T): {RawPerfImg.shape}")

    origin = perf_img.GetOrigin()
    spacing = perf_img.GetSpacing()
    direction = perf_img.GetDirection()

    # -------------------------------------------------
    # 2) Read mask and vessel images
    # -------------------------------------------------
    mask_img = itk.imread(mask_path)
    vessel_img = itk.imread(vessel_path)

    mask_arr = itk.array_view_from_image(mask_img)  # (Z, Y, X) or (Y, X) etc.
    vessel_arr = itk.array_view_from_image(vessel_img)

    # Force them to be at least 3D
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    if vessel_arr.ndim == 2:
        vessel_arr = vessel_arr[None, ...]

    print(f"Original mask shape:   {mask_arr.shape}")
    print(f"Original vessel shape: {vessel_arr.shape}")

    target_shape = (nS, nR, nC)

    # -------------------------------------------------
    # 3) Resample mask & vessels (shape-based) if needed
    #    (nearest-neighbor so labels remain clean)
    # -------------------------------------------------
    def _resample_to_target(arr: np.ndarray, target: tuple) -> np.ndarray:
        if arr.shape == target:
            return arr

        zf = target[0] / arr.shape[0]
        yf = target[1] / arr.shape[1]
        xf = target[2] / arr.shape[2]
        print(
            f"Resampling from {arr.shape} -> {target} with factors (z,y,x)=({zf:.3f},{yf:.3f},{xf:.3f})"
        )

        # order=0 = nearest neighbor (good for masks)
        arr_res = zoom(arr.astype(float), (zf, yf, xf), order=0)
        return arr_res

    mask_resampled = _resample_to_target(mask_arr, target_shape)
    vessel_resampled = _resample_to_target(vessel_arr, target_shape)

    # Binarize / cast to bool
    mask_bool = mask_resampled > 0.5
    vessel_bool = vessel_resampled > 0.5

    print(f"Final mask shape:      {mask_bool.shape}")
    print(f"Final vessel shape:    {vessel_bool.shape}")

    # -------------------------------------------------
    # 4) (Optional) NaN handling / debug
    # -------------------------------------------------
    nan_mask = np.isnan(RawPerfImg)
    print("nan_mask shape:", nan_mask.shape, "nan_mask dtype:", nan_mask.dtype)
    # You can choose to replace NaNs with 0:
    RawPerfImg = np.nan_to_num(RawPerfImg, nan=0.0)

    return RawPerfImg, mask_bool, vessel_bool, origin, spacing, direction
