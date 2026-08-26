"""
Registration-based arterial mask computation.

Strategy:
  - Fixed: patient firstvolume.nii (3D baseline CTP frame)
  - Moving: atlas (reoriented_ccw.nii.gz)
  - unicarl-register is run via subprocess in the 'unicarl' conda env (CUDA required)
  - The resulting composite transform maps atlas → patient space
  - Atlas slices 1–4 contain the ICA / skull-base territory; we warp that binary
    slab mask into patient space and return it as the AIF candidate region
  - The warped mask is cached to avoid re-running registration on subsequent calls
"""

import os
import subprocess

import itk
import numpy as np

# The reference atlas is not redistributed with this repository. Point
# PERFUSION_ATLAS_PATH at your local copy, or pass atlas_path= explicitly.
ATLAS_ENV_VAR = "PERFUSION_ATLAS_PATH"


def resolve_atlas_path(atlas_path: str | None = None) -> str | None:
    """Return an explicit atlas path, else $PERFUSION_ATLAS_PATH, else None."""
    return atlas_path or os.environ.get(ATLAS_ENV_VAR) or None


# Atlas slices (0-indexed, ITK/pipeline Z order) that correspond to
# ICA / skull-base level.  Empirically confirmed: slices 1-4 in the
# atlas map to patient slices 2-6 after registration.
ATLAS_ICA_SLICES = slice(1, 5)  # slices 1, 2, 3, 4


def _run_unicarl(
    patient_firstvolume: str, transform_out: str, atlas_path: str
) -> bool:
    """
    Run unicarl-register (unicarl conda env, requires CUDA).
    Fixed = patient firstvolume, Moving = atlas  →  atlas warped to patient space.
    Returns True on success.
    """
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "unicarl",
        "unicarl-register",
        "--fixed",
        patient_firstvolume,
        "--moving",
        atlas_path,
        "--transform_out",
        transform_out,
        "--io_iterations",
        "50",
    ]
    print("[REGISTER] Running unicarl-register (CUDA)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[REGISTER] FAILED:", result.stderr[-500:])
        return False
    combined = (result.stdout + result.stderr).lower()
    if "cuda" not in combined:
        print("[REGISTER] Warning: CUDA not detected in unicarl output")
    print("[REGISTER] Registration complete.")
    return True


def _build_ica_mask_in_patient_space(
    transform_path: str,
    patient_firstvolume: str,
    atlas_path: str,
) -> np.ndarray:
    """
    Warp the atlas ICA slab mask to patient space using the composite
    ITK transform, then return a boolean (Z, Y, X) array.
    """
    atlas_itk = itk.imread(atlas_path, itk.F)
    atlas_np = itk.array_view_from_image(atlas_itk)  # (Z, Y, X)
    patient_itk = itk.imread(patient_firstvolume, itk.F)

    # Build binary slab mask in atlas space
    ica_np = np.zeros_like(atlas_np)
    ica_np[ATLAS_ICA_SLICES] = 1.0
    ica_img = itk.image_view_from_array(ica_np.astype(np.float32))
    ica_img.CopyInformation(atlas_itk)

    # Read composite transform (atlas → patient)
    T_list = itk.transformread(transform_path)
    T = T_list[0]

    # Resample atlas mask into patient space
    ImageType = itk.Image[itk.F, 3]
    resampler = itk.ResampleImageFilter[ImageType, ImageType].New()
    resampler.SetInput(ica_img)
    resampler.SetTransform(T)
    resampler.SetReferenceImage(patient_itk)
    resampler.SetUseReferenceImage(True)
    interp = itk.NearestNeighborInterpolateImageFunction[
        ImageType, itk.D
    ].New()
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(0)
    resampler.Update()

    warped = itk.array_view_from_image(
        resampler.GetOutput()
    ).copy()  # (Z, Y, X)
    mask = warped > 0.5
    n_slices = int((mask.any(axis=(1, 2))).sum())
    print(
        f"[REGISTER] ICA mask in patient space: {mask.sum()} voxels across {n_slices} slices"
    )
    return mask


def get_arterial_mask(
    patient_firstvolume: str,
    cache_dir: str,
    patient_id: str,
    atlas_path: str | None = None,
) -> np.ndarray | None:
    """
    Return a (Z, Y, X) boolean array of ICA-level voxels in patient space.

    Caches the transform hdf5 to <cache_dir>/<patient_id>_trans.hdf5 so
    re-running the batch skips registration.  Returns None on failure.

    Parameters
    ----------
    patient_firstvolume : str
        Path to firstvolume.nii (or any 3D baseline CTP frame).
    cache_dir : str
        Directory to cache the registration transform.
    patient_id : str
        Short label used for the cache filename.
    atlas_path : str, optional
        Reference atlas. Defaults to $PERFUSION_ATLAS_PATH.
    """
    atlas_path = resolve_atlas_path(atlas_path)
    if atlas_path is None:
        print(
            f"[REGISTER] No atlas configured (set ${ATLAS_ENV_VAR}), skipping."
        )
        return None

    if not os.path.exists(atlas_path):
        print(f"[REGISTER] Atlas not found at {atlas_path}, skipping.")
        return None

    if not os.path.exists(patient_firstvolume):
        print(
            f"[REGISTER] firstvolume not found: {patient_firstvolume}, skipping."
        )
        return None

    os.makedirs(cache_dir, exist_ok=True)
    transform_path = os.path.join(cache_dir, f"{patient_id}_trans.hdf5")

    if not os.path.exists(transform_path):
        ok = _run_unicarl(patient_firstvolume, transform_path, atlas_path)
        if not ok:
            return None

    try:
        return _build_ica_mask_in_patient_space(
            transform_path, patient_firstvolume, atlas_path
        )
    except Exception as e:
        print(f"[REGISTER] Mask build failed: {e}")
        return None
