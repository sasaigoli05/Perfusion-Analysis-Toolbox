#!/usr/bin/env python3
"""
preprocess.py — DICOM → NIfTI (dcm2niix) → brain mask (skull-free, eroded) → vessel mask
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess, shutil, time
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
import itk

# =============================
# Tunables
# =============================
MIN_PRE_FRAMES = 3
K_STD = 2.0
MORPH_RADIUS = (1, 1, 1)
KEEP_LARGEST_CC = True
ERODE_BRAIN_VOX = 2
SOFT_TISSUE_RANGE = (-20, 80)
# =============================


def _run_dcm2niix(dicom_dir: Path, out_dir: Path, tag: str) -> Path:
    exe = shutil.which("dcm2niix")
    if not exe:
        raise FileNotFoundError("dcm2niix not found on PATH.")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"image_{tag}"

    cmd = [
        exe,
        "-z",
        "y",
        "-o",
        str(out_dir),
        "-f",
        base,
        "-b",
        "n",
        "-v",
        "1",
        str(dicom_dir),
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"dcm2niix failed ({proc.returncode}).")

    time.sleep(0.2)
    nii_path = out_dir / f"{base}.nii.gz"
    if not nii_path.exists():
        raise FileNotFoundError("No NIfTI file found after conversion.")
    print(f"[OK] NIfTI written: {nii_path}")
    return nii_path


def _extract_3d_geometry(img):
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    direction = img.GetDirection()
    origin3 = tuple(origin[:3]) if len(origin) >= 3 else (0.0, 0.0, 0.0)
    spacing3 = tuple(spacing[:3]) if len(spacing) >= 3 else (1.0, 1.0, 1.0)
    dir_vals = np.array(direction, dtype=float)
    if img.GetDimension() >= 4 and dir_vals.size >= 16:
        direction3 = tuple(dir_vals.reshape(4, 4)[:3, :3].ravel())
    elif dir_vals.size == 9:
        direction3 = tuple(dir_vals)
    else:
        direction3 = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    return origin3, spacing3, direction3


def _find_pre_and_ttp(arr4d: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """
    arr4d: (T, Z, Y, X), mask: (Z, Y, X) boolean
    Returns (pre_idx, ttp_idx) using global mean curve inside mask.

    TTP search is restricted to the first 50% of frames so that a late
    CTA second-injection (combined CTP+CTA protocols) cannot be mistaken
    for the first-pass bolus peak.
    """
    T = arr4d.shape[0]
    mean_t = np.array(
        [
            arr4d[t][mask].mean() if mask.any() else arr4d[t].mean()
            for t in range(T)
        ]
    )

    first_pass_end = max(T // 2, min(5, T - 1))
    ttp = int(np.argmax(mean_t[:first_pass_end]))

    if ttp < max(2, MIN_PRE_FRAMES):
        pre = max(0, ttp - 1)
    else:
        base_end = max(MIN_PRE_FRAMES, ttp // 3)
        base_mean, base_std = (
            mean_t[:base_end].mean(),
            mean_t[:base_end].std() + 1e-6,
        )
        pre_candidates = np.where(
            mean_t[:ttp] <= base_mean + K_STD * base_std
        )[0]
        pre = (
            int(pre_candidates[-1]) if pre_candidates.size else max(0, ttp - 1)
        )
    print(
        f"[CTP] T={T} | first-pass TTP={ttp} (searched frames 0–{first_pass_end-1}) | pre={pre}"
    )
    return pre, ttp


# ---------- Brain mask (skull-free + eroded) ----------
def _make_brain_mask(nii_path: Path, out_mask_path: Path) -> Path:
    if sitk is None:
        raise ImportError(
            "SimpleITK is required for brain masking. pip install SimpleITK"
        )
    img = sitk.ReadImage(str(nii_path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)  # (T,Z,Y,X) or (Z,Y,X)
    origin3, spacing3, direction3 = _extract_3d_geometry(img)

    if arr.ndim == 4:
        T = arr.shape[0]
        m0 = (arr != 0).sum(axis=0) >= (0.5 * T)
        pre, _ = _find_pre_and_ttp(arr, m0)
        base_arr = np.median(arr[: max(1, pre + 1)], axis=0).astype(np.float32)
    else:
        base_arr = arr.astype(np.float32)

    base = sitk.GetImageFromArray(base_arr)
    base.SetOrigin(origin3)
    base.SetSpacing(spacing3)
    base.SetDirection(direction3)

    low, high = SOFT_TISSUE_RANGE
    soft = sitk.BinaryThreshold(
        base,
        lowerThreshold=float(low),
        upperThreshold=float(high),
        insideValue=1,
        outsideValue=0,
    )
    soft = sitk.BinaryMorphologicalClosing(
        soft, MORPH_RADIUS if MORPH_RADIUS else (1, 1, 1)
    )
    cc = sitk.ConnectedComponent(soft)
    soft = sitk.RelabelComponent(cc, sortByObjectSize=True)
    soft = sitk.BinaryThreshold(soft, 1, 1, 1, 0)
    brain_mask = sitk.BinaryErode(soft, (int(ERODE_BRAIN_VOX),) * 3)

    sitk.WriteImage(brain_mask, str(out_mask_path))
    print(f"[OK] Brain mask saved: {out_mask_path}")
    return out_mask_path


# ---------- Vessel mask (first-pass peak + bolus-width scoring) ----------
def _vessel_mask_numpy(arr4d, brain_arr, pre, first_pass_end):
    from scipy.ndimage import label as cc_label

    Z, Y, X = brain_arr.shape
    baseline = arr4d[: max(1, pre + 1)].mean(axis=0)
    fp_ctc = np.clip(arr4d[:first_pass_end] - baseline, 0, None)
    fp_peak = fp_ctc.max(axis=0)
    fp_width = np.maximum(
        np.sum(fp_ctc > (fp_peak / 2.0)[np.newaxis], axis=0).astype(float), 1.0
    )

    zz, rr, cc = np.where(brain_arr)
    peaks = fp_peak[zz, rr, cc]
    widths = fp_width[zz, rr, cc]
    pos = peaks[peaks > 0]
    if len(pos) < 10:
        return np.zeros((Z, Y, X), dtype=np.uint8)

    def _norm01(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-9 else np.zeros_like(x)

    score = _norm01(peaks) - 0.5 * _norm01(widths)
    selected = (score >= np.percentile(score, 98)) & (
        peaks >= 2.0 * float(np.median(pos))
    )

    score_map = np.zeros((Z, Y, X), dtype=np.float32)
    score_map[zz[selected], rr[selected], cc[selected]] = score[selected]
    vessel_bin = (score_map > 0).astype(np.uint8)

    labeled, _ = cc_label(vessel_bin)
    counts = np.bincount(labeled.ravel())
    keep = np.where((counts >= 2) & (counts <= 3000))[0]
    return np.isin(labeled, keep).astype(np.uint8)


def _make_vessel_mask(
    nii_path: Path,
    brain_mask_path: Path,
    out_vessel_path: Path,
) -> Path:
    """Score voxels by first-pass peak / bolus-width ratio to identify arterial candidates."""
    img4d = itk.imread(str(nii_path), itk.F)
    arr4d = itk.GetArrayFromImage(img4d).astype(np.float32)  # (T, Z, Y, X)
    del img4d
    if arr4d.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI, got shape {arr4d.shape}")
    T, Z, Y, X = arr4d.shape

    brain_img = itk.imread(str(brain_mask_path))
    brain_arr = itk.GetArrayFromImage(brain_img).astype(bool)  # (Z, Y, X)
    if brain_arr.shape != (Z, Y, X):
        raise ValueError(
            f"Mask shape {brain_arr.shape} != CTP spatial {(Z, Y, X)}"
        )

    pre, _ = _find_pre_and_ttp(arr4d, brain_arr)
    first_pass_end = max(T // 2, min(5, T - 1))
    vessel_arr = _vessel_mask_numpy(arr4d, brain_arr, pre, first_pass_end)

    ref_mask = itk.imread(str(brain_mask_path))
    result_itk = itk.GetImageFromArray(vessel_arr)
    result_itk.CopyInformation(ref_mask)
    itk.imwrite(result_itk, str(out_vessel_path))
    print(
        f"[CTP] Wrote vessel mask: {out_vessel_path}  (voxels={int(vessel_arr.sum())})"
    )
    return out_vessel_path


# ---------- Orchestrator ----------
def preprocess_from_dicom(image_directory: str, save_path: str) -> dict:
    dicom_dir = Path(image_directory).expanduser().resolve()
    out_dir = Path(save_path).expanduser().resolve() / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    nii_path = _run_dcm2niix(dicom_dir, out_dir, ts)

    brain_mask_path = out_dir / f"brain_mask_{ts}.nii.gz"
    _make_brain_mask(nii_path, brain_mask_path)

    vessel_mask_path = out_dir / f"vessel_mask_{ts}.nii.gz"
    _make_vessel_mask(nii_path, brain_mask_path, vessel_mask_path)

    print("\nOutputs generated:")
    print(f"  NIfTI:        {nii_path}")
    print(f"  Brain mask:   {brain_mask_path}")
    print(f"  Vessel mask:  {vessel_mask_path}")
    print("============================================\n")

    return {
        "image_path": str(nii_path),
        "mask_path": str(brain_mask_path),
        "vessel_path": str(vessel_mask_path),
    }


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="DICOM -> NIfTI -> brain mask -> vessel mask"
    )
    p.add_argument("dicom_dir", help="Path to DICOM directory")
    p.add_argument("save_path", help="Output directory")
    args = p.parse_args()
    preprocess_from_dicom(args.dicom_dir, args.save_path)
