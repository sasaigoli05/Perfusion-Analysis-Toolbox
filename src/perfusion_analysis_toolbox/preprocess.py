#!/usr/bin/env python3
# preprocess.py — DICOM dir -> NIfTI (dcm2niix) -> brain mask (skull-free, eroded) -> vessel mask (peak-pre with rim suppression + SITK hysteresis-like enhancement)
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import subprocess, shutil, time
import numpy as np
import SimpleITK as sitk

# =============================
# Tunables / Parameters
# =============================
THRESH_MODE       = "otsu"   # "otsu" or "fixed"
FIXED_THRESH_VAL  = 30       # used only if THRESH_MODE == "fixed" (0..255)
MIN_PRE_FRAMES    = 3
K_STD             = 2.0
MORPH_RADIUS      = (1, 1, 1)   # for small openings/closings
KEEP_LARGEST_CC   = True
ERODE_BRAIN_VOX   = 2          # pull brain mask away from skull (voxels)
RIM_DISTANCE_MM   = 2.0        # suppress detections within this distance to brain boundary
MIN_VOXELS_KEEP   = 50         # drop tiny vessel blobs (< this many voxels)
SOFT_TISSUE_RANGE = (-20, 80)  # HU-ish range used to strip bone on baseline volume
# =============================


# ---------- Helpers ----------
def _run_dcm2niix(dicom_dir: Path, out_dir: Path, tag: str) -> Path:
    exe = shutil.which("dcm2niix")
    if not exe:
        raise FileNotFoundError("dcm2niix not found on PATH.")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"image_{tag}"

    cmd = [exe, "-z", "y", "-o", str(out_dir), "-f", base, "-b", "n", "-v", "1", str(dicom_dir)]
    print(f"[INFO] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout: print(proc.stdout.strip())
    if proc.stderr: print(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"dcm2niix failed ({proc.returncode}).")

    time.sleep(0.2)
    nii_path = out_dir / f"{base}.nii.gz"
    if not nii_path.exists():
        raise FileNotFoundError("No NIfTI file found after conversion.")
    print(f"[OK] NIfTI written: {nii_path}")
    return nii_path


def _extract_3d_geometry(img: sitk.Image):
    origin = img.GetOrigin(); spacing = img.GetSpacing(); direction = img.GetDirection()
    origin3 = tuple(origin[:3]) if len(origin) >= 3 else (0.0, 0.0, 0.0)
    spacing3 = tuple(spacing[:3]) if len(spacing) >= 3 else (1.0, 1.0, 1.0)
    dir_vals = np.array(direction, dtype=float)
    if img.GetDimension() >= 4 and dir_vals.size >= 16:
        direction3 = tuple(dir_vals.reshape(4, 4)[:3, :3].ravel())
    elif dir_vals.size == 9:
        direction3 = tuple(dir_vals)
    else:
        direction3 = (1,0,0, 0,1,0, 0,0,1)
    return origin3, spacing3, direction3


def _find_pre_and_ttp(arr4d: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """
    arr4d: (T, Z, Y, X), mask: (Z, Y, X) boolean
    Returns (pre_idx, ttp_idx) using global mean curve inside mask.
    """
    T = arr4d.shape[0]
    mean_t = np.array([arr4d[t][mask].mean() if mask.any() else arr4d[t].mean() for t in range(T)])
    ttp = int(np.argmax(mean_t))
    if ttp < max(2, MIN_PRE_FRAMES):
        pre = max(0, ttp - 1)
    else:
        base_end = max(MIN_PRE_FRAMES, ttp // 3)
        base_mean, base_std = mean_t[:base_end].mean(), mean_t[:base_end].std() + 1e-6
        pre_candidates = np.where(mean_t[:ttp] <= base_mean + K_STD * base_std)[0]
        pre = int(pre_candidates[-1]) if pre_candidates.size else max(0, ttp - 1)
    print(f"[CTP] T={T} | TTP={ttp} | pre={pre}")
    print(f"[Curve] mean_t min/max: {mean_t.min():.1f}/{mean_t.max():.1f}")
    return pre, ttp


# ---------- Brain mask (skull-free + eroded) ----------
def _make_brain_mask(nii_path: Path, out_mask_path: Path) -> Path:
    img = sitk.ReadImage(str(nii_path), sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(img)  # (T,Z,Y,X) or (Z,Y,X)
    origin3, spacing3, direction3 = _extract_3d_geometry(img)

    # choose baseline 3D volume (median up to pre)
    if arr.ndim == 4:
        T = arr.shape[0]
        # crude auto-mask for curve estimation
        m0 = (arr != 0).sum(axis=0) >= (0.5 * T)
        pre, _ = _find_pre_and_ttp(arr, m0)
        base_arr = np.median(arr[:max(1, pre+1)], axis=0).astype(np.float32)
    else:
        base_arr = arr.astype(np.float32)

    base = sitk.GetImageFromArray(base_arr)
    base.SetOrigin(origin3); base.SetSpacing(spacing3); base.SetDirection(direction3)

    # skull strip via soft-tissue window (remove bone)
    low, high = SOFT_TISSUE_RANGE
    soft = sitk.BinaryThreshold(base, lowerThreshold=float(low), upperThreshold=float(high), insideValue=1, outsideValue=0)

    # morphology: close holes, keep largest CC
    soft = sitk.BinaryMorphologicalClosing(soft, MORPH_RADIUS if MORPH_RADIUS else (1,1,1))
    cc   = sitk.ConnectedComponent(soft)
    soft = sitk.RelabelComponent(cc, sortByObjectSize=True)
    soft = sitk.BinaryThreshold(soft, 1, 1, 1, 0)  # largest only

    # erode inward to avoid skull rim in (peak-pre)
    brain_mask = sitk.BinaryErode(soft, (int(ERODE_BRAIN_VOX),)*3)

    sitk.WriteImage(brain_mask, str(out_mask_path))
    print(f"[OK] Brain mask saved: {out_mask_path}")
    return out_mask_path


# ---------- Vessel mask (peak-pre + rim suppression + SITK hysteresis-like) ----------
def _make_vessel_mask(
    nii_path: Path,
    brain_mask_path: Path,
    out_vessel_path: Path,
    THRESH_MODE: str = THRESH_MODE,
    FIXED_THRESH: int = FIXED_THRESH_VAL
) -> Path:

    # load CTP and geometry
    img4d = sitk.ReadImage(str(nii_path), sitk.sitkFloat32)
    arr4d = sitk.GetArrayFromImage(img4d)
    if arr4d.ndim != 4:
        raise ValueError(f"Expected 4D image, got shape {arr4d.shape}")
    T, Z, Y, X = arr4d.shape

    origin3 = tuple(img4d.GetOrigin()[:3])
    spacing3 = tuple(img4d.GetSpacing()[:3])
    dir_flat = img4d.GetDirection()
    if len(dir_flat) >= 9:
        n = int(round(len(dir_flat) ** 0.5))
        direction3 = tuple(np.array(dir_flat).reshape(n, n)[:3, :3].ravel())
    else:
        direction3 = tuple(dir_flat)

    # read brain mask
    if not (brain_mask_path and Path(brain_mask_path).exists()):
        raise FileNotFoundError("Brain mask not found. Run _make_brain_mask first.")
    brain_img = sitk.ReadImage(str(brain_mask_path), sitk.sitkUInt8)
    brain_arr = sitk.GetArrayFromImage(brain_img).astype(bool)
    if brain_arr.shape != (Z, Y, X):
        raise ValueError(f"Mask shape {brain_arr.shape} != CTP spatial {(Z,Y,X)}")

    # timepoints
    pre, ttp = _find_pre_and_ttp(arr4d, brain_arr)

    # (peak - pre) within brain
    peak_arr = arr4d[ttp].astype(np.float32)
    pre_arr  = arr4d[pre].astype(np.float32)
    diff_arr = np.clip(peak_arr - pre_arr, 0, None)
    diff_arr *= brain_arr.astype(np.float32)

    # diagnostics
    if brain_arr.any():
        pr = np.percentile(diff_arr[brain_arr], [90,95,98,99,99.5])
        print(f"[Diff percentiles in brain] 90:{pr[0]:.1f} 95:{pr[1]:.1f} 98:{pr[2]:.1f} 99:{pr[3]:.1f} 99.5:{pr[4]:.1f}")

    # wrap as 3D image
    diff_img = sitk.GetImageFromArray(diff_arr)
    diff_img.SetOrigin(origin3); diff_img.SetSpacing(spacing3); diff_img.SetDirection(direction3)

    # distance-from-boundary rim suppression
    # (keep voxels at least RIM_DISTANCE_MM from boundary)
    dist_map     = sitk.SignedMaurerDistanceMap(brain_img, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
    far_from_rim = sitk.BinaryThreshold(dist_map, lowerThreshold=float(RIM_DISTANCE_MM), upperThreshold=1e9, insideValue=1, outsideValue=0)
    brain_core   = sitk.BinaryErode(brain_img, (1,1,1))  # slightly tighter interior

    # rescale 0..255 for thresholding
    rescaled = sitk.RescaleIntensity(diff_img, 0, 255)

    # compute threshold on central brain so skull rim cannot dominate
    central_mask = sitk.And(brain_img, far_from_rim)

    if THRESH_MODE.lower() == "otsu":
        masked  = sitk.Mask(rescaled, central_mask)
        mask_img = sitk.OtsuThreshold(masked, 0, 1, 200)
    else:
        mask_img = sitk.BinaryThreshold(rescaled, lowerThreshold=int(FIXED_THRESH), upperThreshold=255, insideValue=1, outsideValue=0)

    # --- SimpleITK-only hysteresis-like enhancement ---
    # Build "weak" and "strong" maps via percentiles; then keep weak components that touch strong.
    arr_resc = sitk.GetArrayFromImage(rescaled)
    central  = sitk.GetArrayFromImage(central_mask).astype(bool)
    if central.any():
        low_p  = float(np.percentile(arr_resc[central], 95))
        high_p = float(np.percentile(arr_resc[central], 99))

        weak  = sitk.BinaryThreshold(rescaled, lowerThreshold=low_p,  upperThreshold=255, insideValue=1, outsideValue=0)
        strong= sitk.BinaryThreshold(rescaled, lowerThreshold=high_p, upperThreshold=255, insideValue=1, outsideValue=0)

        # Connected components on weak; keep only labels that intersect strong
        cc_weak = sitk.ConnectedComponent(weak)
        rel     = sitk.RelabelComponent(cc_weak, sortByObjectSize=False)  # preserve labels
        stats   = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(rel)

        keep = sitk.Image(rel.GetSize(), sitk.sitkUInt8); keep.CopyInformation(rel)
        for L in stats.GetLabels():
            lbl = sitk.Equal(rel, int(L))
            # intersect with strong
            if sitk.GetArrayFromImage(sitk.And(lbl, strong)).any():
                keep = sitk.Or(keep, lbl)

        # union with main mask, still constrained to brain
        mask_img = sitk.Or(mask_img, sitk.And(keep, brain_img))

    # cleanup:
    # (a) suppress near-boundary rim
    mask_img = sitk.And(mask_img, far_from_rim)
    # (b) confine to brain core
    mask_img = sitk.And(mask_img, brain_core)
    # (c) small opening to break spurs
    mask_img = sitk.BinaryMorphologicalOpening(mask_img, MORPH_RADIUS if MORPH_RADIUS else (1,1,1))

    # (d) drop tiny components
    cc = sitk.ConnectedComponent(mask_img)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    final = sitk.Image(cc.GetSize(), sitk.sitkUInt8); final.CopyInformation(cc)
    for L in stats.GetLabels():
        if stats.GetNumberOfPixels(L) >= int(MIN_VOXELS_KEEP):
            final = sitk.Or(final, sitk.Equal(cc, int(L)))
    mask_img = final

    # write
    sitk.WriteImage(mask_img, str(out_vessel_path))
    nvox = int(sitk.GetArrayFromImage(mask_img).sum())
    print(f"[CTP] Wrote vessel mask: {out_vessel_path}  (voxels={nvox})")
    return out_vessel_path


# ---------- Orchestrator ----------
def preprocess_from_dicom(image_directory: str, save_path: str, overwrite: bool = True):
    dicom_dir = Path(image_directory).expanduser().resolve()
    out_dir   = Path(save_path).expanduser().resolve() / "preprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Convert DICOM folder to NIfTI
    nii_path = _run_dcm2niix(dicom_dir, out_dir, ts)

    # Step 2: Brain mask (skull-free + eroded)
    brain_mask_path = out_dir / f"brain_mask_{ts}.nii.gz"
    _make_brain_mask(nii_path, brain_mask_path)

    # Step 3: Vessel mask (peak-pre + rim suppression + hysteresis-like)
    vessel_mask_path = out_dir / f"vessel_mask_{ts}.nii.gz"
    _make_vessel_mask(nii_path, brain_mask_path, vessel_mask_path)

    # --- STOP HERE for inspection ---
    print("\n✅ Outputs generated. Inspect before pipeline continues:")
    print(f"  NIfTI:        {nii_path}")
    print(f"  Brain mask:   {brain_mask_path}")
    print(f"  Vessel mask:  {vessel_mask_path}")
    print("============================================\n")

    # raise SystemExit(0)  # <-- remove/comment this line when ready to continue

    return {
        "image_path": str(nii_path),
        "mask_path": str(brain_mask_path),
        "vessel_path": str(vessel_mask_path),
    }


# ---------- CLI (optional) ----------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="DICOM -> NIfTI -> brain mask -> vessel mask")
    p.add_argument("dicom_dir", help="Path to DICOM directory")
    p.add_argument("save_path", help="Output directory")
    args = p.parse_args()
    preprocess_from_dicom(args.dicom_dir, args.save_path)
