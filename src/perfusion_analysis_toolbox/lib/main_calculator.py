#!/usr/bin/env python3

import argparse
import glob
import json
import logging
import os

import itk
import numpy as np
from dataclasses import dataclass
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from . import moco
from .. import normalise
from . import utils
from .ParamsCalculator import ctc as ctc
from .ParamsCalculator import cbv as cbv
from .ParamsCalculator import cbf as cbf
from .ParamsCalculator import aif as aif
from .ParamsCalculator import mtt as mtt


# In-plane smoothing applied to the CTC before deconvolution. Axis 0 (Z) is
# left untouched: slices are thick (~4mm) and cross-slice blur would wash out
# ischemic contrast.
SMOOTH_FWHM_MM = 4.0

# Fallback in-plane voxel size (mm) used only when the reference image reports
# no usable spacing.
DEFAULT_VOXEL_SIZE_MM = 0.391



@dataclass
class PerfusionMaps:
    """
    What a run produces, in memory.

    `run()` used to write NIfTIs and return nothing, so any caller that wanted
    the maps had to read them back off disk -- and had to have a disk. The
    files are still written when a save path is given; this is the same data
    returned directly.
    """

    cbf: np.ndarray
    cbv: np.ndarray
    mtt: np.ndarray
    tmax: np.ndarray
    ctc: np.ndarray
    aif: dict
    extras: dict

    def as_dict(self) -> dict:
        d = {"cbf": self.cbf, "cbv": self.cbv, "mtt": self.mtt,
             "tmax": self.tmax}
        d.update(self.extras)
        return d


class MainCalculator:
    """
    Main perfusion calculator.

    Assumes:
    - raw_perf is a 4D numpy array (Z, Y, X, T)
    - mask and vessels are 3D numpy arrays (Z, Y, X)

    Image geometry (spacing, origin, direction) is read from the reference
    images at config.image_path / config.mask_path rather than passed in, so
    the outputs always inherit the input headers. A caller working purely in
    memory leaves both None and states config.tr, optionally with
    config.voxel_spacing; writing NIfTIs then has no headers to copy and is
    refused.
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

        # A save path is optional: callers working in memory get the maps
        # back from run() and never touch the filesystem.
        if self.savepath:
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
    # Internal helper: CTC conditioning before deconvolution
    # ------------------------------------------------------------------
    def _denoise_ctc(self, CTC, method: str, voxel_size_mm: float):
        """Condition the CTC for deconvolution. Returns a new array."""
        import SimpleITK as sitk

        fwhm = float(getattr(self.config, "smooth_fwhm", SMOOTH_FWHM_MM))
        t_sigma = float(getattr(self.config, "temporal_sigma", 1.0))
        iters = int(getattr(self.config, "curvature_iterations", 5))

        parts = [p for p in method.split("+") if p]
        unknown = set(parts) - {"gaussian", "temporal", "curvature", "tips", "none"}
        if unknown:
            raise ValueError(f"unknown denoise method(s): {sorted(unknown)}")

        out = CTC

        if "gaussian" in parts and fwhm > 0:
            sigma = (fwhm / voxel_size_mm) / 2.355
            print(
                f"[MAIN] Denoise: gaussian {fwhm}mm FWHM at "
                f"{voxel_size_mm:.4f}mm/px -> sigma={sigma:.2f}px (in-plane)"
            )
            sm = np.zeros_like(out)
            for t in range(out.shape[-1]):
                vol = out[..., t].copy().astype(float)
                vol[~self.mask] = 0.0
                v = gaussian_filter(vol, sigma=[0, sigma, sigma])
                v[~self.mask] = 0.0
                sm[..., t] = v
            out = sm

        if "curvature" in parts:
            print(
                f"[MAIN] Denoise: curvature flow, {iters} iterations, "
                f"in-plane per slice (edge-preserving)"
            )
            sm = np.zeros_like(out)
            nZ = out.shape[0]
            for t in range(out.shape[-1]):
                vol = out[..., t].astype(np.float32)
                for z in range(nZ):
                    sl = sitk.GetImageFromArray(vol[z])
                    # 2D CurvatureFlow is stable below timeStep 0.25; 0.125 is
                    # the conventional choice and leaves margin.
                    sl = sitk.CurvatureFlow(sl, 0.125, iters)
                    sm[z, ..., t] = sitk.GetArrayFromImage(sl)
                sm[..., t][~self.mask] = 0.0
            out = sm

        if "tips" in parts:
            radius = int(getattr(self.config, "tips_radius", 3))
            h = float(getattr(self.config, "tips_h", 10.0))
            sigma_s = max(radius / 2.0, 1e-6)
            print(
                f"[MAIN] Denoise: TIPS bilateral, radius={radius}, h={h} HU "
                f"(in-plane, profile-similarity weighted)"
            )
            out = self._tips_filter(out, radius, h, sigma_s)

        if "temporal" in parts and t_sigma > 0:
            print(f"[MAIN] Denoise: temporal gaussian sigma={t_sigma:.2f} frames")
            sm = gaussian_filter(
                out.astype(float), sigma=[0, 0, 0, t_sigma], mode="nearest"
            )
            sm[~self.mask] = 0.0
            out = sm.astype(CTC.dtype)

        if not parts or parts == ["none"]:
            print("[MAIN] Denoise: disabled")

        return out

    def _tips_filter(self, ctc, radius: int, h: float, sigma_s: float):
        """
        Time-Intensity Profile Similarity bilateral filter (Mendrik 2011).

        A Gaussian blur averages a voxel with its neighbours regardless of
        whether they are the same tissue, which is why it suppresses noise and
        lesion contrast in equal measure. A conventional bilateral filter
        weights by single-frame intensity, which in a perfusion series is
        mostly noise at any one time point.

        TIPS weights instead by how similar two voxels' entire concentration-
        time curves are. Two voxels in the same tissue share a curve shape even
        when any single frame is noisy, so they average together; a voxel in
        core and its neighbour in penumbra have different curves and stay
        separate. That is exactly the boundary the pipeline currently blurs
        away.

        Applied in-plane only, matching the convention used elsewhere: slices
        are thick and cross-slice mixing washes out ischaemic contrast.
        """
        out = np.zeros_like(ctc)
        offsets = [
            (dy, dx)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        two_h2 = 2.0 * h * h
        two_s2 = 2.0 * sigma_s * sigma_s

        for z in range(ctc.shape[0]):
            plane = ctc[z].astype(np.float32)          # (Y, X, T)
            num = np.zeros_like(plane)
            den = np.zeros(plane.shape[:2], np.float32)

            for dy, dx in offsets:
                shifted = np.roll(plane, (dy, dx), axis=(0, 1))
                # Mean squared difference between the two time courses, in HU^2.
                d2 = np.mean((plane - shifted) ** 2, axis=2)
                w = np.exp(-(dy * dy + dx * dx) / two_s2) * np.exp(-d2 / two_h2)
                num += w[..., None] * shifted
                den += w

            np.divide(num, den[..., None], out=num, where=den[..., None] > 0)
            out[z] = num

        out[~self.mask] = 0.0
        return out

    def _bayes_kwargs(self) -> dict:
        """Tuning passed through to the parametric estimator, if selected."""
        out = {}
        for attr, key in (
            ("bayes_iterations", "iterations"),
            ("bayes_spatial", "spatial_weight"),
            ("bayes_prior_weight", "prior_weight"),
        ):
            val = getattr(self.config, attr, None)
            if val is not None:
                out[key] = val
        return out

    # ------------------------------------------------------------------
    # Internal helper: frame interval
    # ------------------------------------------------------------------
    def _resolve_tr(self, ref_4d) -> float:
        """
        Frame interval in seconds, in decreasing order of trust.

        1. --tr, if the caller states it explicitly.
        2. An acquisition_*.json written by preprocess.py, which measures it
           from the DICOM acquisition times.
        3. The 4th spacing of the image header.

        There is deliberately no default. The pipeline previously assumed
        1.5s for every study; every result in a cohort that happens to sample
        at 1.5s looks fine, and every result on any other protocol is silently
        wrong -- wrong AIF lead-time gates, wrong Tmax, wrong CBF, with maps
        that still look plausible. Refusing to run is the only safe behaviour
        when the timing is genuinely unknown.
        """
        override = getattr(self.config, "tr", None)
        if override:
            tr = float(override)
            print(f"[MAIN] TR = {tr:.4f}s (from --tr)")
            return self._check_tr(tr)

        if not self.image_path:
            raise ValueError(
                "No frame interval available: set PerfusionSettings.tr when "
                "calling with arrays rather than file paths. The pipeline "
                "does not assume one -- a wrong TR silently corrupts AIF "
                "gating, Tmax, MTT and CBF while still producing plausible "
                "maps."
            )

        image_dir = os.path.dirname(os.path.abspath(self.image_path))
        for name in sorted(glob.glob(os.path.join(image_dir, "acquisition_*.json"))):
            try:
                with open(name) as fh:
                    meta = json.load(fh)
                tr = float(meta["tr_s"])
                print(f"[MAIN] TR = {tr:.4f}s (measured, {os.path.basename(name)})")
                return self._check_tr(tr)
            except (OSError, ValueError, KeyError):
                continue

        # Read pixdim[4] with nibabel rather than ITK. ITK substitutes 1.0 for
        # an unset 4th spacing, which is indistinguishable from a real 1s
        # protocol -- so trusting it reintroduces exactly the silent wrong
        # answer this method exists to prevent. nibabel reports the stored 0.0.
        header_tr = None
        try:
            import nibabel as nib

            zooms = nib.load(self.image_path).header.get_zooms()
            if len(zooms) >= 4:
                header_tr = float(zooms[3])
        except Exception:
            header_tr = None

        if header_tr and header_tr > 0 and header_tr != 1.0:
            print(
                f"[MAIN] TR = {header_tr:.4f}s (image header pixdim[4]). This "
                f"is a converter estimate; run preprocess.py on the DICOMs, or "
                f"pass --tr, for the measured value."
            )
            return self._check_tr(header_tr)

        detail = (
            "pixdim[4] is exactly 1.0, which is the NIfTI default for an unset "
            "value and cannot be distinguished from a real 1s protocol"
            if header_tr == 1.0
            else f"pixdim[4] is {header_tr}"
        )
        raise ValueError(
            "Cannot determine the frame interval (TR): no acquisition_*.json "
            f"beside the image, and {detail}. Pass --tr <seconds>, or run "
            "`python -m perfusion_analysis_toolbox.preprocess` on the DICOM "
            "series so it can be measured from the acquisition times. "
            "Guessing here would silently corrupt AIF timing, Tmax and CBF."
        )

    @staticmethod
    def _check_tr(tr: float) -> float:
        if not (0.1 <= tr <= 30.0):
            raise ValueError(
                f"Implausible frame interval TR={tr:.4f}s. CT perfusion is "
                f"acquired at roughly 0.5-4s per frame; this is far outside "
                f"that, so the header or the --tr value is wrong."
            )
        return tr

    # ------------------------------------------------------------------
    # Internal helper: in-plane voxel size from image metadata
    # ------------------------------------------------------------------
    def _voxel_spacing(self, ref_4d, ref_3d):
        """
        Voxel size in mm as (x, y, z), from whichever source exists.

        The header wins when there is one. An in-memory caller supplies it
        through PerfusionSettings.voxel_spacing; without that, isotropic
        1 mm is assumed, which leaves motion correction working in voxels
        rather than millimetres.
        """
        for ref in (ref_4d, ref_3d):
            if ref is not None:
                sp = [float(v) for v in list(ref.GetSpacing())[:3]]
                if len(sp) == 3 and all(v > 0 for v in sp):
                    return sp
        configured = getattr(self.config, "voxel_spacing", None)
        if configured:
            return [float(v) for v in configured][:3]
        print("[MAIN] No voxel spacing available; assuming 1mm isotropic.")
        return [1.0, 1.0, 1.0]

    def _in_plane_voxel_size_mm(self, ref_img) -> float:
        """
        Mean in-plane voxel size (mm) from the reference image spacing.

        ITK spacing is (x, y, z), so the in-plane components are the first
        two. Falls back to DEFAULT_VOXEL_SIZE_MM when the header carries no
        real spacing (e.g. a synthetic image left at unit spacing).
        """
        sp = (list(ref_img.GetSpacing()) if ref_img is not None
              else list(getattr(self.config, "voxel_spacing", None) or ()))
        in_plane = [float(v) for v in sp[:2] if float(v) > 0]
        if len(in_plane) != 2:
            print(
                f"[MAIN] No usable in-plane spacing in header ({sp}); "
                f"falling back to {DEFAULT_VOXEL_SIZE_MM} mm"
            )
            return DEFAULT_VOXEL_SIZE_MM
        return sum(in_plane) / 2.0

    def _load_aif_mask(self):
        """Arterial mask from --aif-mask, in (Z, Y, X) to match the CTC."""
        path = getattr(self.config, "aif_mask", None)
        if not path:
            return None
        import nibabel as nib
        arr = np.asanyarray(nib.load(str(path)).dataobj)
        # NIfTI is (X, Y, Z); everything inside the calculator is (Z, Y, X).
        arr = np.transpose(arr, (2, 1, 0)) > 0
        if arr.shape != self.mask.shape:
            raise ValueError(
                f"--aif-mask shape {arr.shape} != brain mask {self.mask.shape}"
            )
        return arr

    # ------------------------------------------------------------------
    def run(self):
        """Run the pipeline and return the maps. See `main_cal`."""
        return self.main_cal()

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------
    def main_cal(self):
        print("\n[MAIN] Starting perfusion pipeline...")

        # Load reference images. A caller working purely in memory has no
        # files to read; it states the frame interval and, optionally, the
        # voxel spacing instead. Geometry is only strictly required to write
        # NIfTIs, which is checked at the point of writing.
        ref_3d = itk.imread(self.mask_path) if self.mask_path else None
        ref_4d = itk.imread(self.image_path) if self.image_path else None

        tr = self._resolve_tr(ref_4d)

        # Step 0: rigid motion correction.
        # This has to happen before anything reads a voxel time course.
        # Uncorrected motion both smears the concentration curves and sweeps
        # bone edges through voxels, which then present as huge spurious
        # enhancement and can win AIF selection outright.
        if getattr(self.config, "motion_correction", True):
            spacing = self._voxel_spacing(ref_4d, ref_3d)
            self.raw_perf, motion = moco.correct_motion(self.raw_perf, spacing)
            if self.savepath:
                with open(f"{self.savepath}/motion.json", "w") as fh:
                    json.dump(motion, fh, indent=2)
        else:
            print("[MAIN] Motion correction disabled.")

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
        # The AIF search region is the vessel mask. An atlas-registered
        # arterial mask was tried here and removed: warped onto a patient it
        # captured only 13% of that scan's own arterial voxels, because
        # inter-subject vascular variability defeats a global transform. An
        # external mask can still be supplied per-scan via
        # PerfusionSettings.aif_mask, which is how an anatomical vessel
        # segmentation is used.
        vessels = self.vessels
        AIF = aif.DSC_aif(
            self.mask.copy(),
            CTC,
            vessels,
            self.savepath,
            mode=getattr(self.config, "aif_mode", "voxel"),
            TR=tr,
            aif_mask=self._load_aif_mask(),
        )
        print("[MAIN] AIF complete.")

        # Step 3: Denoise the CTC for deconvolution (AFTER AIF).
        #
        # Deconvolution amplifies noise, so the CTC has to be conditioned
        # first. The default has always been an in-plane Gaussian blur, which
        # works but is indiscriminate: it flattens the lesion boundary along
        # with the noise, compressing the rCBF distribution toward its median
        # and under-calling ischaemic core. Blurring the vendor's own map by
        # 4-6mm reproduces our distribution almost exactly, which is what
        # identified this as the cause rather than a suspicion about it.
        #
        # The alternatives here attack noise without spending contrast:
        #   temporal  -- smooth along time only. The concentration-time curve
        #                is smooth in t while the noise is not, and nothing is
        #                mixed between neighbouring voxels, so no spatial
        #                contrast is lost at all.
        #   curvature -- edge-preserving diffusion in-plane. Smooths within
        #                regions but not across boundaries, so a lesion edge
        #                survives. Run slice by slice, matching the in-plane
        #                only convention used elsewhere (slices are thick and
        #                cross-slice blur would wash out ischaemic contrast).
        # A bilateral filter was measured at ~46 min per study and rejected.
        method = str(getattr(self.config, "denoise_method", "gaussian"))
        voxel_size_mm = self._in_plane_voxel_size_mm(ref_3d)
        CTC = self._denoise_ctc(CTC, method, voxel_size_mm)

        # Step 4: Compute maps from smoothed CTC
        print("[MAIN] Computing CBV...")
        CBV = cbv.cal(self.mask.copy(), CTC, AIF, self.config)

        print("[MAIN] Computing CBF / Tmax...")
        # `extras` collects maps a method can produce that the others cannot.
        # The parametric estimator fits a transit-time distribution, so MTT and
        # CTH come straight out of it rather than from dividing CBV by CBF.
        extras = {}
        CBF, TMAX = cbf.cal(
            CTC,
            AIF,
            self.mask.copy(),
            svd_frac=float(self.config.svd_frac),
            method=getattr(self.config, "deconv_method", "svd"),
            max_delay_s=float(getattr(self.config, "max_delay_s", 16.0)),
            extras=extras,
            bayes_kwargs=self._bayes_kwargs(),
        )

        # Tmax needs the block-circulant formulation. Under the plain
        # Toeplitz matrix a late-filling voxel has its residue wrapped and
        # truncated, so argmax lands on frame 0: 68% of brain voxels read
        # exactly zero, and it fails worst in the delayed tissue Tmax exists
        # to find.
        method = getattr(self.config, "deconv_method", "svd")
        if method in ("svd", "bayes") and np.mean(TMAX[self.mask] == 0) > 0.4:
            print(
                f"[MAIN] WARNING: {np.mean(TMAX[self.mask] == 0):.0%} of brain "
                f"has Tmax=0 under '{method}'. Tmax and anything derived from "
                "it (penumbra, mismatch) are not usable; use "
                "--deconv-method osvd."
            )

        tmax_offset = 0.0
        if getattr(self.config, "tmax_reference", True):
            TMAX, tmax_offset = normalise.reference_tmax(TMAX, self.mask)
            print(f"[MAIN] Tmax referenced to normal tissue "
                  f"(offset {tmax_offset:.2f}s)")

        print("[MAIN] Computing MTT...")
        MTT = mtt.cal(CBV, CBF, self.mask.copy())

        # Step 5: Write outputs, if a destination was given
        if not self.savepath:
            print("[MAIN] Pipeline complete (in-memory).\n")
            return PerfusionMaps(cbf=CBF, cbv=CBV, mtt=MTT, tmax=TMAX,
                                 ctc=CTC, aif=AIF, extras=dict(extras))
        if ref_3d is None:
            raise ValueError(
                "Cannot write NIfTIs without reference geometry: set "
                "PerfusionSettings.image_path and .mask_path so the outputs "
                "inherit the input headers, or drop save_path and take the "
                "maps from the returned PerfusionMaps."
            )
        print("[MAIN] Writing NIfTI outputs to:", self.savepath)
        self._write_like(CBF, ref_3d, f"{self.savepath}/cbf.nii.gz")
        self._write_like(CBV, ref_3d, f"{self.savepath}/cbv.nii.gz")
        self._write_like(MTT, ref_3d, f"{self.savepath}/mtt.nii.gz")
        # Both deconvolution paths return Tmax in seconds.
        self._write_like(
            TMAX.astype(np.float32), ref_3d, f"{self.savepath}/tmax.nii.gz"
        )
        for name, arr in extras.items():
            self._write_like(
                np.asarray(arr, np.float32), ref_3d,
                f"{self.savepath}/{name}.nii.gz",
            )
            print(f"[MAIN] Wrote {name}.nii.gz")
        # CTC may be shorter than ref_4d after CTA truncation — write without metadata
        try:
            self._write_like(CTC, ref_4d, f"{self.savepath}/ctc.nii.gz")
        except Exception:
            ctc_img = itk.image_view_from_array(CTC.astype(np.float32))
            itk.imwrite(ctc_img, f"{self.savepath}/ctc.nii.gz")

        print("[MAIN] Pipeline complete.\n")
        return PerfusionMaps(cbf=CBF, cbv=CBV, mtt=MTT, tmax=TMAX, ctc=CTC,
                             aif=AIF, extras=dict(extras))
