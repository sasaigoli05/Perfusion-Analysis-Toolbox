#!/usr/bin/env python3
"""
Rigid motion correction for 4D CTP series.

A perfusion acquisition runs 30-60 frames over about a minute, and stroke
patients move. Uncorrected, that motion drags tissue across voxel boundaries
between frames, which does two things the rest of the pipeline cannot recover
from: it smears the concentration-time curves, and -- worse -- it sweeps
high-attenuation bone edges through voxels that then look like they enhanced
by hundreds of Hounsfield units. Those spurious edge voxels score as strongly
"arterial" and can win AIF selection outright.

Each frame is registered rigidly to a pre-bolus reference. Mutual information
is used as the metric because the thing that legitimately changes between
frames -- contrast enhancement -- is exactly what an intensity-difference
metric would try to register away.
"""

import numpy as np
import SimpleITK as sitk


# Frames averaged to build the registration reference. These sit before the
# bolus arrives, so the reference carries anatomy without enhancement.
N_REFERENCE_FRAMES = 4

# Registration runs on a downsampled pyramid; full resolution buys nothing
# for a 6-parameter rigid fit and costs a great deal of time.
SHRINK_FACTORS = [4, 2]
SMOOTHING_SIGMAS = [2.0, 1.0]

# A frame whose fitted translation exceeds this is not head motion, it is a
# failed registration. The frame is left uncorrected and flagged.
MAX_PLAUSIBLE_TRANSLATION_MM = 20.0

# Seed for the registration metric's random sampler.
#
# SimpleITK defaults this to sitk.sitkWallClock, which reseeds from the clock on
# every call. That made the whole pipeline irreproducible: identical inputs and
# flags gave different motion corrections and therefore different maps, with
# every brain voxel differing and a median relative difference of 0.80% in CBF
# between consecutive runs. A fixed seed costs nothing and makes a run
# repeatable, which is a precondition for reporting a number.
METRIC_SAMPLING_SEED = 20260905


def _to_sitk(vol_zyx, spacing_xyz):
    """(Z,Y,X) numpy -> sitk image carrying the right physical spacing."""
    img = sitk.GetImageFromArray(
        np.ascontiguousarray(vol_zyx, dtype=np.float32)
    )
    img.SetSpacing([float(s) for s in spacing_xyz])
    return img


def _register_frame(fixed, moving):
    """Rigid registration of one frame. Returns (transform, converged)."""
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10, METRIC_SAMPLING_SEED)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(SHRINK_FACTORS)
    reg.SetSmoothingSigmasPerLevel(SMOOTHING_SIGMAS)
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg.SetInitialTransform(initial, inPlace=False)

    try:
        return reg.Execute(fixed, moving), True
    except RuntimeError:
        return initial, False


def correct_motion(raw_perf, spacing_xyz, verbose=True):
    """
    Rigidly align every frame of a 4D series to a pre-bolus reference.

    Parameters
    ----------
    raw_perf : (Z, Y, X, T) array
    spacing_xyz : voxel size in mm, in ITK (x, y, z) order
    verbose : print per-frame displacement

    Returns
    -------
    corrected : (Z, Y, X, T) float32 array
    motion : dict of per-frame diagnostics -- translation and rotation for
        each frame, the resulting displacement in mm, and the indices of any
        frames whose registration was rejected. Written out as QC by the
        caller; a series with large residual motion should not be trusted
        even after correction.
    """
    raw_perf = np.asarray(raw_perf, dtype=np.float32)
    nZ, nY, nX, nT = raw_perf.shape

    n_ref = min(N_REFERENCE_FRAMES, nT)
    reference = _to_sitk(raw_perf[..., :n_ref].mean(axis=-1), spacing_xyz)

    corrected = np.empty_like(raw_perf)
    translations = np.zeros((nT, 3), dtype=float)
    rotations = np.zeros((nT, 3), dtype=float)
    rejected = []

    for t in range(nT):
        moving = _to_sitk(raw_perf[..., t], spacing_xyz)
        transform, ok = _register_frame(reference, moving)

        params = np.asarray(transform.GetParameters(), dtype=float)
        rot, trans = params[:3], params[3:6]
        shift = float(np.linalg.norm(trans))

        if not ok or shift > MAX_PLAUSIBLE_TRANSLATION_MM:
            # Leave the frame alone rather than applying a bad transform.
            rejected.append(t)
            corrected[..., t] = raw_perf[..., t]
            continue

        resampled = sitk.Resample(
            moving,
            reference,
            transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        corrected[..., t] = sitk.GetArrayFromImage(resampled)
        translations[t] = trans
        rotations[t] = rot

    displacement = np.linalg.norm(translations, axis=1)
    motion = {
        "translation_mm": translations.tolist(),
        "rotation_rad": rotations.tolist(),
        "displacement_mm": displacement.tolist(),
        "median_displacement_mm": float(np.median(displacement)),
        "max_displacement_mm": float(displacement.max()) if nT else 0.0,
        "rejected_frames": rejected,
        "n_frames": int(nT),
    }

    if verbose:
        print(
            f"[MOCO] {nT} frames aligned to mean of first {n_ref}: "
            f"median displacement {motion['median_displacement_mm']:.2f} mm, "
            f"max {motion['max_displacement_mm']:.2f} mm"
        )
        if rejected:
            print(
                f"[MOCO] {len(rejected)} frame(s) left uncorrected "
                f"(registration failed or implausible): {rejected}"
            )

    return corrected, motion
