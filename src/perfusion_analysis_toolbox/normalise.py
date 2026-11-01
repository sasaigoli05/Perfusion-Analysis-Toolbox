"""
Normalisation of perfusion maps to a reference value.

A perfusion map in arbitrary units only becomes interpretable once it is
divided by a reference drawn from the same scan. Which reference is chosen
turns out to matter more than it looks.

The clinical convention is relative CBF against the whole-brain median, cut at
0.30. That reference sits between grey and white matter, and white matter
normally carries roughly half the flow of grey matter -- so white matter starts
much closer to the threshold than grey matter does, before any stroke. Measured
on one cohort scan, against the *vendor's own* core:

    white matter flagged as core   12%
    grey matter flagged as core     3%

a four-fold difference driven by tissue type rather than by ischaemia. This is
a property of the convention, not of any one package, which is also a plausible
contributor to the large between-package disagreement reported in the
literature.

Normalising each voxel against its mirror-image voxel in the opposite
hemisphere removes the bias by construction: the mirror voxel is the same
tissue type. The cost is that it assumes one hemisphere is relatively spared,
so it is unsuitable for bilateral disease, and it is unreliable close to the
midline where mirroring is ill-conditioned.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

MODES = ("brain_median", "noncore_median", "contralateral")

# Voxels nearer the midline than this are excluded from the contralateral
# ratio: the mirror of a near-midline voxel is itself, so the ratio collapses
# to 1 and nothing is ever flagged there.
MIDLINE_EXCLUSION_MM = 6.0


def find_midsagittal(brain, spacing_xyz, max_shift_mm=25.0, max_angle_deg=12.0):
    """
    Locate the midsagittal plane by maximising left-right symmetry.

    Returns (shift_vox, angle_deg) describing how to move the volume so that a
    plain left-right flip about the image centre maps the brain onto itself.

    Heads are not centred or straight in the scanner, so flipping about the
    image centre is wrong by a centimetre or more on many scans -- enough to
    compare grey matter against white. The search is coarse-to-fine on a
    downsampled mask, which is both faster and less sensitive to noise than
    optimising on the image itself.
    """
    m = brain.astype(np.float32)
    # Collapse z: the midsagittal plane is near-vertical, so a single in-plane
    # projection carries the information and the search is 2D rather than 3D.
    proj = m.sum(axis=2)
    small = ndi.zoom(proj, 0.25, order=1)
    sx = float(spacing_xyz[0]) * 4.0          # mm per downsampled voxel

    best, best_score = (0.0, 0.0), -np.inf
    for angle in np.arange(-max_angle_deg, max_angle_deg + 0.1, 2.0):
        rot = ndi.rotate(small, angle, reshape=False, order=1)
        for shift_mm in np.arange(-max_shift_mm, max_shift_mm + 0.1, sx):
            sh = ndi.shift(rot, (shift_mm / sx, 0), order=1)
            flipped = sh[::-1, :]
            denom = np.linalg.norm(sh) * np.linalg.norm(flipped)
            if denom <= 0:
                continue
            score = float((sh * flipped).sum() / denom)
            if score > best_score:
                best_score, best = score, (shift_mm / float(spacing_xyz[0]), angle)
    return best[0], best[1], best_score


def mirror_volume(vol, shift_vox, angle_deg):
    """Left-right mirror of `vol` about the detected midsagittal plane."""
    v = ndi.shift(vol, (shift_vox, 0, 0), order=1, mode="nearest")
    if abs(angle_deg) > 1e-6:
        v = ndi.rotate(v, angle_deg, axes=(0, 1), reshape=False, order=1,
                       mode="nearest")
    v = v[::-1, :, :]
    if abs(angle_deg) > 1e-6:
        v = ndi.rotate(v, -angle_deg, axes=(0, 1), reshape=False, order=1,
                       mode="nearest")
    return ndi.shift(v, (-shift_vox, 0, 0), order=1, mode="nearest")


def contralateral_ratio(cbf, brain, spacing_xyz, smooth_mm=4.0, midline=None):
    """
    Each voxel divided by its mirror-image voxel in the other hemisphere.

    The mirror value is smoothed first. A single mirrored voxel is as noisy as
    the voxel being normalised, so dividing one by the other doubles the noise;
    smoothing only the denominator keeps the numerator's detail while making
    the reference stable.

    Returns (ratio, valid) -- `valid` excludes voxels whose mirror falls
    outside the brain and voxels too near the midline to mirror meaningfully.
    """
    if midline is None:
        midline = find_midsagittal(brain, spacing_xyz)
    shift_vox, angle_deg = midline[0], midline[1]

    sigma = [smooth_mm / (2.355 * float(s)) for s in spacing_xyz[:3]]
    sigma[2] = 0.0                      # slices are far apart; do not mix them

    masked = np.where(brain, cbf, 0.0).astype(np.float32)
    num = ndi.gaussian_filter(masked, sigma)
    den = ndi.gaussian_filter(brain.astype(np.float32), sigma)
    smoothed = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-3)

    mirror_cbf = mirror_volume(smoothed, shift_vox, angle_deg)
    mirror_brain = mirror_volume(brain.astype(np.float32), shift_vox, angle_deg) > 0.5

    # Distance from the detected midline, in mm.
    nx = cbf.shape[0]
    xs = (np.arange(nx) - (nx - 1) / 2.0 - shift_vox) * float(spacing_xyz[0])
    dist = np.abs(xs)[:, None, None] * np.ones_like(cbf, dtype=np.float32)

    valid = brain & mirror_brain & (mirror_cbf > 0) & (dist > MIDLINE_EXCLUSION_MM)
    ratio = np.zeros_like(cbf, dtype=np.float32)
    np.divide(cbf, mirror_cbf, out=ratio, where=valid)
    return ratio, valid


def to_relative(cbf, brain, mode="brain_median", exclude=None, spacing_xyz=None):
    """
    Relative CBF under the requested reference.

    Returns (relative_map, valid_mask). For the global modes every brain voxel
    is valid; for 'contralateral' the mask is narrower, and callers must use it
    rather than the brain mask or they will compare against zeros.
    """
    if mode == "contralateral":
        if spacing_xyz is None:
            raise ValueError("contralateral normalisation needs voxel spacing")
        return contralateral_ratio(cbf, brain, spacing_xyz)

    pool = brain & (cbf > 0)
    if mode == "noncore_median" and exclude is not None:
        pool = pool & ~exclude
    ref = float(np.median(cbf[pool])) if pool.any() else 0.0
    if ref <= 0:
        return np.zeros_like(cbf), brain.copy()
    return np.where(brain, cbf / ref, 0.0).astype(np.float32), brain.copy()


# ----------------------------------------------------------------- Tmax
def reference_tmax(tmax, brain, percentile=25.0):
    """
    Re-reference Tmax so that normal tissue reads approximately zero.

    Tmax is the delay between the arterial input function and the peak of the
    tissue residue function, so it inherits whatever delay is already in the
    AIF. Sweeping a synthetic AIF across one study shows the 25th percentile
    of brain Tmax following the AIF's lead almost one-for-one:

        AIF lead   7.6s   6.1s   4.6s   1.5s
        Tmax p25   6.1s   4.6s   3.1s   1.5s

    So an AIF taken three seconds earlier shifts every voxel by three seconds,
    and the fraction of brain past the clinical 6-second threshold moves with
    it -- 49% to 88% across that sweep. The absolute value is not comparable
    between scans, let alone between software packages.

    Most of a stroke brain is not ischaemic, so a low percentile of the Tmax
    distribution estimates that scan's normal transit delay. Subtracting it
    leaves a delay relative to the patient's own normal tissue, which is what
    the 6-second threshold is meant to express.

    A percentile rather than the histogram mode: Tmax is quantised to whole
    frames and piles up at the search cap, so the mode is unstable -- it
    returned offsets of 12.3, 4.7, 7.8 and 0.1 seconds across that same sweep,
    tracking nothing.

    Returns (referenced_tmax, offset_seconds).
    """
    v = np.asarray(tmax)[brain]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.asarray(tmax).copy(), 0.0
    offset = float(np.percentile(v, percentile))
    out = np.where(brain, np.asarray(tmax) - offset, 0.0)
    return np.maximum(out, 0.0).astype(np.float32), offset
