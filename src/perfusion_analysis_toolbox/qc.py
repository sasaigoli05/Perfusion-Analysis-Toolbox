#!/usr/bin/env python3
"""
Quality control for a completed perfusion run.

These are sanity checks, not accuracy predictions. That distinction is
deliberate and was measured: across a 16-study cohort, no reference-free
signal predicted map accuracy. The strongest apparent predictor -- the
fraction of brain called ischaemic -- turned out to be measuring lesion size
rather than map quality (reference lesion volume alone correlates 0.85 with
Dice), and Pearson correlation against the reference retained essentially no
predictive power once lesion size was controlled for (partial r = 0.08).

So this module does not tell you whether a map is accurate. It tells you
whether a map is *implausible* -- that the AIF is not arterial, that the
patient moved more than registration could absorb, that the brain mask is the
wrong size, that the acquisition violates an assumption the pipeline makes.
Those are the failures worth catching automatically, because each one has a
physical meaning and a defensible threshold.

Every check reports PASS, WARN or FAIL with the value that triggered it, so a
reviewer can see what was and was not verified rather than trusting a score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


PASS, WARN, FAIL = "PASS", "WARN", "FAIL"

# Arterial contrast peaks at a few hundred HU. Much above this is bone or a
# motion-swept edge that has scored as arterial; much below is not an artery.
AIF_PEAK_MIN_HU = 50.0
AIF_PEAK_MAX_HU = 800.0

# The AIF must precede the parenchymal peak: contrast reaches the artery before
# the tissue it supplies. A negative or very long lead means the wrong voxel.
AIF_LEAD_MIN_S = 0.5
AIF_LEAD_MAX_S = 12.0

# Rigid registration absorbs small motion. Beyond this the frames are no longer
# describing the same anatomy.
MOTION_WARN_MM = 2.0
MOTION_FAIL_MM = 5.0

# An adult brain is roughly 1100-1500 mL; the mask is eroded and covers only
# the slab the CTP acquired, so the plausible window is wide.
BRAIN_VOL_MIN_ML = 500.0
BRAIN_VOL_MAX_ML = 2000.0

# More than this fraction of the brain below the core threshold is not a
# lesion, it is a failed map.
CORE_FRACTION_FAIL = 0.50
CORE_FRACTION_WARN = 0.35

# A constant frame interval is assumed throughout. Beyond this the assumption
# is materially violated.
TR_CV_WARN = 0.10

CORE_THRESHOLD = 0.30


def _check(name: str, status: str, value, detail: str) -> dict:
    return {"check": name, "status": status, "value": value, "detail": detail}


def check_timing(timing: dict | None) -> list[dict]:
    if timing is None:
        return [_check("acquisition_timing", WARN, None,
                       "No acquisition_*.json; TR came from the image header "
                       "or --tr and was not measured from the DICOMs.")]
    out = [_check("tr_measured", PASS, round(timing["tr_s"], 4),
                  f"TR {timing['tr_s']:.4f}s from {timing['n_frames']} frames")]
    cv = timing.get("tr_cv")
    if cv is not None and cv > TR_CV_WARN:
        out.append(_check(
            "uniform_sampling", WARN, round(cv, 3),
            f"Frame interval varies (CV={cv:.2f}, "
            f"{timing['tr_min_s']:.2f}-{timing['tr_max_s']:.2f}s). The pipeline "
            f"models a constant interval, so Tmax, MTT and the AIF fit are "
            f"approximations for this study."))
    else:
        out.append(_check("uniform_sampling", PASS,
                          round(cv, 3) if cv is not None else None,
                          "Frame interval is effectively constant."))
    return out


def check_aif(aif: dict) -> list[dict]:
    out = []
    peak = float(aif.get("peak_gv", 0.0))
    if peak < AIF_PEAK_MIN_HU:
        out.append(_check("aif_peak", FAIL, round(peak, 1),
                          f"AIF peak {peak:.0f} HU is too low to be arterial."))
    elif peak > AIF_PEAK_MAX_HU:
        out.append(_check("aif_peak", FAIL, round(peak, 1),
                          f"AIF peak {peak:.0f} HU exceeds arterial contrast; "
                          f"likely bone or a motion-swept edge."))
    else:
        out.append(_check("aif_peak", PASS, round(peak, 1),
                          f"AIF peak {peak:.0f} HU is arterial."))

    lead = aif.get("lead_time_s")
    if lead is None:
        out.append(_check("aif_lead_time", WARN, None, "No lead time recorded."))
    elif not (AIF_LEAD_MIN_S <= float(lead) <= AIF_LEAD_MAX_S):
        out.append(_check("aif_lead_time", FAIL, round(float(lead), 2),
                          f"AIF leads the parenchymal peak by {lead:.1f}s, "
                          f"outside {AIF_LEAD_MIN_S}-{AIF_LEAD_MAX_S}s. The "
                          f"selected voxel is probably not arterial."))
    else:
        out.append(_check("aif_lead_time", PASS, round(float(lead), 2),
                          f"AIF leads tissue by {lead:.1f}s."))

    if aif.get("suspect"):
        out.append(_check("aif_fit", FAIL, aif.get("suspect_reasons"),
                          "The AIF fit was flagged suspect: "
                          f"{aif.get('suspect_reasons')}"))
    else:
        out.append(_check("aif_fit", PASS, None, "AIF fit passed its gates."))

    # Selection method. A fallback is a WARN, not a FAIL: the curve is sound,
    # but it did not come from the method the run asked for, so any cohort
    # analysis that claims anatomical AIF selection has to exclude or flag it.
    if aif.get("used_fallback"):
        out.append(_check("aif_selection", WARN, aif.get("aif_selection"),
                          f"AIF came from {aif.get('aif_selection')}. "
                          f"Anatomical selection was requested but produced "
                          f"no usable arterial voxels."))
    elif aif.get("aif_selection"):
        out.append(_check("aif_selection", PASS, aif.get("aif_selection"),
                          f"AIF selected by {aif.get('aif_selection')}."))

    gates = aif.get("gates", {})
    n_pass = gates.get("n_passing")
    if n_pass is not None and n_pass < 5:
        out.append(_check("aif_candidates", WARN, n_pass,
                          f"Only {n_pass} voxels survived AIF gating; the "
                          f"curve rests on very few samples."))
    return out


def check_motion(motion: dict | None) -> list[dict]:
    if motion is None:
        return [_check("motion", WARN, None, "No motion.json; correction off?")]
    mx = float(motion.get("max_displacement_mm", 0.0))
    rej = motion.get("rejected_frames") or []
    status = PASS if mx < MOTION_WARN_MM else (
        WARN if mx < MOTION_FAIL_MM else FAIL)
    out = [_check("motion_displacement", status, round(mx, 2),
                  f"Peak frame displacement {mx:.2f} mm "
                  f"(median {motion.get('median_displacement_mm', 0):.2f} mm).")]
    if rej:
        out.append(_check("motion_rejected_frames", WARN, len(rej),
                          f"{len(rej)} frame(s) failed registration and were "
                          f"left uncorrected: {rej}"))
    return out


def check_maps(cbf: np.ndarray, mask: np.ndarray, vox_mL: float) -> list[dict]:
    out = []
    brain = mask & np.isfinite(cbf)
    n = int(brain.sum())
    vol = n * vox_mL

    if not (BRAIN_VOL_MIN_ML <= vol <= BRAIN_VOL_MAX_ML):
        out.append(_check("brain_volume", FAIL, round(vol, 1),
                          f"Masked brain volume {vol:.0f} mL is outside "
                          f"{BRAIN_VOL_MIN_ML:.0f}-{BRAIN_VOL_MAX_ML:.0f} mL; "
                          f"the brain mask is wrong."))
    else:
        out.append(_check("brain_volume", PASS, round(vol, 1),
                          f"Masked brain volume {vol:.0f} mL."))

    bad = int((~np.isfinite(cbf[mask])).sum())
    out.append(_check("finite_values", PASS if bad == 0 else FAIL, bad,
                      "All CBF values finite." if bad == 0 else
                      f"{bad} non-finite CBF voxels."))

    positive = cbf[brain & (cbf > 0)]
    if positive.size < 0.5 * max(n, 1):
        out.append(_check("nonzero_fraction", FAIL,
                          round(positive.size / max(n, 1), 3),
                          f"Only {100*positive.size/max(n,1):.0f}% of brain has "
                          f"positive CBF; deconvolution largely failed."))
        return out
    out.append(_check("nonzero_fraction", PASS,
                      round(positive.size / max(n, 1), 3),
                      f"{100*positive.size/max(n,1):.0f}% of brain has positive CBF."))

    rel = positive / np.median(positive)
    frac = float((rel < CORE_THRESHOLD).mean())
    status = PASS if frac < CORE_FRACTION_WARN else (
        WARN if frac < CORE_FRACTION_FAIL else FAIL)
    out.append(_check("core_fraction", status, round(frac, 3),
                      f"{100*frac:.1f}% of brain below rCBF {CORE_THRESHOLD}. "
                      + ("Plausible." if status == PASS else
                         "Implausibly large for a lesion; suspect the map, "
                         "not the patient.")))
    return out


def run_qc(save_path: str | Path, mask_path: str | Path,
           timing_path: str | Path | None = None) -> dict:
    """Collect every check for one completed run."""
    import nibabel as nib

    save = Path(save_path)
    checks: list[dict] = []

    timing = None
    if timing_path and Path(timing_path).exists():
        timing = json.loads(Path(timing_path).read_text())
    checks += check_timing(timing)

    aif_file = save / "aif_params.json"
    if aif_file.exists():
        checks += check_aif(json.loads(aif_file.read_text()))
    else:
        checks.append(_check("aif", FAIL, None, "No aif_params.json written."))

    motion_file = save / "motion.json"
    checks += check_motion(
        json.loads(motion_file.read_text()) if motion_file.exists() else None)

    cbf_file = save / "cbf.nii.gz"
    if cbf_file.exists():
        img = nib.load(str(cbf_file))
        cbf = img.get_fdata().astype(np.float32)
        mask = nib.load(str(mask_path)).get_fdata() > 0.5
        vox_mL = abs(np.linalg.det(img.affine[:3, :3])) / 1000.0
        checks += check_maps(cbf, mask, vox_mL)
    else:
        checks.append(_check("cbf_map", FAIL, None, "No cbf.nii.gz written."))

    n_fail = sum(1 for c in checks if c["status"] == FAIL)
    n_warn = sum(1 for c in checks if c["status"] == WARN)
    verdict = FAIL if n_fail else (WARN if n_warn else PASS)

    report = {"verdict": verdict, "n_fail": n_fail, "n_warn": n_warn,
              "checks": checks}
    (save / "qc.json").write_text(json.dumps(report, indent=2))
    return report


def format_report(report: dict) -> str:
    lines = [f"QC verdict: {report['verdict']}  "
             f"({report['n_fail']} fail, {report['n_warn']} warn)", "-" * 72]
    for c in report["checks"]:
        lines.append(f"  [{c['status']:4s}] {c['check']:24s} {c['detail']}")
    if report["verdict"] == PASS:
        lines.append("")
        lines.append("Note: these are plausibility checks. They do not "
                     "establish that the map is accurate.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="QC a completed perfusion run.")
    ap.add_argument("save_path")
    ap.add_argument("mask_path")
    ap.add_argument("--timing-path", default=None)
    a = ap.parse_args()
    print(format_report(run_qc(a.save_path, a.mask_path, a.timing_path)))
