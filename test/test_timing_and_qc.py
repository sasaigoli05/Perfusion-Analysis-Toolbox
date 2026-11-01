#!/usr/bin/env python3
"""
Tests for acquisition timing resolution and QC gating.

The timing tests exist because the pipeline previously hardcoded a 1.5s frame
interval. That is invisible on any cohort acquired at 1.5s and silently
corrupts AIF timing, Tmax and CBF on every other protocol, so the important
behaviour to pin down is that an unknown TR *stops the run* rather than
defaulting.
"""

import argparse
import sys

import numpy as np
import pytest

sys.path.append("src")


def _cfg(**kw):
    base = dict(image_path="/nonexistent.nii.gz", mask_path="", vessel_path="",
                save_path="", tr=None)
    base.update(kw)
    return argparse.Namespace(**base)


class _FakeImage:
    def __init__(self, spacing):
        self._s = spacing

    def GetSpacing(self):
        return self._s


def _calculator(cfg):
    from perfusion_analysis_toolbox.lib.main_calculator import MainCalculator

    obj = MainCalculator.__new__(MainCalculator)
    obj.config = cfg
    obj.image_path = cfg.image_path
    return obj


def test_tr_explicit_override_wins():
    obj = _calculator(_cfg(tr=2.25))
    assert obj._resolve_tr(_FakeImage([1.0, 1.0, 1.0, 1.0])) == pytest.approx(2.25)


def test_tr_refuses_when_unknown():
    """ITK reports 1.0 for an unset 4th spacing; that must not be trusted."""
    obj = _calculator(_cfg())
    with pytest.raises(ValueError, match="Cannot determine the frame interval"):
        obj._resolve_tr(_FakeImage([0.4, 0.4, 4.0, 1.0]))


def test_tr_rejects_implausible_values():
    obj = _calculator(_cfg(tr=500.0))
    with pytest.raises(ValueError, match="Implausible frame interval"):
        obj._resolve_tr(_FakeImage([0.4, 0.4, 4.0, 1.0]))


def test_dicom_time_parsing():
    from perfusion_analysis_toolbox.preprocess import _dicom_time_to_seconds

    assert _dicom_time_to_seconds("044209.976000") == pytest.approx(
        4 * 3600 + 42 * 60 + 9.976, abs=1e-6
    )


def test_qc_flags_non_arterial_aif():
    from perfusion_analysis_toolbox import qc

    # A peak far above arterial contrast is bone or a motion-swept edge.
    checks = qc.check_aif({"peak_gv": 1900.0, "lead_time_s": 3.0,
                           "suspect": False, "gates": {"n_passing": 30}})
    peak = next(c for c in checks if c["check"] == "aif_peak")
    assert peak["status"] == qc.FAIL


def test_qc_flags_implausible_core_fraction():
    from perfusion_analysis_toolbox import qc

    # A map where most of the brain sits below the core threshold is broken,
    # not a patient with a whole-brain infarct.
    # 40% of the brain at a tenth of the median: the median stays with the
    # majority at 1.0, so those voxels land far below the 0.30 core threshold.
    mask = np.ones((10, 40, 40), bool)
    cbf = np.ones(mask.shape, np.float32)
    cbf[:4] = 0.1
    checks = qc.check_maps(cbf, mask, vox_mL=0.06)
    core = next(c for c in checks if c["check"] == "core_fraction")
    assert core["status"] in (qc.WARN, qc.FAIL)


def test_qc_passes_a_plausible_map():
    from perfusion_analysis_toolbox import qc

    rng = np.random.default_rng(1)
    mask = np.ones((10, 40, 40), bool)
    cbf = rng.normal(1.0, 0.15, mask.shape).astype(np.float32)
    cbf = np.abs(cbf)
    checks = qc.check_maps(cbf, mask, vox_mL=0.06)
    assert all(c["status"] == qc.PASS for c in checks), [
        c for c in checks if c["status"] != qc.PASS
    ]


def test_qc_warns_on_non_uniform_sampling():
    from perfusion_analysis_toolbox import qc

    checks = qc.check_timing({"tr_s": 1.52, "tr_cv": 0.34, "tr_min_s": 1.46,
                              "tr_max_s": 3.08, "n_frames": 36})
    uniform = next(c for c in checks if c["check"] == "uniform_sampling")
    assert uniform["status"] == qc.WARN


# ---------------------------------------------------------------- public API
def test_settings_defaults_are_the_validated_configuration():
    from perfusion_analysis_toolbox import PerfusionSettings

    s = PerfusionSettings()
    assert s.deconv_method == "svd"
    assert s.svd_frac == 0.20
    assert s.denoise_method == "gaussian"
    assert s.motion_correction is True
    assert s.tr is None          # measured from the image unless overridden


def test_settings_replace_rejects_typos():
    """
    A silently-ignored misspelling would mean running a different
    configuration than the one reported, which is the kind of error that
    invalidates a comparison without anything looking wrong.
    """
    import pytest
    from perfusion_analysis_toolbox import PerfusionSettings

    assert PerfusionSettings().replace(deconv_method="bayes").deconv_method == "bayes"
    with pytest.raises(TypeError):
        PerfusionSettings().replace(deconv_methd="bayes")


def test_settings_from_namespace_ignores_extra_cli_arguments():
    import argparse
    from perfusion_analysis_toolbox import PerfusionSettings

    ns = argparse.Namespace(deconv_method="bayes", svd_frac=0.3,
                            save_path="/tmp/x", vessel_path="/tmp/v")
    s = PerfusionSettings.from_namespace(ns)
    assert s.deconv_method == "bayes" and s.svd_frac == 0.3
    assert not hasattr(s, "save_path")


def test_public_api_is_importable_without_itk():
    """Settings and normalisation must not drag in the heavy imaging stack."""
    import perfusion_analysis_toolbox as pat

    assert pat.PerfusionSettings is not None
    assert callable(pat.to_relative)
    assert callable(pat.compute_perfusion)


# --------------------------------------------------------------- AIF fallback
#
# Anatomical AIF selection fails on real data: on the validation cohort the
# vessel segmentation returned no arterial voxels in four studies and lost all
# of them to the brain-mask intersection in a fifth. That used to raise and
# strand the study. It now falls back to voxel scoring and says so in the
# saved record, which is what these tests pin down -- a silent fallback would
# be worse than the exception it replaced.


def _synthetic_study(nz=6, ny=24, nx=24, nt=30, seed=0):
    """A brain mask, a concentration series with a bolus, and no vessels.

    The brain has to dwarf the arterial region. The parenchymal reference is
    a median over brain voxels, so an artery that is a large fraction of the
    mask drags that reference earlier and the lead-time gate then rejects the
    very voxels it was meant to keep.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((nz, ny, nx), bool)
    mask[1:-1, 2:-2, 2:-2] = True

    t = np.arange(nt, dtype=float)
    conc = np.zeros((nz, ny, nx, nt), float)
    tissue = 40.0 * np.exp(-((t - 14.0) ** 2) / 18.0)
    conc[mask] = tissue + rng.normal(0, 0.5, (int(mask.sum()), nt))

    # An arterial region: earlier, narrower, much higher. Peak at frame 11
    # against tissue at 14 is a 4.5s lead at TR=1.5, inside the 1-8s window
    # the selector enforces.
    #
    # It has to be comfortably larger than the 30 candidates the fitter keeps.
    # Voxel scoring relaxes its lead-time gate when too few voxels survive, and
    # a relaxed gate pulls tissue into the fit and flags the run -- which would
    # make this fixture test the relaxation path instead of the fallback.
    art = 300.0 * np.exp(-((t - 11.0) ** 2) / 4.0)
    arterial = np.zeros_like(mask)
    arterial[1:3, 3:7, 3:8] = True
    conc[arterial] = art + rng.normal(0, 0.5, (int(arterial.sum()), nt))
    return mask, conc, arterial


def _run_aif(tmp_path, aif_mask):
    from perfusion_analysis_toolbox.lib.ParamsCalculator import aif as aifmod
    import json

    mask, conc, _ = _synthetic_study()
    aifmod.DSC_aif(mask, conc, vessels=None, savepath=str(tmp_path),
                   mode="voxel", TR=1.5, aif_mask=aif_mask)
    return json.loads((tmp_path / "aif_params.json").read_text())


def test_aif_falls_back_when_mask_is_empty(tmp_path):
    """An all-zero arterial mask must not stop the run."""
    mask, _, arterial = _synthetic_study()
    rec = _run_aif(tmp_path, np.zeros_like(mask))

    assert rec["used_fallback"] is True
    assert rec["aif_selection"] == "global_arterial_score (fallback from external_mask)"
    assert rec["gates"]["fallback"]["reason"] == "supplied aif_mask is empty"
    assert rec["gates"]["fallback"]["n_supplied"] == 0


def test_aif_falls_back_when_mask_misses_the_brain(tmp_path):
    """Voxels outside the brain mask are the commoner real failure."""
    mask, _, arterial = _synthetic_study()
    outside = ~mask
    rec = _run_aif(tmp_path, outside)

    assert rec["used_fallback"] is True
    assert rec["gates"]["fallback"]["reason"] == (
        "supplied aif_mask selects no brain voxels")
    assert rec["gates"]["fallback"]["n_supplied"] == int(outside.sum())


def test_fallback_is_not_reported_as_a_bad_fit(tmp_path):
    """The curve is sound; only the selection method changed."""
    mask, _, arterial = _synthetic_study()
    rec = _run_aif(tmp_path, np.zeros_like(mask))

    assert not any("fallback" in r for r in rec.get("suspect_reasons", []))

    from perfusion_analysis_toolbox import qc
    checks = {c["check"]: c for c in qc.check_aif(rec)}
    assert checks["aif_selection"]["status"] == qc.WARN
    assert checks["aif_fit"]["status"] != qc.FAIL


def test_usable_mask_is_not_a_fallback(tmp_path):
    """The normal path still records anatomical selection."""
    mask, _, arterial = _synthetic_study()
    rec = _run_aif(tmp_path, arterial)

    assert rec["used_fallback"] is False
    assert rec["aif_selection"] == "external_mask"
    assert rec["gates"]["n_in_mask"] == int(arterial.sum())


# ------------------------------------------------------- in-memory library use
#
# The point of the package API is that another pipeline can hand it arrays and
# get arrays back, with no files anywhere. That path used to raise: the
# calculator read its reference geometry from config.image_path unconditionally,
# so a caller with no paths hit `itk.imread(None)`. These pin the contract.


def _synthetic_series(nz=4, ny=20, nx=20, nt=30, tr=1.5, seed=0):
    """A raw 4D series with an arterial region leading the tissue."""
    rng = np.random.default_rng(seed)
    brain = np.zeros((nz, ny, nx), bool)
    brain[1:3, 3:-3, 3:-3] = True
    vessel = np.zeros((nz, ny, nx), bool)
    vessel[1:3, 6:10, 6:11] = True

    t = np.arange(nt, dtype=float) * tr

    def bolus(amp, start):
        c = np.zeros(nt)
        late = t > start
        c[late] = ((t[late] - start) ** 3) * np.exp(-(t[late] - start) / 2.0)
        return c / (c.max() or 1.0) * amp

    raw = np.full((nz, ny, nx, nt), 40.0)
    raw[vessel] += bolus(300.0, 6.0)
    raw[brain & ~vessel] += bolus(60.0, 10.5)
    raw += rng.normal(0, 1.0, raw.shape)
    return raw, brain, vessel


def test_compute_perfusion_runs_from_arrays_alone():
    from perfusion_analysis_toolbox import PerfusionSettings, compute_perfusion

    raw, brain, vessel = _synthetic_series()
    maps = compute_perfusion(
        raw, brain, vessel,
        PerfusionSettings(tr=1.5, voxel_spacing=(0.39, 0.39, 4.0)),
    )
    for name in ("cbf", "cbv", "mtt", "tmax"):
        arr = getattr(maps, name)
        assert arr.shape == brain.shape, name
        assert np.isfinite(arr).all(), name
    # The AIF must have been fitted on the stated time base, not a default.
    assert maps.aif["TR"] == 1.5


def test_in_memory_call_still_refuses_to_guess_the_frame_interval():
    from perfusion_analysis_toolbox import PerfusionSettings, compute_perfusion

    raw, brain, vessel = _synthetic_series()
    # No tr and no image_path to read one from. Assuming a value here is the
    # failure mode the timing work exists to prevent, so this must raise.
    with pytest.raises(ValueError, match="frame interval"):
        compute_perfusion(raw, brain, vessel, PerfusionSettings())


def test_writing_without_reference_geometry_is_refused(tmp_path):
    from perfusion_analysis_toolbox import PerfusionSettings, compute_perfusion

    raw, brain, vessel = _synthetic_series()
    # NIfTI outputs inherit the input headers; with no reference image there
    # are none to inherit, and writing a map with invented geometry would be
    # worse than not writing it.
    with pytest.raises(ValueError, match="reference geometry"):
        compute_perfusion(raw, brain, vessel, PerfusionSettings(tr=1.5),
                          save_path=str(tmp_path / "out"))
