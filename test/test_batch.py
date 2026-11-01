#!/usr/bin/env python3

"""
Batch orchestration.

The point of a batch is that it survives its own failures: eighty studies
started overnight must not be lost because study three had no readable
acquisition times. These tests drive the batch with `run_one` replaced, so
they exercise the scheduling, the error containment and the summary without
needing DICOM on disk or dcm2niix on PATH.
"""

from __future__ import annotations

import csv
import json
import sys

import pytest

sys.path.append("src")

from perfusion_analysis_toolbox import run_study  # noqa: E402


def _make_studies(root, names):
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / "slice.dcm").write_bytes(b"")
    return root


# ------------------------------------------------------------ find_studies
def test_find_studies_returns_immediate_subdirectories_in_order(tmp_path):
    _make_studies(tmp_path, ["c", "a", "b"])
    (tmp_path / "loose.dcm").write_bytes(b"")
    found = run_study.find_studies(tmp_path)
    assert [p.name for p in found] == ["a", "b", "c"]


def test_find_studies_honours_the_glob(tmp_path):
    _make_studies(tmp_path, ["sub-01", "sub-02", "notes"])
    found = run_study.find_studies(tmp_path, "sub-*")
    assert [p.name for p in found] == ["sub-01", "sub-02"]


# ------------------------------------------------------------------- batch
def test_one_failing_study_does_not_stop_the_batch(tmp_path, monkeypatch):
    _make_studies(tmp_path / "in", ["a", "b", "c"])

    def fake_run_one(dicom_dir, out_dir, args):
        if dicom_dir.name == "b":
            raise RuntimeError("dcm2niix failed (2).")
        (out_dir / "provenance.json").write_text("{}")
        return {"study": dicom_dir.name, "dicom_dir": str(dicom_dir),
                "out_dir": str(out_dir), "status": "ok", "verdict": "PASS",
                "tr_s": 1.5, "error": None, "seconds": 0.1}

    monkeypatch.setattr(run_study, "run_one", fake_run_one)
    rc = run_study.main(["--batch", str(tmp_path / "in"), str(tmp_path / "out")])

    # A batch that ran every study to a verdict exits 0 even with failures in
    # it; the verdicts live in the summary. A non-zero exit would make every
    # wrapper script treat a mostly-good cohort as a crash.
    assert rc == 0

    rows = list(csv.DictReader((tmp_path / "out" / "summary.csv").open()))
    assert [r["study"] for r in rows] == ["a", "b", "c"]
    assert {r["study"]: r["status"] for r in rows} == {
        "a": "ok", "b": "error", "c": "ok"}
    failed = next(r for r in rows if r["study"] == "b")
    assert "dcm2niix failed" in failed["error"]


def test_the_failure_detail_is_kept_beside_the_study(tmp_path, monkeypatch):
    _make_studies(tmp_path / "in", ["a"])

    def boom(dicom_dir, out_dir, args):
        raise RuntimeError("exploded")

    monkeypatch.setattr(run_study, "run_one", boom)
    run_study.main(["--batch", str(tmp_path / "in"), str(tmp_path / "out")])

    log = (tmp_path / "out" / "a" / "run.log").read_text()
    assert "Traceback" in log and "exploded" in log


def test_skipped_studies_stay_in_the_summary(tmp_path, monkeypatch):
    _make_studies(tmp_path / "in", ["a", "b"])
    done = tmp_path / "out" / "a"
    done.mkdir(parents=True)
    (done / "provenance.json").write_text(json.dumps(
        {"qc_verdict": "PASS", "acquisition": {"tr_s": 2.0}}))

    def fake_run_one(dicom_dir, out_dir, args):
        return {"study": dicom_dir.name, "dicom_dir": str(dicom_dir),
                "out_dir": str(out_dir), "status": "ok", "verdict": "PASS",
                "tr_s": 1.5, "error": None, "seconds": 0.1}

    monkeypatch.setattr(run_study, "run_one", fake_run_one)
    run_study.main(["--batch", str(tmp_path / "in"), str(tmp_path / "out"),
                    "--skip-existing"])

    rows = {r["study"]: r for r in
            csv.DictReader((tmp_path / "out" / "summary.csv").open())}
    # A resumed batch whose table lists only what ran this time does not
    # describe the cohort, which is the one thing the table is for.
    assert set(rows) == {"a", "b"}
    assert rows["a"]["status"] == "skipped"
    assert rows["a"]["verdict"] == "PASS"
    assert rows["a"]["tr_s"] == "2.0"
    assert rows["b"]["status"] == "ok"


def test_empty_batch_root_is_an_error_not_an_empty_success(tmp_path):
    (tmp_path / "in").mkdir()
    rc = run_study.main(["--batch", str(tmp_path / "in"), str(tmp_path / "out")])
    assert rc == 2


def test_single_study_mode_is_unaffected_by_the_batch_flags(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()

    seen = {}

    def fake_run_one(dicom_dir, out_dir, args):
        seen["dicom_dir"] = dicom_dir
        return {"study": dicom_dir.name, "status": "qc_fail", "verdict": "FAIL",
                "tr_s": 1.5, "error": None, "seconds": 0.1}

    monkeypatch.setattr(run_study, "run_one", fake_run_one)
    rc = run_study.main([str(study), str(tmp_path / "out")])

    # Without --batch the study runs directly, and a QC failure is still a
    # non-zero exit so a single-study caller can branch on it.
    assert seen["dicom_dir"] == study.resolve()
    assert rc == 1
    assert not (tmp_path / "out" / "summary.csv").exists()
