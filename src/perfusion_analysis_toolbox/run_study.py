#!/usr/bin/env python3
"""
One command from a DICOM series to quality-controlled perfusion maps.

Until now the pipeline had to be driven in three manual steps -- run
preprocess.py, inspect the masks, then invoke perfusion-compute-cli with the
right paths -- which is workable for the person who wrote it and a barrier for
anyone else. This wires the steps together, carries the measured acquisition
timing from the DICOM headers through to the deconvolution, and finishes with a
QC report so a run that went wrong says so instead of quietly writing a
plausible-looking map.

    python -m perfusion_analysis_toolbox.run_study <dicom_dir> <out_dir>

With --batch, the first argument is a directory of studies rather than one
study, and every immediate subdirectory is run:

    python -m perfusion_analysis_toolbox.run_study --batch <studies_dir> <out_dir>

A batch never stops on a failure. One study that cannot be converted, or whose
QC fails, is recorded and the rest continue -- the alternative is a loop that
dies on study 3 of 80 overnight. Per-study console output goes to a log beside
that study's maps, and a summary table of every study lands in the output root.

Every tuning flag from perfusion-compute-cli is accepted and passed through.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dicom_dir",
                    help="Directory of the 4D CTP DICOM series, or, with "
                         "--batch, a directory of such directories.")
    ap.add_argument("out_dir", help="Directory for maps, masks and QC report.")

    batch = ap.add_argument_group("batch")
    batch.add_argument("--batch", action="store_true",
                       help="Treat dicom_dir as a directory of studies and "
                            "run every immediate subdirectory.")
    batch.add_argument("--glob", default="*", metavar="PATTERN",
                       help="Which subdirectories to treat as studies "
                            "(default: all).")
    batch.add_argument("--jobs", type=int, default=1, metavar="N",
                       help="Studies to run concurrently (default: 1). Each "
                            "study is a separate process; memory, not cores, "
                            "is usually the limit.")
    batch.add_argument("--skip-existing", action="store_true",
                       dest="skip_existing",
                       help="Skip studies that already have a "
                            "provenance.json, so an interrupted batch "
                            "resumes instead of restarting.")

    ap.add_argument("--tr", type=float, default=None,
                    help="Override the measured frame interval (seconds).")
    ap.add_argument("--svd_frac", type=float, default=0.20)
    ap.add_argument("--deconv-method", default="svd",
                    choices=["svd", "osvd", "osvd_oi", "bayes"],
                    dest="deconv_method")
    ap.add_argument("--smooth-fwhm", type=float, default=4.0,
                    dest="smooth_fwhm")
    ap.add_argument("--denoise-method", default="gaussian",
                    dest="denoise_method")
    ap.add_argument("--temporal-sigma", type=float, default=1.0,
                    dest="temporal_sigma")
    ap.add_argument("--curvature-iterations", type=int, default=5,
                    dest="curvature_iterations")
    ap.add_argument("--aif-mode", default="voxel",
                    choices=["voxel", "region"], dest="aif_mode")
    ap.add_argument("--max_delay_s", type=float, default=16.0)
    ap.add_argument("--kh", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--no-motion-correction", action="store_false",
                    dest="motion_correction")
    ap.add_argument("--core-threshold", type=float, default=None,
                    help=(
                        "Relative CBF below which tissue is called core. The "
                        "clinical 0.30 is calibrated for other software; this "
                        "pipeline's equivalent threshold is configuration-"
                        "specific and should be measured against a reference "
                        "on your own data before being relied on."
                    ))
    return ap


# --------------------------------------------------------------- one study
def run_one(dicom_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    """
    Preprocess, compute maps and QC one study.

    Returns a record of what happened. Raises only for programming errors --
    a study that cannot be processed comes back with status 'error' so a
    batch can carry on.
    """
    from . import preprocess as pre
    from . import qc as qc_mod
    from .lib.main_calculator import MainCalculator
    from .lib.signal_reader import read_signal

    started = time.time()
    record = {"study": dicom_dir.name, "dicom_dir": str(dicom_dir),
              "out_dir": str(out_dir), "status": "error", "verdict": None,
              "tr_s": None, "error": None, "seconds": None}

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 1/3  preprocess: DICOM -> NIfTI, acquisition timing, masks")
    print("=" * 72)
    prep = pre.preprocess_from_dicom(str(dicom_dir), str(out_dir))

    maps_dir = out_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STEP 2/3  perfusion maps")
    print("=" * 72)

    # MainCalculator reads its settings off a namespace, and resolves TR from
    # the acquisition JSON that preprocess just wrote unless overridden here.
    cfg = argparse.Namespace(
        image_path=prep["image_path"],
        mask_path=prep["mask_path"],
        vessel_path=prep["vessel_path"],
        save_path=str(maps_dir),
        tr=args.tr,
        svd_frac=args.svd_frac,
        deconv_method=args.deconv_method,
        smooth_fwhm=args.smooth_fwhm,
        denoise_method=args.denoise_method,
        temporal_sigma=args.temporal_sigma,
        curvature_iterations=args.curvature_iterations,
        aif_mode=args.aif_mode,
        max_delay_s=args.max_delay_s,
        kh=args.kh,
        rho=args.rho,
        motion_correction=args.motion_correction,
    )

    raw, mask, vessels, _o, _s, _d = read_signal(
        cfg.image_path, cfg.mask_path, cfg.vessel_path
    )
    MainCalculator(raw_perf=raw, mask=mask, vessels=vessels,
                   config=cfg, save_path=str(maps_dir)).run()

    print("=" * 72)
    print("STEP 3/3  quality control")
    print("=" * 72)
    report = qc_mod.run_qc(maps_dir, cfg.mask_path, prep.get("timing_path"))
    print(qc_mod.format_report(report))

    acquisition = (json.loads(Path(prep["timing_path"]).read_text())
                   if prep.get("timing_path") else None)

    # Provenance: what produced these maps. Without this a map on disk is an
    # orphan and its numbers cannot be reproduced or defended.
    provenance = {
        "dicom_dir": str(dicom_dir),
        "settings": {k: v for k, v in vars(cfg).items()},
        "acquisition": acquisition,
        "qc_verdict": report["verdict"],
        "core_threshold": args.core_threshold,
    }
    try:
        from . import __version__ as pkg_version
        provenance["version"] = pkg_version
    except Exception:
        pass
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    print()
    print(f"maps       : {maps_dir}")
    print(f"QC report  : {maps_dir / 'qc.json'}")
    print(f"provenance : {out_dir / 'provenance.json'}")

    record.update(
        status="ok" if report["verdict"] != "FAIL" else "qc_fail",
        verdict=report["verdict"],
        tr_s=(acquisition or {}).get("tr_s"),
        seconds=round(time.time() - started, 1),
    )
    return record


def _run_one_logged(dicom_dir: Path, out_dir: Path,
                    args: argparse.Namespace) -> dict:
    """
    run_one with its console output sent to a log beside the maps.

    Two reasons this is not optional in a batch: eighty studies of pipeline
    chatter interleaved by several worker processes is unreadable, and the
    detail of a study that went wrong is worth keeping next to its output
    rather than scrolled off a terminal.
    """
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    try:
        with open(log_path, "w") as fh:
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                record = run_one(dicom_dir, out_dir, args)
    except BaseException as exc:          # noqa: BLE001 - a batch must not die
        record = {
            "study": dicom_dir.name, "dicom_dir": str(dicom_dir),
            "out_dir": str(out_dir), "status": "error", "verdict": None,
            "tr_s": None, "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.time() - started, 1),
        }
        with open(log_path, "a") as fh:
            fh.write("\n" + traceback.format_exc())
    record["log"] = str(log_path)
    return record


# ------------------------------------------------------------------- batch
def find_studies(root: Path, pattern: str = "*") -> list[Path]:
    """
    Immediate subdirectories of `root`, in name order.

    One directory is one study, which is the same contract the single-study
    path already has: dcm2niix is pointed at it and expected to yield one
    series. A vendor export holding twenty-odd series is not a study
    directory -- point this at the level where each folder is one acquisition.
    """
    return sorted(p for p in root.glob(pattern) if p.is_dir())


def _record_for_existing(study: Path, dest: Path) -> dict:
    """Summary row for a study skipped because it was already done."""
    rec = {"study": study.name, "dicom_dir": str(study), "out_dir": str(dest),
           "status": "skipped", "verdict": None, "tr_s": None,
           "error": None, "seconds": None, "log": str(dest / "run.log")}
    try:
        prov = json.loads((dest / "provenance.json").read_text())
        rec["verdict"] = prov.get("qc_verdict")
        rec["tr_s"] = (prov.get("acquisition") or {}).get("tr_s")
    except (OSError, ValueError):
        pass
    return rec


def run_batch(root: Path, out_root: Path, args: argparse.Namespace) -> int:
    studies = find_studies(root, args.glob)
    if not studies:
        print(f"error: no subdirectories matching {args.glob!r} in {root}",
              file=sys.stderr)
        return 2

    planned, skipped = [], []
    for study in studies:
        dest = out_root / study.name
        if args.skip_existing and (dest / "provenance.json").exists():
            skipped.append(_record_for_existing(study, dest))
        else:
            planned.append((study, dest))

    print(f"{len(studies)} studies in {root}")
    if skipped:
        print(f"  {len(skipped)} already done, skipping")
    print(f"  {len(planned)} to run, {args.jobs} at a time")
    print()

    out_root.mkdir(parents=True, exist_ok=True)
    # Skipped studies stay in the summary. A resumed batch whose table only
    # lists the studies that happened to run in the last invocation does not
    # describe the cohort, which is the one thing the table is for.
    records: list[dict] = list(skipped)
    started = time.time()

    MARKS = {"ok": "ok  ", "qc_fail": "QC  ", "error": "ERR ",
             "skipped": "skip"}

    def note(i: int, rec: dict) -> None:
        mark = MARKS.get(rec["status"], "?   ")
        detail = rec["error"] or rec["verdict"] or ""
        print(f"[{i:>3}/{len(planned)}] {mark} {rec['study']:<40} "
              f"{rec['seconds']:>6}s  {detail}")

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(_run_one_logged, study, dest, args): study
                for study, dest in planned
            }
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                records.append(rec)
                note(i, rec)
    else:
        for i, (study, dest) in enumerate(planned, 1):
            rec = _run_one_logged(study, dest, args)
            records.append(rec)
            note(i, rec)

    records.sort(key=lambda r: r["study"])
    summary_json = out_root / "summary.json"
    summary_json.write_text(json.dumps(records, indent=2))

    summary_csv = out_root / "summary.csv"
    fields = ["study", "status", "verdict", "tr_s", "seconds", "error", "log"]
    with open(summary_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    counts = {k: sum(r["status"] == k for r in records)
              for k in ("ok", "qc_fail", "error", "skipped")}
    print()
    print(f"{counts['ok']} ok, {counts['qc_fail']} QC failed, "
          f"{counts['error']} errored, {counts['skipped']} already done "
          f"in {time.time() - started:.0f}s")
    print(f"summary    : {summary_csv}")

    # A batch that ran to completion exits 0 even with failures in it: the
    # per-study verdicts are in the summary, and a non-zero exit here would
    # make every wrapper script treat a mostly-good cohort as a crash.
    return 0


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dicom_dir = Path(args.dicom_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not dicom_dir.is_dir():
        print(f"error: {dicom_dir} is not a directory", file=sys.stderr)
        return 2
    if args.jobs < 1:
        print("error: --jobs must be at least 1", file=sys.stderr)
        return 2

    if args.batch:
        return run_batch(dicom_dir, out_dir, args)

    record = run_one(dicom_dir, out_dir, args)
    return 1 if record["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
