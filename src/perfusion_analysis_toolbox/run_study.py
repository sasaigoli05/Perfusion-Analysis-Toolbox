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

Every tuning flag from perfusion-compute-cli is accepted and passed through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dicom_dir", help="Directory of the 4D CTP DICOM series.")
    ap.add_argument("out_dir", help="Directory for maps, masks and QC report.")

    ap.add_argument("--tr", type=float, default=None,
                    help="Override the measured frame interval (seconds).")
    ap.add_argument("--svd_frac", type=float, default=0.20)
    ap.add_argument("--deconv-method", default="svd",
                    choices=["svd", "osvd", "osvd_oi"], dest="deconv_method")
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
                        "specific and should be measured with "
                        "tools/cohort/calibrate.py before being relied on."
                    ))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dicom_dir = Path(args.dicom_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not dicom_dir.is_dir():
        print(f"error: {dicom_dir} is not a directory", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    from . import preprocess as pre
    from . import qc as qc_mod
    from .lib.main_calculator import MainCalculator
    from .lib.signal_reader import read_signal

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

    # Provenance: what produced these maps. Without this a map on disk is an
    # orphan and its numbers cannot be reproduced or defended.
    provenance = {
        "dicom_dir": str(dicom_dir),
        "settings": {k: v for k, v in vars(cfg).items()},
        "acquisition": json.loads(Path(prep["timing_path"]).read_text())
        if prep.get("timing_path") else None,
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

    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
