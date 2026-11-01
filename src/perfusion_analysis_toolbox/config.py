#!/usr/bin/env python3
"""Command-line configuration for the perfusion parameter calculator."""

import argparse


def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CTP Perfusion Parameter Calculator"
    )

    # --- Inputs / outputs -------------------------------------------------
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to 4D CTP NIfTI image.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to brain mask NIfTI.",
    )
    parser.add_argument(
        "--vessel-path",
        type=str,
        required=True,
        help="Path to vessel mask / centerlines image.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save output perfusion maps.",
    )

    parser.add_argument(
        "--tr",
        type=float,
        default=None,
        help=(
            "Frame interval in seconds. Normally measured from the DICOM "
            "acquisition times by preprocess.py and read automatically; set "
            "this only to override, or when running on a NIfTI whose header "
            "carries no timing."
        ),
    )

    # --- Preprocessing ----------------------------------------------------
    parser.add_argument(
        "--no-motion-correction",
        action="store_false",
        dest="motion_correction",
        help=(
            "Skip rigid frame-to-frame motion correction. On by default; "
            "disable only to reproduce an older run."
        ),
    )

    parser.add_argument(
        "--aif-mode",
        type=str,
        choices=["voxel", "region"],
        default="voxel",
        dest="aif_mode",
        help=(
            "AIF candidate selection. 'voxel' scores individual voxels; "
            "'region' requires connected components of plausible size with "
            "a smooth mean curve, which flags failures far better but "
            "currently scores slightly lower against the reference. Region "
            "mode falls back to voxel scoring when nothing survives."
        ),
    )

    parser.add_argument(
        "--aif-mask",
        type=str,
        default=None,
        dest="aif_mask",
        help=(
            "NIfTI mask of arterial voxels to take the AIF from, overriding "
            "candidate scoring. The curve is still extracted and "
            "gamma-variate fitted the usual way -- the mask supplies only "
            "*where* to look. Intended for an external vessel segmentation "
            "that knows arterial anatomy, which voxel scoring does not."
        ),
    )

    parser.add_argument(
        "--smooth-fwhm",
        type=float,
        default=4.0,
        dest="smooth_fwhm",
        help=(
            "In-plane Gaussian FWHM (mm) applied to the CTC before "
            "deconvolution. Suppresses noise so deconvolution is stable, but "
            "compresses the rCBF distribution toward its median, which "
            "under-calls ischaemic core. 0 disables smoothing."
        ),
    )

    parser.add_argument(
        "--denoise-method",
        type=str,
        default="gaussian",
        dest="denoise_method",
        help=(
            "CTC conditioning before deconvolution. 'gaussian' is an in-plane "
            "blur (default, uses --smooth-fwhm) and costs lesion contrast; "
            "'temporal' smooths along time only and costs none; 'curvature' "
            "is edge-preserving in-plane diffusion. Combine with '+', e.g. "
            "'curvature+temporal'. 'none' disables conditioning."
        ),
    )
    parser.add_argument(
        "--tips-radius",
        type=int,
        default=3,
        dest="tips_radius",
        help="In-plane neighbourhood radius for --denoise-method tips.",
    )
    parser.add_argument(
        "--tips-h",
        type=float,
        default=10.0,
        dest="tips_h",
        help=(
            "TIPS profile-similarity scale in HU. Voxels whose time courses "
            "differ by more than roughly this RMS amount are not averaged "
            "together, so it sets how sharply lesion boundaries are preserved."
        ),
    )
    parser.add_argument(
        "--temporal-sigma",
        type=float,
        default=1.0,
        dest="temporal_sigma",
        help="Temporal smoothing sigma in frames (--denoise-method temporal).",
    )
    parser.add_argument(
        "--curvature-iterations",
        type=int,
        default=5,
        dest="curvature_iterations",
        help="Curvature-flow iterations (--denoise-method curvature).",
    )

    # --- Deconvolution ----------------------------------------------------
    parser.add_argument(
        "--svd_frac",
        type=float,
        default=0.20,
        help=(
            "Truncated-SVD threshold, as a fraction of the largest singular "
            "value. Lower values retain more components (less smoothing, "
            "more noise)."
        ),
    )

    parser.add_argument(
        "--deconv-method",
        type=str,
        choices=["svd", "osvd", "osvd_oi", "bayes"],
        default="svd",
        dest="deconv_method",
        help=(
            "Deconvolution: 'svd' is the Toeplitz form; 'osvd' is "
            "block-circulant and insensitive to bolus delay; 'osvd_oi' is "
            "block-circulant with per-voxel oscillation-index "
            "regularisation, which chooses the truncation level itself and "
            "ignores --svd_frac entirely; 'bayes' is MAP parametric "
            "deconvolution over a gamma transit-time distribution, "
            "regularised by physiological and spatial priors rather than by "
            "smoothing, and also yields MTT and CTH maps directly."
        ),
    )
    parser.add_argument(
        "--max_delay_s",
        type=float,
        default=16.0,
        help=(
            "oSVD only. Longest residue-peak delay treated as physiological; "
            "peaks beyond it are wrap-around artefact and are not searched."
        ),
    )

    parser.add_argument(
        "--bayes-iterations",
        type=int,
        default=3,
        dest="bayes_iterations",
        help=(
            "Iterated-conditional-modes sweeps for --deconv-method bayes. "
            "Each sweep re-solves every voxel against its neighbours' current "
            "shape estimate; the maps stop moving after two or three."
        ),
    )
    parser.add_argument(
        "--bayes-spatial",
        type=float,
        default=1.0,
        dest="bayes_spatial",
        help=(
            "Weight of the spatial prior on the transit-time shape parameters "
            "(0 disables it, leaving per-voxel MAP). Amplitude is never "
            "spatially regularised, so lesion contrast is unaffected."
        ),
    )
    parser.add_argument(
        "--bayes-prior-weight",
        type=float,
        default=1.0,
        dest="bayes_prior_weight",
        help="Weight of the physiological parameter priors (0 disables them).",
    )

    # --- CBV scaling ------------------------------------------------------
    parser.add_argument(
        "--kh",
        type=float,
        default=1.0,
        help="Haematocrit correction factor used to scale CBV.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="Brain tissue density (g/mL) used to scale CBV.",
    )

    return parser.parse_args()
