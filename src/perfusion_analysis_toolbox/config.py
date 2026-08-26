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
