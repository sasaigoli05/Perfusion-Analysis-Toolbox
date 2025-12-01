#!/usr/bin/env python3
# config.py (REVERTED VERSION — SIMPLE 4-PATH MODE)

import argparse

def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTP Perfusion Parameter Calculator")

    # === REQUIRED STATIC INPUTS (old behavior) ===
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to 4D CTP NIfTI image."
    )

    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to brain mask NIfTI."
    )

    parser.add_argument(
        "--vessel-path",
        type=str,
        required=True,
        help="Path to vessel mask / centerlines image."
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Directory to save output perfusion maps."
    )

    # === Keep your constants / algorithm settings ===
    parser.add_argument(
        "--image_type",
        type=str,
        default="CTP",
        help="Image type: CTP or MRP."
    )

    parser.add_argument(
        "--k_ct",
        type=float,
        default=1.0,
        help="Constant k_ct (g/ml/HU) for CTP"
    )
    parser.add_argument(
        "--k_mr",
        type=float,
        default=1.0,
        help="Constant k_mr for MRP"
    )
    parser.add_argument(
        "--TE",
        type=float,
        default=0.025,
        help="TE (ms) for MRP"
    )
    parser.add_argument(
        "--TR",
        type=float,
        default=1.55,
        help="TR (s) for MRP"
    )

    parser.add_argument(
        "--use_filter",
        type=bool,
        default=True,
        help="Use low-pass filtering for CTC/MRP."
    )
    parser.add_argument(
        "--mrp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for MRP bolus arrival time."
    )
    parser.add_argument(
        "--ctp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for CTP bolus arrival time."
    )
    parser.add_argument(
        "--to_tensor",
        type=bool,
        default=True,
        help="Convert arrays to torch.tensor."
    )
    parser.add_argument(
        "--mask",
        type=list,
        default=[[], [0, 138], [0, 138]],
        help="Background mask / range settings."
    )

    return parser.parse_args()
