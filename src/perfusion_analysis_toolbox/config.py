#!/usr/bin/env python3
# config.py

import argparse

def parse_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CTP Colormaps Calculation")

    # Mutually exclusive source: DICOM dir OR NIfTI
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--image-directory",
        type=str,
        help="Directory containing a single DICOM series (e.g., 001-100 .dcm slices)."
    )
    src_group.add_argument(
        "--image-path",
        type=str,
        help="Path to input NIfTI image (if already preprocessed)."
    )

    # These become optional because they’ll be generated if using --image-directory
    parser.add_argument("--mask-path", type=str, help="Path to mask image (NIfTI)")
    parser.add_argument("--vessel-path", type=str, help="Path to vessel centerlines (e.g., .mha)")

    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to folder to save results"
    )

    parser.add_argument(
        "--image_type", type=str, default="CTP", help="Image type: CTP/MRP"
    )

    ################## Constants Settings ##################
    parser.add_argument("--k_ct", type=float, default=1.0, help="Constant k_ct (g/ml/HU) for CTP")
    parser.add_argument("--k_mr", type=float, default=1.0, help="Constant k_mr for MRP")
    parser.add_argument("--TE", type=float, default=0.025, help="Constant TE (ms) for MRP")
    parser.add_argument("--TR", type=float, default=1.55, help="Constant TR (s) for MRP")

    parser.add_argument(
        "--use_filter",
        type=bool,
        default=True,
        help="Whether to use low-pass filtering for CTC. Usually need filter for MRP, not for CTP.",
    )
    parser.add_argument(
        "--mrp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for finding MRP bolus arrival time",
    )
    parser.add_argument(
        "--ctp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for finding CTP bolus arrival time",
    )
    parser.add_argument(
        "--to_tensor",
        type=bool,
        default=True,
        help="Whether to convert arrays to torch.tensor",
    )
    parser.add_argument(
        "--mask",
        type=list,
        default=[[], [0, 138], [0, 138]],
        help="Used as BackGround Code for MRP, while BrainMask -300 for CTP (UNC)",
    )

    return parser.parse_args()
