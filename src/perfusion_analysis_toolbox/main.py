#!/usr/bin/env python3

"""
Command line utility to compute perfusion parameters over a given CT input image.
"""

import torch
from pathlib import Path

from .lib.utils import datestr
from .lib.signal_reader import read_signal
from .lib.main_calculator import MainCalculator
from .config import parse_config
from .preprocess import preprocess_from_dicom  # NEW

def _resolve_inputs_from_config(config):
    """
    Handle two modes:
      A) --image-directory provided -> run preprocess to get image/mask/vessel paths
      B) --image-path provided -> use given paths (mask/vessel may be required externally)
    Returns (image_path, mask_path, vessel_path) as strings.
    """
    if config.image_directory:
        outputs = preprocess_from_dicom(
            image_directory=config.image_directory,
            save_path=config.save_path,
            overwrite=True,
        )
        return outputs["image_path"], outputs["mask_path"], outputs["vessel_path"]

    # else: classic mode
    missing = [k for k in ("mask_path", "vessel_path") if getattr(config, k) in (None, "")]
    if missing:
        raise ValueError(
            f"Missing required arguments in classic mode: {', '.join(missing)}. "
            "Either provide --image-directory OR provide --image-path, --mask-path, and --vessel-path."
        )
    return config.image_path, config.mask_path, config.vessel_path

def main():
    """
    Application entrypoint.

    Parse arguments, (optionally) preprocess DICOMs, then compute perfusion parameters.
    """
    print(datestr())
    config = parse_config()

    device = torch.device("cpu")

    # Resolve file paths either via preprocess (DICOM) or as provided (NIfTI)
    image_path, mask_path, vessel_path = _resolve_inputs_from_config(config)

    # Read and preprocess (brain-region extraction, low-pass filtering)
    # raw signal (size: (slice, row, column, time))
    RawPerfImg, mask, vessels, origin, spacing, direction = read_signal(
        image_path,
        mask_path,
        vessel_path,
    )

    # Calculate perfusion parameters
    calculator = MainCalculator(
        RawPerfImg,
        mask,
        vessels,
        origin,
        spacing,
        direction,
        config,
        config.save_path,
        device,
    )
    calculator.run()

if __name__ == "__main__":
    main()
