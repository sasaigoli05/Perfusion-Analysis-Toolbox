#!/usr/bin/env python3

"""
Simplified command-line utility to compute perfusion parameters
using ONLY direct file paths to image, mask, vessel mask, and output.
"""

from .lib.utils import datestr
from .lib.signal_reader import read_signal
from .lib.main_calculator import MainCalculator
from .config import parse_config


def main():
    """
    Application entrypoint.

    Required:
        --image-path    → 4D CTP NIfTI
        --mask-path     → Brain mask NIfTI
        --vessel-path   → Vessel mask NIfTI
        --save-path     → Output folder
    """
    print(datestr())
    config = parse_config()

    # --- REQUIRED INPUTS ---
    if not config.image_path:
        raise ValueError("ERROR: --image-path is required.")

    if not config.mask_path:
        raise ValueError("ERROR: --mask-path is required.")

    if not config.vessel_path:
        raise ValueError("ERROR: --vessel-path is required.")

    if not config.save_path:
        raise ValueError("ERROR: --save-path is required.")

    image_path = config.image_path
    mask_path = config.mask_path
    vessel_path = config.vessel_path

    # --- LOAD RAW SIGNAL + MASKS ---
    RawPerfImg, mask, vessels, origin, spacing, direction = read_signal(
        image_path,
        mask_path,
        vessel_path,
    )

    # --- RUN PERFUSION CALCULATIONS ---
    calculator = MainCalculator(
        raw_perf=RawPerfImg,
        mask=mask,
        vessels=vessels,
        origin=origin,
        spacing=spacing,
        direction=direction,
        config=config,
        save_path=config.save_path,
    )

    calculator.run()


if __name__ == "__main__":
    main()
