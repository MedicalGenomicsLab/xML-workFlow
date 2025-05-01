# --- 100 characters ------------------------------------------------------------------------------
# Created by: Khoa Tran | 2025-04-23 | FileIO utilities
# --------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List


def check_file_path(path: str | Path, suffixes: List | None) -> str:
    """Check if the given path is a valid file
    if suffixes is provided, check if the file ends with one of the suffixes.

    Args:
        - value (str):      The value to check.
        - suffixex (List):  The suffix to check for.

    Returns:
        - str: The value if it ends with the specified suffix.

    Raises:
        - argparse.ArgumentTypeError: If the value does not end with the specified suffix.
    """
    # If path is str, convert to Path
    if not isinstance(path, Path):
        path = Path(path)

    # Check if path exists
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path {path} does not exist")

    # Check if path is a file
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Path {path} is not a file")

    # Check if path has the correct suffix
    if suffixes is None:
        return path
    elif isinstance(suffixes, str):
        # Check if path has the correct suffix
        if not path.endswith(suffixes):
            raise argparse.ArgumentTypeError(
                f"Argument must end with one of the following suffixes: {suffixes}"
            )

        return path
    elif isinstance(suffixes, Path):
        if not suffixes:
            raise argparse.ArgumentTypeError("Suffixes list is empty")
        if not isinstance(suffixes, list):
            raise argparse.ArgumentTypeError("Suffixes must be a list")
        if not all(isinstance(suffix, str) for suffix in suffixes):
            raise argparse.ArgumentTypeError("All suffixes must be strings")

        # Check if path has the correct suffix
        if not path.suffix in suffixes:
            raise argparse.ArgumentTypeError(
                f"Argument must end with one of the following suffixes: {', '.join(suffixes)}"
            )

        return path
    else:
        raise argparse.ArgumentTypeError(
            f"Argument must be a string or Path. Got {type(path)}"
        )
