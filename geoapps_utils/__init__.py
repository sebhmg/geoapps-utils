#  Copyright (c) 2022-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

__version__ = "0.4.0-alpha.1"

from pathlib import Path

# from .importing import assets_path as assets_path_impl


def assets_path() -> Path:
    """Return the path to the assets folder."""
    # return assets_path_impl(__file__)
    #
    parent = Path(__file__).parent
    folder_name = f"{parent.name}-assets"
    assets_folder = parent.parent / folder_name
    if not assets_folder.is_dir():
        raise RuntimeError(f"Assets folder not found: {assets_folder}")

    return assets_folder


__all__ = ["assets_path"]
