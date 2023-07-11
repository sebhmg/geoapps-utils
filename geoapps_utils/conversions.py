#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import numpy as np


def hex_to_rgb(hex_color):
    """
    Convert hex color code to RGB
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def string_to_numeric(text: str) -> int | float | str:
    """Converts numeric string representation to int or string if possible."""
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return np.nan if text == "nan" else text
