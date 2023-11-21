#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from geoh5py.objects import Octree


def hex_to_rgb(hex_color: str) -> list[int]:
    """
    Convert hex color code to RGB

    :param hex_color: Hex color code.

    :return rgb: RGB color code.
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]


def string_to_numeric(text: str) -> int | float | str:
    """
    Converts numeric string representation to int, float, or nan if possible.

    :param text: String to be converted.

    :return: Int, float, or nan if possible, otherwise returns the original string.
    """
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return np.nan if text == "nan" else text


def treemesh_2_octree(workspace, treemesh, **kwargs):
    index_array, levels = getattr(treemesh, "_ubc_indArr")
    ubc_order = getattr(treemesh, "_ubc_order")

    index_array = index_array[ubc_order] - 1
    levels = levels[ubc_order]

    origin = treemesh.x0.copy()
    origin[2] += treemesh.h[2].size * treemesh.h[2][0]
    mesh_object = Octree.create(
        workspace,
        origin=origin,
        u_count=treemesh.h[0].size,
        v_count=treemesh.h[1].size,
        w_count=treemesh.h[2].size,
        u_cell_size=treemesh.h[0][0],
        v_cell_size=treemesh.h[1][0],
        w_cell_size=-treemesh.h[2][0],
        octree_cells=np.c_[index_array, levels],
        **kwargs,
    )

    return mesh_object
