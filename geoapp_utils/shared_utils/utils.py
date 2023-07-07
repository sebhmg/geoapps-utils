#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import re
from uuid import UUID

import numpy as np
from discretize import TensorMesh, TreeMesh
from geoh5py.objects import DrapeModel
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from scipy.spatial import cKDTree

from geoapps_utils.utils.string import string_to_numeric
from geoapps_utils.utils.surveys import compute_alongline_distance


def hex_to_rgb(hex_color):
    """
    Convert hex color code to RGB
    """
    code = hex_color.lstrip("#")
    return [int(code[i : i + 2], 16) for i in (0, 2, 4)]

