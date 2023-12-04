#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from uuid import UUID

import numpy as np
from geoh5py import Workspace
from geoh5py.shared import Entity
from scipy.interpolate import interp1d


def get_locations(workspace: Workspace, entity: UUID | Entity):
    """
    Returns entity's centroids or vertices.

    If no location data is found on the provided entity, the method will
    attempt to call itself on its parent.

    :param workspace: Geoh5py Workspace entity.
    :param entity: Object or uuid of entity containing centroid or
        vertex location data.

    :return: Array shape(*, 3) of x, y, z location data

    """
    locations = None

    if isinstance(entity, UUID):
        entity_obj = workspace.get_entity(entity)[0]
    elif isinstance(entity, Entity):
        entity_obj = entity
    if entity_obj is None:
        return None

    if hasattr(entity_obj, "centroids"):
        locations = entity_obj.centroids
    elif hasattr(entity_obj, "vertices"):
        locations = entity_obj.vertices
    elif (
        getattr(entity_obj, "parent", None) is not None
        and entity_obj.parent is not None
    ):
        locations = get_locations(workspace, entity_obj.parent)

    return locations


def resample_locations(locations: np.ndarray, increment: float) -> np.ndarray:
    """
    Resample locations along a sequence of positions at a given increment.

    :param locations: Array of shape (n, 3) of x, y, z locations.
    :param increment: Minimum distance between points along the curve.

    :return: Array of shape (n, 3) of x, y, z locations.
    """
    distance = np.cumsum(
        np.r_[0, np.linalg.norm(locations[1:, :] - locations[:-1, :], axis=1)]
    )
    new_distances = np.sort(
        np.unique(np.r_[distance, np.arange(0, distance[-1], increment)])
    )

    resampled = []
    for axis in locations.T:
        interpolator = interp1d(distance, axis, kind="linear")
        resampled.append(interpolator(new_distances))

    return np.c_[resampled].T
