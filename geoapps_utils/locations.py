#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from uuid import UUID

from geoh5py import Workspace
from geoh5py.data import Data
from geoh5py.objects import Points
from geoh5py.objects.grid_object import GridObject


def get_locations(workspace: Workspace, entity: UUID | Points | GridObject | Data):
    """
    Returns entity's centroids or vertices.

    If no location data is found on the provided entity, the method will
    attempt to call itself on its parent.

    :param workspace: Geoh5py Workspace entity.
    :param entity: Object or uuid of entity containing centroid or
        vertex location data.

    :return: Array shape(*, 3) of x, y, z location data

    """
    if isinstance(entity, UUID):
        entity_obj = workspace.get_entity(entity)[0]
    else:
        entity_obj = entity

    if not isinstance(entity_obj, (Points, GridObject, Data)):
        raise TypeError(
            f"Entity must be of type Points, GridObject or Data, {type(entity_obj)} provided."
        )

    if isinstance(entity_obj, Points):
        locations = entity_obj.vertices
    elif isinstance(entity_obj, GridObject):
        locations = entity_obj.vertices
    else:
        locations = get_locations(workspace, entity_obj.parent)

    return locations
