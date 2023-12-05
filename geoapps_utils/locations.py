#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from uuid import UUID

from geoh5py import Workspace
from geoh5py.shared import Entity


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
    entity_obj = entity
    if isinstance(entity, UUID):
        obj = workspace.get_entity(entity)[0]
        if isinstance(obj, Entity):
            entity_obj = obj

    if hasattr(entity_obj, "centroids"):
        locations = entity_obj.centroids
    elif hasattr(entity_obj, "vertices"):
        locations = entity_obj.vertices
    elif (
        isinstance(entity_obj, Entity)
        and getattr(entity_obj, "parent", None) is not None
        and entity_obj.parent is not None
    ):
        locations = get_locations(workspace, entity_obj.parent)

    return locations
