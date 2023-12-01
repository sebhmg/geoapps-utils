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
from geoh5py.objects import Curve
from geoh5py.shared import Entity
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay


def densify_curve(curve: Curve, increment: float) -> np.ndarray:
    """
    Refine a curve by adding points along the curve at a given increment.

    :param curve: Curve object to be refined.
    :param increment: Distance between points along the curve.

    :return: Array of shape (n, 3) of x, y, z locations.
    """
    locations = []
    for part in curve.unique_parts:
        logic = curve.parts == part
        if logic is not None and curve.cells is not None and curve.vertices is not None:
            cells = curve.cells[np.all(logic[curve.cells], axis=1)]
            vert_ind = np.r_[cells[:, 0], cells[-1, 1]]
            locs = curve.vertices[vert_ind, :]
            locations.append(resample_locations(locs, increment))

    return np.vstack(locations)


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


def running_mean(
    values: np.array, width: int = 1, method: str = "centered"
) -> np.array:
    """
    Compute a running mean of an array over a defined width.

    :param values: Input values to compute the running mean over
    :param width: Number of neighboring values to be used
    :param method: Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: Averaged array values of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padded = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padded)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padded = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padded)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean


def traveling_salesman(locs: np.ndarray) -> np.ndarray:
    """
    Finds the order of a roughly linear point set.
    Uses the point furthest from the mean location as the starting point.
    :param: locs: Cartesian coordinates of points lying either roughly within a plane or a line.
    :param: return_index: Return the indices of the end points in the original array.
    """
    mean = locs[:, :2].mean(axis=0)
    current = np.argmax(np.linalg.norm(locs[:, :2] - mean, axis=1))
    order = [current]
    mask = np.ones(locs.shape[0], dtype=bool)
    mask[current] = False

    for _ in range(locs.shape[0] - 1):
        remaining = np.where(mask)[0]
        ind = np.argmin(np.linalg.norm(locs[current, :2] - locs[remaining, :2], axis=1))
        current = remaining[ind]
        order.append(current)
        mask[current] = False

    return np.asarray(order)


def filter_edges(
    vertices: np.ndarray,
    edge: np.ndarray,
    connected_edges: np.ndarray,
    ids: np.ndarray,
    max_angle: float,
) -> list[np.ndarray]:
    """
    Filter edges by angle and line_id.

    :param vertices: Vertices of points.
    :param edge: Edge to add to.
    :param connected_edges: Edges connected to starting edge.
    :param ids: Ids of points.
    :param max_angle: Maximum angle between points in a curve, in radians.

    :return: List of possible partial curves.
    """
    possible_edges = []
    for connected_edge in connected_edges:
        new_point = list(set(connected_edge) - set(edge))
        if len(new_point) == 0:
            continue
        new_point = new_point[0]
        # Filter edges by line_id
        if ids[new_point] in ids[edge]:
            continue
        # Filter edges by angle
        # Determine which edge connects to new point
        connecting_point = connected_edge[connected_edge != new_point][0]

        if edge[0] == connecting_point:
            # Adding edge to start of curve
            vec1 = [
                vertices[new_point][0] - vertices[connecting_point][0],
                vertices[new_point][1] - vertices[connecting_point][1],
            ]
            vec2 = [
                vertices[edge[1]][0] - vertices[connecting_point][0],
                vertices[edge[1]][1] - vertices[connecting_point][1],
            ]
            new_edge = np.concatenate(([new_point], edge))

        elif edge[-1] == connecting_point:
            # Adding edge to end of curve
            vec1 = [
                vertices[edge[-2]][0] - vertices[connecting_point][0],
                vertices[edge[-2]][1] - vertices[connecting_point][1],
            ]
            vec2 = [
                vertices[new_point][0] - vertices[connecting_point][0],
                vertices[new_point][1] - vertices[connecting_point][1],
            ]
            new_edge = np.concatenate((edge, [new_point]))

        angle = np.arccos(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )

        if angle < max_angle:
            continue

        possible_edges.append(new_edge)

    return possible_edges


# def combine_edges(
#     current_edge: np.ndarray,
#     full_edges: np.ndarray,
#     vertices: np.ndarray,
#     ids: np.ndarray,
#     max_angle: float,
#     min_length: int,
# ) -> list[np.ndarray]:
#     """
#     Combine edges into a single curve.
#
#     :param current_edge: Starting curve to add an edge to.
#     :param full_edges: All possible edges to add to current edge. Length 2 array.
#     :param vertices: Vertices of points.
#     :param ids: Ids of points.
#     :param max_angle: Maximum angle between points in a curve, in radians.
#     :param min_length: Minimum number of points in a curve.
#
#     :return: List of complete possible curves.
#     """
#     return_edges = []
#     # Find all edges connected to this edge
#     edge_bounds = {current_edge[0], current_edge[-1]}
#     inds = np.array([len(edge_bounds & set(edg)) == 1 for edg in full_edges])
#     connected_edges = full_edges[inds]
#
#     # Remove edges with duplicate ids and with invalid angles
#     possible_edges = filter_edges(
#         vertices, current_edge, connected_edges, ids, max_angle
#     )
#
#     if len(possible_edges) == 0 and len(current_edge) >= min_length:
#         return current_edge
#     for possible_edge in possible_edges:
#         new_edges = combine_edges(
#             possible_edge, full_edges, vertices, ids, max_angle, min_length
#         )
#         if len(new_edges) > 0 and isinstance(new_edges[0], np.intc):
#             return_edges.append(possible_edge)
#         else:
#             return_edges += new_edges
#
#     return return_edges


def find_curves(  # pylint: disable=too-many-locals
    vertices: np.ndarray,
    ids: np.ndarray,
    min_edges: int,
    max_distance: float,
    max_angle: float,
) -> list[list[list[float]]]:
    """
    Find curves in a set of points.

    :param vertices: Vertices for points.
    :param ids: Ids for points.
    :param min_edges: Minimum number of points in a curve.
    :param max_distance: Maximum distance between points in a curve.
    :param max_angle: Maximum angle between points in a curve, in radians.

    :return: List of curves.
    """
    tri = Delaunay(vertices, qhull_options="QJ")
    if not hasattr(tri, "simplices"):
        return []

    edges = np.vstack(
        (
            tri.simplices[:, :2],  # pylint: disable=no-member
            tri.simplices[:, 1:],  # pylint: disable=no-member
            tri.simplices[:, ::2],  # pylint: disable=no-member
        )
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    distances = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    edges = edges[distances <= max_distance, :]

    # Check if both in columns have same id
    edge_ids = ids[edges]
    edges = edges[edge_ids[:, 0] != edge_ids[:, 1]]

    # Compute vectors for each edge
    vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]

    mask = np.ones(vertices.shape[0], dtype=bool)
    out_curves = []
    for ind in range(edges.shape[0]):
        if not np.any(mask[edges[ind]]):
            continue

        mask[edges[ind]] = False
        path = [edges[ind]]
        path, actives = walk_edges(path, ind, edges, vectors, max_angle, mask=mask)
        path, actives = walk_edges(
            path, ind, edges, vectors, max_angle, direction="backward", mask=mask
        )
        if len(path) < min_edges:
            continue

        out_curves.append(path)

    return out_curves


def walk_edges(path, ind, edges, vectors, max_angle, direction="forward", mask=None):
    """
    Find all edges connected to a point.

    :param path: Current list of edges forming a path.
    :param ind: Index of incoming edge.
    :param edges: All edges.
    :param vectors: Direction of the edges.
    :param max_angle: Maximum angle between points in a curve, in radians.
    :param direction: Direction to walk in.
    :param mask: Mask for nodes that have already been visited.

    :return: Edges connected to point.
    """
    node = 1 if direction == "forward" else 0

    if mask is None:
        mask = np.ones(edges.max() + 1, dtype=bool)
        mask[np.hstack(path).flatten()] = False

    neighbours = np.where(
        np.any(edges == edges[ind][node], axis=1) & np.any(mask[edges], axis=1)
    )[0]

    if len(neighbours) == 0:
        return path, mask

    dot = np.dot(vectors[ind], vectors[neighbours].T)
    vec_lengths = np.linalg.norm(vectors[neighbours], axis=1)
    angle = np.arccos(dot / (np.linalg.norm(vectors[ind]) * vec_lengths))

    # Filter large angles
    keep = angle < max_angle

    if not np.any(keep):
        return path, mask

    dot, neighbours, vec_lengths, angle = (
        dot[keep],
        neighbours[keep],
        vec_lengths[keep],
        angle[keep],
    )

    # Minimize the torque
    sub_ind = np.argmin(angle * vec_lengths)

    fork = neighbours[sub_ind]
    mask[edges[fork]] = False

    # Reverse the edge if necessary
    if dot[sub_ind] < 0:
        edges[fork] = edges[fork][::-1]

    path.append(edges[fork].tolist())
    path, mask = walk_edges(path, fork, edges, vectors, max_angle, mask=mask)

    return path, mask
