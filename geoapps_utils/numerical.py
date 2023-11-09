#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


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
    min_angle: float,
) -> list[np.ndarray]:
    """
    Filter edges by angle and line_id.

    :param vertices: Vertices of points.
    :param edge: Edge to add to.
    :param connected_edges: Edges connected to starting edge.
    :param ids: Ids of points.
    :param min_angle: Minimum angle between points in a curve, in radians.

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

        if angle < min_angle:
            continue

        possible_edges.append(new_edge)

    return possible_edges


def combine_edges(
    current_edge: np.ndarray,
    full_edges: np.ndarray,
    vertices: np.ndarray,
    ids: np.ndarray,
    min_angle: float,
    min_length: int,
) -> list[np.ndarray]:
    """
    Combine edges into a single curve.

    :param current_edge: Starting curve to add an edge to.
    :param full_edges: All possible edges to add to current edge. Length 2 array.
    :param vertices: Vertices of points.
    :param ids: Ids of points.
    :param min_angle: Minimum angle between points in a curve, in radians.
    :param min_length: Minimum number of points in a curve.

    :return: List of complete possible curves.
    """
    return_edges = []
    # Find all edges connected to this edge
    edge_bounds = {current_edge[0], current_edge[-1]}
    inds = np.array([len(edge_bounds & set(edg)) == 1 for edg in full_edges])
    connected_edges = full_edges[inds]

    # Remove edges with duplicate ids and with invalid angles
    possible_edges = filter_edges(
        vertices, current_edge, connected_edges, ids, min_angle
    )

    if len(possible_edges) == 0 and len(current_edge) >= min_length:
        return current_edge
    for possible_edge in possible_edges:
        new_edges = combine_edges(
            possible_edge, full_edges, vertices, ids, min_angle, min_length
        )
        if len(new_edges) > 0 and isinstance(new_edges[0], np.intc):
            return_edges.append(possible_edge)
        else:
            return_edges += new_edges

    return return_edges


def find_curves(  # pylint: disable=too-many-locals
    vertices: np.ndarray,
    ids: np.ndarray,
    min_length: int,
    max_distance: float,
    min_angle: float,
) -> list[list[list[float]]]:
    """
    Find curves in a set of points.

    :param vertices: Vertices for points.
    :param ids: Ids for points.
    :param min_length: Minimum number of points in a curve.
    :param max_distance: Maximum distance between points in a curve.
    :param min_angle: Minimum angle between points in a curve, in radians.

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

    # Find all valid paths connecting the segments
    curves = []
    for edge in edges:
        new_curves = combine_edges(edge, edges, vertices, ids, min_angle, min_length)
        curves += new_curves

    # Remove duplicate curves
    out_curves: list[np.ndarray] = []
    for curve in curves:
        add_curve = True
        for out_curve in out_curves:
            if len(set(curve) & set(out_curve)) == len(curve):
                add_curve = False
                break
        if add_curve:
            out_curves.append(curve)

    out_curves = [vertices[curve] for curve in out_curves]

    return out_curves
