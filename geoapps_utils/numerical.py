#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import numpy as np
from geoh5py.objects import Points
from scipy.spatial.distance import cdist
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


def filter_curves(curves, min_length, min_angle) -> list[list[list[float]]]:
    """
    Filter curves based on length and angle.

    :param curves: List of curves.
    :param min_length: Minimum number of points in a curve.
    :param min_angle: Minimum angle between points in a curve, in radians.

    :return: List of filtered curves.
    """
    filtered_curves = []
    sub_curves = []
    for curve in curves:
        # Check that the curve is long enough
        if len(curve) >= min_length:
            # Check that the angles are not too sharp
            for i in range(len(curve) - 2):
                vec1 = [curve[i][0] - curve[i + 1][0], curve[i][1] - curve[i + 1][1]]
                vec2 = [
                    curve[i + 2][0] - curve[i + 1][0],
                    curve[i + 2][1] - curve[i + 1][1],
                ]

                angle = 2 * np.pi - (
                    np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
                )

                left_bound = min_angle
                right_bound = 2 * np.pi - min_angle

                if not left_bound <= angle <= right_bound:
                    if i + 2 >= min_length:
                        sub_curves.append(curve[: i + 2])
                    if len(curve) - (i + 2) >= min_length:
                        sub_curves.append(curve[i + 1 :])
                    break

                if i == (len(curve) - 3):
                    filtered_curves.append(curve)

    if len(sub_curves) > 0:
        filtered_curves += filter_curves(sub_curves, min_length, min_angle)
    return filtered_curves


def filter_edges(vertices, edge, connected_edges, ids, points_used, min_angle):
    possible_edges = []
    all_points_used = []
    for connected_edge in connected_edges:
        #new_point = list(set(connected_edge) - set(points_used))
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
            vec1 = [vertices[new_point][0] - vertices[connecting_point][0],
                    vertices[new_point][1] - vertices[connecting_point][1]]
            vec2 = [
                vertices[edge[1]][0] - vertices[connecting_point][0],
                vertices[edge[1]][1] - vertices[connecting_point][1],
            ]
            new_edge = np.concatenate(([new_point], edge))

        elif edge[-1] == connecting_point:
            vec1 = [vertices[edge[-2]][0] - vertices[connecting_point][0],
                     vertices[edge[-2]][1] - vertices[connecting_point][1]]
            vec2 = [
                vertices[new_point][0] - vertices[connecting_point][0],
                vertices[new_point][1] - vertices[connecting_point][1],
            ]
            new_edge = np.concatenate((edge, [new_point]))

        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        if angle < min_angle:
            continue

        #all_points_used.append(points_used + [new_point])
        possible_edges.append(new_edge)

    return possible_edges, all_points_used


def combine_edges(current_edge, full_edges, vertices, ids, min_angle, points_used, min_length):
    return_edges = []
    #current_points_used = points_used.copy()
    #current_points_used.extend(current_edge)
    #points_used = current_edge

    # Remove edges that have been used
    #full_edges = np.array([edge for edge in full_edges if len(set(edge) & set(points_used)) == 0])

    # Find all edges connected to this edge
    edge_bounds = {current_edge[0], current_edge[-1]}
    inds = np.array([len(edge_bounds & set(edg)) == 1 for edg in full_edges])
    connected_edges = full_edges[inds]

    # Remove edges with duplicate ids and with invalid angles
    #possible_edges, possible_points_used = filter_edges(vertices, current_edge, connected_edges, ids, current_points_used, min_angle)
    possible_edges, possible_points_used = filter_edges(vertices, current_edge, connected_edges, ids,
                                                        points_used, min_angle)

    if len(possible_edges) == 0 and len(current_edge) >= min_length:
        #return_edges.append(current_edge)
        return current_edge
    for ind, possible_edge in enumerate(possible_edges):
        #new_edges = combine_edges(possible_edge, full_edges, vertices, ids, min_angle, possible_points_used[ind], min_length)
        new_edges = combine_edges(possible_edge, full_edges, vertices, ids, min_angle, points_used,
                                  min_length)
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
    min_angle: float
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
    tri = Delaunay(vertices)

    edges = np.vstack((tri.simplices[:, :2], tri.simplices[:, 1:], tri.simplices[:, ::2]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    distances = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    edges = edges[distances <= max_distance, :]
    #distances = distances[distances <= max_distance]

    # Check if both in columns have same id
    edge_ids = ids[edges]
    #distances = distances[edge_ids[:, 0] != edge_ids[:, 1]]
    edges = edges[edge_ids[:, 0] != edge_ids[:, 1]]

    # Find all valid paths connecting the segments
    curves = []
    points_used = []
    for i, edge in enumerate(edges):
        #print(edge, vertices[edge])
        if i == 14:
            print("here")
        points_used = edge
        new_curves = combine_edges(edge, edges, vertices, ids, min_angle, points_used=points_used, min_length=min_length)
        curves += new_curves
        #points_used = list(set(points_used + np.unique(new_curves)))
        #points_used = list(np.unique(points_used.extend(new_curves)))
        """
        for curve in new_curves:
            for point in curve:
                if point not in points_used:
                    points_used.append(point)
        """

    # Remove duplicate curves
    out_curves = []
    for curve in curves:
        add_curve = True
        for out_curve in out_curves:
            if len(set(curve) & set(out_curve)) == len(curve):
                add_curve = False
                break
        if add_curve:
            out_curves.append(curve)

    curves = out_curves

    #"""
    import matplotlib.pyplot as plt

    plt.triplot(vertices[:, 0], vertices[:, 1], tri.simplices.copy())
    plt.plot(vertices[:, 0], vertices[:, 1], 'o')
    plt.show()
    #"""

    import plotly.graph_objects as go


    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=vertices[:, 0],
            y=vertices[:, 1],
            mode="markers",
        ),
    )
    #plt.scatter(vertices[:, 0], vertices[:, 1])
    for curve in curves:
        x = [vertices[point][0] for point in curve]
        y = [vertices[point][1] for point in curve]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
            ),
        )
        #plt.plot(x, y)
        #plt.plot(vertices[edge, 0], vertices[edge, 1])
    #plt.show()
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

    return curves
