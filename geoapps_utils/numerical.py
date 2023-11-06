#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import numpy as np
from geoh5py.objects import Points
from scipy.spatial.distance import cdist


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


def find_curves(  # pylint: disable=too-many-locals
    points: Points, min_length: int, max_distance: float, min_angle: float
) -> list[list[list[float]]]:
    """
    Find curves in a set of points.

    :param points: Points object to find curves in.
    :param min_length: Minimum number of points in a curve.
    :param max_distance: Maximum distance between points in a curve.
    :param min_angle: Minimum angle between points in a curve, in radians.

    :return: List of curves.
    """
    point_vertices = points.vertices
    point_vertices = np.delete(point_vertices, 2, axis=1)
    line_ids = points.get_data("line_ids")[0].values
    channels = points.get_data("channel_group")[0].values

    unique_line_ids = np.unique(line_ids)

    curves: list[list[list[float]]] = []
    for i, line_id in enumerate(unique_line_ids[:-1]):
        # Find adjacent lines
        current_line = point_vertices[line_ids == line_id]
        current_channels = channels[line_ids == line_id]

        adjacent_line = point_vertices[line_ids == unique_line_ids[i + 1]]
        adjacent_channels = channels[line_ids == unique_line_ids[i + 1]]

        end_points = [p[-1] for p in curves]
        for j, point in enumerate(current_line):
            compare_points = adjacent_line[adjacent_channels == current_channels[j]]
            if len(compare_points) == 0:
                continue
            dists = cdist([point], compare_points)

            closest_ind = np.argmin(dists, axis=1)

            # Filter points that are too far
            if dists[:, closest_ind] >= max_distance:
                continue

            closest_point = list(compare_points[closest_ind][0])
            point = list(point)

            if point in end_points:
                ind = end_points.index(point)
                curves[ind].append(closest_point)
            else:
                curves.append([point, closest_point])

    curves = filter_curves(curves, min_length, min_angle)
    return curves
