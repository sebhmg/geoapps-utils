#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations

import numpy as np
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


def filter_curves(curves, min_length, min_angle):
    """ """
    filtered_curves = []
    sub_curves = []
    for curve in curves:
        # Check that the curve is long enough
        if len(curve) >= min_length:
            # Check that the angles are not too sharp
            for i in range(len(curve) - 2):
                # v1 = [curve[i + 1][0] - curve[i][0], curve[i + 1][1] - curve[i][1]]
                v1 = [curve[i][0] - curve[i + 1][0], curve[i][1] - curve[i + 1][1]]
                v2 = [
                    curve[i + 2][0] - curve[i + 1][0],
                    curve[i + 2][1] - curve[i + 1][1],
                ]

                # angle = np.arccos((np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

                angle = 2 * np.pi - (
                    np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                )

                left_bound = min_angle
                right_bound = 2 * np.pi - min_angle
                if left_bound <= angle <= right_bound:
                    filtered_curves.append(curve)
                else:
                    if i + 2 >= min_length:
                        filtered_curves.append(curve[: i + 2])
                    if len(curve) - (i + 2) >= min_length:
                        sub_curves.append(curve[i + 1 :])
    if len(sub_curves) > 0:
        filtered_curves += filter_curves(sub_curves, min_length, min_angle)
    return filtered_curves


def find_curves(survey, points, min_length, max_distance, min_angle):
    """ """
    # min length of chain
    # max distance between anomalies
    # 2 with same point as closest distance
    # need to know end points from previous line

    survey_vertices = survey.vertices
    survey_vertices = np.delete(survey_vertices, 2, axis=1)
    survey_lines = survey.get_data("Line")[0].values

    point_vertices = points.vertices
    point_vertices = np.delete(point_vertices, 2, axis=1)
    channels = points.get_data("channel_group")[0].values

    dist_matrix = cdist(point_vertices, survey_vertices)
    # Closest points
    inds = np.argmin(dist_matrix, axis=1)
    # Line ids
    line_ids = survey_lines[inds]

    unique_line_ids = np.unique(line_ids)

    curves = []
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
