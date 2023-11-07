#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  All rights reserved.

# pylint: disable=import-outside-toplevel
import os
import random
from pathlib import Path

import geoh5py
import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.objects import Grid2D, Points

from geoapps_utils.conversions import string_to_numeric
from geoapps_utils.formatters import string_name
from geoapps_utils.importing import warn_module_not_found
from geoapps_utils.iterables import (
    find_value,
    sorted_alphanumeric_list,
    sorted_children_dict,
)
from geoapps_utils.numerical import find_curves, running_mean
from geoapps_utils.plotting import inv_symlog, symlog


def test_find_curves(tmp_path: Path):  # pylint: disable=too-many-locals
    # Create test data
    # Survey lines
    y_array = np.linspace(0, 100, 20)
    line_ids_array = np.arange(0, len(y_array))

    curve1 = 5 * np.sin(y_array) + 10  # curve
    curve2 = 0.7 * y_array + 20  # crossing lines
    curve3 = -0.4 * y_array + 50
    curve4 = [80] * len(y_array)  # zig-zag
    curve4[5:10] = [90, 80, 70, 80, 90]
    curve5 = [None] * (len(y_array) - 1)  # short line
    curve5[0:1] = [60, 62]  # type: ignore

    curves = [curve1, curve2, curve3, curve4, curve5]

    points_data = []
    line_ids = []
    channel_groups = []
    for channel_group, curve in enumerate(curves):
        for x_coord, y_coord, line_id in zip(curve, y_array, line_ids_array):
            if x_coord is not None:
                points_data.append([x_coord, y_coord, 0])
                line_ids.append(line_id)
                channel_groups.append(channel_group)

    workspace = Workspace.create(tmp_path / "testFindCurves.geoh5")
    with workspace.open(mode="r+"):
        points = Points.create(workspace, vertices=np.array(points_data), name="Points")
        points.add_data({"line_ids": {"values": np.asarray(line_ids, dtype=np.int32)}})
        points.add_data({"channel_group": {"values": np.array(channel_groups)}})
    points = workspace.get_entity("Points")[0]

    result_curves = find_curves(
        points,
        min_length=3,
        max_distance=500,
        min_angle=0,
    )
    assert len(result_curves) == 4

    ind = 0
    for curve in result_curves:
        curve_length = len(curve)
        for i in range(curve_length):
            assert curve[i][0] == points_data[ind + i][0:1]
        ind += curve_length

    result_curves = find_curves(
        points,
        min_length=3,
        max_distance=500,
        min_angle=3 * np.pi / 4,
    )
    assert [len(curve) for curve in result_curves] == [20, 20, 12, 5, 4, 4, 3, 3, 10]


def test_find_value():
    labels = ["inversion_01_model", "inversion_01_data", "inversion_02_model"]
    assert find_value(labels, ["data"]) == "inversion_01_data"
    assert find_value(labels, ["inversion", "02"]) == "inversion_02_model"
    assert find_value(labels, ["inversion"]) == "inversion_02_model"
    assert find_value(labels, ["lskdfjsd"]) is None
    labels = [["inversion_01_model", 1], ["inversion_01_data", 2]]
    assert find_value(labels, ["model"]) == 1
    assert find_value(labels, ["data"]) == 2
    assert find_value(labels, ["lskdjf"]) is None


def test_no_warn_module_not_found(recwarn):
    with warn_module_not_found():
        import os as test_import  # pylint: disable=W0404

    assert test_import == os

    with warn_module_not_found():
        from os import system as test_import_from
    assert test_import_from == os.system

    with warn_module_not_found():
        import geoh5py.objects as test_import_submodule  # pylint: disable=W0404
    assert test_import_submodule == geoh5py.objects

    with warn_module_not_found():
        from geoh5py.objects import ObjectBase as test_import_from_submodule
    assert test_import_from_submodule == geoh5py.objects.ObjectBase

    assert len(recwarn) == 0


def test_plotting_symlog():
    thresh = 1.0
    vals = np.logspace(-6, 6, 13)
    symlog_vals = symlog(vals, thresh)
    inv_symlog_vals = inv_symlog(symlog_vals, thresh)
    thresh_vals = symlog_vals[symlog_vals > thresh]

    assert np.allclose(vals, inv_symlog_vals)
    assert len(thresh_vals) == 6
    assert np.all(np.diff(thresh_vals) > 0.9)


def test_running_mean():
    vec = np.random.randn(100)
    mean_forw = running_mean(vec, method="forward")
    mean_back = running_mean(vec, method="backward")
    mean_cent = running_mean(vec, method="centered")

    mean_test = (vec[1:] + vec[:-1]) / 2

    assert (
        np.linalg.norm(mean_back[:-1] - mean_test) < 1e-12
    ), "Backward averaging does not match expected values."
    assert (
        np.linalg.norm(mean_forw[1:] - mean_test) < 1e-12
    ), "Forward averaging does not match expected values."
    assert (
        np.linalg.norm((mean_test[1:] + mean_test[:-1]) / 2 - mean_cent[1:-1]) < 1e-12
    ), "Centered averaging does not match expected values."


def test_sorted_alphanumeric_list():
    test = [
        "Iteration_3.2e-1_data",
        "Iteration_1_data",
        "Iteration_2_data",
        "Iteration_3_data",
        "Iteration_5.11_data",
        "Iteration_5.2_data",
        "Iteration_6_data",
        "Iteration_7_data",
        "Iteration_8e0_data",
        "Iteration_9.0_data",
        "Iteration_10_data",
        "Iteration_11_data",
        "Iteration_2_model",
        "Iteration_12_model",
        "interp_01",
        "interp_02",
        "interp_11",
        "iteration_2_model",
        "iteration_12_model",
        "topo",
        "uncert",
    ]

    sorted_list = sorted_alphanumeric_list(random.sample(test, len(test)))
    assert all(elem == tester for elem, tester in zip(sorted_list, test))


def test_sorted_children_dict(tmp_path: Path):
    workspace = Workspace(tmp_path / "test.geoh5")
    n_x, n_y = 10, 15
    grid = Grid2D.create(
        workspace,
        origin=[0, 0, 0],
        u_cell_size=20.0,
        v_cell_size=30.0,
        u_count=n_x,
        v_count=n_y,
        name="test_grid",
        allow_move=False,
    )

    grid.add_data({"Iteration_10_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_1_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_5_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_3_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_2_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_4_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_9.0_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_8e0_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_11_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_6_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_7_data": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_02": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_01": {"values": np.ones(10 * 15)}})
    grid.add_data({"interp_11": {"values": np.ones(10 * 15)}})
    grid.add_data({"iteration_2_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"iteration_12_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_2_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"Iteration_12_model": {"values": np.ones(10 * 15)}})
    grid.add_data({"topo": {"values": np.ones(10 * 15)}})
    grid.add_data({"uncert": {"values": np.ones(10 * 15)}})

    data_dict = sorted_children_dict(grid)
    assert data_dict is not None
    data_keys = list(data_dict.keys())
    assert data_keys[0] == "Iteration_1_data"
    assert data_keys[1] == "Iteration_2_data"
    assert data_keys[7] == "Iteration_8e0_data"
    assert data_keys[8] == "Iteration_9.0_data"
    assert data_keys[-2] == "topo"
    assert data_keys[-1] == "uncert"


def test_string_to_numeric():
    assert string_to_numeric("test") == "test"
    assert string_to_numeric("2.1") == 2.1
    assert string_to_numeric("34") == 34
    assert string_to_numeric("1e-2") == 0.01
    assert string_to_numeric("1.05e2") == 105


def test_string_name():
    chars = "!@#$%^&*().,"
    value = "H!e(l@l#o.W$o%r^l&d*"
    assert (
        string_name(value, characters=chars) == "H_e_l_l_o_W_o_r_l_d_"
    ), "string_name validator failed"


def test_warn_module_not_found():
    # pylint: disable=import-error
    # pylint: disable=no-name-in-module

    def noop(_):
        return None

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import nonexisting as test_import
    with pytest.raises(NameError):
        noop(test_import)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from nonexisting import nope as test_import_from
    with pytest.raises(NameError):
        noop(test_import_from)

    with pytest.warns(match="Module 'os.nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import os.nonexisting as test_import_os_submodule
    with pytest.raises(NameError):
        noop(test_import_os_submodule)

    with pytest.warns(match="Module 'os.nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from os.nonexisting import nope as test_import_from_os_submodule
    with pytest.raises(NameError):
        noop(test_import_from_os_submodule)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            import nonexisting.nope as test_import_nonexising_submodule
    with pytest.raises(NameError):
        noop(test_import_nonexising_submodule)

    with pytest.warns(match="Module 'nonexisting' is missing from the environment."):
        with warn_module_not_found():
            from nonexisting.nope import nada as test_import_from_nonexisting_submodule
    with pytest.raises(NameError):
        noop(test_import_from_nonexisting_submodule)
