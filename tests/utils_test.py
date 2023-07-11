#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps_utils package.
#
#  All rights reserved.
#
# pylint: disable=import-outside-toplevel
import os
import random
from pathlib import Path

import geoh5py
import numpy as np
import pytest
from geoh5py import Workspace
from geoh5py.objects import Grid2D

from geoapps_utils.conversions import string_to_numeric
from geoapps_utils.formatters import string_name
from geoapps_utils.importing import warn_module_not_found
from geoapps_utils.iterables import (
    find_value,
    sorted_alphanumeric_list,
    sorted_children_dict,
)
from geoapps_utils.numerical import running_mean


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
