#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  All rights reserved.
#

from __future__ import annotations

import pytest
from geoh5py.workspace import Workspace
from pydantic import ValidationError

from geoapps_utils.driver.data import BaseData


def test_dataclass(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")

    valid_params = {
        "monitoring_directory": workspace.h5file,
        "workspace_geoh5": workspace.h5file,
        "geoh5": workspace.h5file,
        "run_command": "test.driver",
        "title": "test title",
        "conda_environment": "test_env",
        "conda_environment_boolean": True,
        "generate_sweep": True,
        "workspace": workspace,
        "run_command_boolean": False,
    }

    try:
        model = BaseData.model_construct(**valid_params)
        model.model_validate(valid_params)
    except ValidationError:
        pytest.fail()

    output_params = {**model.model_dump()}

    for k, v in output_params.items():
        assert output_params[k] == v

    assert len(output_params) == len(valid_params)

    for k, v in valid_params.items():
        assert output_params[k] == v

    invalid_params = {
        "monitoring_directory": 5,
        "workspace_geoh5": workspace.h5file,
        "geoh5": False,
        "run_command": "test.driver",
        "title": None,
        "conda_environment": "test_env",
        "conda_environment_boolean": True,
        "generate_sweep": True,
        "workspace": workspace,
        "run_command_boolean": False,
    }

    try:
        model = BaseData.model_construct(**invalid_params)
        model.model_validate(invalid_params)
        pytest.fail()
    except ValidationError as e:
        assert len(e.errors()) == 4
        error_params = [error["loc"][0] for error in e.errors()]
        error_types = [error["type"] for error in e.errors()]
        assert "monitoring_directory" in error_params
        assert "geoh5" in error_params
        assert "string_type" in error_types
        assert "path_type" in error_types
