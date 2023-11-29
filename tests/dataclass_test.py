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

    data = BaseData(**valid_params)
    output_params = {
        **data.workspace_data.model_dump(),
        **data.ui_json_data.model_dump(),
    }

    assert len(set(valid_params) & set(output_params)) == len(valid_params)

    for k, v in valid_params.items():
        assert output_params[k] == v

    # Just workspace params because it will fail before validating ui_json params
    invalid_params = {
        "monitoring_directory": 5,
        "workspace_geoh5": [1, 2],
        "geoh5": None,
        "workspace": None,
    }

    try:
        BaseData(**invalid_params)
        pytest.fail()
    except ValidationError as e:
        assert len(e.errors()) == 4
        error_params = [error["loc"][0] for error in e.errors()]
        error_types = [error["type"] for error in e.errors()]
        assert "monitoring_directory" in error_params
        assert "workspace_geoh5" in error_params
        assert "string_type" in error_types
        assert "path_type" in error_types
