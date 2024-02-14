#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import pytest
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, ValidationError

from geoapps_utils.driver.data import BaseData

WORKSPACE = Workspace()
VALID_PARAMETERS = {
    "monitoring_directory": None,
    "workspace_geoh5": WORKSPACE,
    "geoh5": WORKSPACE,
    "run_command": "test.driver",
    "title": "test title",
    "conda_environment": "test_env",
}


def test_dataclass_valid_values():
    model = BaseData(**VALID_PARAMETERS)
    output_params = {**model.model_dump()}

    for k, v in output_params.items():
        assert output_params[k] == v

    assert len(output_params) == len(VALID_PARAMETERS) + 1

    for k, v in VALID_PARAMETERS.items():
        assert output_params[k] == v


def test_dataclass_invalid_values(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")

    invalid_params = {
        "monitoring_directory": 5,
        "workspace_geoh5": workspace.h5file,
        "geoh5": False,
        "run_command": "test.driver",
        "title": None,
        "conda_environment": "test_env",
        "workspace": workspace.h5file,
    }

    with pytest.raises(ValidationError) as e:
        BaseData(**invalid_params)
        assert len(e.errors()) == 6
        error_params = [error["loc"][0] for error in e.errors()]
        error_types = [error["type"] for error in e.errors()]
        for error_param in [
            "monitoring_directory",
            "geoh5",
            "title",
            "conda_environment_boolean",
        ]:
            assert error_param in error_params
        for error_type in ["string_type", "path_type", "missing"]:
            assert error_type in error_types


def test_dataclass_input_file():
    ifile = InputFile(ui_json=VALID_PARAMETERS)
    model = BaseData.build(ifile)

    assert model.geoh5 == WORKSPACE
    assert model.flatten() == VALID_PARAMETERS
    assert model.input_file == ifile


def test_nested_model():
    class GroupParams(BaseModel):
        value: str

    class NestedModel(BaseData):
        """
        Example of nested model
        """

        _name = "nested"
        group: GroupParams

        @classmethod
        def _parse_input(cls, input_data):
            """
            Parse input parameter into dicts for nested models.
            """
            input_data["group"] = {"value": input_data.pop("value")}
            return input_data

    valid_params = VALID_PARAMETERS.copy()
    valid_params["value"] = "test"

    ifile = InputFile(ui_json=valid_params)
    model = NestedModel.build(ifile)

    assert isinstance(model.group, GroupParams)
    assert model.group.value == "test"
    assert model.flatten() == valid_params

    assert model.name == "nested"
