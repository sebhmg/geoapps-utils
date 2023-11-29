#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  All rights reserved.
#

#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path

from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, Extra, ValidationError
from pydantic.dataclasses import dataclass


class BaseData:  # pylint: disable=too-few-public-methods
    """
    Core parameters for all apps.
    """

    _defaults: dict | None = None
    _default_ui_json: dict | None = None
    _free_parameter_keys: list[str] | None = None
    _free_parameter_identifier: str | None = None
    _input_file: InputFile | None = None
    _monitoring_directory: str | None = None
    _ui_json: dict | None = None

    def __init__(
        self,
        input_file: InputFile | None = None,
        workpath: str | Path | None = None,
        **kwargs,
    ):
        """
        Initialize class.

        :param input_file: Input file.
        :param workpath: Working directory.
        """
        self.workpath = workpath
        self.input_file = input_file
        kwargs.update({"workpath": workpath, "input_file": input_file})

        try:
            self.workspace_data = self.WorkspaceData.model_construct(**kwargs)
            self.workspace_data.model_validate(kwargs)
            self.ui_json_data = self.UIJsonData.model_construct(**kwargs)
            self.ui_json_data.model_validate(kwargs)
        except ValidationError as e:
            raise e

    @dataclass
    class WorkspaceData(
        BaseModel,
        arbitrary_types_allowed=True,  # type: ignore
        extra=Extra.ignore,  # type: ignore
    ):
        """
        Workspace related parameters.
        """

        monitoring_directory: str | Path | None = None
        workspace_geoh5: str | Path | None = None
        geoh5: str | Path | None = None
        workspace: str | Workspace | None = None
        workpath: str | Path | None = None

    @dataclass
    class UIJsonData(BaseModel, arbitrary_types_allowed=True, extra=Extra.ignore):  # type: ignore
        """
        Parameters used in the ui.json.
        """

        input_file: InputFile | None = None
        ui_json: dict | None = None
        title: str | None = None
        run_command: str | None = None
        run_command_boolean: bool = False
        conda_environment: str | None = None
        conda_environment_boolean: bool | None = None
        generate_sweep: bool = False
