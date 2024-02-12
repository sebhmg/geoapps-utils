#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from pathlib import Path
from typing import Optional, Union

from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, ConfigDict


class BaseData(BaseModel):
    """
    Core parameters expected by the ui.json file format.

    :param conda_environment: Environment used to run run_command.
    :param geoh5: Current workspace path.
    :param monitoring_directory: Path to monitoring directory, where .geoh5 files
        are automatically processed by GA.
    :param run_command: Command to run the application through GA.
    :param title: Application title.
    :param workspace_geoh5: Current workspace, where results will be exported.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _input_file: InputFile
    _name: str = "base"

    conda_environment: Optional[str] = None
    geoh5: Workspace
    monitoring_directory: Optional[Union[str, Path]] = None
    run_command: str
    title: str
    workspace_geoh5: Optional[Workspace] = None

    @classmethod
    def _parse_input(cls, input_data: Union[InputFile, dict]) -> dict:
        """
        Parse input parameter and values from ui.json data.

        :param input_data: Dictionary of parameters and values.

        :return: Dataclass of application parameters.
        """
        if isinstance(input_data, InputFile) and input_data.data is not None:
            data = input_data.data
            data["_input_file"] = input_data
        elif isinstance(input_data, dict):
            data = input_data
        else:
            raise TypeError("Input data must be a dictionary or InputFile.")

        return data

    def flatten(self) -> dict:
        """
        Flatten the parameters to a dictionary.

        :return: Dictionary of parameters.
        """
        param_dict = dict(self)
        out_dict = {}
        for key, value in param_dict.items():
            if isinstance(value, BaseModel):
                out_dict.update(dict(value))
            else:
                out_dict.update({key: value})

        return out_dict

    @property
    def input_file(self) -> InputFile:
        """Application input file."""
        return self._input_file

    @property
    def name(self) -> str:
        """Application name."""
        return self._name
