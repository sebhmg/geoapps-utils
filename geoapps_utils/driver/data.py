#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from pathlib import Path
from typing import Any, Optional, Union

from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self


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

    _name: str = "base"

    input_file: Optional[InputFile] = None
    conda_environment: Optional[str] = None
    geoh5: Workspace
    monitoring_directory: Optional[Union[str, Path]] = None
    run_command: str
    title: str
    workspace_geoh5: Optional[Workspace] = None

    @classmethod
    def _parse_input(cls, input_data: dict[str, Any]) -> dict[str, Union[dict, Any]]:
        """
        Parse input parameter into dicts for nested models.
        """
        for field, info in cls.model_fields.items():
            if isinstance(info.annotation, type) and issubclass(
                info.annotation, BaseModel
            ):
                field_data = {}
                for sub_field in info.annotation.model_fields:
                    if sub_field in input_data:
                        field_data.update({sub_field: input_data.pop(sub_field)})

                if field in input_data:
                    raise ValueError(
                        f"Field {field} defines both a nested model and a value."
                    )

                input_data.update({field: field_data})

        return input_data

    @classmethod
    def build(cls, input_data: Union[InputFile, dict]) -> Self:
        """
        Build a dataclass from a dictionary or InputFile.

        :param input_data: Dictionary of parameters and values.

        :return: Dataclass of application parameters.
        """

        data = input_data

        if isinstance(input_data, InputFile) and input_data.data is not None:
            data = input_data.data.copy()
            data["input_file"] = input_data

        if not isinstance(data, dict):
            raise TypeError("Input data must be a dictionary or InputFile.")

        nested_data = cls._parse_input(data.copy())

        return cls(**nested_data)

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

        out_dict.pop("input_file", None)

        return out_dict

    @property
    def name(self) -> str:
        """Application name."""
        return self._name
