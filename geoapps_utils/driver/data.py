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
from pydantic import BaseModel, ConfigDict, PydanticUndefinedAnnotation
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

    @staticmethod
    def model_to_dict(
        base_model: BaseModel, data: dict[str, Any]
    ) -> dict[str, Union[dict, Any]]:
        """
        Recursively replace BaseModel objects with dictionary of 'data' values.

        :param base_model: BaseModel object holding data and possibly other nested
            BaseModel objects.
        :param data: Dictionary of parameters and values without nesting structure.
        """
        update = {}
        for field, info in base_model.model_fields.items():
            if isinstance(info.annotation, type) and issubclass(
                info.annotation, BaseModel
            ):
                update[field] = BaseData.model_to_dict(info.annotation, data)
            else:
                if field in data:
                    update[field] = data.get(field, info.default)

        return update

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

        kwargs = BaseData.model_to_dict(cls, data)

        return cls(**kwargs)

    def _recursive_flatten(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively flatten nested dictionary.

        To be used on output of BaseModel.model_dump.

        :param data: Dictionary of parameters and values.
        """
        out_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                out_dict.update(self._recursive_flatten(value))
            else:
                out_dict.update({key: value})

        return out_dict

    def flatten(self) -> dict:
        """
        Flatten the parameters to a dictionary.

        :return: Dictionary of parameters.
        """
        out = self._recursive_flatten(self.model_dump())
        out.pop("input_file", None)

        return out

    @property
    def name(self) -> str:
        """Application name."""
        return self._name
