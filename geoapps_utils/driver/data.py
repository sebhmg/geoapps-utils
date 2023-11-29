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

from pathlib import Path
from typing import Union

from geoh5py.workspace import Workspace
from pydantic import BaseModel


class BaseData(
    BaseModel,
    arbitrary_types_allowed=True,  # type: ignore
    extra="ignore",  # type: ignore
):
    """
    Core parameters expected by the ui.json file format. 

    :param monitoring_directory: Path to monitoring directory, where .geoh5 files
        are automatically processed by GA.
    :param workspace_geoh5: Path of the source .geoh5 file where the ui.json was created.
    :param geoh5: Current workspace path.
    :param workspace: Current workspace, where results will be exported.
    :param title: Application title.
    :param run_command: Command to run the application through GA.
    :param run_command_boolean: Boolean to determine if run command is used.
    :param conda_environment: Conda environment used to run run_command.
    :param conda_environment_boolean: Boolean to determine if conda environment is used.
    """

    monitoring_directory: Union[str, Path]
    workspace_geoh5: Union[str, Path]
    geoh5: Union[str, Path]
    workspace: Union[str, Workspace]
    title: str
    run_command: str
    run_command_boolean: bool
    conda_environment: str
    conda_environment_boolean: bool
