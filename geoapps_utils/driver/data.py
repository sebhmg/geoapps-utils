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
from pydantic.dataclasses import dataclass


@dataclass
class BaseData(
    BaseModel,
    arbitrary_types_allowed=True,  # type: ignore
    extra="ignore",  # type: ignore
):
    """
    Core parameters for all apps.

    :param monitoring_directory: Path to monitoring directory.
    :param workspace_geoh5: Source geoh5 file.
    :param geoh5: Workspace path.
    :param workspace: Workspace.
    :param title: Application title.
    :param run_command: Command to run the application through GA.
    :param run_command_boolean: Boolean to determine if run command is used.
    :param conda_environment: Conda environment.
    :param conda_environment_boolean: Boolean to determine if conda environment is used.
    :param generate_sweep: Boolean to determine whether to run param sweep.
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
    generate_sweep: bool
