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

from geoh5py.workspace import Workspace
from pydantic import BaseModel, Extra
from pydantic.dataclasses import dataclass


@dataclass
class BaseData(
    BaseModel,
    arbitrary_types_allowed=True,  # type: ignore
    extra=Extra.ignore,  # type: ignore
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

    monitoring_directory: str | Path | None = None
    workspace_geoh5: str | Path | None = None
    geoh5: str | Path | None = None
    workspace: str | Workspace | None = None
    title: str | None = None
    run_command: str | None = None
    run_command_boolean: bool | None = None
    conda_environment: str | None = None
    conda_environment_boolean: bool | None = None
    generate_sweep: bool | None = None
