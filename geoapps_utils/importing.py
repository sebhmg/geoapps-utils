#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def warn_module_not_found():
    """
    Context manager to suppress ModuleNotFoundError exceptions, and warn instead.

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with warn_module_not_found():
             from ipywidgets import Widget
         # Execution still resumes here if ipywidget is not found
    """
    try:
        yield
    except ModuleNotFoundError as error:
        module_name = error.name
        err = (
            f"Module '{module_name}' is missing from the environment. "
            f"Consider installing with: 'conda install -c conda-forge {module_name}'"
        )
        warnings.warn(err)


def assets_path(file: str) -> Path:
    """
    Return the path to the assets folder.

    :param file: Pathname of file from which the app was loaded.
    """

    parent = Path(file).parent
    folder_name = f"{parent.name}-assets"
    assets_folder = parent.parent / folder_name
    if not assets_folder.is_dir():
        raise RuntimeError(f"Assets folder not found: {assets_folder}")

    return assets_folder
