#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from geoh5py.shared import Entity
from geoh5py.shared.utils import fetch_active_workspace, str2uuid, uuid2entity
from geoh5py.ui_json import InputFile, InputValidation, utils
from geoh5py.workspace import Workspace


class BaseParams:  # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """
    Stores input parameters to drive a ui.json application.

    Methods
    -------
    is_uuid(p) :
        Returns True if string is valid uuid.
    parent(child_id) :
        Returns parent id for provided child id.
    active() :
        Returns parameters that are not None.
    default(default_ui, param) :
        return default value for param stored in default_ui.

    """

    _defaults: dict | None = None
    _default_ui_json: dict | None = None
    _free_parameter_keys: list[str] | None = None
    _free_parameter_identifier: str | None = None
    _validator: InputValidation | None = None

    def __init__(
        self,
        input_file: InputFile | None = None,
        validate: bool = True,
        validation_options: dict | None = None,
        workpath: str | Path | None = None,
        **kwargs,
    ):
        """
        Initialize class.

        :param input_file: Input file.
        :param validate: Whether to validate params.
        :param validation_options: Optional validation directives.
        :param workpath: Working directory.
        """
        self._conda_environment: str | None = None
        self._conda_environment_boolean: bool | None = None
        self._generate_sweep: bool = False
        self._geoh5: str | None = None
        self._input_file: InputFile | None = None
        self._monitoring_directory: str | None = None
        self._run_command: str | None = None
        self._run_command_boolean: bool = False
        self._title: str | None = None
        self._ui_json: dict[str, Any] | None = None
        self._validate: bool = True
        self._validations: dict[str, Any] | None = None
        self._validation_options: dict | None = None
        self._workpath: Path | None = None
        self._workspace: str | None = None
        self._workspace_geoh5: str | None = None

        self.workpath = workpath  # type: ignore[assignment]
        self.input_file = input_file
        self.validate = validate
        self.validation_options = validation_options

        self._initialize(**kwargs)

    def _initialize(self, **kwargs):
        """
        Custom actions to initialize the class and deal with input values.
        """
        original_validate_state = self.validate
        self.validate = False

        if self._input_file is None:
            self.input_file = InputFile(
                ui_json=self._default_ui_json,
                validations=self.validations,
                validate=False,
            )
        input_file = self.input_file
        assert input_file is not None
        self.update(input_file.data)
        self.validate = original_validate_state
        self.param_names = list(input_file.data.keys())

        # Apply user input
        if any(kwargs):
            self.update(kwargs)

    @property
    def defaults(self):
        """
        Dictionary of default values.
        """
        return self._defaults

    @property
    def ui_json(self) -> dict[str, Any] | None:
        """
        The default ui_json structure stored on InputFile.
        """
        if self._ui_json is None and self.input_file is not None:
            self._ui_json = self.input_file.ui_json

        return self._ui_json

    def update(self, params_dict: dict[str, Any]):
        """
        Update parameters with dictionary contents.

        :param params_dict: Dictionary of parameters.
        """
        if self.input_file is not None:
            params_dict = self.input_file.numify(params_dict)
            if (
                params_dict.get("geoh5", None) is not None
                and self.input_file.geoh5 is None
            ):
                self.geoh5 = params_dict.pop("geoh5")

            assert self.ui_json is not None
            with fetch_active_workspace(self.geoh5):
                params_dict = self.input_file.promote(params_dict)

                for key, value in params_dict.items():
                    if key not in self.ui_json.keys():
                        continue  # ignores keys not in default_ui_json

                    setattr(self, key, value)

                # Set all parameters belonging to groupOptional disabled.
                for key in utils.find_all(self.ui_json, "groupOptional"):
                    if key in params_dict and params_dict[key] is None:
                        for elem in utils.collect(
                            self.ui_json, "group", self.ui_json[key]["group"]
                        ):
                            setattr(self, elem, None)

    @property
    def workpath(self) -> Path | None:
        """
        Working directory.
        """
        if self._workpath is None and self._geoh5 is not None:
            self._workpath = Path(self.geoh5.h5file).parent
        return self._workpath

    @workpath.setter
    def workpath(self, val: str | Path | None):
        if val is not None:
            if not Path(val).is_dir():
                raise ValueError("Provided 'workpath' is not a valid path.")
            val = Path(val).resolve()

        self._workpath = val

    @property
    def validation_options(self) -> dict | None:
        """
        Optional validation directives.
        """
        if self._validation_options is None:
            return {}

        return self._validation_options

    @validation_options.setter
    def validation_options(self, value: dict | None):
        if not isinstance(value, (dict, type(None))):
            raise UserWarning(
                "Input 'validation_options' must a dictionary of options or None."
            )

        self._validation_options = value

    @property
    def validate(self) -> bool:
        """
        Return True if validation is enabled.
        """
        return self._validate

    @validate.setter
    def validate(self, value: bool):
        """
        Set validation state.
        """
        if not isinstance(value, bool):
            raise UserWarning("Input 'validate' must be a boolean.")

        self._validate = value

        if self.input_file is not None:
            self.input_file.validate = value

    def to_dict(self, ui_json_format=False):
        """
        Return params and values dictionary.

        :param ui_json_format: Return dictionary in ui_json format.
        """

        if ui_json_format:
            assert self.input_file is not None
            return self.input_file.stringify(
                self.input_file.demote(self.input_file.ui_json)
            )

        return {k: getattr(self, k) for k in self.param_names if hasattr(self, k)}

    def active_set(self):
        """
        Return list of parameters with non-null entries.
        """
        return [k for k, v in self.to_dict().items() if v is not None]

    def is_uuid(self, param: str) -> bool:
        """
        Check if string is a valid UUID.

        :param param: String to check.

        :return: Whether string contains valid UUID.
        """
        if isinstance(param, str):
            private_attr = getattr(self, "_" + param)
            return isinstance(private_attr, UUID)
        return False

    @property
    def free_parameter_dict(self) -> dict:
        """
        Extract groups of free parameters from the ui_json dictionary that match
        the 'free_parameter_identifier' and 'free_parameter_keys'.

        :return: Dictionary of free parameters.
        """
        free_parameter_dict: dict = {}
        if (
            self._free_parameter_keys is not None
            and self._free_parameter_identifier is not None
            and self._ui_json is not None
        ):
            ui_groups = list(
                {
                    form["group"]
                    for form in utils.collect(self.ui_json, "group").values()
                    if form["group"] is not None
                }
            )
            ui_groups.sort()
            for group in ui_groups:
                if self._free_parameter_identifier in group.lower():
                    # TODO Create a geoh5py validation for
                    #  "allof" -> ["object", "levels", "type", "distance"]
                    free_parameter_dict[group] = {}
                    forms = utils.collect(self.ui_json, "group", group)
                    for label, key in zip(forms, self._free_parameter_keys):
                        if key not in label.lower():
                            raise ValueError(
                                f"Malformed input refinement group {group}. "
                                f"Must contain forms for all "
                                f"of {self._free_parameter_keys} in this order."
                            )
                        free_parameter_dict[group][key] = label

        return free_parameter_dict

    @property
    def free_parameter_keys(self) -> list | None:
        """
        String identifiers for free form parameters.
        """
        return self._free_parameter_keys

    @property
    def validations(self) -> dict[str, Any] | None:
        """
        Encoded parameter validator type and associated validations.
        """
        if self._validations is None and self.input_file is not None:
            self._validations = self.input_file.validations
        return self._validations

    @validations.setter
    def validations(self, validations: dict[str, Any]):
        assert isinstance(
            validations, dict
        ), "Input value must be a dictionary of validations."
        self._validations = validations

    @property
    def geoh5(self):
        """
        Workspace.
        """
        return self._geoh5

    @geoh5.setter
    def geoh5(self, val):
        if val is None:
            self._geoh5 = val
            return
        self.setter_validator(
            "geoh5",
            val,
            fun=lambda x: Workspace(x) if isinstance(val, (str, Path)) else x,
        )

    @property
    def run_command(self):
        """
        Command to run the application through GA.
        """
        return self._run_command

    @run_command.setter
    def run_command(self, val):
        self.setter_validator("run_command", val)

    @property
    def monitoring_directory(self) -> str | None:
        """
        Path to monitoring directory.
        """
        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, val):
        self.setter_validator("monitoring_directory", val)

    @property
    def conda_environment(self):
        """
        Conda environment.
        """
        return self._conda_environment

    @conda_environment.setter
    def conda_environment(self, val):
        self.setter_validator("conda_environment", val)

    @property
    def conda_environment_boolean(self):
        """
        Boolean to determine if conda environment is used.
        """
        return self._conda_environment_boolean

    @conda_environment_boolean.setter
    def conda_environment_boolean(self, val):
        self.setter_validator("conda_environment_boolean", val)

    @property
    def title(self):
        """
        Application title.
        """
        return self._title

    @title.setter
    def title(self, val):
        self.setter_validator("title", val)

    @property
    def workspace_geoh5(self):
        """
        Source geoh5 file.
        """
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, val):
        self.setter_validator("workspace_geoh5", val)

    @property
    def input_file(self) -> InputFile | None:
        """
        An InputFile class holding the associated ui_json and validations.
        """
        return self._input_file

    @input_file.setter
    def input_file(self, ifile: InputFile | None):
        if not isinstance(ifile, (type(None), InputFile)):
            raise TypeError(
                f"Value for 'input_file' must be {InputFile} or None. "
                f"Provided {ifile} of type{type(ifile)}"
            )

        if ifile is not None:
            self.validator = ifile.validators
            self.validations = ifile.validations

        self._input_file = ifile

    def _uuid_promoter(self, uid: str | UUID) -> str | UUID | Entity:
        """
        Promote a string to a UUID or entity if possible.

        :param uid: String or UUID to promote

        :return: Promoted UUID or entity
        """
        if self.geoh5 is None:
            return uid

        uid = str2uuid(uid)
        entity = uuid2entity(uid, self.geoh5)
        if entity is not None:
            return entity
        return uid

    def setter_validator(self, key: str, value, fun=lambda x: x):
        """
        Validate param and set value.

        :param key: Parameter name.
        :param value: Parameter value.
        :param fun: Function to apply to value before setting.
        """
        if value is None:
            setattr(self, f"_{key}", value)
            return

        value = fun(value)

        if (
            self.input_file is not None
            and hasattr(self.input_file, "data")
            and self.input_file.data is not None
        ):
            if value != self.input_file.data[key]:
                self.input_file.set_data_value(key, value)

        setattr(self, f"_{key}", value)

    def write_input_file(
        self,
        name: str | None = None,
        path: str | Path | None = None,
        validate: bool = True,
    ) -> str:
        """
        Write out a ui.json with the current state of parameters.

        :param name: Name of the ui.json file.
        :param path: Path to write the ui.json file.
        :param validate: Validate the ui.json file before writing.

        :return: Path to the ui.json file.
        """
        if self.input_file is None:
            raise ValueError("No input file has been set.")
        original_validate_state = self.input_file.validate
        self.input_file.validate = validate
        self.input_file.data = self.to_dict()
        self.input_file.validate = original_validate_state
        return self.input_file.write_ui_json(name=name, path=path)
