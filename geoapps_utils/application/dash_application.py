#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  geoapps-utils is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import base64
import io
import json
import os
import signal
import socket
import sys
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any

import numpy as np
from dash import Dash, callback_context, dcc, no_update
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
from flask import Flask
from geoh5py.data import Data
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.utils import fetch_active_workspace, is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from PySide2 import QtCore, QtWebEngineWidgets, QtWidgets

from geoapps_utils.application.layout import object_selection_layout
from geoapps_utils.driver.params import BaseParams


class BaseDashApplication(ABC):
    """
    Base class for geoapps dash applications
    """

    _params = None
    _param_class = BaseParams
    _driver_class = None
    _workspace = None
    _app_initializer: dict | None = None
    _ui_json_data: dict | None = None

    def __init__(
        self,
        ui_json: InputFile | None = None,
        ui_json_data: dict | None = None,
        params: BaseParams | None = None,
    ):
        """
        Set initial ui_json_data from input file and open workspace.
        """
        if params is not None:
            # Launched from notebook
            # Params for initialization are coming from params
            # ui_json_data is provided
            self.params = params
        elif (
            ui_json is not None
            and getattr(ui_json, "path", None) is not None
            and Path(ui_json.path).exists()
        ):
            # Launched from terminal
            # Params for initialization are coming from ui_json
            # ui_json_data starts as None
            self.params = self._param_class(ui_json)
            input_file = self.params.input_file
            if input_file is not None:
                ui_json_data = input_file.demote(self.params.to_dict())
        self._ui_json_data = ui_json_data

        assert self.params is not None, "No parameters provided."
        self.workspace = self.params.geoh5
        self.workspace.open()
        if self._driver_class is not None:
            self.driver = self._driver_class(self.params)

    @property
    @abstractmethod
    def app(self) -> Dash:
        """Dash app"""

    def get_data_options(
        self,
        ui_json_data: dict,
        object_uid: str | None,
        object_name: str = "objects",
        trigger: str | None = None,
    ) -> list:
        """
        Get data dropdown options from a given object.

        :param ui_json_data: Uploaded ui.json data to read object from.
        :param object_uid: Selected object in object dropdown.
        :param object_name: Object parameter name in ui.json.
        :param trigger: Callback trigger.

        :return options: Data dropdown options.
        """
        obj = None
        if trigger == "ui_json_data" and object_name in ui_json_data:
            if is_uuid(ui_json_data[object_name]):
                object_uid = ui_json_data[object_name]
            elif self.workspace.get_entity(ui_json_data[object_name])[0] is not None:
                object_uid = self.workspace.get_entity(ui_json_data[object_name])[0].uid

        if object_uid is not None and is_uuid(object_uid):
            for entity in self.workspace.get_entity(uuid.UUID(object_uid)):
                if isinstance(entity, ObjectBase):
                    obj = entity

        if obj:
            options = []
            for child in obj.children:
                if isinstance(child, Data):
                    if child.name != "Visual Parameters":
                        options.append(
                            {"label": child.name, "value": "{" + str(child.uid) + "}"}
                        )
            options = sorted(options, key=lambda d: d["label"])

            return options
        return []

    def get_params_dict(self, update_dict: dict) -> dict:
        """
        Get dict of current params.

        :param update_dict: Dict of parameters with new values to convert to a params dict.

        :return output_dict: Dict of current params.
        """
        if self.params is None:
            return {}

        output_dict: dict[str, Any] = {}
        # Get validations to know expected type for keys in self.params.
        validations = self.params.validations

        # Loop through self.params and update self.params with locals_dict.
        for key in self.params.to_dict():
            if key not in update_dict:
                continue
            if validations is not None:
                if bool in validations[key]["types"] and isinstance(
                    update_dict[key], list
                ):
                    # Convert from dash component checklist to bool
                    if not update_dict[key]:
                        output_dict[key] = False
                    else:
                        output_dict[key] = True
                    continue
                if (
                    float in validations[key]["types"]
                    and int not in validations[key]["types"]
                    and isinstance(update_dict[key], int)
                ):
                    # Checking for values that Dash has given as int when they should be floats.
                    output_dict[key] = float(update_dict[key])
                    continue
            if is_uuid(update_dict[key]):
                output_dict[key] = self.workspace.get_entity(
                    uuid.UUID(update_dict[key])
                )[0]
            else:
                output_dict[key] = update_dict[key]
        return output_dict

    @staticmethod
    def init_vals(layout: list[Component], ui_json_data: dict):
        """
        Initialize dash components in layout from ui_json_data.

        :param layout: Dash layout.
        :param ui_json_data: Uploaded ui.json data.
        """
        for comp in layout:
            BaseDashApplication._init_component(comp, ui_json_data)

    @staticmethod
    def _init_component(comp: Component, ui_json_data: dict):
        if hasattr(comp, "children") and not isinstance(comp, dcc.Markdown):
            BaseDashApplication.init_vals(comp.children, ui_json_data)
            return

        if hasattr(comp, "id") and comp.id in ui_json_data:
            if isinstance(comp, dcc.Store):
                comp.data = ui_json_data[comp.id]
                return

            if isinstance(comp, dcc.Dropdown):
                comp.value = ui_json_data[comp.id]
                if comp.value is None:
                    comp.value = "None"
                if not hasattr(comp, "options"):
                    comp_option = ui_json_data[comp.id]
                    if comp_option is None:
                        comp_option = "None"
                    comp.options = [comp_option]
                return

            if isinstance(comp, dcc.Checklist):
                if ui_json_data[comp.id]:
                    comp.value = [True]
                else:
                    comp.value = []
            else:
                comp.value = ui_json_data[comp.id]

    @staticmethod
    def update_visibility_from_checklist(checklist_val: list[bool]) -> dict:
        """
        Update visibility of a component from a checklist value.

        :param checklist_val: Checklist value.

        :return visibility: Component style.
        """
        if checklist_val:
            return {"display": "block"}
        return {"display": "none"}

    @property
    def params(self) -> BaseParams | None:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: BaseParams):
        assert isinstance(
            params, BaseParams
        ), f"Input parameters must be an instance of {BaseParams}"

        self._params = params

    @property
    def workspace(self):
        """
        Current workspace.
        """
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        # Close old workspace
        if self._workspace is not None:
            self._workspace.close()
        self._workspace = workspace


class ObjectSelection:
    """
    Dash app to select workspace and object.

    Creates temporary workspace with the object, and
    opens a Qt window to run an app.
    """

    _app_initializer: dict | None = None

    def __init__(
        self,
        app_name: str,
        app_initializer: dict,
        app_class: type[BaseDashApplication],
        param_class: type[BaseParams],
        **kwargs,
    ):
        self._app_name = None
        self._app_class = BaseDashApplication
        self._param_class = BaseParams
        self._workspace = None

        self.app_name = app_name
        self.app_class = app_class
        self.param_class = param_class

        app_initializer.update(kwargs)
        self.params = self.param_class(**app_initializer)
        self._app_initializer = {
            key: value
            for key, value in app_initializer.items()
            if key not in self.params.param_names
        }
        self.workspace = self.params.geoh5

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = Dash(
            server=server,
            url_base_pathname=os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )
        self.app.layout = object_selection_layout

        # Set up callbacks
        self.app.callback(
            Output(component_id="objects", component_property="options"),
            Output(component_id="objects", component_property="value"),
            Output(component_id="ui_json_data", component_property="data"),
            Output(component_id="upload", component_property="filename"),
            Output(component_id="upload", component_property="contents"),
            Input(component_id="ui_json_data", component_property="data"),
            Input(component_id="upload", component_property="filename"),
            Input(component_id="upload", component_property="contents"),
            Input(component_id="objects", component_property="value"),
        )(self.update_object_options)
        self.app.callback(
            Output(component_id="launch_app_markdown", component_property="children"),
            State(component_id="objects", component_property="value"),
            Input(component_id="launch_app", component_property="n_clicks"),
        )(self.launch_qt)

    def update_object_options(
        self,
        ui_json_data: dict,
        filename: str,
        contents: str,
        objects: str,
        trigger: str | None = None,
    ) -> tuple:
        """
        This function is called when a file is uploaded. It sets the new workspace, sets the dcc
        ui_json_data component, and sets the new object options and values.

        :param filename: Uploaded filename. Workspace or ui.json.
        :param contents: Uploaded file contents. Workspace or ui.json.
        :param trigger: Dash component which triggered the callback.

        :return object_options: New object dropdown options.
        :return object_value: New object value.
        :return ui_json_data: Uploaded ui_json data.
        :return filename: Return None to reset the filename so the same file can be chosen twice
            in a row.
        :return contents: Return None to reset the contents so the same file can be chosen twice
            in a row.
        """
        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if trigger != "":
            # Remove entities from ui_json_data
            inp_ui_json_data = ui_json_data.copy()
            for key, value in inp_ui_json_data.items():
                if is_uuid(value) or isinstance(value, Entity):
                    setattr(self.params, key, None)
                    del ui_json_data[key]
        if trigger == "objects":
            return no_update, objects, ui_json_data, None, None

        object_options, object_value = no_update, None
        if contents is not None and filename is not None:
            if filename.endswith(".ui.json"):
                # Uploaded ui.json
                _, content_string = contents.split(",")
                ui_json = json.loads(base64.b64decode(content_string))
                self.workspace = Workspace(ui_json["geoh5"], mode="r")
                # Create ifile from ui.json
                ifile = InputFile(ui_json=ui_json)
                self.params = self.param_class(ifile)
                # Demote ifile data so it can be stored as a string
                if getattr(ifile, "data", None) is not None:
                    ui_json_data = ifile.demote(ifile.data.copy())
                    # Get new object value for dropdown from ui.json
                    object_value = ui_json_data["objects"]
            elif filename.endswith(".geoh5"):
                # Uploaded workspace
                _, content_string = contents.split(",")
                self.workspace = Workspace(
                    io.BytesIO(base64.b64decode(content_string)), mode="r"
                )
                # Update self.params with new workspace, but keep unaffected params the same.
                new_params = self.params.to_dict()
                for key, value in new_params.items():
                    if isinstance(value, Entity):
                        new_params[key] = None
                new_params["geoh5"] = self.workspace
                self.params = self.param_class(**new_params)
        elif trigger == "":
            ui_json_data = self._ui_json_data_from_params()
            object_value = ui_json_data["objects"]

            # Get new options for object dropdown
            if self.workspace is not None:
                object_options = [
                    {
                        "label": obj.parent.name + "/" + obj.name,
                        "value": "{" + str(obj.uid) + "}",
                    }
                    for obj in self.workspace.objects
                ]

        return object_options, object_value, ui_json_data, None, None

    def _ui_json_data_from_params(self) -> dict[str, Any]:
        """Returns app json data initialized from self.params"""
        assert (
            self.params.input_file is not None
        ), "No input file provided in parameters."

        ifile = InputFile(
            ui_json=self.params.input_file.ui_json,
            validate=False,
        )
        ifile.update_ui_values(self.params.to_dict())
        ui_json_data = ifile.demote(ifile.data)
        if self._app_initializer is not None:
            ui_json_data.update(self._app_initializer)
        return ui_json_data

    @staticmethod
    def get_port() -> int:
        """
        Loop through a list of ports to find an available port.

        :return port: Available port.
        """
        open_port = None
        for port in np.arange(8050, 8101):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                in_use = sock.connect_ex(("localhost", port)) == 0
            if in_use is False:
                open_port = port
                break
        if open_port is None:
            raise RuntimeError("No open port found.")
        return open_port

    @staticmethod
    def start_server(
        port: int,
        app_class: type[BaseDashApplication],
        ui_json: InputFile | None = None,
        ui_json_data: dict | None = None,
        params: BaseParams | None = None,
    ):
        """
        Launch dash app server using given port.

        :param port: Port for where to launch server.
        :param app_class: Type of app to create.
        :param ui_json: ifile corresponding to the ui_json_data.
        :param ui_json_data: Dict of current params to provide to app init.
        :param params: Current params to pass to new app.
        """
        app = app_class(ui_json=ui_json, ui_json_data=ui_json_data, params=params)
        app.app.run(jupyter_mode="external", host="127.0.0.1", port=port)

    @staticmethod
    def make_qt_window(app_name: str | None, port: int):
        """
        Make Qt window and load dash url with the given port.

        :param app_name: App name to display as Qt window title.
        :param port: Port where the dash app has been launched.
        """
        app = QtWidgets.QApplication(sys.argv)
        browser = QtWebEngineWidgets.QWebEngineView()

        browser.setWindowTitle(app_name)
        browser.load(QtCore.QUrl("http://127.0.0.1:" + str(port)))
        # Brings Qt window to the front
        browser.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        # Setting window size
        browser.resize(1200, 800)
        browser.show()

        app.exec_()  # running the Qt app
        os.kill(os.getpid(), signal.SIGTERM)  # shut down dash server and notebook

    def launch_qt(self, objects: str, n_clicks: int) -> str:
        """
        Launch the Qt app when launch app button is clicked.

        :param objects: Selected object uid.
        :param n_clicks: Number of times button has been clicked; triggers callback.

        :return launch_app_markdown: Empty string since callbacks must have output.
        """
        del n_clicks  # unused argument

        triggers = [c["prop_id"].split(".")[0] for c in callback_context.triggered]
        if not (
            "launch_app" in triggers
            and objects is not None
            and self.workspace is not None
        ):
            return ""

        # Make new workspace with only the selected object
        obj = self.workspace.get_entity(uuid.UUID(objects))[0]
        if not isinstance(obj, Entity):
            return ""

        temp_geoh5 = self.workspace.name + "_" + f"{time():.0f}.geoh5"
        with tempfile.TemporaryDirectory() as tempdir_name:
            tempdir = Path(tempdir_name)
            temp_workspace = Workspace.create(tempdir / temp_geoh5)

            # Update ui.json with temp_workspace to pass initialize scatter plot app
            param_dict = self.params.to_dict()
            param_dict["geoh5"] = temp_workspace

            with fetch_active_workspace(temp_workspace), fetch_active_workspace(
                self.workspace
            ):
                param_dict["objects"] = obj.copy(
                    parent=temp_workspace, copy_children=True
                )

                if hasattr(obj, "property_groups"):
                    self._copy_property_groups(obj.property_groups, param_dict)

            new_params = self.param_class(**param_dict)

            temp_ui_json_name = temp_geoh5.replace(".geoh5", ".ui.json")
            new_params.write_input_file(
                name=temp_ui_json_name,
                path=tempdir_name,
                validate=False,
            )

            ifile = InputFile.read_ui_json(tempdir / temp_ui_json_name)
            ui_json_data = ifile.demote(ifile.data)

        # Start server
        port = ObjectSelection.get_port()
        threading.Thread(
            target=self.start_server,
            args=(port, self.app_class, ifile, ui_json_data, new_params),
            daemon=True,
        ).start()

        # Make Qt window
        self.make_qt_window(self.app_name, port)
        return ""

    @staticmethod
    def _copy_property_groups(source_groups: ObjectBase, param_dict: dict):
        """Copy any property groups over to param_dict of temporary workspace"""
        temp_prop_groups = param_dict["objects"].property_groups
        for group in source_groups:
            if not isinstance(group, PropertyGroup):
                continue
            for temp_group in temp_prop_groups:
                if temp_group.name != group.name:
                    continue
                for key, value in param_dict.items():
                    if hasattr(value, "uid") and value.uid == group.uid:
                        param_dict[key] = temp_group.uid

    @staticmethod
    def run(app_name: str, app_class: BaseDashApplication, ui_json: InputFile):
        """
        Launch Qt app from terminal.

        :param app_name: Name of app to display as Qt window title.
        :param app_class: Type of app to create.
        :param ui_json: Input file to pass to app for initialization.
        """
        # Start server
        port = ObjectSelection.get_port()
        threading.Thread(
            target=ObjectSelection.start_server,
            args=(port, app_class, ui_json),
            daemon=True,
        ).start()

        # Make Qt window
        ObjectSelection.make_qt_window(app_name, port)

    @property
    def app_name(self) -> str | None:
        """
        Name of app that appears as Qt window title.
        """
        return self._app_name

    @app_name.setter
    def app_name(self, val):
        if not isinstance(val, str) and (val is not None):
            raise TypeError("Value for attribute `app_name` should be 'str' or 'None'")
        self._app_name = val

    @property
    def app_class(self) -> type[BaseDashApplication]:
        """
        The kind of app to launch.
        """
        return self._app_class

    @app_class.setter
    def app_class(self, val):
        if not issubclass(val, BaseDashApplication):
            raise TypeError(
                "Value for attribute `app_class` should be a subclass of "
                ":obj:`geoapps_utils.application.BaseDashApplication`"
            )
        self._app_class = val

    @property
    def param_class(self) -> type[BaseParams]:
        """
        The param class associated with the launched app.
        """
        return self._param_class

    @param_class.setter
    def param_class(self, val):
        if not issubclass(val, BaseParams):
            raise TypeError(
                "Value for attribute `param_class` should be a subclass of "
                ":obj:`geoapps_utils.driver.params.BaseParams`"
            )
        self._param_class = val

    @property
    def workspace(self) -> Workspace | None:
        """
        Input workspace.
        """
        return self._workspace

    @workspace.setter
    def workspace(self, val):
        if not isinstance(val, Workspace) and (val is not None):
            raise TypeError(
                "Value for attribute `workspace` should be :obj:`geoh5py.workspace.Workspace`"
                " or 'None'"
            )
        if self._workspace is not None:
            # Close old workspace
            self._workspace.close()
        self._workspace = val
