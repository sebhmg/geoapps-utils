#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.

from __future__ import annotations


def string_name(value: str, characters: str = ".") -> str:
    """
    Find and replace characters in a string with underscores '_'.

    :param value: String to be validated
    :param char: Characters to be replaced

    :return value: Re-formatted string
    """
    for char in characters:
        value = value.replace(char, "_")
    return value
