#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils package.
#
#  All rights reserved.
#

from geoapps_utils import geophysical_systems


def test_geophysical_systems_list():
    expected_systems = [
        "Airborne TEM Survey",
        "AeroTEM (2007)",
        "AeroTEM (2010)",
        "DIGHEM",
        "GENESIS (2014)",
        "GEOTEM 75 Hz - 2082 Pulse",
        "HELITEM2 (25C, 7.5Hz)",
        "HELITEM (35C)",
        "Hummingbird",
        "QUESTEM (1996)",
        "Resolve",
        "SandersGFEM",
        "Skytem 304M (HM)",
        "Skytem 306HP (LM)",
        "Skytem 306M HP (HM)",
        "Skytem 312HP (HM)",
        "Skytem 312HP v2 (HM)",
        "Skytem 312HP v3 (HM)",
        "Skytem 312HP v2 (LM)",
        "Skytem 312HP v3 (LM)",
        "Skytem 516M (HM)",
        "Spectrem (2000)",
        "Spectrem Plus",
        "Spectrem (2003)",
        "VTEM (2007)",
        "VTEM Plus",
        "VTEM Max",
        "Xcite",
        "Zonge 8Hz",
    ]

    systems = geophysical_systems.parameters()
    assert list(systems.keys()).sort() == expected_systems.sort()
