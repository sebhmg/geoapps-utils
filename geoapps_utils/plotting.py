#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps-utils.
#
#  All rights reserved.
#

import numpy as np


def symlog(values, threshold):
    """
    Convert values to log with linear threshold near zero
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values, threshold):
    """
    Compute the inverse symlog mapping
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))


def format_axis(channel, axis, log, threshold, nticks=5):
    """
    Format plot axis ticks and labels
    """
    label = channel

    if log:
        axis = symlog(axis, threshold)

    values = axis[~np.isnan(axis)]
    ticks = np.linspace(values.min(), values.max(), nticks)

    if log:
        label = f"Log({channel})"
        ticklabels = inv_symlog(ticks, threshold)
    else:
        ticklabels = ticks

    return axis, label, ticks, ticklabels.tolist()
