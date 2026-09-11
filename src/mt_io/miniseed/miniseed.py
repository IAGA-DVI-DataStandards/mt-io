# -*- coding: utf-8 -*-
"""

Created on Wed Sep 30 10:20:12 2020

:author: Jared Peacock

:license: MIT

"""

# =============================================================================
# Imports
# =============================================================================
from pathlib import Path

import xarray as xr
from mt_timeseries import RunTS
from obspy import read as obspy_read

# Ensure sps_filters accessor exists for environments where mt_timeseries
# does not auto-register scipy_filters on import.
if not hasattr(xr.DataArray, "sps_filters"):
    import mt_timeseries.scipy_filters  # noqa: F401


# =============================================================================
# read seismic file
# =============================================================================
def read_miniseed(fn):
    """
    Read a miniseed file into a :class:`mth5.timeseries.RunTS` object. Uses
    `Obspy` to read the miniseed.

    :param fn: full path to the miniseed file
    :type fn: string
    :return: RunTS object
    :rtype: :class:`mth5.timeseries.RunTS`

    """

    # obspy does not use Path objects for file names
    if isinstance(fn, Path):
        fn = fn.as_posix()
    try:
        obs_stream = obspy_read(fn, format="MSEED")
    except Exception:
        # not actually miniSEED; let obspy work the format out
        obs_stream = obspy_read(fn)
    run_obj = RunTS()
    run_obj.from_obspy_stream(obs_stream)

    return run_obj
