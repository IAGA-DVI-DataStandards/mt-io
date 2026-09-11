# -*- coding: utf-8 -*-
"""
UoA Collection
==============

Collection of Earth Data PR6-24 (EDL) files combined into runs.

EDL writes one ASCII file per channel into day-numbered folders
(001-366), named ``{station_prefix}YYMMDDhhmmss.{CHANNEL}``. The recorder
also drops a ``config/recorder.ini`` describing the channel layout and
sample rates, which is used here when present.

@author: ben kay (ben@auscope.org.au)
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pandas as pd

from mt_io.collection import Collection
from mt_io.uoa.pr624 import (
    count_samples,
    infer_sample_rate,
    parse_edl_station,
    parse_edl_timestamp,
)

# =============================================================================


class UoACollection(Collection):
    """
    Collection of EDL PR6-24 files into runs based on start and end times.

    Run names are assigned as ``sr{sample_rate}_{index:0{zeros}}``, e.g.
    ``sr10_0001``. A new run is started wherever the recording is not
    contiguous.

    :param file_path: full path to a station directory holding day folders
    :type file_path: string or :class:`pathlib.Path`
    :param survey_id: survey id, default is 'uoa'
    :type survey_id: string
    :param sample_rate: sample rate in Hz, read from recorder.ini when present
    :type sample_rate: float
    :param dipole_length_ex: Ex dipole length in metres
    :type dipole_length_ex: float
    :param dipole_length_ey: Ey dipole length in metres
    :type dipole_length_ey: float

    .. note:: EDL ASCII files carry no header, so the sample rate comes from
     ``config/recorder.ini`` or from the `sample_rate` argument. The station
     id is taken from each file name, so a mid-deployment rename shows up as
     two stations rather than being silently merged.

    :Example:

    .. code-block:: python

        >>> from mt_io.uoa import UoACollection
        >>> uc = UoACollection(r"/path/to/FR17")
        >>> uc.dipole_length_ex = 50.0
        >>> uc.dipole_length_ey = 50.0
        >>> run_dict = uc.get_runs([10])
    """

    CHANNEL_MAP = {
        "BX": ("hx", 0),
        "BY": ("hy", 1),
        "BZ": ("hz", 2),
        "EX": ("ex", 3),
        "EY": ("ey", 4),
    }

    def __init__(self, file_path: str | Path | None = None, **kwargs) -> None:
        self.survey_id = "uoa"
        self.sample_rate = None
        self.dipole_length_ex = 0.0
        self.dipole_length_ey = 0.0
        self.latitude = None
        self.longitude = None
        self.elevation = None

        super().__init__(file_path=file_path, **kwargs)

    def read_recorder_ini(self) -> dict:
        """
        Read ``config/recorder.ini`` if the recorder left one behind.

        :return: sample rate, station id and gain flag, empty if absent
        :rtype: dict
        """
        if self.file_path is None:
            return {}

        matches = sorted(self.file_path.rglob("recorder.ini"))
        if not matches:
            self.logger.debug("No recorder.ini found")
            return {}

        parser = configparser.ConfigParser(strict=False)
        try:
            parser.read(matches[0], encoding="utf-8")
        except (configparser.Error, UnicodeDecodeError) as error:
            self.logger.warning(f"Could not read {matches[0]}: {error}")
            return {}

        if not parser.has_section("recorder"):
            return {}
        section = parser["recorder"]

        info = {"fn": matches[0]}
        station = section.get("station_long_identifier")
        if station:
            info["station"] = station.strip().rstrip("_")

        # channels 0-5 are the MT channels; in practice they share one rate
        rates = {
            section.getint(f"channel_{n}_samplerate", fallback=0) for n in range(6)
        }
        rates.discard(0)
        if len(rates) == 1:
            info["sample_rate"] = float(rates.pop())
        elif rates:
            self.logger.warning(f"Mixed channel sample rates in recorder.ini: {rates}")

        info["high_gain"] = any(
            section.getint(f"channel_{n}_high_gain", fallback=0) for n in range(6)
        )
        return info

    def to_dataframe(
        self,
        sample_rates: list | None = None,
        run_name_zeros: int = 4,
        calibration_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Summarise every EDL file under `file_path`, one row per file.

        :param sample_rates: sample rates to keep, defaults to all found
        :type sample_rates: list, optional
        :param run_name_zeros: number of zeros in the run name, defaults to 4
        :type run_name_zeros: int, optional
        :param calibration_path: path to LEMI-120 .rsp files, defaults to None
        :type calibration_path: str or :class:`pathlib.Path`, optional
        :return: summary table of files
        :rtype: :class:`pandas.DataFrame`
        """
        ini = self.read_recorder_ini()
        sample_rate = self.sample_rate or ini.get("sample_rate")
        if sample_rate is None:
            # fall back to the file stamps: a file that runs on into the next
            # holds the gap between their start times
            sample_rate = infer_sample_rate(self.get_files("BX"))
        if sample_rate is None:
            raise ValueError(
                "Sample rate unknown: no recorder.ini, no sample_rate set, and "
                "the file names do not agree. EDL ASCII files carry no header."
            )
        sample_rate = float(sample_rate)

        if sample_rates is not None:
            wanted = [float(s) for s in sample_rates]
            if sample_rate not in wanted:
                self.logger.warning(
                    f"Sample rate {sample_rate} not in requested {wanted}"
                )
                return pd.DataFrame(columns=self._columns)

        entries = []
        for channel, (component, channel_id) in self.CHANNEL_MAP.items():
            for fn in self.get_files(channel):
                start = parse_edl_timestamp(fn)
                if start is None:
                    self.logger.warning(f"No timestamp in {fn.name}, skipping")
                    continue

                try:
                    n_samples = count_samples(fn)
                except ValueError as error:
                    self.logger.warning(f"{error}, skipping")
                    continue

                # EDL prefixes each file with the station id, so a rename part
                # way through a deployment stays visible here. The separating
                # underscore is not always written, so let the reader find it.
                station = parse_edl_station(fn) or fn.parent.name

                entry = self.get_empty_entry_dict()
                entry["survey"] = self.survey_id
                entry["station"] = station
                entry["run"] = None
                entry["start"] = start.isoformat()
                entry["end"] = (
                    start + pd.Timedelta(seconds=n_samples / sample_rate)
                ).isoformat()
                entry["channel_id"] = channel_id
                entry["component"] = component
                entry["fn"] = fn
                entry["sample_rate"] = sample_rate
                entry["file_size"] = fn.stat().st_size
                entry["n_samples"] = n_samples
                entry["sequence_number"] = 0
                if component == "ex":
                    entry["dipole"] = self.dipole_length_ex
                elif component == "ey":
                    entry["dipole"] = self.dipole_length_ey
                else:
                    entry["dipole"] = 0
                entry["coil_number"] = None
                entry["latitude"] = self.latitude
                entry["longitude"] = self.longitude
                entry["elevation"] = self.elevation
                entry["instrument_id"] = "PR6-24"
                entry["calibration_fn"] = calibration_path
                entries.append(entry)

        if not entries:
            self.logger.warning(f"No EDL files found in {self.file_path}")
            return pd.DataFrame(columns=self._columns)

        df = self._set_df_dtypes(pd.DataFrame(entries))
        self._warn_duplicate_starts(df)
        return self._sort_df(df, run_name_zeros)

    def _warn_duplicate_starts(self, df: pd.DataFrame) -> None:
        """
        Warn when a channel has two files claiming the same start.

        Archives sometimes keep a corrected copy of a day beside the original,
        in a folder such as ``Orig134`` or ``All-169``. Both are found here and
        both would go into the run, so the samples appear twice. Which copy is
        the right one is not something the file names say.

        :param df: summary table
        :type df: :class:`pandas.DataFrame`
        """
        if not len(df):
            return
        counts = df.groupby(["station", "component", "start"]).size()
        repeated = counts[counts > 1]
        if not len(repeated):
            return

        stations = sorted({station for station, _, _ in repeated.index})
        self.logger.warning(
            f"{len(repeated)} duplicated start time(s) in {stations}; the same "
            "samples will appear more than once in a run. Check for a second "
            "copy of a day beside the original."
        )

    def assign_run_names(self, df: pd.DataFrame, zeros: int = 4) -> pd.DataFrame:
        """
        Split into runs wherever the recording is not contiguous.

        File length is a recorder setting, so consecutive files belong to the
        same run when the next one starts where the previous ended rather than
        at any fixed interval. A tolerance of two sample periods absorbs
        rounding in the file name stamp.

        :param df: summary table
        :type df: :class:`pandas.DataFrame`
        :param zeros: number of zeros in the run name, defaults to 4
        :type zeros: int, optional
        :return: summary table with run names
        :rtype: :class:`pandas.DataFrame`
        """
        df = df.copy()
        for station in df.station.unique():
            for sample_rate in df[df.station == station].sample_rate.unique():
                mask = (df.station == station) & (df.sample_rate == sample_rate)
                block = df[mask].sort_values("start")
                tolerance = pd.Timedelta(seconds=2.0 / float(sample_rate))

                boundaries = self._run_boundaries(block, tolerance)

                run_index = 0
                names = []
                previous_start = None
                for _, row in block.iterrows():
                    if previous_start is None or row.start in boundaries:
                        if previous_start is None or row.start != previous_start:
                            run_index += 1
                    names.append(f"sr{int(sample_rate)}_{run_index:0{zeros}}")
                    previous_start = row.start
                df.loc[block.index, "run"] = names

        return df

    @staticmethod
    def _run_boundaries(block: pd.DataFrame, tolerance: pd.Timedelta) -> set:
        """
        Start times where the recording is not continuous.

        Each channel is walked on its own, so a file missing from one of them
        ends the run for all of them rather than being hidden by the others.
        Files that overlap end the run as well: the samples are duplicated and
        the time axis cannot run through them.

        :param block: rows for one station at one sample rate
        :type block: :class:`pandas.DataFrame`
        :param tolerance: how far apart two files may sit and still join
        :type tolerance: :class:`pandas.Timedelta`
        :return: start times that begin a new run
        :rtype: set
        """
        boundaries = set()
        for _, channel in block.groupby("component"):
            channel = channel.sort_values("start")
            previous_end = None
            for _, row in channel.iterrows():
                if previous_end is not None:
                    if abs(row.start - previous_end) > tolerance:
                        boundaries.add(row.start)
                previous_end = row.end
        return boundaries
