# -*- coding: utf-8 -*-
"""
==========
Orange Box
==========

    * reads University of Adelaide / Flinders Orange Box binary files
    * builds the sensor and electrode response

The Orange Box is a legacy 8-channel long-period MT logger. Each file opens
with a 40 byte ASCII header holding the sample count in hex, a 25 character
timestamp and a 4 character filter point, then fixed 21 byte records: three
24-bit channels, two 16-bit, one 8-bit, two more 24-bit, and a trailing byte.
The sample rate is not stored directly, it follows from the filter point as
10e6 / (512 * filter_point).

Channels map as ch0 to Bx, ch1 to Bz, ch2 to By, ch7 to Ex and ch6 to Ey.
By, Ex and Ey are inverted by the hardware. Counts are unsigned about 2**23,
so the reader stores them signed and the response is then a plain gain:

    hx, hz   nT    -> count    2**23 / 70000
    hy       nT    -> count   -2**23 / 70000
    ex, ey   mV/km -> count   -2**23 * L / 100000

Boxes built before the symmetric board was rebridged recorded +/-2.5 V, for
which the electric full scale is 25000 rather than 100000.

@author: ben kay (ben@auscope.org.au)

:license: MIT

"""

from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from loguru import logger
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic, Run, Station
from mt_metadata.timeseries.filters import ChannelResponse, CoefficientFilter
from mt_timeseries import ChannelTS, RunTS

# ==============================================================================
# Hardware Calibration Constants
# ==============================================================================

# ADC characteristics (24-bit sigma-delta, unsigned)
ADC_BITS = 24
ADC_MAX_COUNTS = 2**23  # Signed range: +/-2^23
ADC_ZERO = 2**23  # Zero point for unsigned -> signed conversion

# Bartington fluxgate full-scale range
BARTINGTON_FULL_SCALE_NT = 70000.0  # +/-70,000 nT

# Electric field full-scale (before dipole normalization)
ELECTRIC_FULL_SCALE_UV = 100000.0  # +/-100,000 uV

# Sample format constants
BYTES_PER_SAMPLE = 21  # 3+3+3+2+2+1+3+3 channel bytes plus 1 trailing byte
NCHANNELS = 8


# ==============================================================================
# Calibration Filter Creation Functions
# ==============================================================================


def create_orange_magnetic_filter(
    component: str, invert: bool = False
) -> CoefficientFilter:
    """
    Create the Bartington sensor filter for an Orange Box magnetic channel.

    The sensor covers +/-70,000 nT over the full 24 bit swing, so one nT is
    2**23 / 70000 counts. By is inverted by the hardware.

    :param component: component name ('hx', 'hy' or 'hz')
    :type component: str
    :param invert: negate the gain, True for hy
    :type invert: bool
    :return: coefficient filter, nanoTesla to count
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    mag_filter = CoefficientFilter()
    mag_filter.name = f"orange_magnetic_{component}"
    mag_filter.units_in = "nanoTesla"
    mag_filter.units_out = "count"
    gain = ADC_MAX_COUNTS / BARTINGTON_FULL_SCALE_NT
    mag_filter.gain = -gain if invert else gain
    mag_filter.comments = (
        f"Orange Box {component.upper()}, +/-{BARTINGTON_FULL_SCALE_NT:.0f} nT "
        f"over +/-2**23 counts"
        + (", inverted by the hardware" if invert else "")
    )
    return mag_filter


def create_orange_electric_filter(
    component: str, dipole_length: float, full_scale_uv: float = ELECTRIC_FULL_SCALE_UV
) -> CoefficientFilter:
    """
    Create the electrode filter for an Orange Box electric channel.

    1 mV/km is 1 microVolt per metre, so a field of E across a dipole of
    length L metres gives E * L microVolt, which is E * L / full_scale of the
    24 bit swing. Ex and Ey are inverted by the hardware.

    :param component: component name ('ex' or 'ey')
    :type component: str
    :param dipole_length: dipole length in metres
    :type dipole_length: float
    :param full_scale_uv: electrode full scale, 100000 for +/-10 V boxes and
     25000 for the earlier +/-2.5 V boxes
    :type full_scale_uv: float
    :return: coefficient filter, milliVolt per kilometer to count
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    if dipole_length <= 0:
        dipole_length = 1.0

    elec_filter = CoefficientFilter()
    elec_filter.name = f"orange_electric_{component}_{dipole_length}m"
    elec_filter.units_in = "milliVolt per kilometer"
    elec_filter.units_out = "count"
    elec_filter.gain = -(ADC_MAX_COUNTS * dipole_length) / full_scale_uv
    elec_filter.comments = (
        f"Orange Box {component.upper()}, {dipole_length} m dipole, "
        f"+/-{full_scale_uv:.0f} uV full scale, inverted by the hardware"
    )
    return elec_filter


class OrangeDataReader:
    """
    Reads a single Orange Box binary file.

    **Binary Format**:
        - 3 header lines (ASCII with \\n terminators)
        - Binary data stream (21 bytes per sample)

        - Returns raw counts (no calibration applied)
        - Calibrations described as unapplied filters

    :param file_path: Path to .BIN file
    :type file_path: Path or str
    """

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.n_samples = None
        self.sample_rate = None
        self.start_time = None
        self.filter_point = None
        self.logger = logger

    def parse_header(self, f) -> dict:
        """
        Parse the 3-line ASCII header.

        :param f: Open binary file handle
        :type f: file object
        :return: Dictionary with sample_rate, start_time, filter_point
        :rtype: dict

        **Header Format**:
            Line 1: " 00008CA0 " sample count in hex
            Line 2: "Tue Jun 16 02:01:04 2009"
            Line 3: " 7A1" filter point, sets the sample rate
        """
        # Line 1 is the number of samples in the file, not the rate
        line1 = f.readline().decode("ascii", errors="ignore").strip()
        self.n_samples = int(line1, 16)

        # Line 2: Date/time string
        line2 = f.readline().decode("ascii", errors="ignore").strip()
        try:
            self.start_time = pd.to_datetime(
                line2, format="%a %b %d %H:%M:%S %Y", utc=True
            )
        except Exception as e:
            self.logger.warning(f"Could not parse start time '{line2}': {e}")
            self.start_time = None

        # Line 3: Filter point (4 bytes, hex)
        filter_bytes = f.read(4).decode("ascii", errors="ignore").strip()
        self.filter_point = int(filter_bytes, 16)

        # the logger derives its rate from the filter point
        self.sample_rate = 10_000_000 / (512 * self.filter_point)

        return {
            "n_samples": self.n_samples,
            "sample_rate": self.sample_rate,
            "start_time": self.start_time,
            "filter_point": self.filter_point,
        }

    def read_samples(self, f) -> np.ndarray:
        """
        Read the binary samples, decoding the mixed channel widths.

        :param f: open binary file, positioned after the header
        :type f: file object
        :return: array of shape (n_samples, 8) of raw counts
        :rtype: :class:`numpy.ndarray`

        **Binary Layout** (21 bytes per sample, big endian):
            - channels 0-2: 3 bytes each, 24 bit
            - channels 3-4: 2 bytes each, 16 bit
            - channel 5: 1 byte
            - channels 6-7: 3 bytes each, 24 bit
            - 1 trailing byte

        A 25 character end timestamp follows the last record, so only
        ``n_samples`` records are taken.
        """
        want = self.n_samples * BYTES_PER_SAMPLE if self.n_samples else -1
        buf = f.read(want)
        n = len(buf) // BYTES_PER_SAMPLE
        if self.n_samples and n < self.n_samples:
            self.logger.warning(
                f"{self.file_path.name}: header claims {self.n_samples} samples, "
                f"file holds {n}"
            )
        if n == 0:
            return np.zeros((0, NCHANNELS), dtype=np.int64)

        rec = (
            np.frombuffer(buf[: n * BYTES_PER_SAMPLE], dtype=np.uint8)
            .reshape(n, BYTES_PER_SAMPLE)
            .astype(np.int64)
        )

        def u24(o):
            return (rec[:, o] << 16) | (rec[:, o + 1] << 8) | rec[:, o + 2]

        def u16(o):
            return (rec[:, o] << 8) | rec[:, o + 1]

        counts = np.empty((n, NCHANNELS), dtype=np.int64)
        counts[:, 0] = u24(0)
        counts[:, 1] = u24(3)
        counts[:, 2] = u24(6)
        counts[:, 3] = u16(9)
        counts[:, 4] = u16(11)
        counts[:, 5] = rec[:, 13]
        counts[:, 6] = u24(14)
        counts[:, 7] = u24(17)

        # the 24 bit channels are offset binary about 2**23; store them signed
        # so the response is a plain gain and needs no offset term
        for ch in (0, 1, 2, 6, 7):
            counts[:, ch] -= ADC_ZERO

        self.logger.info(f"Read {n} samples from {self.file_path.name}")
        return counts

    def read(self) -> pd.DataFrame:
        """
        Read Orange Box binary file and return raw counts.

        :return: DataFrame with columns [Bx, Bz, By, Ex, Ey] as raw counts
        :rtype: pd.DataFrame

        **Note**: Only channels 0, 1, 2, 6, 7 are used (Bx, Bz, By, Ey, Ex).
        Channels 3, 4, 5 are present in file but not used for MT processing.
        """
        with open(self.file_path, "rb") as f:
            header = self.parse_header(f)
            samples = self.read_samples(f)

        if samples.size == 0:
            self.logger.warning(f"No samples read from {self.file_path}")
            return pd.DataFrame(columns=["Bx", "Bz", "By", "Ex", "Ey"])

        # Extract MT channels (raw counts, no calibration)
        # Channel mapping: 0=Bx, 1=Bz, 2=By, 6=Ey, 7=Ex
        df = pd.DataFrame(
            {
                "Bx": samples[:, 0],  # Channel 0
                "Bz": samples[:, 1],  # Channel 1
                "By": samples[:, 2],  # Channel 2
                "Ey": samples[:, 6],  # Channel 6
                "Ex": samples[:, 7],  # Channel 7
            }
        )

        self.logger.info(f"Read {len(df)} samples from {self.file_path.name}")
        return df


# ==============================================================================
# Main Reader Class
# ==============================================================================


class OrangeReader:
    """
    MTH5-compatible reader for Orange Box binary files.

        - Stores raw counts in data arrays (no calibrations applied)
        - Describes all calibrations as unapplied filters (filter.applied = False)

    :param files: Single file path or list of .BIN files to read
    :type files: str, Path, or list
    :param kwargs: Additional options for metadata
    :type kwargs: dict

    :Keyword Arguments:
        * **station_id** (str) - Station identifier (required)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 100.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 100.0)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> from mt_io.uoa import read_orange
        >>> run_ts = read_orange('/path/to/HFM1-*.BIN', station_id='ST61',
        ...                      dipole_length_ex=100.0, dipole_length_ey=100.0)
    """

    def __init__(self, files: Union[str, Path, List[Union[str, Path]]], **kwargs):
        self.files = [Path(f) for f in (files if isinstance(files, list) else [files])]
        self.station_id = kwargs.get("station_id", "OrangeBox")
        self.dipole_length_ex = kwargs.get("dipole_length_ex", 100.0)
        self.dipole_length_ey = kwargs.get("dipole_length_ey", 100.0)
        self.latitude = kwargs.get("latitude", 0.0)
        self.longitude = kwargs.get("longitude", 0.0)
        self.elevation = kwargs.get("elevation", 0.0)
        self.logger = logger
        self.data = None
        self.header = None
        self.sample_rate = None
        self.start_time = None

    def read(self) -> RunTS:
        """
        Read Orange Box file(s) and return RunTS with channels hx, hy, hz, ex, ey.

        :return: RunTS object containing ChannelTS for each component
        :rtype: RunTS
        """
        # Read all files and concatenate
        dfs = []
        for file_path in self.files:
            reader = OrangeDataReader(file_path)
            df = reader.read()

            # Store header from first file
            if self.header is None:
                self.header = {
                    "sample_rate": reader.sample_rate,
                    "start_time": reader.start_time,
                    "filter_point": reader.filter_point,
                }
                self.sample_rate = reader.sample_rate
                self.start_time = reader.start_time

            dfs.append(df)

        if not dfs:
            raise ValueError(f"No data read from files: {self.files}")

        # Concatenate all dataframes
        self.data = pd.concat(dfs, ignore_index=True)

        # Build metadata objects (cached for reuse)
        station_meta = self._build_station_metadata()
        run_meta = self._build_run_metadata()

        # Build ChannelTS objects
        ch_objs = []
        mapping = {
            "Bx": ("hx", "magnetic", 1),
            "By": ("hy", "magnetic", 2),
            "Bz": ("hz", "magnetic", 3),
            "Ex": ("ex", "electric", 4),
            "Ey": ("ey", "electric", 5),
        }

        for src, (code, ch_type, ch_num) in mapping.items():
            if src not in self.data.columns:
                self.logger.warning(f"Channel {src} not found in data")
                continue

            series = self.data[src].to_numpy()
            ch_metadata = self._get_channel_metadata(code, ch_num)

            # Create calibration filters
            filters_list = []

            if ch_type == "magnetic":
                # Magnetic calibration (invert for hy)
                invert = code == "hy"
                mag_filter = create_orange_magnetic_filter(code, invert=invert)
                filters_list.append(mag_filter)
            else:  # electric
                # Electric calibration
                dipole_length = (
                    self.dipole_length_ex if code == "ex" else self.dipole_length_ey
                )
                elec_filter = create_orange_electric_filter(code, dipole_length)
                filters_list.append(elec_filter)

            # Create ChannelResponse
            channel_response = None
            if filters_list:
                channel_response = ChannelResponse(filters_list=filters_list)
                for sequence, filter_obj in enumerate(filters_list, start=1):
                    ch_metadata.add_filter(
                        AppliedFilter(
                            name=filter_obj.name, sequence=sequence, applied=False
                        )
                    )

            # Create ChannelTS object
            ch = ChannelTS(
                channel_type=ch_type,
                data=series,
                channel_metadata=ch_metadata,
                run_metadata=run_meta,
                station_metadata=station_meta,
                channel_response=channel_response,
            )

            ch_objs.append(ch)

        return RunTS(
            array_list=ch_objs,
            station_metadata=station_meta,
            run_metadata=run_meta,
        )

    def _build_station_metadata(self) -> Station:
        """Build station metadata object."""
        s = Station()
        s.id = self.station_id
        s.location.latitude = self.latitude
        s.location.longitude = self.longitude
        s.location.elevation = self.elevation
        return s

    def _build_run_metadata(self) -> Run:
        """Build run metadata object."""
        r = Run()
        r.id = "a"  # Default run ID
        r.sample_rate = self.sample_rate if self.sample_rate else 0.0
        r.data_logger.manufacturer = "University of Adelaide"
        r.data_logger.model = "Orange Box"
        r.data_logger.type = "long-period MT"
        r.data_type = "LPMT"
        r.time_period.start = self.start_time.isoformat() if self.start_time else ""
        return r

    def _get_channel_metadata(self, component: str, channel_number: int):
        """Get metadata for a specific channel."""
        if component in ["hx", "hy", "hz"]:
            ch_metadata = Magnetic()
            ch_metadata.type = "magnetic"
            ch_metadata.units = "count"

            azimuth_map = {"hx": 0, "hy": 90, "hz": 0}
            tilt_map = {"hx": 0, "hy": 0, "hz": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = tilt_map.get(component, 0)

            ch_metadata.sensor.manufacturer = "Bartington"
            ch_metadata.sensor.model = "Mag-03"
            ch_metadata.sensor.type = "fluxgate"
        else:
            ch_metadata = Electric()
            ch_metadata.type = "electric"
            ch_metadata.units = "count"

            azimuth_map = {"ex": 0, "ey": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = 0

            if component == "ex":
                ch_metadata.dipole_length = self.dipole_length_ex
            elif component == "ey":
                ch_metadata.dipole_length = self.dipole_length_ey

        ch_metadata.component = component
        ch_metadata.channel_number = channel_number
        ch_metadata.sample_rate = self.sample_rate if self.sample_rate else 0.0
        if self.start_time is not None:
            ch_metadata.time_period.start = self.start_time.isoformat()

        return ch_metadata


# ==============================================================================
# Convenience Function
# ==============================================================================


def read_orange(data_path: Union[str, Path, List[Union[str, Path]]], **kwargs) -> RunTS:
    """
    Read Orange Box binary file(s) and return a RunTS.

    Convenience function to read legacy UoA/Flinders Orange Box long-period
    magnetotelluric binary files with automatic file concatenation.

    :param data_path: Path to .BIN file(s) or list of file paths
    :type data_path: str, Path, or list
    :param kwargs: Additional options (see OrangeReader for details)
    :type kwargs: dict
    :return: RunTS object with channels hx, hy, hz, ex, ey
    :rtype: RunTS

    :Keyword Arguments:
        * **station_id** (str) - Station identifier (required)
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 100.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 100.0)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> from mt_io.uoa import read_orange
        >>> # Single file
        >>> run_ts = read_orange('HFM1-000.BIN', station_id='ST61')
        >>>
        >>> # Multiple files (glob pattern)
        >>> from glob import glob
        >>> files = sorted(glob('HFM1-*.BIN'))
        >>> run_ts = read_orange(files, station_id='ST61',
        ...                      dipole_length_ex=100.0, dipole_length_ey=100.0)
    """
    # Handle glob patterns
    if isinstance(data_path, (str, Path)):
        data_path = Path(data_path)
        if "*" in str(data_path):
            from glob import glob

            files = sorted(glob(str(data_path)))
            if not files:
                raise ValueError(f"No files found matching pattern: {data_path}")
            data_path = files

    reader = OrangeReader(data_path, **kwargs)
    return reader.read()
