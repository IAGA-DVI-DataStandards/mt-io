# -*- coding: utf-8 -*-
"""
=======
PR6-24
=======

    * reads Earth Data PR6-24 (EDL) ASCII files into a RunTS
    * builds the University of Adelaide sensor and interface response

The PR6-24 is a three- or six-channel 24-bit field datalogger. UoA MT
systems paired it with external magnetic sensors, electrodes and analogue
interface electronics. It writes one file per channel, named
{station}YYMMDDhhmmss.{CHANNEL}, either in day-numbered folders (001-366)
or flat. File length is set at acquisition time by mseed_filesize in
recorder.ini, so it varies between deployments.

The recorder writes either ASCII or miniSEED to the same file names, so
the format is detected from the first record header rather than the
extension. Both hold input voltage in microVolt. ASCII carries no header,
so its rate comes from recorder.ini, the caller, or the gap between
consecutive file stamps, snapped to a rate the instrument supports.
miniSEED carries rate, start time and sample count of its own.

Filter gains are stored forward, physical to recorded, as MTH5 divides by
them when removing the response:

    hx, hy   nT    -> uV    142.857         (Bartington Mag-03)
    hz       nT    -> uV    142.857 * 0.4   (15k/10k divider to the logger)
    ex, ey   mV/km -> uV    L * 10          (dipole length, terminal box)

Broadband systems use LEMI-120 coils with a normalized .rsp response, so
the flat-band 200 mV/nT sensitivity is carried as a separate filter.

@author: ben kay (ben@auscope.org.au)

:license: MIT

"""

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from loguru import logger
from mt_metadata.timeseries import AppliedFilter, Electric, Magnetic, Run, Station
from mt_metadata.timeseries.filters import (
    ChannelResponse,
    CoefficientFilter,
    FrequencyResponseTableFilter,
)
from mt_timeseries import ChannelTS, RunTS

# ==============================================================================
# Hardware Calibration Constants
# ==============================================================================

# PR6-24 digitiser ranges. ASCII is written already scaled to microVolt
# (EDM 021 4.2.9), so gain selection changes resolution, not units.
ADC_BITS = 24
ADC_LOW_GAIN_FULL_SCALE_V = 8.388  # ~1 uV per bit
ADC_HIGH_GAIN_FULL_SCALE_V = 0.8388  # ~0.1 uV per bit

# Bz voltage divider (hardware-fixed)
BZ_DIVIDER_R_TOP = 15000.0  # 15 kOhm resistor to sensor
BZ_DIVIDER_R_BOTTOM = 10000.0  # 10 kOhm resistor to ground
BZ_DIVIDER_RATIO = BZ_DIVIDER_R_BOTTOM / (BZ_DIVIDER_R_TOP + BZ_DIVIDER_R_BOTTOM)  # 0.4

# Rates the PR6-24 can be set to, EDM 021 1.3
EDL_SAMPLE_RATES = (
    1, 2, 4, 5, 10, 20, 25, 40, 50, 75, 100, 120, 125, 150,
    200, 250, 300, 375, 500, 600, 750, 1000, 3000,
)

# Standard field layout: ex north, ey east. Anything else is
# recorded in the field notes and passed in by the caller.
NOMINAL_AZIMUTH = {"ex": 0.0, "ey": 90.0}

# Electric field terminal box gain (hardware-fixed)
E_TERMINAL_BOX_GAIN = 10.0  # x10 pre-amplifier

# Bartington Mag-03 fluxgate, +/-10 V over +/-70,000 nT
BARTINGTON_NT_PER_V = 70000.0 / 10.0  # 7000 nT/V
BARTINGTON_NT_PER_UV = BARTINGTON_NT_PER_V / 1e6  # 0.007 nT/uV

# LEMI-120 induction coil, flat-band sensitivity
LEMI120_SENSITIVITY_MV_PER_NT = 400.0  # 400 mV/nT
LEMI120_UV_PER_NT_SPEC = LEMI120_SENSITIVITY_MV_PER_NT * 1e3  # 400,000 uV/nT
LEMI120_NT_PER_UV = 1.0 / LEMI120_UV_PER_NT_SPEC  # 2.5e-6 nT/uV

# Sensitivities forward, physical to recorded, as MTH5 divides by the gain
BARTINGTON_UV_PER_NT = 1.0 / BARTINGTON_NT_PER_UV  # 142.857 uV/nT
LEMI120_UV_PER_NT = LEMI120_UV_PER_NT_SPEC  # 400,000 uV/nT


def read_uoa_coil_response(
    calibration_fn: Union[str, Path], coil_number: Optional[str] = None
) -> FrequencyResponseTableFilter:
    """
    Read LEMI-120 or other coil calibration from .rsp file.

    Auto-detects whether the .rsp file contains normalized or absolute amplitudes:
    - If flat response ~1.0: normalized (nT -> nT), needs DC gain filter
    - If flat response ~200: absolute mV/nT (nT -> mV), no DC gain needed

    :param calibration_fn: Path to .rsp calibration file
    :type calibration_fn: str or Path
    :param coil_number: Optional coil serial number for identification
    :type coil_number: str, optional
    :return: Frequency response table filter
    :rtype: FrequencyResponseTableFilter

    :Example:
        >>> filter_obj = read_uoa_coil_response("l120_sensor01.rsp", coil_number="01")
        >>> print(f"Normalized: {filter_obj.units_in == filter_obj.units_out}")

    **File Format** (.rsp files):

        Line 1: Channel type ('B' for magnetic)
        Line 2: Column headers (freq, amp, phas)
        Lines 3+: Calibration data (frequency Hz, amplitude, phase degrees)

    .. note::
        Auto-detection uses max amplitude: if > 10, assumes absolute (mV/nT).
        If < 10, assumes normalized (relative to flat response).
    """
    calibration_fn = Path(calibration_fn)

    # Read the file (skip first 2 header lines)
    # Format: frequency (Hz), amplitude, phase (degrees)
    cal_data = np.loadtxt(calibration_fn, skiprows=2)

    # Auto-detect normalization by checking amplitude range
    # Normalized files have flat response ~1.0, absolute files ~100-200
    max_amplitude = np.max(cal_data[:, 1])
    is_normalized = max_amplitude < 10.0

    # Create frequency response filter
    fap = FrequencyResponseTableFilter()
    fap.frequencies = cal_data[:, 0]  # Hz
    fap.amplitudes = cal_data[:, 1]
    fap.phases = np.deg2rad(cal_data[:, 2])  # Convert degrees to radians
    fap.name = (
        f"lemi_120_{coil_number}_response" if coil_number else "lemi_120_response"
    )
    fap.type = "fap"
    fap.calibration_date = "1970-01-01T00:00:00+00:00"  # Default

    if is_normalized:
        # Normalized: nT -> nT (relative correction)
        fap.units_in = "nanoTesla"
        fap.units_out = "nanoTesla"
        fap.comments = f"normalized coil response from {calibration_fn.name}"
    else:
        # absolute, nT to milliVolt, so it already carries the sensitivity
        fap.units_in = "nanoTesla"
        fap.units_out = "milliVolt"
        fap.comments = f"coil response in mV/nT from {calibration_fn.name}"

    return fap


def create_mv_to_uv_filter() -> CoefficientFilter:
    """
    Create the milliVolt to microVolt step.

    A coil response given in mV/nT stops one prefix short of the recorded
    units, so this carries it the rest of the way.

    :return: coefficient filter, milliVolt to microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    f = CoefficientFilter()
    f.name = "uoa_millivolt_to_microvolt"
    f.units_in = "milliVolt"
    f.units_out = "microVolt"
    f.gain = 1000.0
    f.comments = "milliVolt to microVolt, to match the recorded units."
    return f


def create_bz_divider_filter() -> CoefficientFilter:
    """
    Create the Bz voltage divider filter.

    The UoA long-period systems put a 15k/10k divider between the Bz fluxgate
    and the logger, so the recorded voltage is 0.4 of the sensor output.

    :return: coefficient filter, sensor microVolt to logger microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    bz_filter = CoefficientFilter()
    bz_filter.name = "uoa_bz_voltage_divider"
    bz_filter.units_in = "microVolt"
    bz_filter.units_out = "microVolt"
    bz_filter.gain = BZ_DIVIDER_RATIO  # 0.4 (forward: sensor -> logger)
    bz_filter.comments = (
        "Bz voltage divider, 15 kOhm/10 kOhm. The logger sees 0.4 of the "
        "sensor output. Bartington fluxgate only."
    )
    return bz_filter


def create_efield_gain_filter(gain: float = E_TERMINAL_BOX_GAIN) -> CoefficientFilter:
    """
    Create the electric field terminal box filter.

    The terminal box holds a x10 pre-amplifier ahead of the logger. Not every
    deployment used one, so the gain is settable and 1.0 means the electrodes
    fed the logger directly.

    :param gain: forward gain, electrode to logger
    :type gain: float
    :return: coefficient filter, electrode microVolt to logger microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    efield_filter = CoefficientFilter()
    efield_filter.name = "uoa_efield_terminal_box_gain"
    efield_filter.units_in = "microVolt"
    efield_filter.units_out = "microVolt"
    efield_filter.gain = gain  # forward: electrode -> logger
    efield_filter.comments = (
        f"E-field terminal box, x{gain:g} pre-amplifier ahead of the logger."
    )
    return efield_filter


def create_lemi120_dc_gain_filter(component: str) -> CoefficientFilter:
    """
    Create the LEMI-120 flat-band sensitivity filter.

    Only needed when the .rsp file holds a normalized response (nT to nT).

    :param component: component name ('hx', 'hy' or 'hz')
    :type component: str
    :return: coefficient filter, nanoTesla to microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    dc_filter = CoefficientFilter()
    dc_filter.name = f"lemi120_dc_gain_{component}"
    dc_filter.units_in = "nanoTesla"
    dc_filter.units_out = "microVolt"
    dc_filter.gain = LEMI120_UV_PER_NT  # 200,000 uV/nT (forward)
    dc_filter.comments = "LEMI-120 flat-band sensitivity: 200 mV/nT"
    return dc_filter


def create_bartington_calibration_filter(component: str) -> CoefficientFilter:
    """
    Create the Bartington Mag-03 fluxgate filter.

    The sensor puts out +/-10 V over +/-70,000 nT, so 142.857 microVolt per nT.
    For Bz this describes the sensor output, ahead of the divider filter.

    :param component: component name ('hx', 'hy' or 'hz')
    :type component: str
    :return: coefficient filter, nanoTesla to microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    bart_filter = CoefficientFilter()
    bart_filter.name = f"uoa_bartington_{component}"
    bart_filter.units_in = "nanoTesla"
    bart_filter.units_out = "microVolt"
    bart_filter.gain = BARTINGTON_UV_PER_NT
    bart_filter.comments = (
        f"Bartington Mag-03 {component.upper()}, +/-10 V over +/-70,000 nT, "
        "so 142.857 uV/nT."
    )
    return bart_filter


def create_dipole_length_filter(
    component: str, dipole_length: float, azimuth: Optional[float] = None
) -> CoefficientFilter:
    """
    Create the dipole length filter for an electric channel.

    1 mV/km is 1 microVolt per metre, so a field E across a dipole of length L
    metres gives E * L microVolt.

    The gain also carries the sign. The electric channels are inverted in
    hardware, so a dipole laid the standard way, ex north and ey east, records
    the negative of the field along it. Laying a dipole south or west reverses
    that again and the two cancel. The azimuth therefore decides the sign, and
    it has to be supplied because nothing downstream applies it: aurora and
    mth5 store `measurement_azimuth` but never rotate on it.

    :param component: component name ('ex' or 'ey')
    :type component: str
    :param dipole_length: dipole length in metres
    :type dipole_length: float
    :param azimuth: as-laid azimuth in degrees, default 0 for ex and 90 for ey
    :type azimuth: float, optional
    :return: coefficient filter, milliVolt per kilometer to microVolt
    :rtype: :class:`mt_metadata.timeseries.filters.CoefficientFilter`
    """
    if dipole_length <= 0:
        dipole_length = 1.0  # Fallback

    nominal = NOMINAL_AZIMUTH[component]
    if azimuth is None:
        azimuth = nominal
    # a dipole within 90 degrees of nominal points the standard way
    reversed_dipole = abs(((azimuth - nominal) + 180) % 360 - 180) > 90
    sign = 1.0 if reversed_dipole else -1.0

    dipole_filter = CoefficientFilter()
    dipole_filter.name = f"uoa_dipole_{component}_{dipole_length:.1f}m"
    dipole_filter.units_in = "milliVolt per kilometer"
    dipole_filter.units_out = "microVolt"
    dipole_filter.gain = sign * dipole_length
    dipole_filter.comments = (
        f"{dipole_length} m dipole laid at {azimuth:.0f} deg, nominal "
        f"{nominal:.0f}; sign {'+' if sign > 0 else '-'} because the hardware "
        f"inverts E and the dipole is "
        f"{'reversed, which cancels it' if reversed_dipole else 'the standard way'}"
    )
    return dipole_filter


# Every EDL file carries a YYMMDDhhmmss stamp, e.g. FR17_131017000000.BX
EDL_TIMESTAMP = re.compile(r"(\d{12})$")


def parse_edl_timestamp(path: Union[str, Path]) -> Optional[datetime]:
    """
    Pull the acquisition start time out of an EDL file name.

    :param path: Path to an EDL data file
    :type path: str or Path
    :return: UTC start time, or None if the name carries no stamp
    :rtype: datetime or None
    """
    match = EDL_TIMESTAMP.search(Path(path).stem)
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(1), "%y%m%d%H%M%S")
    except ValueError:
        return None
    return stamp.replace(tzinfo=timezone.utc)


def sort_by_timestamp(files: List[Union[str, Path]]) -> List[Path]:
    """
    Order EDL files by the stamp in their name.

    Day folders restart at 001 each January, so a deployment that runs over
    new year sorts wrongly by path: 001 comes before 365 and the new year
    lands ahead of December. The stamp carries the year, so sort on that.
    Files with no stamp keep their path order at the end.

    :param files: EDL data files
    :type files: list of str or :class:`pathlib.Path`
    :return: files in chronological order
    :rtype: list of :class:`pathlib.Path`
    """
    stamped, unstamped = [], []
    for fn in files:
        stamp = parse_edl_timestamp(fn)
        (stamped if stamp else unstamped).append((stamp, Path(fn)))
    stamped.sort(key=lambda pair: pair[0])
    unstamped.sort(key=lambda pair: pair[1])
    return [fn for _, fn in stamped] + [fn for _, fn in unstamped]


# A miniSEED record opens with a six digit sequence number and a quality code
MSEED_QUALITY = (b"D", b"R", b"Q", b"M")


def is_miniseed(fn: Union[str, Path]) -> bool:
    """
    Tell miniSEED apart from ASCII by looking at the first record header.

    The PR6-24 writes either format to the same .BX/.BY/.BZ/.EX/.EY names,
    so the extension cannot be used to decide.

    :param fn: path to an EDL data file
    :type fn: str or :class:`pathlib.Path`
    :return: True if the file starts with a miniSEED record header
    :rtype: bool
    """
    with open(fn, "rb") as data_file:
        head = data_file.read(8)
    return len(head) >= 8 and head[:6].isdigit() and head[6:7] in MSEED_QUALITY


def read_edl_miniseed(fn: Union[str, Path]):
    """
    Read one miniSEED channel file.

    Values are the same recorder-scaled microVolt the ASCII files hold, but
    the header also carries the start time, sample rate and sample count, so
    none of those need to be guessed from the file name.

    :param fn: path to a miniSEED EDL file
    :type fn: str or :class:`pathlib.Path`
    :return: samples, start time and sample rate
    :rtype: tuple of (:class:`numpy.ndarray`, datetime, float)
    """
    try:
        from obspy import read as obspy_read
    except ImportError as error:
        raise ImportError(
            "Reading PR6-24 miniSEED needs obspy: pip install mt-io[obspy]"
        ) from error

    stream = obspy_read(Path(fn).as_posix())
    stream.merge(method=0)
    trace = stream[0]
    start = trace.stats.starttime.datetime.replace(tzinfo=timezone.utc)
    return (
        trace.data.astype(float),
        start,
        float(trace.stats.sampling_rate),
    )


def parse_edl_station(path: Union[str, Path]) -> Optional[str]:
    """
    Pull the station id out of an EDL file name.

    Names are {station}YYMMDDhhmmss.{CHANNEL}, so the stamp is a fixed
    twelve digits at the end and whatever precedes it is the station.
    Reading it that way avoids having to know the prefix in advance, which
    matters because field names are not always tidy: CP1L07 was written
    CP1L7_ and CP1L09 was written CP1B09_.

    :param path: path to an EDL data file
    :type path: str or :class:`pathlib.Path`
    :return: station id, or None if the name carries no stamp
    :rtype: str or None
    """
    match = EDL_TIMESTAMP.search(Path(path).stem)
    if match is None:
        return None
    return Path(path).stem[: match.start()].rstrip("_-") or None


def count_samples(fn: Union[str, Path]) -> int:
    """
    Count samples in an EDL file, whichever format it is in.

    ASCII holds one value per line. miniSEED carries the count in its record
    headers, which is read without unpacking the data.

    An empty file is allowed to report zero. Bytes with no samples in them is
    not a data file at all and raises, because returning zero there puts a run
    of no length into a collection and silently breaks contiguity.

    :param fn: path to an EDL data file
    :type fn: str or :class:`pathlib.Path`
    :return: number of samples
    :rtype: int
    :raises ValueError: if the file holds bytes but no samples
    """
    if is_miniseed(fn):
        from obspy import read as obspy_read

        stream = obspy_read(Path(fn).as_posix(), headonly=True)
        return int(sum(trace.stats.npts for trace in stream))

    total = 0
    with open(fn, "rb") as data_file:
        while True:
            chunk = data_file.read(1 << 20)
            if not chunk:
                break
            total += chunk.count(b"\n")

    if total == 0 and Path(fn).stat().st_size > 0:
        raise ValueError(f"{Path(fn).name} holds bytes but no samples")

    return total


def snap_sample_rate(rate: float, tolerance: float = 0.02) -> Optional[float]:
    """
    Snap an estimated rate to one the PR6-24 can actually be set to.

    :param rate: estimated sample rate in Hz
    :type rate: float
    :param tolerance: allowed fractional error, defaults to 0.02
    :type tolerance: float, optional
    :return: the matching rate from EDL_SAMPLE_RATES, or None
    :rtype: float or None
    """
    if rate <= 0:
        return None
    nearest = min(EDL_SAMPLE_RATES, key=lambda valid: abs(valid - rate))
    if abs(nearest - rate) / nearest <= tolerance:
        return float(nearest)
    return None


def infer_sample_rate(
    files: List[Union[str, Path]], max_pairs: int = 25
) -> Optional[float]:
    """
    Work out the sample rate from consecutive file names.

    Each file is stamped with its start time, so a file that runs on into the
    next one holds ``start(next) - start(this)`` seconds of data and the rate
    follows from its sample count. Startup files can overlap or leave gaps, so
    the most common answer is taken rather than the average.

    :param files: EDL data files for one channel
    :type files: list of str or :class:`pathlib.Path`
    :param max_pairs: how many consecutive pairs to sample, defaults to 25
    :type max_pairs: int, optional
    :return: sample rate in Hz, or None if the files do not agree
    :rtype: float or None
    """
    stamped = sorted(
        ((parse_edl_timestamp(f), Path(f)) for f in files if parse_edl_timestamp(f)),
        key=lambda pair: pair[0],
    )
    if len(stamped) < 2:
        return None

    # sample across the whole deployment, not just the start, because setup
    # fragments at the beginning can overlap and skew the vote
    pairs = list(zip(stamped, stamped[1:]))
    if len(pairs) > max_pairs:
        step = len(pairs) / max_pairs
        pairs = [pairs[int(i * step)] for i in range(max_pairs)]

    counts = Counter()
    for (t_this, fn_this), (t_next, _) in pairs:
        span = (t_next - t_this).total_seconds()
        if span <= 0:
            continue
        try:
            n_samples = count_samples(fn_this)
        except ValueError:
            continue
        rate = snap_sample_rate(n_samples / span)
        if rate is not None:
            counts[rate] += 1

    if not counts:
        return None
    rate, votes = counts.most_common(1)[0]
    if votes < 2:
        logger.warning(f"Sample rate not consistent across files: {dict(counts)}")
        return None
    logger.info(f"Inferred sample rate {rate} Hz from {votes} file pairs")
    return float(rate)


def decimate_series(
    data: np.ndarray, factor: int, max_stage: int = 8
) -> np.ndarray:
    """
    Decimate one channel with anti-alias filtering.

    Long-period EDL is often recorded at 10 Hz when 1 Hz covers the band of
    interest, and the extra rate only adds storage and bands that sit in the
    fluxgate noise floor. Large factors are split into stages because a single
    high order FIR is numerically poor.

    :param data: samples
    :type data: :class:`numpy.ndarray`
    :param factor: integer decimation factor
    :type factor: int
    :param max_stage: largest factor per stage, defaults to 8
    :type max_stage: int, optional
    :return: decimated samples
    :rtype: :class:`numpy.ndarray`
    """
    from scipy.signal import decimate as scipy_decimate

    if factor <= 1:
        return np.asarray(data, dtype=float)

    stages, remaining = [], int(factor)
    for candidate in range(min(max_stage, remaining), 1, -1):
        while remaining % candidate == 0 and remaining > 1:
            stages.append(candidate)
            remaining //= candidate
    if remaining > 1:
        stages.append(remaining)

    out = np.asarray(data, dtype=float)
    for stage in stages:
        out = scipy_decimate(out, stage, ftype="fir", zero_phase=True)
    return out


# ==============================================================================
# EDL Data File Reader
# ==============================================================================


class UoADataReader:
    """
    Reads EDL ASCII data files for a single channel.

    Handles multiple file organization patterns:
    - Day-numbered folders (001-366)
    - Flat directory structure
    - Pre-concatenated single files

    :param channel: Channel name (BX, BY, BZ, EX, EY, TP, etc.)
    :type channel: str
    :param data_path: Path to data directory or specific file
    :type data_path: Path
    :param station_prefix: Station identifier prefix (e.g., 'EDL_', 'MT001_')
    :type station_prefix: str, optional

    :Example:
        >>> reader = UoADataReader('BX', Path('/data/site1'), station_prefix='EDL_')
        >>> data = reader.read()
        >>> print(f"Read {len(data)} samples")
    """

    def __init__(
        self, channel: str, data_path, station_prefix: Optional[str] = None
    ):
        self.channel = channel.upper()
        # a list comes from a collection, which has already grouped the files
        # into runs, so it is kept as given rather than searched for
        if isinstance(data_path, (list, tuple, set)):
            self.given_files = [Path(f) for f in data_path]
            self.data_path = None
        else:
            self.given_files = None
            self.data_path = Path(data_path)
        self.station_prefix = station_prefix or ""
        self.files: List[Path] = []
        # (start, sample_rate) per file, filled in for miniSEED
        self.segments: List[tuple] = []
        self.logger = logger

    def find_files(self) -> List[Path]:
        """
        Find all data files for this channel.

        Searches for {station_prefix}*.{channel} anywhere below data_path,
        which covers day folders and flat directories alike, then falls back
        to *.{channel} for files that carry no station prefix. Results are
        ordered by the stamp in the file name, not by path.

        Archives packed on a Mac carry an AppleDouble "._name" sidecar beside
        every file. They share the data extension but hold resource fork bytes,
        so they are dropped here rather than parsed.

        Given a list of files, only the ones for this channel are taken.

        :return: List of file paths sorted chronologically
        :rtype: list of Path
        """
        if self.given_files is not None:
            found = [
                f
                for f in self.given_files
                if f.suffix.lstrip(".").upper() == self.channel
                and not f.name.startswith("._")
            ]
            if found:
                return sort_by_timestamp(found)
            self.logger.warning(f"No {self.channel} files in the list given")
            return []

        if self.data_path.is_dir():
            found = [f for f in self.data_path.glob(f"**/*.{self.channel}")
                     if not f.name.startswith("._")]
            if found and self.station_prefix:
                want = self.station_prefix.rstrip("_-").lower()
                matched = [
                    f for f in found
                    if (parse_edl_station(f) or "").lower() == want
                ]
                if matched:
                    found = matched
                else:
                    seen = sorted(
                        {parse_edl_station(f) or "?" for f in found}
                    )
                    self.logger.warning(
                        f"No {self.channel} files for station "
                        f"{self.station_prefix.rstrip('_-')}; using all "
                        f"{len(found)} found instead, from {seen}"
                    )
            if found:
                found = sort_by_timestamp(found)
                self.logger.info(f"Found {len(found)} files for {self.channel}")
                return found

        # If data_path is a file, use it directly
        elif self.data_path.is_file():
            self.logger.info(
                f"Using specified file for {self.channel}: {self.data_path.name}"
            )
            return [self.data_path]

        self.logger.warning(
            f"No files found for channel {self.channel} in {self.data_path}"
        )
        return []

    def read(self) -> np.ndarray:
        """
        Read and concatenate all files for this channel.

        :return: Array of float values in microVolt (uV)
        :rtype: np.ndarray

        **File Format**:
            - ASCII text, one sample per line
            - Values in microVolt (uV)
            - No header
            - a file that will not parse is skipped, with an error logged
        """
        files = self.find_files()
        self.files = files
        if not files:
            self.logger.error(f"No data files found for channel {self.channel}")
            return np.array([])

        all_data = []
        self.segments = []
        for file_path in files:
            try:
                if is_miniseed(file_path):
                    data, start, rate = read_edl_miniseed(file_path)
                    self.segments.append((start, rate))
                else:
                    data = np.loadtxt(file_path, dtype=float, comments=None)
                    if data.ndim > 1:
                        data = data.flatten()

                all_data.append(data)
                self.logger.debug(f"Read {len(data)} samples from {file_path.name}")

            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                continue

        if not all_data:
            return np.array([])

        # Concatenate all data
        combined = np.concatenate(all_data)
        self.logger.info(
            f"Channel {self.channel}: {len(combined)} total samples from {len(files)} file(s)"
        )

        return combined


# ==============================================================================
# Main EDL Reader Class
# ==============================================================================


class UoAReader:
    """
    MTH5-compatible reader for Earth Data PR6-24 ASCII data files.

    Builds the UoA response chain alongside the data:
    - Bz voltage divider, 15 kOhm / 10 kOhm
    - electric field terminal box, x10
    - Bartington fluxgate or LEMI-120 coil sensitivity

    **File Discovery**:
        - Day-numbered folders (001-366)
        - Flat directories
        - Pre-concatenated files
        - ASCII or miniSEED, detected per file
        - Works without recorder.ini or GPS files

    :param data_path: Path to data directory or file
    :type data_path: str or Path
    :param sensor_type: 'bartington' or 'lemi120' (default: 'bartington')
    :type sensor_type: str, optional
    :param kwargs: Additional parameters (see below)
    :type kwargs: dict

    :Keyword Arguments:
        * **sample_rate** (float) - Sample rate in Hz, inferred from the file
          stamps when not given
        * **station_id** (str) - Station identifier (default: auto-detect from path)
        * **station_prefix** (str) - File prefix like 'EDL_', 'MT001_' (default: '')
        * **dipole_length_ex** (float) - Ex dipole length in meters (default: 1.0)
        * **dipole_length_ey** (float) - Ey dipole length in meters (default: 1.0)
        * **ex_azimuth** (float) - as-laid Ex azimuth in degrees (default: 0)
        * **ey_azimuth** (float) - as-laid Ey azimuth in degrees (default: 90)
        * **efield_gain** (float) - E terminal box gain, 1.0 if none was used
          (default: 10.0)
        * **declination** (float) - magnetic declination in degrees (default: 0)
        * **geographic_name** (str) - site name from the deployment notes
        * **acquired_by** (str) - operator from the deployment notes
        * **data_logger_id** (str) - recorder serial from the deployment notes
        * **magnetometer_id** (str) - sensor serial from the deployment notes
        * **decimate_to** (float) - decimate to this rate in Hz before the
          response is attached, e.g. 1.0 for long period (default: None)
        * **calibration_fn_bx** (str) - LEMI-120 .rsp file for Bx (if lemi120 mode)
        * **calibration_fn_by** (str) - LEMI-120 .rsp file for By (if lemi120 mode)
        * **calibration_fn_bz** (str) - LEMI-120 .rsp file for Bz (if lemi120 mode)
        * **latitude** (float) - Station latitude in decimal degrees
        * **longitude** (float) - Station longitude in decimal degrees
        * **elevation** (float) - Station elevation in meters

    :Example:
        >>> # Long-period with Bartington
        >>> reader = UoAReader('/data/site1', sensor_type='bartington',
        ...                     sample_rate=10.0,
        ...                     dipole_length_ex=50.0,
        ...                     dipole_length_ey=50.0)
        >>> run_ts = reader.read()
        >>>
        >>> # Broadband with LEMI-120
        >>> reader = UoAReader('/data/site1', sensor_type='lemi120',
        ...                     sample_rate=500.0,
        ...                     calibration_fn_bx='l120_01.rsp',
        ...                     calibration_fn_by='l120_02.rsp',
        ...                     calibration_fn_bz='l120_03.rsp',
        ...                     dipole_length_ex=50.0,
        ...                     dipole_length_ey=50.0)
        >>> run_ts = reader.read()

    .. note::
        - Sample rate is inferred from consecutive file names when not given
        - Samples stay as recorded logger voltage in microVolt
        - The Bz divider and terminal box gain are attached as response
          response filters, which aurora divides back out
        - For accurate E-field, provide actual dipole lengths in meters
    """

    def __init__(self, data_path, sensor_type: str = "bartington", **kwargs):
        self.logger = logger
        # a directory, a single file, or the file list a collection built for
        # one run
        if isinstance(data_path, (list, tuple, set)):
            self.data_path = [Path(f) for f in data_path]
        else:
            self.data_path = Path(data_path)
        self.sensor_type = sensor_type.lower()

        # Required parameters
        # may be None; read() then works it out from the file stamps
        self.sample_rate = kwargs.get("sample_rate", None)

        # Station identification
        self.station_id = kwargs.get("station_id", None)
        self.station_prefix = kwargs.get("station_prefix", "")

        # Dipole lengths for electric field conversion
        self.dipole_length_ex = kwargs.get("dipole_length_ex", 1.0)
        self.dipole_length_ey = kwargs.get("dipole_length_ey", 1.0)

        # as-laid electrode azimuths from the field notes. The standard is ex
        # north and ey east; south or west reverses the channel and the reader
        # corrects for it, because nothing downstream does.
        self.ex_azimuth = kwargs.get("ex_azimuth", NOMINAL_AZIMUTH["ex"])
        self.ey_azimuth = kwargs.get("ey_azimuth", NOMINAL_AZIMUTH["ey"])

        # optional decimation, applied to the samples before the response is
        # attached so nothing is lost. RunTS.decimate drops channel_response
        # while leaving the metadata claiming it is applied.
        self.decimate_to = kwargs.get("decimate_to", None)

        # identifiers and site description, all optional and all from the
        # deployment notes rather than anything in the data files
        # not every deployment ran the x10 terminal box, so allow 1.0
        self.efield_gain = kwargs.get("efield_gain", E_TERMINAL_BOX_GAIN)
        self.declination = kwargs.get("declination", 0.0)
        self.geographic_name = kwargs.get("geographic_name", None)
        self.acquired_by = kwargs.get("acquired_by", None)
        self.data_logger_id = kwargs.get("data_logger_id", None)
        self.magnetometer_id = kwargs.get("magnetometer_id", None)

        # Calibration files (LEMI-120 mode only)
        self.calibration_fn_bx = kwargs.get("calibration_fn_bx", None)
        self.calibration_fn_by = kwargs.get("calibration_fn_by", None)
        self.calibration_fn_bz = kwargs.get("calibration_fn_bz", None)

        # Optional location metadata
        self.latitude = kwargs.get("latitude", None)
        self.longitude = kwargs.get("longitude", None)
        self.elevation = kwargs.get("elevation", None)

        # Data storage
        self.data = None
        self.n_samples = 0
        self.start_time: Optional[datetime] = None

    def _get_station_id(self) -> str:
        """
        Determine station ID from kwargs or path.

        Priority:
        1. Explicit station_id parameter
        2. Parent directory name

        :return: Station identifier
        :rtype: str
        """
        if self.station_id:
            return self.station_id

        # a list holds the files of one run, which sit beside each other
        if isinstance(self.data_path, list):
            if not self.data_path:
                return "unknown"
            first = self.data_path[0]
            return parse_edl_station(first) or first.parent.name

        # Use parent directory name
        return (
            self.data_path.parent.name
            if self.data_path.is_file()
            else self.data_path.name
        )

    def read(self) -> RunTS:
        """
        Read EDL data files and return a RunTS object.

        - Stores RAW microVolt in data array
        - Describes the hardware response as filters, applied=True

        Process:
        1. Find and read all 5 MT channels (BX, BY, BZ, EX, EY)
        2. Store RAW microVolt
        3. Build calibration filters based on sensor type
        4. Build metadata objects
        5. Create ChannelTS objects with filters
        6. Return RunTS with all channels

        :return: RunTS object containing ChannelTS for bx, by, bz, ex, ey
        :rtype: RunTS

        **Calibration Filters Created**:

        For Bartington (long-period):
            - hx, hy: Bartington filter, nT to uV
            - hz: Bartington filter then the divider, nT to uV
            - ex, ey: dipole filter then the terminal box, mV/km to uV

        For LEMI-120 (broadband):
            - hx, hy, hz: .rsp response, plus the flat-band gain if normalized
            - ex, ey: dipole filter then the terminal box, mV/km to uV
        """
        source = (
            f"{len(self.data_path)} files"
            if isinstance(self.data_path, list)
            else self.data_path
        )
        self.logger.info(f"Reading EDL data from {source}")
        self.logger.info(
            f"Sensor type: {self.sensor_type}, Sample rate: {self.sample_rate} Hz"
        )

        # Define MT channels
        channels = ["BX", "BY", "BZ", "EX", "EY"]
        channel_data = {}
        channel_files = {}
        channel_segments = {}

        # Read each channel
        for channel in channels:
            reader = UoADataReader(channel, self.data_path, self.station_prefix)
            data = reader.read()

            if len(data) == 0:
                self.logger.warning(f"No data found for channel {channel}")
                continue

            channel_data[channel] = data
            channel_files[channel] = reader.files
            channel_segments[channel] = reader.segments

        # Check we have data
        if not channel_data:
            raise ValueError(f"No channel data found in {self.data_path}")

        # Find minimum length (trim all to same length)
        min_length = min(len(data) for data in channel_data.values())
        self.n_samples = min_length

        for channel in channel_data:
            channel_data[channel] = channel_data[channel][:min_length]

        self.logger.info(
            f"Read {min_length} samples across {len(channel_data)} channels"
        )

        header_rates = {rate for segs in channel_segments.values() for _, rate in segs}
        if self.sample_rate is None and header_rates:
            if len(header_rates) > 1:
                self.logger.warning(f"Mixed sample rates in headers: {header_rates}")
            self.sample_rate = max(header_rates)
            self.logger.info(f"Sample rate {self.sample_rate} Hz from miniSEED headers")
        if self.sample_rate is None:
            for files in channel_files.values():
                self.sample_rate = infer_sample_rate(files)
                if self.sample_rate is not None:
                    break
            if self.sample_rate is None:
                raise ValueError(
                    "sample_rate could not be determined; pass sample_rate explicitly"
                )

        native_rate = None
        if self.decimate_to:
            ratio = self.sample_rate / float(self.decimate_to)
            if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"Cannot decimate {self.sample_rate} Hz to "
                    f"{self.decimate_to} Hz, the factor is not a whole number"
                )
            factor = int(round(ratio))
            if factor > 1:
                for channel in list(channel_data):
                    channel_data[channel] = decimate_series(
                        channel_data[channel], factor
                    )
                native_rate = self.sample_rate
                self.sample_rate = float(self.decimate_to)
                self.n_samples = min(len(v) for v in channel_data.values())
                self.logger.info(
                    f"Decimated by {factor} to {self.sample_rate} Hz, "
                    f"{self.n_samples} samples"
                )

        # miniSEED carries the start time; ASCII does not, so fall back to
        # the stamp in the file name.
        stamps = [start for segs in channel_segments.values() for start, _ in segs]
        if not stamps:
            stamps = [
                stamp
                for files in channel_files.values()
                for stamp in (parse_edl_timestamp(f) for f in files)
                if stamp is not None
            ]
        self.start_time = min(stamps) if stamps else None
        if self.start_time is None:
            self.logger.warning(
                "No YYMMDDhhmmss stamp in any file name; start time left unset"
            )
        else:
            self.logger.info(f"Run starts {self.start_time.isoformat()}")

        # Files are concatenated end to end, so warn if they do not join up.
        # File length is a recorder setting, so compare total samples against
        # the wall-clock span rather than assuming a fixed file duration.
        # count_samples reads the file, so it is in the rate the file was
        # written at, which is not self.sample_rate once decimation has run
        file_rate = native_rate if native_rate else self.sample_rate
        for channel, files in channel_files.items():
            if channel not in channel_data or len(files) < 2:
                continue
            stamped = sorted(
                ((parse_edl_timestamp(f), f) for f in files if parse_edl_timestamp(f)),
                key=lambda pair: pair[0],
            )
            if len(stamped) < 2 or is_miniseed(stamped[-1][1]):
                continue
            # the run ends where the last file ends, not at its start stamp
            span = (stamped[-1][0] - stamped[0][0]).total_seconds() + (
                count_samples(stamped[-1][1]) / file_rate
            )
            recorded = len(channel_data[channel]) / self.sample_rate
            missing = span - recorded
            if missing > 1.0 / self.sample_rate:
                self.logger.warning(
                    f"{channel}: {missing:.1f} s missing between files; samples "
                    "after a gap will be dated early. Use UoACollection to split "
                    "the deployment into runs."
                )

        station_meta = self._build_station_metadata()
        run_meta = self._build_run_metadata()
        if self.start_time is not None:
            run_meta.time_period.start = self.start_time.isoformat()

        # Build ChannelTS objects with RAW data and calibration filters
        ch_objs: List[ChannelTS] = []

        # Mapping: file channel name -> (MTH5 code, type, channel_number)
        mapping = {
            "BX": ("hx", "magnetic", 0),
            "BY": ("hy", "magnetic", 1),
            "BZ": ("hz", "magnetic", 2),
            "EX": ("ex", "electric", 3),
            "EY": ("ey", "electric", 4),
        }

        for src, (code, ch_type, ch_num) in mapping.items():
            if src not in channel_data:
                self.logger.warning(f"Channel {src} not found in data")
                continue

            # Get RAW data in microVolt from ASCII file
            raw_uv = channel_data[src]

            # Build channel metadata
            ch_metadata = self._get_channel_metadata(code, ch_num)
            if self.start_time is not None:
                ch_metadata.time_period.start = self.start_time.isoformat()

            # Response filters describing what is in the data
            if ch_type == "magnetic":
                channel_response = self._create_magnetic_filters(code)
            else:  # electric
                channel_response = self._create_electric_filters(code)

            # Update metadata to reference filters
            if channel_response is not None:
                for stage, filter_obj in enumerate(
                    channel_response.filters_list, start=1
                ):
                    # applied=True means the response is present in the data,
                    # which is what tells aurora to divide it back out
                    ch_metadata.add_filter(
                        AppliedFilter(
                            name=filter_obj.name, stage=stage, applied=True
                        )
                    )

            # Create ChannelTS object with RAW data (microVolt)
            ch = ChannelTS(
                channel_type=ch_type,
                data=raw_uv,  # Store RAW microVolt (NOT calibrated)
                channel_metadata=ch_metadata,
                run_metadata=run_meta,
                station_metadata=station_meta,
                channel_response=channel_response,
            )

            ch_objs.append(ch)

        # Return RunTS with all channels
        return RunTS(
            array_list=ch_objs,
            station_metadata=station_meta,
            run_metadata=run_meta,
        )

    def _create_magnetic_filters(self, component: str) -> Optional[ChannelResponse]:
        """
        Build the response chain for a magnetic channel.

        Ordered physical to recorded: sensor calibration first, then the Bz
        divider where it applies.

        :param component: component name ('hx', 'hy' or 'hz')
        :type component: str
        :return: channel response, or None if a .rsp file is missing
        :rtype: :class:`mt_metadata.timeseries.filters.ChannelResponse` or None
        """
        filters = []

        if self.sensor_type == "bartington":
            # nT -> sensor microVolt, then the divider on Bz only
            filters.append(create_bartington_calibration_filter(component))

            if component == "hz":
                # Bz sits behind the 15k/10k divider on its way to the logger
                filters.append(create_bz_divider_filter())

        elif self.sensor_type == "lemi120":
            # LEMI-120 induction coils (broadband)
            # Load frequency response from .rsp file
            cal_file_map = {
                "hx": self.calibration_fn_bx,
                "hy": self.calibration_fn_by,
                "hz": self.calibration_fn_bz,
            }

            cal_fn = cal_file_map.get(component)
            if cal_fn is None:
                self.logger.warning(f"No calibration file specified for {component}")
                return None

            try:
                coil_filter = read_uoa_coil_response(cal_fn, coil_number=component)

                filters.append(coil_filter)

                # A normalized .rsp is nT -> nT, so the nT -> microVolt DC
                # sensitivity has to follow it to reach recorded units.
                if coil_filter.units_in == coil_filter.units_out == "nanoTesla":
                    filters.append(create_lemi120_dc_gain_filter(component))
                elif coil_filter.units_out == "milliVolt":
                    # the table already holds the sensitivity, it just stops at
                    # milliVolt while the logger records microVolt
                    filters.append(create_mv_to_uv_filter())
            except Exception as e:
                self.logger.error(f"Error reading calibration file {cal_fn}: {e}")
                return None
        else:
            raise ValueError(f"Unknown sensor type: {self.sensor_type}")

        if filters:
            return ChannelResponse(filters_list=filters)
        return None

    def _create_electric_filters(self, component: str) -> Optional[ChannelResponse]:
        """
        Build the response chain for an electric channel.

        Ordered physical to recorded: dipole length first, then the terminal
        box gain.

        :param component: component name ('ex' or 'ey')
        :type component: str
        :return: channel response
        :rtype: :class:`mt_metadata.timeseries.filters.ChannelResponse`
        """
        filters = []

        # mV/km -> electrode microVolt, then the terminal box
        dipole_length = (
            self.dipole_length_ex if component == "ex" else self.dipole_length_ey
        )
        if dipole_length <= 0:
            self.logger.warning(
                f"Dipole length for {component} is {dipole_length}m, using 1m"
            )
            dipole_length = 1.0

        azimuth = self.ex_azimuth if component == "ex" else self.ey_azimuth
        filters.append(
            create_dipole_length_filter(component, dipole_length, azimuth)
        )
        filters.append(create_efield_gain_filter(self.efield_gain))

        return ChannelResponse(filters_list=filters)

    def _build_station_metadata(self) -> Station:
        """
        Build Station metadata object.

        Uses provided location or defaults to zeros.

        :return: Station metadata
        :rtype: Station
        """
        s = Station()
        s.id = self._get_station_id()

        # Location (use provided or default to 0,0,0)
        s.location.latitude = self.latitude if self.latitude is not None else 0.0
        s.location.longitude = self.longitude if self.longitude is not None else 0.0
        s.location.elevation = self.elevation if self.elevation is not None else 0.0

        s.location.declination.value = self.declination

        # UoA long-period fluxgates were squared to magnetic north, so the
        # channels sit in the geomagnetic frame, not the geographic one
        s.orientation.reference_frame = "geomagnetic"
        if self.geographic_name:
            s.geographic_name = self.geographic_name
        if self.acquired_by:
            s.acquired_by.name = self.acquired_by

        return s

    def _build_run_metadata(self) -> Run:
        """
        Build Run metadata object.

        :return: Run metadata
        :rtype: Run
        """
        r = Run()
        r.id = "a"  # Default run ID
        r.sample_rate = self.sample_rate

        # Data logger information
        r.data_logger.model = "PR6-24"
        r.data_logger.manufacturer = "Earth Data"
        if self.data_logger_id:
            r.data_logger.id = str(self.data_logger_id)
        r.data_logger.type = "digitizer"

        # Data type
        r.data_type = "BBMT" if self.sensor_type == "lemi120" else "LPMT"

        return r

    def _get_channel_metadata(self, component: str, channel_number: int):
        """
        Build channel metadata object.

        :param component: Component name ('hx', 'hy', 'hz', 'ex', 'ey')
        :type component: str
        :param channel_number: Channel number (0-4)
        :type channel_number: int
        :return: Channel metadata (Magnetic or Electric)
        :rtype: Magnetic or Electric
        """
        if component in ["hx", "hy", "hz"]:
            ch_metadata = Magnetic()
            ch_metadata.type = "magnetic"
            # UoA fluxgates were set to magnetic north
            azimuth_map = {"hx": 0, "hy": 90, "hz": 0}
            tilt_map = {"hx": 0, "hy": 0, "hz": 90}
            ch_metadata.measurement_azimuth = azimuth_map.get(component, 0)
            ch_metadata.measurement_tilt = tilt_map.get(component, 0)
            # Store RAW microVolt (following MTH5 standard like Zen, Phoenix, NIMS)
            ch_metadata.units = "microVolt"

            # Sensor information
            ch_metadata.sensor.manufacturer = (
                "Bartington" if self.sensor_type == "bartington" else "LEMI"
            )
            ch_metadata.sensor.model = (
                "Mag-03" if self.sensor_type == "bartington" else "LEMI-120"
            )
            ch_metadata.sensor.type = (
                "fluxgate" if self.sensor_type == "bartington" else "induction coil"
            )
            if self.magnetometer_id:
                ch_metadata.sensor.id = str(self.magnetometer_id)

        else:  # electric
            ch_metadata = Electric()
            ch_metadata.type = "electric"
            # Store RAW microVolt (following MTH5 standard)
            ch_metadata.units = "microVolt"

            # Store dipole length in metadata for filter application
            # the dipole filter carries the sign that brings the channel
            # into the nominal north/east frame, so report that frame here
            if component == "ex":
                ch_metadata.dipole_length = self.dipole_length_ex
            else:
                ch_metadata.dipole_length = self.dipole_length_ey
            ch_metadata.measurement_azimuth = NOMINAL_AZIMUTH[component]

            ch_metadata.measurement_tilt = 0.0

        ch_metadata.component = component
        ch_metadata.channel_number = channel_number
        ch_metadata.sample_rate = self.sample_rate

        return ch_metadata


# ==============================================================================
# Convenience Function (Entry Point)
# ==============================================================================


def read_uoa(data_path: Union[str, Path], **kwargs) -> RunTS:
    """
    Read Earth Data PR6-24 (EDL) ASCII data into a RunTS.

    Thin wrapper around :class:`UoAReader`; see that class for the full list
    of keyword arguments and for what the response filters describe.

    :param data_path: path to a station directory or a single file
    :type data_path: str or :class:`pathlib.Path`
    :param kwargs: passed straight through to :class:`UoAReader`
    :type kwargs: dict
    :return: run time series with the hardware response attached
    :rtype: :class:`mt_timeseries.RunTS`

    :Example:

    .. code-block:: python

        >>> from mt_io.uoa import read_uoa
        >>> run = read_uoa("/data/FR17", sensor_type="bartington",
        ...                sample_rate=10.0, dipole_length_ex=50.0,
        ...                dipole_length_ey=50.0)
    """
    reader = UoAReader(data_path, **kwargs)
    return reader.read()
