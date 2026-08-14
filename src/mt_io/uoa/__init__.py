"""
=====================================
University of Adelaide MT instruments
=====================================

Readers for the two loggers UoA and Flinders used for long-period and
broadband MT.

**Earth Data PR6-24** (:func:`read_uoa`, :class:`UoAReader`)
    Three- or six-channel 24-bit field datalogger, paired with external
    sensors and analogue interface electronics. One file per channel, written
    as either ASCII or miniSEED under the same names, so the format is
    detected from the file header. Both hold input voltage in microVolt.
    Sensors are Bartington Mag-03 fluxgates for long period or LEMI-120
    induction coils for broadband. The Bz channel sits behind a 15 kOhm /
    10 kOhm divider in fluxgate mode, and the electric channels behind a x10
    terminal box.

**Orange Box** (:func:`read_orange`, :class:`OrangeReader`)
    Legacy eight-channel long-period logger, roughly 2000 to 2014. Binary,
    21 bytes per sample, with a 40 byte ASCII header. Bartington fluxgates
    over +/-70,000 nT. By, Ex and Ey are inverted by the hardware. Electric
    full scale is 100000 uV, or 25000 uV on boxes predating the +/-10 V
    rewiring.

:class:`UoACollection` groups PR6-24 files into runs for MTH5.

Filter gains are stored forward, physical to recorded, because MTH5 divides
by them when removing the response.

@author: ben kay (ben@auscope.org.au)

:license: MIT

"""

from .pr624 import read_uoa, UoAReader
from .orange import read_orange, OrangeReader
from .uoa_collection import UoACollection

__all__ = [
    "read_uoa",
    "UoAReader",
    "read_orange",
    "OrangeReader",
    "UoACollection",
]
