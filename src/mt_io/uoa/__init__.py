"""
University of Adelaide (UoA) Magnetotelluric Instrument Readers

Created on Fri Nov 7 12:22:00 2025

@author: ben kay (ben@auscope.org.au)

Instruments Supported
---------------------

1. **Earth Data Logger PR6-24** (read_uoa)
   - 6-channel broadband/long-period MT system
   - ASCII or miniSEED, one file per channel, both in microvolts
   - format detected from the file header, not the extension
   - Hardware: Bz voltage divider (fluxgate only; 15 kOhm/10 kOhm), E-field terminal box (x10 gain)
   - Sensors: LEMI-120 induction coils or Bartington Mag-03 fluxgates

2. **Orange Box** (read_orange)
   - Legacy 8-channel long-period MT system (UoA/Flinders)
   - Binary format: 18 bytes per sample, 8 channels
   - Sensors: Bartington fluxgates, non-polarizing electrodes
   - Used in: ~2000 to ~2014


Hardware Calibration Notes
--------------------------

**PR6-24 (EDL)**:
- Bz channel: 15 kOhm/10 kOhm voltage divider (Bartington mode only)
- Ex, Ey channels: x10 gain from terminal box (both modes)
- ADC: 24-bit, +/-8.388V range, ~1 uV per count

**Orange Box**:
- Magnetic: 24-bit ADC, +/-70,000 nT full-scale
- Electric: 24-bit ADC, +/-100,000 uV full-scale (dipole-normalized)
- By channel: Inverted by hardware convention
- Ex, Ey channels: Inverted by hardware convention


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
