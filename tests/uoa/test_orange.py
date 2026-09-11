# -*- coding: utf-8 -*-
"""
Test the Orange Box binary decoder against a file built to the spec.

The format came from MAGALL2.C and mt_transform.for.

@author: bkay
"""

# =============================================================================
# Imports
# =============================================================================
import shutil
import tempfile
import unittest
from pathlib import Path

from mt_io.uoa.orange import (
    ADC_ZERO,
    BYTES_PER_SAMPLE,
    NCHANNELS,
    OrangeDataReader,
)

# =============================================================================
# Build a file to the documented layout
# =============================================================================

FILTER_POINT = 0x7A1  # 1953, the rate the boxes were run at
START_STRING = "Tue Jun 16 02:01:04 2009"

# raw value per channel. The 24 bit channels are offset binary about ADC_ZERO.
SAMPLES = [
    [
        ADC_ZERO + 100,
        ADC_ZERO - 50,
        ADC_ZERO + 7,
        1234,
        4321,
        9,
        ADC_ZERO + 11,
        ADC_ZERO - 3,
    ],
    [
        ADC_ZERO + 200,
        ADC_ZERO - 60,
        ADC_ZERO + 8,
        1235,
        4322,
        10,
        ADC_ZERO + 12,
        ADC_ZERO - 4,
    ],
    [ADC_ZERO, ADC_ZERO, ADC_ZERO, 0, 0, 0, ADC_ZERO, ADC_ZERO],
]


def encode_record(values):
    """Pack one sample into the 21 byte record, big endian."""
    record = bytearray(BYTES_PER_SAMPLE)

    def write_24(offset, value):
        record[offset] = (value >> 16) & 0xFF
        record[offset + 1] = (value >> 8) & 0xFF
        record[offset + 2] = value & 0xFF

    def write_16(offset, value):
        record[offset] = (value >> 8) & 0xFF
        record[offset + 1] = value & 0xFF

    write_24(0, values[0])
    write_24(3, values[1])
    write_24(6, values[2])
    write_16(9, values[3])
    write_16(11, values[4])
    record[13] = values[5] & 0xFF
    write_24(14, values[6])
    write_24(17, values[7])
    record[20] = 0  # trailing byte
    return bytes(record)


def write_orange_file(path, samples=SAMPLES, end_stamp=True):
    """Write a whole Orange Box file, header and records."""
    with open(path, "wb") as fid:
        fid.write(f" {len(samples):08X} \n".encode("ascii"))
        fid.write(f"{START_STRING}\n".encode("ascii"))
        fid.write(f" {FILTER_POINT:03X}".encode("ascii"))
        for values in samples:
            fid.write(encode_record(values))
        if end_stamp:
            # 25 characters follow the last record
            fid.write(b"Tue Jun 16 02:11:04 2009\n")


# =============================================================================


class TestOrangeDataReader(unittest.TestCase):
    """Test decoding a file written to the documented layout"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.fn = Path(cls.temp_dir) / "ST61.BIN"
        write_orange_file(cls.fn)
        cls.obj = OrangeDataReader(cls.fn)
        cls.df = cls.obj.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_sample_count_is_read_from_header(self):
        """Test that the first header field is a sample count"""
        self.assertEqual(self.obj.n_samples, len(SAMPLES))

    def test_filter_point(self):
        """Test the filter point"""
        self.assertEqual(self.obj.filter_point, FILTER_POINT)

    def test_sample_rate_from_filter_point(self):
        """Test that the sample rate comes from the filter point"""
        self.assertAlmostEqual(
            self.obj.sample_rate, 10_000_000 / (512 * FILTER_POINT), places=9
        )
        self.assertAlmostEqual(self.obj.sample_rate, 10.00064, places=5)
        self.assertNotEqual(self.obj.sample_rate, 10.0)

    def test_start_time(self):
        """Test that the start time comes from the header"""
        self.assertEqual(self.obj.start_time.year, 2009)
        self.assertEqual(self.obj.start_time.month, 6)
        self.assertEqual(self.obj.start_time.day, 16)

    def test_record_count_stops_before_the_end_stamp(self):
        """Test that exactly n_samples records are read"""
        self.assertEqual(len(self.df), len(SAMPLES))

    def test_channel_map(self):
        """0=Bx, 1=Bz, 2=By, 6=Ey, 7=Ex"""
        self.assertEqual(
            sorted(self.df.columns), sorted(["Bx", "By", "Bz", "Ex", "Ey"])
        )
        self.assertEqual(self.df["Bx"].iloc[0], 100)  # channel 0
        self.assertEqual(self.df["Bz"].iloc[0], -50)  # channel 1
        self.assertEqual(self.df["By"].iloc[0], 7)  # channel 2
        self.assertEqual(self.df["Ey"].iloc[0], 11)  # channel 6
        self.assertEqual(self.df["Ex"].iloc[0], -3)  # channel 7

    def test_offset_binary_about_2_23(self):
        """Test that ADC_ZERO stores as zero"""
        for column in ("Bx", "By", "Bz", "Ex", "Ey"):
            self.assertEqual(self.df[column].iloc[2], 0)

    def test_second_sample_decodes_independently(self):
        """Test that the second sample decodes correctly"""
        self.assertEqual(self.df["Bx"].iloc[1], 200)
        self.assertEqual(self.df["Bz"].iloc[1], -60)
        self.assertEqual(self.df["Ex"].iloc[1], -4)

    def test_unused_channels_are_decoded_but_dropped(self):
        """Test that channels 3, 4 and 5 are decoded but not returned"""
        with open(self.fn, "rb") as fid:
            self.obj.parse_header(fid)
            counts = self.obj.read_samples(fid)
        self.assertEqual(counts.shape, (len(SAMPLES), NCHANNELS))
        self.assertEqual(counts[0, 3], 1234)
        self.assertEqual(counts[0, 4], 4321)
        self.assertEqual(counts[0, 5], 9)


class TestOrangeTruncatedFile(unittest.TestCase):
    """Test a file shorter than its header claims"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fn = Path(self.temp_dir) / "SHORT.BIN"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_short_file_returns_what_is_there(self):
        """Test that a short file returns the records present"""
        with open(self.fn, "wb") as fid:
            fid.write(f" {3:08X} \n".encode("ascii"))
            fid.write(f"{START_STRING}\n".encode("ascii"))
            fid.write(f" {FILTER_POINT:03X}".encode("ascii"))
            for values in SAMPLES[:2]:
                fid.write(encode_record(values))

        df = OrangeDataReader(self.fn).read()
        self.assertEqual(len(df), 2)
        self.assertEqual(df["Bx"].iloc[0], 100)

    def test_no_records_gives_an_empty_frame(self):
        """Test that a header with no records gives an empty frame"""
        with open(self.fn, "wb") as fid:
            fid.write(f" {2:08X} \n".encode("ascii"))
            fid.write(f"{START_STRING}\n".encode("ascii"))
            fid.write(f" {FILTER_POINT:03X}".encode("ascii"))

        df = OrangeDataReader(self.fn).read()
        self.assertEqual(len(df), 0)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
