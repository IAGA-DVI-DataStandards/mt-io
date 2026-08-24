# -*- coding: utf-8 -*-
"""
Test grouping EDL files into runs.

Fixtures are written here so the tests run anywhere. Checks against real
archives live in the project scripts.

@author: bkay
"""

# =============================================================================
# Imports
# =============================================================================
import shutil
import tempfile
import unittest
from pathlib import Path

from mt_io.uoa import UoACollection, read_uoa
from mt_io.uoa.pr624 import count_samples

# =============================================================================

CHANNELS = ["BX", "BY", "BZ", "EX", "EY"]
SAMPLE_RATE = 10.0
N_SAMPLES = 60  # 6 s per file at 10 Hz


def write_edl_files(directory, station, stamps, n_samples=N_SAMPLES):
    """One ASCII file per channel per stamp, one value per line."""
    body = "\n".join(str(1000 + i) for i in range(n_samples)) + "\n"
    for stamp in stamps:
        for channel in CHANNELS:
            (directory / f"{station}_{stamp}.{channel}").write_text(body)


# =============================================================================


class TestRunSplitting(unittest.TestCase):
    """Test that a gap starts a new run"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.path = Path(cls.temp_dir)
        # three files six seconds apart, then a gap, then two more
        write_edl_files(
            cls.path,
            "TEST01",
            [
                "240101000000",
                "240101000006",
                "240101000012",
                "240101000100",
                "240101000106",
            ],
        )
        cls.collection = UoACollection(cls.path)
        cls.collection.sample_rate = SAMPLE_RATE
        cls.df = cls.collection.to_dataframe(sample_rates=[SAMPLE_RATE])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_every_file_is_listed(self):
        """Test that every file is listed"""
        self.assertEqual(len(self.df), 5 * len(CHANNELS))

    def test_two_runs(self):
        """Test that there are two runs"""
        self.assertEqual(self.df.run.nunique(), 2)

    def test_run_names_carry_the_sample_rate(self):
        self.assertEqual(
            sorted(self.df.run.unique()), ["sr10_0001", "sr10_0002"]
        )

    def test_first_run_holds_three_stamps(self):
        first = self.df[self.df.run == "sr10_0001"]
        self.assertEqual(len(first), 3 * len(CHANNELS))

    def test_second_run_holds_two_stamps(self):
        second = self.df[self.df.run == "sr10_0002"]
        self.assertEqual(len(second), 2 * len(CHANNELS))

    def test_contiguous_files_are_not_split(self):
        """Test that contiguous files stay in one run"""
        other = Path(tempfile.mkdtemp())
        try:
            write_edl_files(
                other, "TEST02",
                ["240101000000", "240101000006", "240101000012"],
            )
            collection = UoACollection(other)
            collection.sample_rate = SAMPLE_RATE
            df = collection.to_dataframe(sample_rates=[SAMPLE_RATE])
            self.assertEqual(df.run.nunique(), 1)
        finally:
            shutil.rmtree(other)


class TestSidecarsAreSkipped(unittest.TestCase):
    """Test that AppleDouble sidecars are skipped"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir)
        write_edl_files(self.path, "TEST01", ["240101000000", "240101000006"])
        # a Mac archive carries one beside every file
        for channel in CHANNELS:
            for stamp in ("240101000000", "240101000006"):
                (self.path / f"._TEST01_{stamp}.{channel}").write_bytes(
                    b"\x00\x05\x16\x07" + b"\x00" * 100
                )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_files_skips_sidecars(self):
        collection = UoACollection(self.path)
        found = collection.get_files("BX")
        self.assertEqual(len(found), 2)
        self.assertFalse(any(Path(f).name.startswith("._") for f in found))

    def test_run_structure_survives_sidecars(self):
        """Test that sidecars do not fragment the runs"""
        collection = UoACollection(self.path)
        collection.sample_rate = SAMPLE_RATE
        df = collection.to_dataframe(sample_rates=[SAMPLE_RATE])
        self.assertEqual(len(df), 2 * len(CHANNELS))
        self.assertEqual(df.run.nunique(), 1)


class TestCountSamples(unittest.TestCase):
    """Test count_samples"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_counts_ascii_lines(self):
        fn = self.path / "TEST01_240101000000.BX"
        fn.write_text("1\n2\n3\n")
        self.assertEqual(count_samples(fn), 3)

    def test_empty_file_is_zero(self):
        """Test that an empty file counts zero"""
        fn = self.path / "TEST01_240101000006.BX"
        fn.write_bytes(b"")
        self.assertEqual(count_samples(fn), 0)

    def test_bytes_with_no_samples_raises(self):
        """Test that a file with bytes but no samples raises"""
        fn = self.path / "._TEST01_240101000000.BX"
        fn.write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 100)
        with self.assertRaises(ValueError):
            count_samples(fn)


class TestPartialChannelSets(unittest.TestCase):
    """Test deployments that did not record every channel"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _read(self, channels):
        body = "\n".join(str(1000 + i) for i in range(N_SAMPLES)) + "\n"
        for stamp in ("240101000000", "240101000006"):
            for channel in channels:
                (self.path / f"TEST01_{stamp}.{channel}").write_text(body)
        return read_uoa(
            self.path, station_id="TEST01", sample_rate=SAMPLE_RATE,
            dipole_length_ex=50.0, dipole_length_ey=50.0,
        )

    def test_no_bz(self):
        """Test four component, which is the common broadband layout"""
        run = self._read(["BX", "BY", "EX", "EY"])
        self.assertEqual(sorted(run.channels), ["ex", "ey", "hx", "hy"])
        self.assertEqual(run.dataset.sizes["time"], 2 * N_SAMPLES)

    def test_magnetics_only(self):
        """Test a magnetics only deployment"""
        run = self._read(["BX", "BY", "BZ"])
        self.assertEqual(sorted(run.channels), ["hx", "hy", "hz"])

    def test_two_magnetic_channels(self):
        """Test bx and by alone"""
        run = self._read(["BX", "BY"])
        self.assertEqual(sorted(run.channels), ["hx", "hy"])

    def test_electrics_only(self):
        """Test an electrics only deployment"""
        run = self._read(["EX", "EY"])
        self.assertEqual(sorted(run.channels), ["ex", "ey"])

    def test_no_channels_raises(self):
        """Test that finding nothing at all is an error"""
        with self.assertRaises(ValueError):
            read_uoa(self.path, station_id="TEST01", sample_rate=SAMPLE_RATE)

    def test_collection_handles_partial_sets(self):
        """Test that the collection groups a partial set into one run"""
        self._read(["BX", "BY", "EX", "EY"])
        collection = UoACollection(self.path)
        collection.sample_rate = SAMPLE_RATE
        df = collection.to_dataframe(sample_rates=[SAMPLE_RATE])
        self.assertEqual(sorted(df.component.unique()), ["ex", "ey", "hx", "hy"])
        self.assertEqual(df.run.nunique(), 1)


class TestReadFromFileList(unittest.TestCase):
    """Test reading from a list of files"""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.path = Path(cls.temp_dir)
        write_edl_files(cls.path, "TEST01", ["240101000000", "240101000006"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def test_reads_a_list_of_files(self):
        files = sorted(self.path.glob("TEST01_*"))
        run_ts = read_uoa(
            files, station_id="TEST01", sample_rate=SAMPLE_RATE,
            dipole_length_ex=50.0, dipole_length_ey=50.0,
        )
        self.assertEqual(sorted(run_ts.channels), ["ex", "ey", "hx", "hy", "hz"])
        self.assertEqual(run_ts.dataset.sizes["time"], 2 * N_SAMPLES)

    def test_a_subset_of_the_list_is_honoured(self):
        """Test that only the files given are read"""
        files = sorted(self.path.glob("TEST01_240101000000.*"))
        run_ts = read_uoa(
            files, station_id="TEST01", sample_rate=SAMPLE_RATE,
            dipole_length_ex=50.0, dipole_length_ey=50.0,
        )
        self.assertEqual(run_ts.dataset.sizes["time"], N_SAMPLES)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    unittest.main()
