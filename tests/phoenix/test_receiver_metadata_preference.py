# -*- coding: utf-8 -*-
"""
Test which receiver metadata file is picked when both are present.

recmeta.json is what the receiver was programmed with before the survey.
EMpower writes empower_recmeta.json afterwards, so it carries any correction
to a mislabelled serial or an as-measured dipole and has to win.

Created on August 25, 2026
"""

# =============================================================================
# Imports
# =============================================================================
import tempfile
from pathlib import Path

import pytest

from mt_io.phoenix.readers.receiver_metadata import (
    find_receiver_metadata,
    RECEIVER_METADATA_NAMES,
)

# =============================================================================


@pytest.fixture
def folder():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_empower_is_preferred(folder):
    """Test that EMpower's file wins when both are there"""
    (folder / "recmeta.json").write_text("{}")
    (folder / "empower_recmeta.json").write_text("{}")

    assert find_receiver_metadata(folder).name == "empower_recmeta.json"


def test_recmeta_is_the_fallback(folder):
    """Test that recmeta.json is used when EMpower has not seen the data"""
    (folder / "recmeta.json").write_text("{}")

    assert find_receiver_metadata(folder).name == "recmeta.json"


def test_empower_alone(folder):
    """Test that EMpower's file is used on its own"""
    (folder / "empower_recmeta.json").write_text("{}")

    assert find_receiver_metadata(folder).name == "empower_recmeta.json"


def test_neither_gives_none(folder):
    """Test that a folder with no receiver metadata returns None"""
    assert find_receiver_metadata(folder) is None


def test_accepts_a_string_path(folder):
    """Test that a string path works as well as a Path"""
    (folder / "recmeta.json").write_text("{}")

    assert find_receiver_metadata(str(folder)).name == "recmeta.json"


def test_names_are_in_preference_order():
    """Test that the search order puts EMpower first"""
    assert RECEIVER_METADATA_NAMES == (
        "empower_recmeta.json",
        "recmeta.json",
    )
