"""
Unit Tests — Session Logger
=============================
Tests the SessionLogger class: CSV creation, row writing,
flush mechanics, and lifecycle management.
"""

import pytest
import os
import sys
import csv
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.session_logger import SessionLogger, FIELDNAMES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def logger():
    """SessionLogger writing to a temp directory."""
    tmp_dir = tempfile.mkdtemp()
    sl = SessionLogger(log_dir=tmp_dir)
    yield sl
    sl.stop()
    # Cleanup
    if sl.filepath and os.path.exists(sl.filepath):
        os.remove(sl.filepath)
    if os.path.exists(tmp_dir):
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ── Tests ─────────────────────────────────────────────────────────────

class TestLifecycle:
    """Test start/stop lifecycle."""

    def test_not_active_initially(self, logger):
        assert logger.is_active is False

    def test_start_activates(self, logger):
        logger.start()
        assert logger.is_active is True
        assert logger.filepath is not None
        assert os.path.exists(logger.filepath)

    def test_stop_deactivates(self, logger):
        logger.start()
        logger.stop()
        assert logger.is_active is False

    def test_filepath_contains_session(self, logger):
        path = logger.start()
        assert "session_" in path
        assert path.endswith(".csv")


class TestCSVFormat:
    """Test CSV file format."""

    def test_header_matches_fieldnames(self, logger):
        logger.start()
        logger.stop()

        with open(logger.filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == FIELDNAMES

    def test_row_written_correctly(self, logger):
        logger.start()
        logger.log(
            bpm=72.5,
            hrv_rmssd=45.3,
            ear=0.3012,
            brow_furrow=0.3500,
            blink_count=5,
            stress_label="Normal",
            confidence=0.85,
            signal_quality="Good",
            signal_snr=5.2,
        )
        logger.stop()

        with open(logger.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["stress_label"] == "Normal"
            assert rows[0]["bpm"] == "72.5"

    def test_multiple_rows(self, logger):
        logger.start()
        for i in range(10):
            logger.log(bpm=70 + i, stress_label="Normal")
        logger.stop()

        with open(logger.filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 10

    def test_row_count_tracked(self, logger):
        logger.start()
        assert logger.row_count == 0
        logger.log(bpm=72)
        assert logger.row_count == 1
        logger.log(bpm=73)
        assert logger.row_count == 2


class TestEdgeCases:
    """Test edge cases and safety."""

    def test_log_without_start_does_nothing(self, logger):
        logger.log(bpm=72)  # Should not raise
        assert logger.row_count == 0

    def test_double_stop_safe(self, logger):
        logger.start()
        logger.stop()
        logger.stop()  # Should not raise

    def test_finalise_alias(self, logger):
        logger.start()
        logger.finalise()
        assert logger.is_active is False
