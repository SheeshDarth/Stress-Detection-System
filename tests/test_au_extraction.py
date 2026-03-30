"""
Unit Tests — AU Extraction Pipeline
=====================================
Tests the AUExtractor class from src.visual.au_extraction.
"""

import pytest
import numpy as np
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.visual.au_extraction import AUExtractor


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def au_extractor():
    """Create an AUExtractor instance for the test module."""
    au = AUExtractor(fps=30, adaptive_blink=True)
    yield au
    au.release()


# ── Tests ─────────────────────────────────────────────────────────────

class TestAUExtractorInit:
    """Test AUExtractor initialization."""

    def test_model_loads(self, au_extractor):
        """FaceLandmarker model loads successfully."""
        assert au_extractor.landmarker is not None

    def test_default_params(self, au_extractor):
        """Verify default parameters are set correctly."""
        assert au_extractor.fps == 30
        assert au_extractor.blink_threshold == 0.21
        assert au_extractor.adaptive_blink is True

    def test_buffers_initialized_empty(self, au_extractor):
        au_extractor.reset_buffers()
        assert len(au_extractor.ear_buffer) == 0
        assert len(au_extractor.brow_buffer) == 0
        assert len(au_extractor.nose_positions) == 0
        assert au_extractor.blink_count == 0


class TestDistanceHelper:
    """Test the static distance helper."""

    def test_zero_distance(self):
        p = np.array([0.0, 0.0, 0.0])
        assert AUExtractor._distance(p, p) == pytest.approx(0.0)

    def test_unit_distance(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 0.0, 0.0])
        assert AUExtractor._distance(p1, p2) == pytest.approx(1.0)

    def test_3d_distance(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([4.0, 6.0, 3.0])
        expected = np.sqrt(9 + 16 + 0)  # 5.0
        assert AUExtractor._distance(p1, p2) == pytest.approx(expected)


class TestEMASmoothing:
    """Test exponential moving average."""

    def test_ema_first_value(self, au_extractor):
        result = au_extractor._ema(None, 0.5)
        assert result == pytest.approx(0.5)

    def test_ema_smoothing(self, au_extractor):
        prev = 0.5
        value = 1.0
        result = au_extractor._ema(prev, value)
        # alpha=0.33 → 0.33*1.0 + 0.67*0.5 = 0.665
        assert result == pytest.approx(0.665, abs=0.01)


class TestBufferReset:
    """Test buffer reset functionality."""

    def test_reset_clears_all(self, au_extractor):
        # Add some data
        au_extractor.ear_buffer.append(0.3)
        au_extractor.brow_buffer.append(0.2)
        au_extractor.blink_count = 5

        au_extractor.reset_buffers()

        assert len(au_extractor.ear_buffer) == 0
        assert len(au_extractor.brow_buffer) == 0
        assert au_extractor.blink_count == 0
        assert au_extractor._ema_ear is None
        assert au_extractor._calibrated is False


class TestFrameProcessingWithSyntheticImage:
    """Test process_frame with a synthetic image (no real face expected)."""

    def test_no_face_returns_none(self, au_extractor):
        """A blank image should return None (no face detected)."""
        au_extractor.reset_buffers()
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = au_extractor.process_frame(blank)
        assert result is None

    def test_window_features_not_enough_data(self, au_extractor):
        """Window features should be None with insufficient buffer data."""
        au_extractor.reset_buffers()
        result = au_extractor.get_window_features(10)
        assert result is None
