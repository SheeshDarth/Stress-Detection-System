"""
Unit Tests — rPPG Extraction Pipeline
=======================================
Tests the RPPGExtractor class: POS algorithm, spectral analysis,
detrending, bandpass filtering, and HRV computation.
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.physiological.rppg_extraction import RPPGExtractor


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def rppg():
    """Fresh RPPGExtractor for each test."""
    return RPPGExtractor(fps=30, window_seconds=10)


@pytest.fixture
def synthetic_75bpm_rgb():
    """
    Synthetic 10-second RGB signal with a 75 BPM (1.25 Hz) component.
    Returns shape (300, 3).
    """
    t = np.linspace(0, 10, 300)
    hr_signal = 0.5 * np.sin(2 * np.pi * 1.25 * t)
    r = 150 + hr_signal + np.random.default_rng(42).normal(0, 0.1, 300)
    g = 120 + hr_signal * 1.5 + np.random.default_rng(43).normal(0, 0.1, 300)
    b = 100 + hr_signal * 0.5 + np.random.default_rng(44).normal(0, 0.1, 300)
    return np.column_stack([r, g, b])


# ── Tests ─────────────────────────────────────────────────────────────

class TestPOSSimple:
    """Test the simple (non-overlap-add) POS algorithm."""

    def test_returns_array(self, synthetic_75bpm_rgb):
        pulse = RPPGExtractor._pos_simple(synthetic_75bpm_rgb)
        assert isinstance(pulse, np.ndarray)
        assert len(pulse) == len(synthetic_75bpm_rgb)

    def test_zero_mean(self, synthetic_75bpm_rgb):
        pulse = RPPGExtractor._pos_simple(synthetic_75bpm_rgb)
        assert abs(np.mean(pulse)) < 0.01

    def test_short_input(self):
        short = np.array([[100, 100, 100]])
        pulse = RPPGExtractor._pos_simple(short)
        assert len(pulse) == 0  # too short


class TestPOSOverlapAdd:
    """Test overlap-add POS."""

    def test_returns_correct_length(self, rppg, synthetic_75bpm_rgb):
        pulse = rppg._pos_overlap_add(synthetic_75bpm_rgb)
        assert len(pulse) == len(synthetic_75bpm_rgb)

    def test_zero_mean(self, rppg, synthetic_75bpm_rgb):
        pulse = rppg._pos_overlap_add(synthetic_75bpm_rgb)
        assert abs(np.mean(pulse)) < 0.5

    def test_falls_back_to_simple_for_short(self, rppg):
        """Short signals should fall back to simple POS."""
        short = np.random.rand(10, 3) * 100 + 100
        pulse = rppg._pos_overlap_add(short)
        assert len(pulse) > 0


class TestDetrend:
    """Test polynomial detrending."""

    def test_removes_linear_trend(self, rppg):
        x = np.linspace(0, 10, 300) + np.sin(np.linspace(0, 6 * np.pi, 300))
        detrended = rppg._detrend(x, order=1)
        # After removing linear trend, mean should be near zero
        assert abs(np.mean(detrended)) < 1.0

    def test_short_signal(self, rppg):
        x = np.array([1.0, 2.0])
        result = rppg._detrend(x, order=2)
        assert len(result) == 2


class TestBandpass:
    """Test Butterworth bandpass filter."""

    def test_output_length(self, rppg):
        x = np.random.randn(300)
        filtered = rppg._bandpass(x, lo=0.7, hi=4.0)
        assert len(filtered) == 300

    def test_attenuates_dc(self, rppg):
        """DC component should be removed by bandpass."""
        x = np.ones(300) * 5.0 + np.sin(np.linspace(0, 8 * np.pi, 300))
        filtered = rppg._bandpass(x)
        assert abs(np.mean(filtered)) < 0.5


class TestBPMWelch:
    """Test Welch PSD-based BPM estimation."""

    def test_detects_75bpm(self, rppg, synthetic_75bpm_rgb):
        pulse = RPPGExtractor._pos_simple(synthetic_75bpm_rgb)
        detrended = rppg._detrend(pulse)
        filtered = rppg._bandpass(detrended)
        bpm = rppg._bpm_welch(filtered)
        # Should be approximately 75 BPM (±10)
        assert 65 <= bpm <= 85, f"Expected ~75 BPM, got {bpm:.1f}"

    def test_short_signal_returns_zero(self, rppg):
        x = np.random.randn(30)
        bpm = rppg._bpm_welch(x)
        assert bpm == 0.0


class TestHRVRMSSD:
    """Test HRV RMSSD computation."""

    def test_returns_float(self, rppg, synthetic_75bpm_rgb):
        pulse = RPPGExtractor._pos_simple(synthetic_75bpm_rgb)
        filtered = rppg._bandpass(rppg._detrend(pulse))
        hrv = rppg._hrv_rmssd(filtered)
        assert isinstance(hrv, float)
        assert hrv >= 0.0

    def test_short_signal_returns_zero(self, rppg):
        x = np.random.randn(30)
        assert rppg._hrv_rmssd(x) == 0.0


class TestSNR:
    """Test signal-to-noise ratio computation."""

    def test_returns_float(self, rppg, synthetic_75bpm_rgb):
        pulse = RPPGExtractor._pos_simple(synthetic_75bpm_rgb)
        filtered = rppg._bandpass(rppg._detrend(pulse))
        snr = rppg._compute_snr(filtered)
        assert isinstance(snr, float)

    def test_short_signal_returns_zero(self, rppg):
        x = np.random.randn(30)
        assert rppg._compute_snr(x) == 0.0


class TestReset:
    """Test state reset."""

    def test_reset_clears_buffers(self, rppg):
        rppg.rgb_buffer.append(np.array([100, 100, 100]))
        rppg._bpm_history.append(72.0)
        rppg.current_bpm = 72.0

        rppg.reset()

        assert len(rppg.rgb_buffer) == 0
        assert len(rppg._bpm_history) == 0
        assert rppg.current_bpm == 0.0
        assert rppg.signal_snr == 0.0
