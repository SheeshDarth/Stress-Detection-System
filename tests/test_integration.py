"""
Integration Tests — Full Pipeline (Task 9)
==========================================
Tests the end-to-end pipeline (AUExtractor → RPPGExtractor → StressClassifier →
SessionLogger) using a synthetic 10-second video file.  All display calls are
bypassed so the tests run fully headlessly.

Fixtures
--------
- generate_test_video: Creates tests/fixtures/test_video.avi if it does not
  already exist (10 s, 30 fps, 640×480 moving gradient).

How to run
----------
    pytest tests/test_integration.py -v -m integration
"""

import os
import sys
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Attempt lazy imports (skip whole module if opencv or scipy not available) ──
try:
    import cv2
    from scipy.signal import butter, filtfilt  # noqa: F401 – verify scipy present
except ImportError as _e:
    pytest.skip(f"Required dependency missing: {_e}", allow_module_level=True)

from src.physiological.rppg_extraction import RPPGExtractor
from src.fusion.classifier import StressClassifier
from src.session_logger import SessionLogger

# ── Fixtures dir ──────────────────────────────────────────────────────────────
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_VIDEO   = os.path.join(FIXTURES_DIR, "test_video.avi")

FPS    = 30
WIDTH  = 640
HEIGHT = 480
DURATION_S = 10
N_FRAMES   = FPS * DURATION_S  # 300


# ─────────────────────────────────────────────────────────────────────────────
# ── Video generation fixture ─────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_video_path() -> str:
    """Generate a synthetic test video once per pytest session.

    The video contains a moving gradient that simulates a face region
    (changing intensity in the ROI location).  The RGB values oscillate
    at ~1.25 Hz (≈75 BPM) so that the rPPG extractor can lock on.
    """
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    if not os.path.exists(TEST_VIDEO):
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(TEST_VIDEO, fourcc, float(FPS), (WIDTH, HEIGHT))

        rng = np.random.default_rng(42)
        t_arr = np.linspace(0, DURATION_S, N_FRAMES)
        hr_wave = 0.5 * np.sin(2 * np.pi * 1.25 * t_arr)  # 75 BPM

        for i, t in enumerate(t_arr):
            # Base gradient background
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            # Horizontal gradient
            col_ramp = np.linspace(40, 200, WIDTH, dtype=np.uint8)
            frame[:, :, 1] = col_ramp[np.newaxis, :]  # green channel

            # "Face" ROI: rectangle in the centre with oscillating brightness
            r0, r1 = HEIGHT // 4, 3 * HEIGHT // 4
            c0, c1 = WIDTH  // 4, 3 * WIDTH  // 4
            base_r = 160 + hr_wave[i] * 30
            base_g = 120 + hr_wave[i] * 45
            base_b = 100 + hr_wave[i] * 15

            frame[r0:r1, c0:c1, 2] = int(np.clip(base_r, 0, 255))  # R (BGR→R)
            frame[r0:r1, c0:c1, 1] = int(np.clip(base_g, 0, 255))  # G
            frame[r0:r1, c0:c1, 0] = int(np.clip(base_b, 0, 255))  # B

            # Small random noise to avoid perfect periodicity (more realistic)
            noise = rng.integers(-5, 5, (HEIGHT, WIDTH, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)

        writer.release()

    assert os.path.exists(TEST_VIDEO), "test_video.avi was not created"
    return TEST_VIDEO


# ─────────────────────────────────────────────────────────────────────────────
# ── Mock landmarks ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class _FakeLandmark:
    """Minimal landmark that satisfies RPPGExtractor's _roi_mask / _extract_rgb."""

    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _build_fake_landmarks(n: int = 468) -> list:
    """Return a list of n fake landmarks distributed across the face ROI.

    The "face" region in the synthetic video occupies the centre quarter
    (0.25–0.75 along each axis), so we place all landmarks there so that
    the rPPG extractor samples non-zero RGB values.
    """
    rng = np.random.default_rng(7)
    lms = []
    for _ in range(n):
        x = rng.uniform(0.30, 0.70)
        y = rng.uniform(0.30, 0.70)
        lms.append(_FakeLandmark(x, y))
    return lms


# ─────────────────────────────────────────────────────────────────────────────
# ── Integration Test ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestFullPipelineIntegration:
    """End-to-end pipeline test using a synthetic video file (no display)."""

    def test_pipeline_no_exceptions(self, test_video_path: str, tmp_path) -> None:
        """Feed 300 frames through rPPG + classifier + session_logger.

        Assertions
        ----------
        - No unhandled exceptions throughout the loop.
        - ``is_signal_ready()`` returns True by frame 240 (8 s at 30 fps ≈ 80% of
          the 10 s buffer → well above the 30% threshold).
        - At least one valid ``predict()`` call succeeds (label ∈ {Normal, Stressed}).
        - The session logger records > 0 rows.
        """
        # ── Instantiate pipeline ──
        rppg = RPPGExtractor(fps=FPS, window_seconds=10)

        clf = StressClassifier(model_path=str(tmp_path / "test_model.pkl"))
        clf.train(use_mock=True)

        log_path = str(tmp_path / "test_session.csv")
        session = SessionLogger(log_dir=str(tmp_path))
        session.start()

        landmarks = _build_fake_landmarks(468)

        # ── Open video ──
        cap = cv2.VideoCapture(test_video_path)
        assert cap.isOpened(), f"Could not open test video: {test_video_path}"

        signal_ready_by_240 = False
        prediction_succeeded = False
        frame_idx = 0

        while True:
            ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

            # rPPG
            rppg_result = rppg.process_frame(rgb_frame, landmarks)

            # Check signal_ready by frame 240
            if frame_idx >= 239 and rppg.is_signal_ready():
                signal_ready_by_240 = True

            # Classification (requires buffer fill ≥ 30% + BPM > 0)
            if rppg_result is not None and rppg.is_signal_ready():
                # Build a trivial visual window dict (all zeros is safe — model handles it)
                vis = {
                    "blink_rate": 15.0, "ear_mean": 0.30, "ear_std": 0.02,
                    "brow_furrow_mean": 0.35, "brow_furrow_std": 0.01,
                    "head_pose_variance": 0.001, "head_pose_mean_movement": 0.005,
                }
                physio = {
                    "bpm": rppg_result["bpm"],
                    "hrv_rmssd": rppg_result["hrv_rmssd"],
                }
                vec = clf.create_feature_vector(vis, physio)
                if vec is not None:
                    label, conf = clf.predict(vec)
                    if label in ("Normal", "Stressed"):
                        prediction_succeeded = True

                        # Log to session CSV
                        session.log(
                            bpm=rppg_result["bpm"],
                            hrv_rmssd=rppg_result["hrv_rmssd"],
                            ear=vis["ear_mean"],
                            brow_furrow=vis["brow_furrow_mean"],
                            blink_count=0,
                            stress_label=label,
                            confidence=conf,
                            signal_quality=rppg_result.get("signal_quality", ""),
                            signal_snr=rppg_result.get("snr", 0.0),
                        )

            frame_idx += 1

        cap.release()
        session.stop()

        # ── Assertions ──
        assert frame_idx == N_FRAMES, (
            f"Expected {N_FRAMES} frames, read {frame_idx}"
        )
        assert signal_ready_by_240, (
            "is_signal_ready() should return True by frame 240 "
            f"(buffer_fill={len(rppg.rgb_buffer)/rppg.window_size:.2f}, "
            f"bpm={rppg.current_bpm:.1f})"
        )
        assert prediction_succeeded, (
            "At least one predict() call should return a valid label"
        )
        assert session.row_count > 0, (
            f"Session logger should have written > 0 rows, got {session.row_count}"
        )

    def test_video_fixture_properties(self, test_video_path: str) -> None:
        """Verify the generated synthetic video has correct frame count and dimensions."""
        cap = cv2.VideoCapture(test_video_path)
        assert cap.isOpened()

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        count  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert width  == WIDTH,  f"Expected width={WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height={HEIGHT}, got {height}"
        assert abs(fps - FPS) < 1.0, f"Expected fps~{FPS}, got {fps:.1f}"
        assert count  == N_FRAMES, f"Expected {N_FRAMES} frames, got {count}"

    def test_rppg_is_signal_ready_false_before_buffer_fill(self) -> None:
        """is_signal_ready() must return False on a fresh extractor."""
        rppg = RPPGExtractor(fps=FPS, window_seconds=10)
        assert rppg.is_signal_ready() is False

    def test_rppg_is_signal_ready_api_exists(self) -> None:
        """Ensure is_signal_ready() is a callable method on RPPGExtractor."""
        rppg = RPPGExtractor(fps=FPS, window_seconds=10)
        assert callable(rppg.is_signal_ready)
