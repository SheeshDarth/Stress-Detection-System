"""
Session Logger — CSV Export of Real-Time Metrics
==================================================
Writes timestamped stress detection metrics to a CSV file
for post-session analysis and reproducibility.

Usage
-----
  logger = SessionLogger()
  logger.log(bpm=72, hrv=45, ear=0.30, brow=0.35,
             stress_label="Normal", confidence=0.85)
  logger.finalise()   # Call on session end
"""

import os
import csv
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


FIELDNAMES = [
    "timestamp",
    "elapsed_seconds",
    "bpm",
    "hrv_rmssd",
    "ear",
    "brow_furrow",
    "blink_count",
    "stress_label",
    "confidence",
    "signal_quality",
    "signal_snr",
]


class SessionLogger:
    """Buffers and writes per-frame stress metrics to CSV."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self._active = False
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._filepath: Optional[str] = None
        self._start_time: float = 0.0
        self._row_count: int = 0
        self._flush_interval: int = 30  # flush every N rows

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> str:
        """Begin a new logging session.  Returns the log file path."""
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._filepath = os.path.join(self.log_dir, f"session_{ts}.csv")

        self._file = open(self._filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        self._writer.writeheader()

        self._start_time = time.time()
        self._row_count = 0
        self._active = True
        return self._filepath

    def stop(self) -> None:
        """Flush and close the current log file."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def filepath(self) -> Optional[str]:
        return self._filepath

    @property
    def row_count(self) -> int:
        return self._row_count

    # ── recording ─────────────────────────────────────────────────────

    def log(
        self,
        bpm: float = 0.0,
        hrv_rmssd: float = 0.0,
        ear: float = 0.0,
        brow_furrow: float = 0.0,
        blink_count: int = 0,
        stress_label: str = "",
        confidence: float = 0.0,
        signal_quality: str = "",
        signal_snr: float = 0.0,
    ) -> None:
        """Append one row of metrics to the CSV log."""
        if not self._active or self._writer is None:
            return

        now = time.time()
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_seconds": f"{now - self._start_time:.2f}",
            "bpm": f"{bpm:.1f}",
            "hrv_rmssd": f"{hrv_rmssd:.1f}",
            "ear": f"{ear:.4f}",
            "brow_furrow": f"{brow_furrow:.4f}",
            "blink_count": blink_count,
            "stress_label": stress_label,
            "confidence": f"{confidence:.3f}",
            "signal_quality": signal_quality,
            "signal_snr": f"{signal_snr:.2f}",
        }
        self._writer.writerow(row)
        self._row_count += 1

        # Periodic flush to avoid data loss on crash
        if self._row_count % self._flush_interval == 0:
            self._file.flush()

    # ── alias ─────────────────────────────────────────────────────────

    def finalise(self) -> None:
        """Alias for stop(), called at session end."""
        self.stop()
