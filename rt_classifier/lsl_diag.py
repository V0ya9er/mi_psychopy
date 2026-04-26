"""LSL timestamp diagnostics for real-time classification debugging.

Tracks LSL data quality: timestamp gaps, jitter, sample loss, and saves windows
for offline analysis. Used to diagnose why real-time classification accuracy
differs from offline replay on the same XDF data.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Cap for jitter samples to avoid unbounded memory growth
_JITTER_SAMPLES_CAP = 10000


@dataclass
class TrialDiagnostics:
    """Per-trial diagnostic statistics."""
    trial_id: int = 0
    total_samples_received: int = 0
    expected_samples: int = 0
    gap_count: int = 0
    max_gap_s: float = 0.0
    total_lost_samples: float = 0.0
    jitter_samples: list[float] = field(default_factory=list)
    non_monotonic_count: int = 0
    start_time: float = 0.0
    end_time: float = 0.0


class LSLDiagnostics:
    """Tracks LSL data quality: timestamp gaps, jitter, sample loss, and saves windows.

    Thread-safe implementation for use in real-time classification loop.

    Usage:
        diag = LSLDiagnostics(sample_rate=250.0, save_dir="diag")

        # In main loop:
        diag.on_eeg_chunk(samples, timestamps)
        diag.on_trial_start(trial_id)
        # ... after classification ...
        diag.on_window_classified(window, result, trial_id, win_idx)
        diag.on_trial_end(trial_id)

        # At session end:
        summary = diag.get_summary()
    """

    def __init__(self, sample_rate: float, save_dir: str | Path | None = None) -> None:
        """Initialize LSL diagnostics.

        Args:
            sample_rate: Expected sample rate (e.g., 250.0 Hz)
            save_dir: Directory to save diagnostic files (None = no saving)
        """
        self._sample_rate = sample_rate
        self._save_dir = Path(save_dir) if save_dir else None
        self._lock = threading.Lock()

        # Session-wide statistics
        self._total_samples_received: int = 0
        self._total_gaps: int = 0
        self._max_gap_s: float = 0.0
        self._total_lost_samples: float = 0.0
        self._jitter_samples: list[float] = []
        self._total_non_monotonic: int = 0

        # Per-trial state
        self._current_trial: TrialDiagnostics | None = None
        self._trial_counter: int = 0

        # Chunk tracking for gap detection
        self._last_chunk_end_ts: float | None = None
        self._session_start_time: float | None = None

        # Create save directory if needed
        if self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            logger.info("LSL diagnostics: saving windows to %s", self._save_dir)

    def on_eeg_chunk(self, samples: np.ndarray, timestamps: np.ndarray) -> None:
        """Called after every pull_eeg_chunk.

        Tracks:
        - Total samples received
        - Gap detection: if time since last chunk > 2 * expected_interval
        - Non-monotonic timestamps (reordered data)
        - Per-chunk timestamp jitter stats

        Args:
            samples: EEG samples array (n_samples, n_channels)
            timestamps: LSL timestamps array (n_samples,)
        """
        if len(timestamps) == 0:
            return

        with self._lock:
            n_samples = len(samples)
            self._total_samples_received += n_samples

            # Track session start time
            if self._session_start_time is None:
                self._session_start_time = timestamps[0]

            # Gap detection
            if self._last_chunk_end_ts is not None:
                inter_chunk_gap = timestamps[0] - self._last_chunk_end_ts
                expected_interval = n_samples / self._sample_rate

                # Gap threshold: 2x expected interval
                if inter_chunk_gap > expected_interval * 2:
                    self._total_gaps += 1
                    gap_duration = inter_chunk_gap - expected_interval
                    self._max_gap_s = max(self._max_gap_s, gap_duration)

                    # Estimate lost samples
                    lost = gap_duration * self._sample_rate
                    self._total_lost_samples += lost

                    logger.warning(
                        "LSL gap detected: %.1fms between chunks (expected %.1fms, ~%.0f samples lost)",
                        inter_chunk_gap * 1000,
                        expected_interval * 1000,
                        lost,
                    )

                    # Update per-trial stats if active
                    if self._current_trial is not None:
                        self._current_trial.gap_count += 1
                        self._current_trial.max_gap_s = max(
                            self._current_trial.max_gap_s, gap_duration
                        )
                        self._current_trial.total_lost_samples += lost

            # Non-monotonic detection
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                non_mono_mask = diffs <= 0
                if np.any(non_mono_mask):
                    reorder_count = int(np.sum(non_mono_mask))
                    self._total_non_monotonic += reorder_count
                    logger.warning(
                        "Non-monotonic timestamps: %d samples out of order", reorder_count
                    )
                    if self._current_trial is not None:
                        self._current_trial.non_monotonic_count += reorder_count

            # Jitter detection
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                expected_dt = 1.0 / self._sample_rate
                jitter = np.abs(diffs - expected_dt) * 1000  # in ms

                # Cap jitter samples to avoid unbounded growth
                new_jitter = jitter.tolist()
                remaining_capacity = _JITTER_SAMPLES_CAP - len(self._jitter_samples)
                if remaining_capacity > 0:
                    self._jitter_samples.extend(new_jitter[:remaining_capacity])

                # Update per-trial jitter
                if self._current_trial is not None:
                    remaining_trial = _JITTER_SAMPLES_CAP - len(
                        self._current_trial.jitter_samples
                    )
                    if remaining_trial > 0:
                        self._current_trial.jitter_samples.extend(
                            new_jitter[:remaining_trial]
                        )

            # Update tracking state
            self._last_chunk_end_ts = timestamps[-1]

            # Update per-trial sample count
            if self._current_trial is not None:
                self._current_trial.total_samples_received += n_samples

    def on_window_classified(
        self,
        window: np.ndarray,
        result: Any,
        trial_id: int,
        win_idx: int,
    ) -> None:
        """Called after each window classification.

        If save_dir is set, saves:
        - {save_dir}/trial_{trial_id}_win_{win_idx}.npy — raw window data
        - {save_dir}/trial_{trial_id}_win_{win_idx}_result.json — classification result

        Args:
            window: Window data array (n_samples, n_channels)
            result: ClassificationResult object
            trial_id: Current trial ID
            win_idx: Window index within trial
        """
        if self._save_dir is None:
            return

        base_name = f"trial_{trial_id}_win_{win_idx}"

        # Save window data
        window_path = self._save_dir / f"{base_name}.npy"
        np.save(window_path, window)

        # Save result as JSON
        result_path = self._save_dir / f"{base_name}_result.json"
        result_dict = {
            "label": result.label,
            "confidence": result.confidence,
            "cca_score": result.cca_score,
            "mi_score": result.mi_score,
            "timestamp": result.timestamp,
            "trial_id": result.trial_id,
            "cca_label": result.cca_label,
            "mi_label": result.mi_label,
        }
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    def on_trial_start(self, trial_id: int) -> None:
        """Reset per-trial counters.

        Args:
            trial_id: Trial ID from tracker
        """
        with self._lock:
            self._current_trial = TrialDiagnostics(trial_id=trial_id)
            self._trial_counter += 1

    def on_trial_end(self, trial_id: int) -> dict[str, Any]:
        """Return per-trial diagnostic summary.

        Args:
            trial_id: Trial ID from tracker

        Returns:
            Dict with per-trial statistics:
            - total_samples_received vs expected
            - gap_count, max_gap_s
            - avg_jitter_ms, max_jitter_ms
            - sample_loss_rate (estimated)
        """
        with self._lock:
            if self._current_trial is None:
                return {
                    "trial_id": trial_id,
                    "total_samples_received": 0,
                    "expected_samples": 0,
                    "gap_count": 0,
                    "max_gap_s": 0.0,
                    "avg_jitter_ms": 0.0,
                    "max_jitter_ms": 0.0,
                    "sample_loss_rate": 0.0,
                    "non_monotonic_count": 0,
                }

            trial = self._current_trial

            # Compute jitter stats
            avg_jitter = 0.0
            max_jitter = 0.0
            if trial.jitter_samples:
                avg_jitter = float(np.mean(trial.jitter_samples))
                max_jitter = float(np.max(trial.jitter_samples))

            # Compute sample loss rate
            total_expected = trial.total_samples_received + trial.total_lost_samples
            sample_loss_rate = (
                trial.total_lost_samples / total_expected if total_expected > 0 else 0.0
            )

            result = {
                "trial_id": trial_id,
                "total_samples_received": trial.total_samples_received,
                "expected_samples": int(total_expected),
                "gap_count": trial.gap_count,
                "max_gap_s": trial.max_gap_s,
                "avg_jitter_ms": avg_jitter,
                "max_jitter_ms": max_jitter,
                "sample_loss_rate": sample_loss_rate,
                "non_monotonic_count": trial.non_monotonic_count,
            }

            # Reset for next trial
            self._current_trial = None

            return result

    def get_summary(self) -> dict[str, Any]:
        """Return session-wide diagnostic summary.

        Returns:
            Dict with session statistics:
            - total_samples_received
            - total_gaps
            - max_gap_s
            - avg_jitter_ms
            - max_jitter_ms
            - sample_loss_rate
            - total_non_monotonic
            - total_trials
        """
        with self._lock:
            # Compute jitter stats
            avg_jitter = 0.0
            max_jitter = 0.0
            if self._jitter_samples:
                avg_jitter = float(np.mean(self._jitter_samples))
                max_jitter = float(np.max(self._jitter_samples))

            # Compute sample loss rate
            total_expected = self._total_samples_received + self._total_lost_samples
            sample_loss_rate = (
                self._total_lost_samples / total_expected if total_expected > 0 else 0.0
            )

            return {
                "total_samples_received": self._total_samples_received,
                "total_gaps": self._total_gaps,
                "max_gap_s": self._max_gap_s,
                "avg_jitter_ms": avg_jitter,
                "max_jitter_ms": max_jitter,
                "sample_loss_rate": sample_loss_rate,
                "total_non_monotonic": self._total_non_monotonic,
                "total_trials": self._trial_counter,
            }
