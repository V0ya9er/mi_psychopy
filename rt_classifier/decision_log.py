"""Decision logger and CSV-based FeedbackSink.

CSV fields: timestamp_s, trial_id, label, confidence, cca_score, mi_score,
            cca_label, mi_label, window_samples, note
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from rt_classifier.config import ClassificationResult

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "timestamp_s", "trial_id", "label", "confidence",
    "cca_score", "mi_score", "cca_label", "mi_label",
    "window_samples", "note",
]


class DecisionLogger:
    """Log classification decisions to a CSV file."""

    def __init__(self, output_dir: Path, run_name: str) -> None:
        self._run_dir = output_dir / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._run_dir / "decisions.csv"
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        logger.info(f"Decision log: {self._csv_path}")

    def log_decision(self, result: ClassificationResult, cca_label: str, mi_label: str, window_samples: int, note: str = "") -> None:
        self._writer.writerow({
            "timestamp_s": f"{result.timestamp:.6f}",
            "trial_id": result.trial_id,
            "label": result.label,
            "confidence": f"{result.confidence:.4f}",
            "cca_score": f"{result.cca_score:.4f}",
            "mi_score": f"{result.mi_score:.4f}",
            "cca_label": cca_label,
            "mi_label": mi_label,
            "window_samples": window_samples,
            "note": note,
        })
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
            logger.info(f"Decision log closed: {self._csv_path}")


class CSVFeedbackSink:
    """FeedbackSink that writes classification results to CSV."""

    def __init__(self, output_dir: Path, run_name: str) -> None:
        self._logger = DecisionLogger(output_dir, run_name)

    def push_result(self, result: ClassificationResult) -> None:
        self._logger.log_decision(
            result,
            cca_label=result.cca_label,
            mi_label=result.mi_label,
            window_samples=0,
        )

    def close(self) -> None:
        self._logger.close()
