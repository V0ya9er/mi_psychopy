"""Decision fusion and trial tracking for the real-time classifier.

DecisionFusion: combines CCA (SSVEP) and MI engine results.
TrialTracker: manages trial state based on marker events and computes
              cumulative decisions via majority vote.
"""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from markers import MARKERS

from rt_classifier.cca_engine import CCAEngine
from rt_classifier.mi_engine import MIEngine
from rt_classifier.config import ClassifierConfig, ClassificationResult, TrialState

logger = logging.getLogger(__name__)

# Marker values for trial events (from markers.py)
MARKER_SSVEP_LEFT = MARKERS["ssvep_left"]      # 61
MARKER_SSVEP_RIGHT = MARKERS["ssvep_right"]    # 62
MARKER_TASK_OFF = MARKERS["task_off"]           # 29
MARKER_RT_SSVEP_LEFT = MARKERS["rt_ssvep_left"]    # 181
MARKER_RT_SSVEP_RIGHT = MARKERS["rt_ssvep_right"]  # 182
MARKER_RT_TASK_OFF = MARKERS["rt_task_off"]         # 189


class DecisionFusion:
    """Combine CCA and MI classification results.

    Fusion strategy:
    - MI unavailable: use CCA result directly
    - MI+CCA agree: confidence = (cca_conf + mi_conf) / 2
    - MI+CCA disagree: use CCA label, confidence = cca_conf * 0.7
    """

    def __init__(self, config: ClassifierConfig, cca_engine: CCAEngine, mi_engine: MIEngine) -> None:
        self._config = config
        self._cca = cca_engine
        self._mi = mi_engine

    def classify(self, window: np.ndarray, timestamp: float = 0.0, trial_id: int = 0) -> ClassificationResult:
        """Classify a window using CCA + MI fusion."""
        cca_label, cca_conf = self._cca.classify(window)
        cca_score = cca_conf  # CCA score is the confidence itself

        if not self._mi.available:
            return ClassificationResult(
                label=cca_label, confidence=cca_conf,
                cca_score=cca_score, mi_score=0.0,
                timestamp=timestamp, trial_id=trial_id,
                cca_label=cca_label, mi_label="unknown",
            )

        mi_label, mi_conf = self._mi.classify(window)
        mi_score = mi_conf

        if cca_label == mi_label:
            # Agreement: average confidence
            fused_conf = (cca_conf + mi_conf) / 2.0
            return ClassificationResult(
                label=cca_label, confidence=fused_conf,
                cca_score=cca_score, mi_score=mi_score,
                timestamp=timestamp, trial_id=trial_id,
                cca_label=cca_label, mi_label=mi_label,
            )
        else:
            # Disagreement: CCA wins, confidence downweighted
            return ClassificationResult(
                label=cca_label, confidence=cca_conf * 0.7,
                cca_score=cca_score, mi_score=mi_score,
                timestamp=timestamp, trial_id=trial_id,
                cca_label=cca_label, mi_label=mi_label,
            )


class TrialTracker:
    """Track trial state from marker events and compute cumulative decisions."""

    def __init__(self) -> None:
        self._state = TrialState.WAITING
        self._trial_id = 0
        self._condition: str = ""
        self._labels: list[str] = []
        self._confidences: list[float] = []

    @property
    def state(self) -> TrialState:
        return self._state

    @property
    def trial_id(self) -> int:
        return self._trial_id

    @property
    def condition(self) -> str:
        return self._condition

    def on_marker(self, marker_value: int) -> None:
        """Process a marker event."""
        if marker_value in (MARKER_SSVEP_LEFT, MARKER_RT_SSVEP_LEFT):
            self._trial_id += 1
            self._condition = "left"
            self._state = TrialState.ACTIVE
            self._labels.clear()
            self._confidences.clear()
            logger.info(f"Trial {self._trial_id} started: condition=left (marker={marker_value})")
        elif marker_value in (MARKER_SSVEP_RIGHT, MARKER_RT_SSVEP_RIGHT):
            self._trial_id += 1
            self._condition = "right"
            self._state = TrialState.ACTIVE
            self._labels.clear()
            self._confidences.clear()
            logger.info(f"Trial {self._trial_id} started: condition=right (marker={marker_value})")
        elif marker_value in (MARKER_TASK_OFF, MARKER_RT_TASK_OFF):
            if self._state == TrialState.ACTIVE:
                self._state = TrialState.ENDED
                logger.info(f"Trial {self._trial_id} ended")

    def add_classification(self, label: str, confidence: float) -> None:
        """Record a single-window classification result for the current trial."""
        if self._state == TrialState.ACTIVE:
            self._labels.append(label)
            self._confidences.append(confidence)

    def get_cumulative_decision(self) -> tuple[str, float]:
        """Get the cumulative decision for the current/last trial via majority vote."""
        if not self._labels:
            return "unknown", 0.0
        counter = Counter(self._labels)
        best_label = counter.most_common(1)[0][0]
        # Average confidence for the winning label
        winning_confs = [c for label, c in zip(self._labels, self._confidences) if label == best_label]
        avg_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0.0
        return best_label, avg_conf

    def reset(self) -> None:
        """Reset to WAITING state (new trial can begin)."""
        self._state = TrialState.WAITING
        self._labels.clear()
        self._confidences.clear()
