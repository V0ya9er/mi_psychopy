from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class TrialState(Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    ENDED = "ended"


@dataclass(frozen=True)
class ClassifierConfig:
    # LSL stream query parameters
    eeg_stream_type: str = "EEG"
    marker_stream_type: str = "Markers"
    lsl_connect_timeout_s: float = 30.0

    # Classifier parameters
    window_size_s: float = 1.0
    stride_s: float = 0.25
    cca_n_harmonics: int = 3
    confidence_threshold: float = 0.6

    # SSVEP frequency mapping
    left_freq_hz: float = 10.0
    right_freq_hz: float = 15.0

    # MI model checkpoint path (optional)
    mi_checkpoint_path: str | Path = ""

    # Feedback LSL stream parameters
    feedback_stream_name: str = "rt_classifier"
    feedback_stream_type: str = "classification_result"

    # Channel selection (indices in 8-channel layout: C3,Cz,C4,P3,Pz,P4,O1,O2)
    ssvep_channels: tuple[int, ...] = (6, 7)  # O1, O2
    mi_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # C3,Cz,C4,P3,Pz,P4

    # Sample rate
    sample_rate: float = 250.0

    # Logging
    log_output_dir: str = "logs"
    run_name: str = "rt_classifier"


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    cca_score: float
    mi_score: float
    timestamp: float
    trial_id: int
    cca_label: str = ""
    mi_label: str = ""
