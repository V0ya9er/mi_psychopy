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
    # LSL stream query parameters — resolve by stream NAME (OpenBCI GUI defaults)
    eeg_stream_name: str = "obci_eeg1"       # TimeSeriesRaw stream
    marker_stream_name: str = "obci_eeg2"     # Marker stream
    lsl_connect_timeout_s: float = 30.0

    # Classifier parameters
    window_size_s: float = 1.5
    stride_s: float = 0.25
    cca_n_harmonics: int = 2   # 2 avoids 30Hz cross-contamination between 10Hz(3rd) and 15Hz(2nd)
    confidence_threshold: float = 0.15  # margin-based confidence — much lower range than ratio-based

    # SSVEP frequency mapping
    left_freq_hz: float = 10.0
    right_freq_hz: float = 15.0

    # MI model checkpoint path — default points to best DeepConvNet fold
    mi_checkpoint_path: str | Path = (
        r"D:\CSDIY\EEG\OLM\mi_benchmark\results"
        r"\0418_eegnet_deepconvnet_shallowconvnet_mi_ssvep_branchnet_mi_ssvep_branchnet_mi_only_mi_ssvep_branchnet_ssvep_only_logreg_svm_random_forest_fbcsp_lda"
        r"\deepconvnet\artifacts\fold_2.pth"
    )

    # Feedback LSL stream parameters
    feedback_stream_name: str = "rt_classifier"
    feedback_stream_type: str = "classification_result"

    # Channel selection (indices in 8-channel layout: C3,Cz,C4,P3,Pz,P4,O1,O2)
    # FBCCA SSVEP: Pz+O1+O2 (indices 4,6,7) gives 94% trial accuracy on real data.
    # Including motor channels (C3,Cz,C4) adds alpha-rhythm noise that confuses CCA.
    ssvep_channels: tuple[int, ...] = (4, 6, 7)  # Pz, O1, O2
    mi_channels: tuple[int, ...] = (0, 1, 2, 3, 4, 5)  # C3,Cz,C4,P3,Pz,P4

    # Both-sides flicker mode: when True, uses channel asymmetry (O1 vs O2)
    # to disambiguate gaze direction when both frequencies are present
    ssvep_both_sides_mode: bool = False

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
