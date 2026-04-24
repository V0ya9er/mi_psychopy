"""Real-time SSVEP + MI hybrid classifier for OpenBCI LSL streams."""

__version__ = "0.1.0"

from rt_classifier.cca_engine import CCAEngine
from rt_classifier.mi_engine import MIEngine
from rt_classifier.feedback import FeedbackSink, LSLFeedbackSink
from rt_classifier.config import ClassifierConfig, ClassificationResult, TrialState
from rt_classifier.buffer import RingBuffer
from rt_classifier.decision import DecisionFusion, TrialTracker
from rt_classifier.decision_log import DecisionLogger

__all__ = [
    "CCAEngine",
    "MIEngine",
    "FeedbackSink",
    "LSLFeedbackSink",
    "ClassifierConfig",
    "ClassificationResult",
    "TrialState",
    "RingBuffer",
    "DecisionFusion",
    "TrialTracker",
    "DecisionLogger",
]
