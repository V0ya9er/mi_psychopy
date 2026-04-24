"""FeedbackSink protocol and implementations for classification results.

- FeedbackSink: protocol (push_result + close)
- LSLFeedbackSink: push results to LSL outlet
- AudioFeedbackSink: stub for future audio feedback engine
- CompositeFeedbackSink: broadcast to multiple sinks
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from rt_classifier.config import ClassificationResult

logger = logging.getLogger(__name__)


@runtime_checkable
class FeedbackSink(Protocol):
    """Protocol for classification result feedback."""

    def push_result(self, result: ClassificationResult) -> None: ...
    def close(self) -> None: ...


class LSLFeedbackSink:
    """Push classification results to an LSL outlet.

    Outlet format: 6 channels float32
    [label_int, confidence, cca_score, mi_score, timestamp, trial_id]
    """

    def __init__(self, stream_name: str = "rt_classifier", stream_type: str = "classification_result") -> None:
        import pylsl

        info = pylsl.StreamInfo(
            stream_name, stream_type,
            channel_count=6,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_float32,
            source_id="rt_classifier_py",
        )
        chns = info.desc().append_child("channels")
        for label in ["label_int", "confidence", "cca_score", "mi_score", "timestamp", "trial_id"]:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)

        info.desc().append_child_value("label_mapping", "0=left,1=right")
        info.desc().append_child_value("freq_mapping", "left=10Hz,right=15Hz")
        info.desc().append_child_value("classifier_version", "1.0.0")

        self._outlet = pylsl.StreamOutlet(info)

    def push_result(self, result: ClassificationResult) -> None:
        label_int = 0 if result.label == "left" else 1
        self._outlet.push_sample([
            float(label_int), result.confidence, result.cca_score,
            result.mi_score, result.timestamp, float(result.trial_id),
        ])

    def close(self) -> None:
        self._outlet = None


class AudioFeedbackSink:
    """Future: audio feedback based on classification results.

    Will map classification labels to audio cues (e.g., directional sound,
    pitch mapping) for closed-loop neurofeedback.
    """

    def push_result(self, result: ClassificationResult) -> None:
        pass  # TODO: implement audio feedback

    def close(self) -> None:
        pass


class CompositeFeedbackSink:
    """Broadcast results to multiple FeedbackSink instances."""

    def __init__(self, sinks: list[FeedbackSink] | None = None) -> None:
        self._sinks: list[FeedbackSink] = list(sinks or [])

    def add(self, sink: FeedbackSink) -> None:
        self._sinks.append(sink)

    def push_result(self, result: ClassificationResult) -> None:
        for sink in self._sinks:
            try:
                sink.push_result(result)
            except Exception as e:
                logger.warning(f"FeedbackSink error: {e}")

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as e:
                logger.warning(f"FeedbackSink close error: {e}")
