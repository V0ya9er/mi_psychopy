"""LSL I/O for the real-time classifier: EEG/Marker subscription + result publication.

Stream resolution: by stream NAME — fast and unambiguous.
  - EEG stream:   name="obci_eeg1" (OpenBCI GUI default for TimeSeriesRaw)
  - Marker stream: name="obci_eeg2" (OpenBCI GUI default for Marker)
Configurable via ClassifierConfig.eeg_stream_name / marker_stream_name.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from rt_classifier.config import ClassifierConfig
from rt_classifier.feedback import LSLFeedbackSink
from rt_classifier.config import ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class EEGStreamInfo:
    """Metadata about the connected EEG stream."""
    sample_rate: float
    n_channels: int
    channel_names: list[str]


class LSLSubscriber:
    """Subscribe to EEG and Marker LSL streams (non-blocking)."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        self._config = config or ClassifierConfig()
        self._eeg_inlet: Any = None
        self._marker_inlet: Any = None
        self._eeg_info: EEGStreamInfo | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def eeg_info(self) -> EEGStreamInfo | None:
        return self._eeg_info

    def connect(self, timeout_s: float = 30.0) -> bool:
        """Resolve and connect to EEG + Marker streams. Returns True on success."""
        import pylsl

        # Resolve EEG stream by name
        eeg_streams = pylsl.resolve_byprop(
            "name", self._config.eeg_stream_name,
            timeout=timeout_s, minimum=1,
        )
        if not eeg_streams:
            logger.error(
                f"EEG stream (name={self._config.eeg_stream_name!r}) not found "
                f"within {timeout_s}s"
            )
            return False

        self._eeg_inlet = pylsl.StreamInlet(eeg_streams[0], max_buflen=360)
        info = eeg_streams[0]
        sr = info.nominal_srate()
        n_ch = info.channel_count()
        ch_names = self._read_channel_names(info, n_ch)
        self._eeg_info = EEGStreamInfo(sample_rate=sr, n_channels=n_ch, channel_names=ch_names)
        logger.info(f"Connected to EEG stream: name={info.name()!r} {n_ch}ch @ {sr}Hz")

        # Resolve Marker stream by name
        marker_streams = pylsl.resolve_byprop(
            "name", self._config.marker_stream_name,
            timeout=5.0, minimum=1,
        )
        if not marker_streams:
            logger.warning(
                f"Marker stream (name={self._config.marker_stream_name!r}) not found — "
                f"classifier will operate without marker events"
            )
        else:
            self._marker_inlet = pylsl.StreamInlet(marker_streams[0])
            minfo = marker_streams[0]
            logger.info(
                f"Connected to Marker stream: name={minfo.name()!r} "
                f"type={minfo.type()!r}"
            )

        self._connected = True
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_channel_names(info: Any, n_ch: int) -> list[str]:
        """Try to read channel names from stream description XML."""
        try:
            ch_elem = info.desc().child("channels")
            names: list[str] = []
            for i in range(n_ch):
                ch = ch_elem.child("channel")
                label = ch.child_value("label")
                names.append(label or f"ch{i}")
            if names and len(names) == n_ch:
                return names
        except Exception:
            pass
        return [f"ch{i}" for i in range(n_ch)]

    # ------------------------------------------------------------------
    # Data pull methods
    # ------------------------------------------------------------------

    def pull_eeg_chunk(self, max_samples: int = 256) -> tuple[np.ndarray, np.ndarray] | None:
        """Non-blocking pull of EEG data chunk. Returns (samples, timestamps) or None."""
        if not self._eeg_inlet:
            return None
        try:
            samples, timestamps = self._eeg_inlet.pull_chunk(timeout=0.0, max_samples=max_samples)
            if not samples:
                return None
            return np.array(samples), np.array(timestamps)
        except Exception:
            return None

    def pull_marker(self, timeout: float = 0.0) -> tuple[float, int] | None:
        """Pull a single marker (non-blocking by default). Returns (timestamp, marker_value) or None."""
        if not self._marker_inlet:
            return None
        try:
            sample, ts = self._marker_inlet.pull_sample(timeout=timeout)
            if sample is not None:
                return (ts, int(sample[0]))
            return None
        except Exception:
            return None

    def disconnect(self) -> None:
        self._eeg_inlet = None
        self._marker_inlet = None
        self._connected = False
        self._eeg_info = None


class LSLPublisher:
    """Publish classification results via LSL outlet."""

    def __init__(self, config: ClassifierConfig | None = None) -> None:
        cfg = config or ClassifierConfig()
        self._sink = LSLFeedbackSink(
            stream_name=cfg.feedback_stream_name,
            stream_type=cfg.feedback_stream_type,
        )

    def publish(self, result: ClassificationResult) -> None:
        self._sink.push_result(result)

    def close(self) -> None:
        self._sink.close()
