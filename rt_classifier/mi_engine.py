"""MI model inference adapter using pre-trained MISSVEPBranchNet.

Preprocessing pipeline: detrend → notch 50Hz → bandpass 4-40Hz → z-score.
Algorithms copied from mi_benchmark/mi_dataset.py — do NOT import from
mi_dataset (it depends on pyxdf and other heavy libraries).

If no checkpoint is available, the engine degrades gracefully:
available=False, classify() returns ("unknown", 0.0).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rt_classifier.config import ClassifierConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Preprocessing helpers — adapted from mi_benchmark/mi_dataset.py

def _detrend(data: np.ndarray) -> np.ndarray:
    """Linear detrend along axis 0 (time axis). Shape: (samples, channels)."""
    from scipy.signal import detrend as scipy_detrend
    return scipy_detrend(data, axis=0, type="linear")


def _notch(data: np.ndarray, freq: float = 50.0, sample_rate: float = 250.0, quality: float = 30.0) -> np.ndarray:
    """IIR notch filter along axis 0. Shape: (samples, channels)."""
    from scipy.signal import filtfilt, iirnotch
    b, a = iirnotch(w0=freq, Q=quality, fs=sample_rate)
    return filtfilt(b, a, data, axis=0)


def _bandpass(data: np.ndarray, low: float = 4.0, high: float = 40.0, sample_rate: float = 250.0, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter along axis 0. Shape: (samples, channels)."""
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * sample_rate
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, data, axis=0)


def _zscore(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalisation. Shape: (samples, channels)."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (data - mean) / std


class MIEngine:
    """MI classification engine using pre-trained MISSVEPBranchNet."""

    def __init__(self, config: ClassifierConfig) -> None:
        self._config: ClassifierConfig = config
        self._model: Any = None
        self._label_mapping: dict[int, str] = {}
        self._available: bool = False
        checkpoint_path = config.mi_checkpoint_path
        if checkpoint_path and str(checkpoint_path).strip():
            self.load_checkpoint(Path(str(checkpoint_path)))

    @property
    def available(self) -> bool:
        return self._available

    def load_checkpoint(self, path: Path) -> bool:
        """Load a pre-trained MISSVEPBranchNet checkpoint. Returns True on success."""
        try:
            import torch
            from models.cnn_models import MISSVEPBranchNet

            if not path.exists():
                logger.warning(f"Checkpoint not found: {path}")
                return False

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            num_channels = checkpoint.get("num_channels", 8)
            time_points = checkpoint.get("time_points", 256)
            num_classes = len(checkpoint.get("label_mapping", {}))
            if num_classes < 2:
                num_classes = 2
            dropout = checkpoint.get("dropout", 0.5)

            model = MISSVEPBranchNet(
                chans=num_channels,
                classes=num_classes,
                time_points=time_points,
                dropout_rate=dropout,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self._model = model
            # label_mapping is {original_label: mapped_int}, e.g. {61: 0, 62: 1}
            # Build reverse mapping: {mapped_int: "left"/"right"}
            raw_mapping = checkpoint.get("label_mapping", {})
            class_names = checkpoint.get("class_names", {})
            self._label_mapping = {}
            for orig, mapped in raw_mapping.items():
                name = class_names.get(orig, str(orig))
                self._label_mapping[int(mapped)] = name

            self._available = True
            logger.info(f"Loaded MI model from {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load MI checkpoint: {e}")
            self._available = False
            return False

    def preprocess(self, window: np.ndarray) -> torch.Tensor:
        """Preprocess EEG window for MI model inference.
        Pipeline: detrend → notch 50Hz → bandpass 4-40Hz → z-score → reshape
        Input shape: (n_samples, n_channels)
        Output shape: (1, 1, n_channels, n_samples)
        """
        import torch
        data = window.copy().astype(np.float64)
        data = _detrend(data)
        data = _notch(data, freq=50.0, sample_rate=self._config.sample_rate)
        data = _bandpass(data, low=4.0, high=40.0, sample_rate=self._config.sample_rate)
        data = _zscore(data)
        # (n_samples, n_channels) → (n_channels, n_samples) → (1, 1, C, T)
        data = data.T[np.newaxis, np.newaxis, :, :]
        return torch.from_numpy(data).float()

    def classify(self, window: np.ndarray) -> tuple[str, float]:
        """Classify MI from a window of EEG data. Returns (label, confidence)."""
        if not self._available:
            return "unknown", 0.0
        try:
            import torch
            with torch.no_grad():
                tensor = self.preprocess(window)
                output = self._model(tensor)
                probs = torch.softmax(output, dim=-1)
                conf, pred = probs.max(dim=-1)
                label_idx = int(pred.item())
                label = self._label_mapping.get(label_idx, f"class_{label_idx}")
                return label, float(conf.item())
        except Exception as e:
            logger.warning(f"MI classification error: {e}")
            return "unknown", 0.0
