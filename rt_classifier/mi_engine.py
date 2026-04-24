"""MI model inference adapter using pre-trained deep learning models.

Supports DeepConvNet, ShallowConvNet, EEGNet, and MISSVEPBranchNet checkpoints
trained by mi_benchmark.

Preprocessing pipeline: detrend → notch 50Hz → bandpass 4-40Hz → z-score.
Algorithms copied from mi_benchmark/mi_dataset.py — do NOT import from
mi_dataset (it depends on pyxdf and other heavy libraries).

If no checkpoint is available, the engine degrades gracefully:
available=False, classify() returns ("unknown", 0.0).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rt_classifier.config import ClassifierConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Marker-value → human-readable label mapping.
# Matches the marker conventions used in markers.py.
_MARKER_LABEL_MAP: dict[int, str] = {
    11: "left", 12: "right", 13: "rest",           # cue
    21: "left", 22: "right", 23: "rest",            # mi
    41: "left", 42: "right",                         # ao_mi
    51: "left", 52: "right",                         # mi_only
    61: "left", 62: "right",                         # ssvep
    101: "left", 102: "right",                       # arrow_cue
    111: "left", 112: "right",                       # arrow_mi
    181: "left", 182: "right",                       # rt_ssvep
}

# Model name → (class_name, constructor_kwargs_keys)
_MODEL_REGISTRY: dict[str, tuple[str, tuple[str, ...]]] = {
    "deepconvnet": ("DeepConvNetModel", ("chans", "classes", "time_points", "dropout_rate")),
    "shallowconvnet": ("ShallowConvNetModel", ("chans", "classes", "time_points", "dropout_rate")),
    "eegnet": ("EEGNet", ("chans", "classes", "time_points", "dropout_rate")),
    "mi_ssvep_branchnet": ("MISSVEPBranchNet", ("chans", "classes", "time_points", "dropout_rate")),
}


def _ensure_models_path() -> None:
    """Add mi_benchmark directory to sys.path so ``from models import …`` works."""
    # mi_benchmark is a sibling of mi_psychopy
    mi_benchmark = Path(__file__).resolve().parent.parent.parent / "mi_benchmark"
    candidate = str(mi_benchmark)
    if candidate not in sys.path and mi_benchmark.is_dir():
        sys.path.insert(0, candidate)
        logger.info(f"Added to sys.path: {candidate}")


# ---------------------------------------------------------------------------
# Preprocessing helpers — adapted from mi_benchmark/mi_dataset.py
# ---------------------------------------------------------------------------

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
    from scipy.signal import butter, sosfiltfilt
    nyquist = 0.5 * sample_rate
    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def _zscore(data: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score normalisation. Shape: (samples, channels)."""
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (data - mean) / std


class MIEngine:
    """MI classification engine using pre-trained deep learning models.

    Automatically detects the model architecture from the checkpoint's
    ``model_name`` field and instantiates the correct class.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self._config: ClassifierConfig = config
        self._model: Any = None
        self._label_mapping: dict[int, str] = {}
        self._available: bool = False
        self._expected_time_points: int = 256
        checkpoint_path = config.mi_checkpoint_path
        if checkpoint_path and str(checkpoint_path).strip():
            self.load_checkpoint(Path(str(checkpoint_path)))

    @property
    def available(self) -> bool:
        return self._available

    def load_checkpoint(self, path: Path) -> bool:
        """Load a pre-trained model checkpoint. Returns True on success."""
        try:
            import torch

            if not path.exists():
                logger.warning(f"Checkpoint not found: {path}")
                return False

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            model_name = checkpoint.get("model_name", "deepconvnet").lower()
            num_channels = checkpoint.get("num_channels", 8)
            time_points = checkpoint.get("time_points", 256)
            num_classes = len(checkpoint.get("label_mapping", {}))
            if num_classes < 2:
                num_classes = 2
            dropout = checkpoint.get("dropout", 0.5)

            self._expected_time_points = time_points

            # Resolve model class — models is added to sys.path by _ensure_models_path()
            _ensure_models_path()
            from models.cnn_models import DeepConvNetModel, ShallowConvNetModel, MISSVEPBranchNet  # pyright: ignore[reportMissingImports]

            _MODEL_CLASSES = {
                "deepconvnet": DeepConvNetModel,
                "shallowconvnet": ShallowConvNetModel,
                "mi_ssvep_branchnet": MISSVEPBranchNet,
                # EEGNet lives in its own file
            }
            # Try to import EEGNet separately (may not exist)
            try:
                from models.eegnet import EEGNet  # pyright: ignore[reportMissingImports]
                _MODEL_CLASSES["eegnet"] = EEGNet
            except ImportError:
                pass

            model_cls = _MODEL_CLASSES.get(model_name)
            if model_cls is None:
                logger.warning(f"Unknown model_name '{model_name}' in checkpoint. "
                               f"Available: {list(_MODEL_CLASSES.keys())}")
                return False

            model = model_cls(
                chans=num_channels,
                classes=num_classes,
                time_points=time_points,
                dropout_rate=dropout,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self._model = model

            # Build reverse label mapping: {mapped_int: "left"/"right"}
            # checkpoint.label_mapping: {original_marker: mapped_int} e.g. {61: 0, 62: 1}
            # checkpoint.class_names: {mapped_int: "class_XX"} e.g. {0: "class_61", 1: "class_62"}
            raw_mapping = checkpoint.get("label_mapping", {})
            class_names = checkpoint.get("class_names", {})
            self._label_mapping = {}
            for orig, mapped in raw_mapping.items():
                # Look up by mapped int key in class_names
                name = class_names.get(mapped, class_names.get(str(mapped), None))
                if name is not None:
                    # Try to resolve marker-based names (e.g. "class_61" → "left")
                    if isinstance(name, str) and name.startswith("class_"):
                        try:
                            marker_val = int(name.replace("class_", ""))
                            name = _MARKER_LABEL_MAP.get(marker_val, name)
                        except ValueError:
                            pass
                if name is None:
                    name = str(orig)
                self._label_mapping[int(mapped)] = name

            self._available = True
            logger.info(f"Loaded {model_name} from {path} "
                        f"(ch={num_channels}, T={time_points}, classes={num_classes})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load MI checkpoint: {e}")
            self._available = False
            return False

    def preprocess(self, window: np.ndarray) -> "torch.Tensor":
        """Preprocess EEG window for MI model inference.
        Pipeline: detrend → notch 50Hz → bandpass 4-40Hz → z-score → reshape
        Input shape: (n_samples, n_channels)
        Output shape: (1, 1, n_channels, expected_time_points)
        """
        import torch
        data = window.copy().astype(np.float64)
        data = _detrend(data)
        data = _notch(data, freq=50.0, sample_rate=self._config.sample_rate)
        data = _bandpass(data, low=4.0, high=40.0, sample_rate=self._config.sample_rate)
        data = _zscore(data)

        # Trim or pad to match model's expected time_points
        n_samples, n_channels = data.shape
        expected = self._expected_time_points
        if n_samples > expected:
            data = data[-expected:, :]
        elif n_samples < expected:
            pad = np.zeros((expected - n_samples, n_channels), dtype=data.dtype)
            data = np.concatenate([pad, data], axis=0)

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
