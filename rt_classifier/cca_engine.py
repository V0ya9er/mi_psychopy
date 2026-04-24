"""CCA-based SSVEP frequency detection engine.

Adapted from mi_benchmark/verify_ssvep_signal.py:cca_ssvep_score()
(lines 446-490). Algorithm copied faithfully — do NOT import from
verify_ssvep_signal (it depends on mi_benchmark's mi_dataset).
"""
from __future__ import annotations

import numpy as np

from rt_classifier.config import ClassifierConfig


class CCAEngine:
    """CCA-based SSVEP frequency detection engine.

    Uses Canonical Correlation Analysis to detect which target frequency
    (left or right SSVEP) is present in the occipital EEG channels.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self._config = config
        self._freqs = [config.left_freq_hz, config.right_freq_hz]
        self._labels = ["left", "right"]
        self._n_harmonics = config.cca_n_harmonics
        self._ssvep_channels = config.ssvep_channels  # (6, 7) = O1, O2

    def classify(self, window: np.ndarray) -> tuple[str, float]:
        """Classify SSVEP frequency from a window of EEG data.

        Args:
            window: shape (n_samples, n_channels) — full 8-channel EEG window

        Returns:
            (label, confidence): label is "left" or "right", confidence in (0,1)
        """
        # Extract SSVEP channels (O1, O2)
        ssvep_data = window[:, self._ssvep_channels]  # (n_samples, n_ssvep_ch)
        n_samples = ssvep_data.shape[0]
        sample_rate = self._config.sample_rate

        # Compute CCA score for each target frequency
        scores = [
            self._cca_score(ssvep_data, freq, sample_rate, n_samples)
            for freq in self._freqs
        ]

        # Sort scores descending
        sorted_indices = np.argsort(scores)[::-1]
        best_label = self._labels[sorted_indices[0]]
        max_score = scores[sorted_indices[0]]
        second_score = scores[sorted_indices[1]]
        confidence = max_score / (max_score + second_score + 1e-6)

        return best_label, float(confidence)

    # ------------------------------------------------------------------
    # Core CCA — copied from verify_ssvep_signal.py:cca_ssvep_score()
    # ------------------------------------------------------------------

    def _cca_score(
        self,
        data: np.ndarray,
        target_freq: float,
        sample_rate: float,
        n_samples: int,
    ) -> float:
        """Compute CCA correlation between data and reference signals.

        Reference: verify_ssvep_signal.py:446-490 cca_ssvep_score()
        """
        t = np.arange(n_samples) / sample_rate

        # Build reference signals: base freq + harmonics (sin + cos each)
        # Harmonics: h = 1 .. n_harmonics  (NOT 0 .. n_harmonics)
        ref_signals = []
        for h in range(1, self._n_harmonics + 1):
            ref_signals.append(np.sin(2 * np.pi * h * target_freq * t))
            ref_signals.append(np.cos(2 * np.pi * h * target_freq * t))
        Y = np.stack(ref_signals, axis=1)  # (n_samples, 2*n_harmonics)

        X = data  # (n_samples, n_channels)

        # Centre
        X_c = X - X.mean(axis=0, keepdims=True)
        Y_c = Y - Y.mean(axis=0, keepdims=True)

        n_x = X_c.shape[1]
        n_y = Y_c.shape[1]

        # Covariance matrices
        denom = max(n_samples - 1, 1)
        C_xx = (X_c.T @ X_c) / denom
        C_yy = (Y_c.T @ Y_c) / denom
        C_xy = (X_c.T @ Y_c) / denom

        # Regularisation (matches verify_ssvep_signal.py)
        reg = 1e-6
        C_xx_reg = C_xx + reg * np.eye(n_x)
        C_yy_reg = C_yy + reg * np.eye(n_y)

        try:
            C_xx_inv = np.linalg.inv(C_xx_reg)
            C_yy_inv = np.linalg.inv(C_yy_reg)
            M = C_xx_inv @ C_xy @ C_yy_inv @ C_xy.T
            eigenvalues = np.linalg.eigvalsh(M)
            eigenvalues = np.clip(eigenvalues, 0, 1)
            return float(np.sqrt(np.max(eigenvalues)))
        except np.linalg.LinAlgError:
            return 0.0
