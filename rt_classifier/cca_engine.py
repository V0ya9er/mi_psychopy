"""Filter Bank CCA (FBCCA) SSVEP frequency detection engine.

Replaces the basic CCA engine with FBCCA (Chen et al. 2015) which decomposes
the EEG into multiple frequency sub-bands before CCA, providing much better
discrimination — especially for 10Hz SSVEP which sits in the alpha band.

Algorithm:
  1. Decompose EEG into N sub-bands (default 5) via bandpass filtering.
     Sub-band k covers [8+8k, 8+8(k+1)] Hz (Chen 2015 design):
       Band 1: [8-16]Hz, Band 2: [16-24]Hz, ..., Band 5: [40-48]Hz
  2. For each sub-band, compute CCA correlation between filtered EEG and
     reference sin/cos signals for each candidate frequency.
  3. Combine correlations across sub-bands using fixed weights:
     w_n = n^(-1.25) + 0.25  (from Chen et al. 2015)
  4. The frequency with the highest weighted sum wins.

Key advantages over basic CCA:
- Harmonics in higher sub-bands (20Hz, 30Hz) are outside the alpha band,
  so they discriminate SSVEP from alpha without contamination.
- No training data required — weights are fixed by the formula.

Reference:
  Chen, X. et al. (2015). Filter bank canonical correlation analysis for
  implementing a high-speed SSVEP-based brain-computer interface.
  Journal of Neural Engineering, 12(4), 046008.
"""
from __future__ import annotations

import numpy as np

from rt_classifier.config import ClassifierConfig


# Default number of filter-bank sub-bands (from Chen 2015)
_NUM_SUB_BANDS = 5

# Number of harmonics in reference signals.
# 2 harmonics avoids two problems:
#   1) 15Hz 3rd harmonic (45Hz) spurious-correlates with noise in band 5 [40-48]
#   2) 10Hz 3rd harmonic (30Hz) = 15Hz 2nd harmonic → cross-contamination
# With harmonics=2: 10Hz refs={10,20}Hz, 15Hz refs={15,30}Hz — clean separation
# across sub-bands.
_NUM_HARMONICS = 2


class CCAEngine:
    """Filter Bank CCA (FBCCA) SSVEP frequency detection engine.

    Uses Chen et al. 2015 FBCCA: filter-bank decomposition + standard CCA
    per sub-band + weighted score combination.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self._config = config
        self._freqs = [config.left_freq_hz, config.right_freq_hz]
        self._labels = ["left", "right"]
        self._n_harmonics = _NUM_HARMONICS
        self._ssvep_channels = config.ssvep_channels
        self._sample_rate = config.sample_rate
        self._both_sides_mode = config.ssvep_both_sides_mode

        # Pre-compute filter-bank coefficients for each sub-band
        # Chen 2015 standard design: sub-band k covers [k*bandwidth, (k+1)*bandwidth] Hz
        # starting from a base of bandwidth Hz.  Default bandwidth = 8 Hz gives:
        #   Band 1: [8, 16]  — captures 10Hz and 15Hz fundamentals
        #   Band 2: [16, 24] — captures 2nd harmonic of 10Hz (20Hz)
        #   Band 3: [24, 32] — captures 2nd harmonic of 15Hz (30Hz), 3rd of 10Hz
        #   Band 4: [32, 40] — higher harmonics
        #   Band 5: [40, 48] — even higher harmonics
        # Key advantage: higher bands are free from alpha contamination,
        # so they discriminate SSVEP correctly even when band 1 is ambiguous.
        nyquist = 0.5 * self._sample_rate
        bandwidth = 8.0  # Hz per sub-band (Chen 2015)
        base = bandwidth  # start at 8 Hz

        self._fb_sos: list[np.ndarray] = []
        try:
            from scipy.signal import butter
            for k in range(_NUM_SUB_BANDS):
                low_hz = base + k * bandwidth
                high_hz = base + (k + 1) * bandwidth
                low = max(low_hz / nyquist, 0.01)
                high = min(high_hz / nyquist, 0.99)
                if low < high:
                    self._fb_sos.append(
                        butter(4, [low, high], btype="band", output="sos")
                    )
                else:
                    self._fb_sos.append(np.array([]))  # unusable sub-band
        except ImportError:
            self._fb_sos = []

        # Fixed weights from Chen 2015: w_n = n^(-1.25) + 0.25
        self._fb_weights = np.power(
            np.arange(1, _NUM_SUB_BANDS + 1, dtype=np.float64), -1.25
        ) + 0.25

        # CCA regularization — strong enough to prevent spurious perfect
        # correlations when many channels carry narrowband alpha rhythm.
        # Empirically validated on real EEG: 0.1 balances left/right detection.
        self._cca_reg = 0.1

    def classify(self, window: np.ndarray) -> tuple[str, float]:
        """Classify SSVEP frequency using FBCCA.

        Args:
            window: shape (n_samples, n_channels) — full 8-channel EEG window

        Returns:
            (label, confidence): label is "left" or "right", confidence in (0,1)
        """
        # Extract SSVEP channels
        ssvep_data = window[:, self._ssvep_channels]  # (n_samples, n_ssvep_ch)
        n_samples = ssvep_data.shape[0]

        # If no filter-bank filters available, fall back to basic CCA
        if not self._fb_sos:
            return self._basic_cca_classify(ssvep_data, n_samples)

        # FBCCA with intra-band normalization:
        # For each sub-band, compute CCA scores for ALL candidate frequencies,
        # then normalize WITHIN the band by computing the relative score:
        #   rel_i = score_i / sum(score_j for all j)
        # This eliminates per-band amplitude bias (e.g., alpha rhythm inflating
        # both 10Hz and 15Hz equally in band 1).  The relative proportion of
        # each frequency's contribution is what matters.
        num_freqs = len(self._freqs)
        weighted_rel = np.zeros(num_freqs)

        for fb_i, sos in enumerate(self._fb_sos):
            if sos is None or len(sos) == 0:
                continue
            # Filter into this sub-band
            filtered = self._sos_filter(sos, ssvep_data)
            weight = self._fb_weights[fb_i]

            # Compute raw CCA scores for all frequencies in this band
            band_scores = np.array([
                self._cca_score(filtered, freq, self._sample_rate, n_samples)
                for freq in self._freqs
            ])

            # Intra-band normalization: compute relative proportion
            band_total = band_scores.sum()
            if band_total > 1e-8:
                rel_scores = band_scores / band_total
            else:
                rel_scores = np.ones(num_freqs) / num_freqs  # uniform if all zero

            # Weight and accumulate
            weighted_rel += weight * rel_scores

        # Determine winner from normalized relative scores
        sorted_indices = np.argsort(weighted_rel)[::-1]
        best_label = self._labels[sorted_indices[0]]
        max_rel = weighted_rel[sorted_indices[0]]
        second_rel = weighted_rel[sorted_indices[1]]

        # Confidence: margin between top two relative scores.
        margin = max_rel - second_rel
        confidence = float(np.clip(margin, 0.0, 1.0))

        # Both-sides flicker: use channel asymmetry to disambiguate
        if self._both_sides_mode and margin < 0.1 * max_rel:
            asym_label, asym_conf = self._classify_by_asymmetry(
                window, n_samples,
            )
            if asym_label is not None:
                best_label = asym_label
                confidence = asym_conf

        return best_label, confidence

    # ------------------------------------------------------------------
    # Fallback: basic CCA (no filter bank)
    # ------------------------------------------------------------------

    def _basic_cca_classify(
        self, ssvep_data: np.ndarray, n_samples: int,
    ) -> tuple[str, float]:
        """Fallback basic CCA when filter-bank is unavailable."""
        scores = [
            self._cca_score(ssvep_data, freq, self._sample_rate, n_samples)
            for freq in self._freqs
        ]
        sorted_indices = np.argsort(scores)[::-1]
        best_label = self._labels[sorted_indices[0]]
        margin = scores[sorted_indices[0]] - scores[sorted_indices[1]]
        confidence = float(np.clip(margin, 0.0, 1.0))
        return best_label, confidence

    # ------------------------------------------------------------------
    # Filter-bank helper
    # ------------------------------------------------------------------

    @staticmethod
    def _sos_filter(sos: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Apply SOS zero-phase filter along axis 0.

        Args:
            sos: second-order sections from scipy.signal.butter
            data: shape (n_samples, n_channels)

        Returns:
            Filtered data, same shape.
        """
        if data.shape[0] < 24:
            return data
        try:
            from scipy.signal import sosfiltfilt
            return sosfiltfilt(sos, data, axis=0)
        except Exception:
            return data

    # ------------------------------------------------------------------
    # Both-sides flicker: channel asymmetry detection
    # ------------------------------------------------------------------

    def _classify_by_asymmetry(
        self,
        window: np.ndarray,
        n_samples: int,
    ) -> tuple[str | None, float]:
        """Use channel asymmetry to detect gaze direction in both-sides mode."""
        o1_idx = 6  # C3,Cz,C4,P3,Pz,P4,O1,O2
        o2_idx = 7

        o1_data = window[:, o1_idx:o1_idx+1]
        o2_data = window[:, o2_idx:o2_idx+1]

        left_freq = self._freqs[0]
        right_freq = self._freqs[1]

        # Compute FBCCA-style scores per channel
        o1_left = self._compute_rho(o1_data, left_freq, n_samples)
        o1_right = self._compute_rho(o1_data, right_freq, n_samples)
        o2_left = self._compute_rho(o2_data, left_freq, n_samples)
        o2_right = self._compute_rho(o2_data, right_freq, n_samples)

        o1_best = max(o1_left, o1_right)
        o2_best = max(o2_left, o2_right)

        asymmetry = o2_best - o1_best
        total = o1_best + o2_best + 1e-6
        norm_asym = abs(asymmetry) / total

        if norm_asym < 0.05:
            return None, 0.0

        if asymmetry > 0:
            return "left", float(np.clip(norm_asym + 0.3, 0.3, 0.9))
        else:
            return "right", float(np.clip(norm_asym + 0.3, 0.3, 0.9))

    def _compute_rho(
        self, data: np.ndarray, target_freq: float, n_samples: int,
    ) -> float:
        """Compute FBCCA weighted score for a single-channel signal."""
        rho = 0.0
        for fb_i, sos in enumerate(self._fb_sos):
            if sos is None or len(sos) == 0:
                continue
            filtered = self._sos_filter(sos, data)
            score = self._cca_score(filtered, target_freq, self._sample_rate, n_samples)
            rho += self._fb_weights[fb_i] * score
        return rho

    # ------------------------------------------------------------------
    # Core CCA — eigenvalue-based (no sklearn dependency)
    # ------------------------------------------------------------------

    def _cca_score(
        self,
        data: np.ndarray,
        target_freq: float,
        sample_rate: float,
        n_samples: int,
    ) -> float:
        """Compute CCA correlation between data and reference signals."""
        t = np.arange(n_samples) / sample_rate

        # Build reference signals: base freq + harmonics (sin + cos each)
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

        # Regularisation — uses self._cca_reg for anti-overfitting
        reg = self._cca_reg
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
