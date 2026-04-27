#!/usr/bin/env python
"""Offline replay: feed recorded XDF data through FBCCA + MI classifier.

Simulates the real-time classifier pipeline on pre-recorded data to evaluate
FBCCA and DeepConvNet performance without live EEG hardware.

Usage:
    python replay_offline.py                          # default XDF path
    python replay_offline.py path/to/run-1.xdf        # custom XDF
    python replay_offline.py --no-mi                   # FBCCA only
    python replay_offline.py --window 1.5 --stride 0.25

Output: per-trial classification report + overall accuracy for FBCCA and MI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

try:
    import pyxdf
except ImportError:
    sys.exit("ERROR: pyxdf not installed. Run: pip install pyxdf")

from rt_classifier.config import ClassifierConfig, TrialState
from rt_classifier.buffer import RingBuffer
from rt_classifier.cca_engine import CCAEngine
from rt_classifier.mi_engine import MIEngine
from rt_classifier.decision import DecisionFusion, TrialTracker

# Marker values (from markers.py)
MARKER_RT_LEFT = 181
MARKER_RT_RIGHT = 182
MARKER_RT_OFF = 189
MARKER_SSVEP_LEFT = 61
MARKER_SSVEP_RIGHT = 62
MARKER_TASK_OFF = 29
MARKER_CUE_LEFT = 11
MARKER_CUE_RIGHT = 12

# All markers that start a left trial
_LEFT_START_MARKERS = {MARKER_RT_LEFT, MARKER_SSVEP_LEFT, MARKER_CUE_LEFT}
_RIGHT_START_MARKERS = {MARKER_RT_RIGHT, MARKER_SSVEP_RIGHT, MARKER_CUE_RIGHT}
_ALL_END_MARKERS = {MARKER_RT_OFF, MARKER_TASK_OFF}


def load_xdf(xdf_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load XDF and return EEG data, EEG timestamps, markers, marker timestamps.

    Returns:
        eeg_data: (n_samples, n_channels) float64
        eeg_ts: (n_samples,) float64 timestamps
        marker_vals: (n_markers,) int marker values (non-zero only)
        marker_ts: (n_markers,) float64 timestamps
    """
    print(f"Loading XDF: {xdf_path}")
    streams, _ = pyxdf.load_xdf(xdf_path)

    # Find EEG stream (prefer obci_eeg1 or first 8-channel stream)
    eeg_stream = None
    marker_stream = None
    for s in streams:
        info = s["info"]
        name = info["name"][0]
        n_ch = int(info["channel_count"][0])
        if name == "obci_eeg1" or (eeg_stream is None and n_ch >= 8):
            eeg_stream = s
        if name == "obci_eeg_marker" or name == "obci_eeg2" or (marker_stream is None and n_ch == 1 and "marker" in name.lower()):
            marker_stream = s

    if eeg_stream is None:
        sys.exit("ERROR: No EEG stream found in XDF file")

    eeg_data = np.array(eeg_stream["time_series"], dtype=np.float64)
    eeg_ts = np.array(eeg_stream["time_stamps"], dtype=np.float64)

    # Extract non-zero markers
    if marker_stream is not None:
        raw_markers = np.array(marker_stream["time_series"], dtype=np.float64)[:, 0]
        raw_ts = np.array(marker_stream["time_stamps"], dtype=np.float64)
        nonzero = raw_markers != 0
        marker_vals = raw_markers[nonzero].astype(int)
        marker_ts = raw_ts[nonzero]
    else:
        marker_vals = np.array([], dtype=int)
        marker_ts = np.array([], dtype=np.float64)

    print(f"  EEG: {eeg_data.shape[0]} samples, {eeg_data.shape[1]} channels, "
          f"duration={eeg_ts[-1]-eeg_ts[0]:.1f}s")
    print(f"  Markers: {len(marker_vals)} non-zero events, "
          f"unique={np.unique(marker_vals)}")

    return eeg_data, eeg_ts, marker_vals, marker_ts


def replay(
    xdf_path: str,
    window_size_s: float = 1.5,
    stride_s: float = 0.25,
    mi_checkpoint: str = "",
    both_sides: bool = False,
    sample_rate: float = 250.0,
) -> None:
    """Replay XDF data through the classifier pipeline and report results."""

    eeg_data, eeg_ts, marker_vals, marker_ts = load_xdf(xdf_path)

    # Config
    config = ClassifierConfig(
        window_size_s=window_size_s,
        stride_s=stride_s,
        mi_checkpoint_path=mi_checkpoint or ClassifierConfig.mi_checkpoint_path,
        ssvep_both_sides_mode=both_sides,
        sample_rate=sample_rate,
    )

    # Initialize engines
    cca_engine = CCAEngine(config)
    mi_engine = MIEngine(config)
    fusion = DecisionFusion(config, cca_engine, mi_engine)

    print(f"\n--- Classifier Setup ---")
    print(f"FBCCA: {len(cca_engine._fb_sos)} filter-bank sub-bands")
    print(f"MI model available: {mi_engine.available}")
    if mi_engine.available:
        print(f"MI model: {type(mi_engine._model).__name__}")
    print(f"Window: {window_size_s}s ({int(window_size_s * sample_rate)} samples)")
    print(f"Stride: {stride_s}s")

    # Build trial timeline from markers
    trials: list[dict] = []
    current_trial = None
    for i, val in enumerate(marker_vals):
        if val in _LEFT_START_MARKERS:
            current_trial = {"condition": "left", "start_ts": marker_ts[i],
                             "end_ts": None, "marker": val}
        elif val in _RIGHT_START_MARKERS:
            current_trial = {"condition": "right", "start_ts": marker_ts[i],
                             "end_ts": None, "marker": val}
        elif val in _ALL_END_MARKERS and current_trial is not None:
            current_trial["end_ts"] = marker_ts[i]
            trials.append(current_trial)
            current_trial = None

    print(f"\n--- Trial Timeline ---")
    print(f"Total trials: {len(trials)}")
    n_left = sum(1 for t in trials if t["condition"] == "left")
    n_right = sum(1 for t in trials if t["condition"] == "right")
    print(f"Left trials: {n_left}, Right trials: {n_right}")

    # Classify each trial
    window_samples = int(window_size_s * sample_rate)
    stride_samples = int(stride_s * sample_rate)

    # Results storage
    trial_results: list[dict] = []

    print(f"\n--- Per-Window Classification ---")
    print(f"{'Trial':>5} {'Cond':>6} {'FBCCA':>6} {'MI':>6} {'Fusion':>7} {'Conf':>6} "
          f"{'FBCCA_sc':>9} {'MI_sc':>9}")
    print("-" * 70)

    for trial_idx, trial in enumerate(trials):
        condition = trial["condition"]
        start_ts = trial["start_ts"]
        end_ts = trial["end_ts"]
        if end_ts is None:
            end_ts = eeg_ts[-1]

        # Find sample indices for this trial
        start_idx = np.searchsorted(eeg_ts, start_ts)
        end_idx = np.searchsorted(eeg_ts, end_ts)
        trial_data = eeg_data[start_idx:end_idx]

        if len(trial_data) < window_samples:
            print(f"  Trial {trial_idx+1}: too short ({len(trial_data)} < {window_samples}), skipping")
            continue

        # Sliding window classification
        fbcca_labels = []
        mi_labels = []
        fusion_labels = []
        fusion_confs = []
        fbcca_scores = []
        mi_scores = []

        for win_start in range(0, len(trial_data) - window_samples + 1, stride_samples):
            window = trial_data[win_start:win_start + window_samples]

            # FBCCA
            fbcca_label, fbcca_conf = cca_engine.classify(window)
            fbcca_labels.append(fbcca_label)
            fbcca_scores.append(fbcca_conf)

            # MI
            if mi_engine.available:
                mi_label, mi_conf = mi_engine.classify(window)
            else:
                mi_label, mi_conf = "unknown", 0.0
            mi_labels.append(mi_label)
            mi_scores.append(mi_conf)

            # Fusion
            result = fusion.classify(window, timestamp=0.0, trial_id=trial_idx + 1)
            fusion_labels.append(result.label)
            fusion_confs.append(result.confidence)

        # Majority vote for this trial
        from collections import Counter
        fbcca_majority = Counter(fbcca_labels).most_common(1)[0][0] if fbcca_labels else "unknown"
        mi_majority = Counter(mi_labels).most_common(1)[0][0] if mi_labels else "unknown"
        fusion_majority = Counter(fusion_labels).most_common(1)[0][0] if fusion_labels else "unknown"

        avg_conf = np.mean(fusion_confs) if fusion_confs else 0.0
        avg_fbcca = np.mean(fbcca_scores) if fbcca_scores else 0.0
        avg_mi = np.mean(mi_scores) if mi_scores else 0.0

        trial_results.append({
            "condition": condition,
            "fbcca_majority": fbcca_majority,
            "mi_majority": mi_majority,
            "fusion_majority": fusion_majority,
            "fbcca_correct": fbcca_majority == condition,
            "mi_correct": mi_majority == condition,
            "fusion_correct": fusion_majority == condition,
            "n_windows": len(fbcca_labels),
            "fbcca_labels": fbcca_labels,
            "mi_labels": mi_labels,
            "fusion_labels": fusion_labels,
            "avg_fbcca_score": avg_fbcca,
            "avg_mi_score": avg_mi,
            "avg_fusion_conf": avg_conf,
        })

        # Per-window detail line
        fbcca_mark = "V" if fbcca_majority == condition else "X"
        mi_mark = "V" if mi_majority == condition else "X"
        fusion_mark = "V" if fusion_majority == condition else "X"
        print(f"{trial_idx+1:>5} {condition:>6} {fbcca_majority:>5}{fbcca_mark} "
              f"{mi_majority:>5}{mi_mark} {fusion_majority:>6}{fusion_mark} "
              f"{avg_conf:>6.3f} {avg_fbcca:>9.4f} {avg_mi:>9.4f}")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SUMMARY REPORT")
    print(f"{'='*70}")

    n_total = len(trial_results)
    if n_total == 0:
        print("No trials classified.")
        return

    fbcca_correct = sum(1 for r in trial_results if r["fbcca_correct"])
    mi_correct = sum(1 for r in trial_results if r["mi_correct"])
    fusion_correct = sum(1 for r in trial_results if r["fusion_correct"])

    print(f"\nTrial-level accuracy (majority vote):")
    print(f"  FBCCA:  {fbcca_correct}/{n_total} = {fbcca_correct/n_total:.1%}")
    print(f"  MI:     {mi_correct}/{n_total} = {mi_correct/n_total:.1%}")
    print(f"  Fusion: {fusion_correct}/{n_total} = {fusion_correct/n_total:.1%}")

    # Per-condition breakdown
    for cond in ["left", "right"]:
        cond_results = [r for r in trial_results if r["condition"] == cond]
        if not cond_results:
            continue
        n = len(cond_results)
        fbcca_c = sum(1 for r in cond_results if r["fbcca_correct"])
        mi_c = sum(1 for r in cond_results if r["mi_correct"])
        fusion_c = sum(1 for r in cond_results if r["fusion_correct"])
        print(f"\n  {cond.upper()} trials ({n}):")
        print(f"    FBCCA:  {fbcca_c}/{n} = {fbcca_c/n:.1%}")
        print(f"    MI:     {mi_c}/{n} = {mi_c/n:.1%}")
        print(f"    Fusion: {fusion_c}/{n} = {fusion_c/n:.1%}")

    # Window-level accuracy
    all_fbcca = []
    all_mi = []
    all_fusion = []
    all_conditions = []
    for r in trial_results:
        all_fbcca.extend(r["fbcca_labels"])
        all_mi.extend(r["mi_labels"])
        all_fusion.extend(r["fusion_labels"])
        all_conditions.extend([r["condition"]] * r["n_windows"])

    n_windows = len(all_fbcca)
    fbcca_win_correct = sum(1 for lbl, cond in zip(all_fbcca, all_conditions) if lbl == cond)
    mi_win_correct = sum(1 for lbl, cond in zip(all_mi, all_conditions) if lbl == cond)
    fusion_win_correct = sum(1 for lbl, cond in zip(all_fusion, all_conditions) if lbl == cond)

    print(f"\nWindow-level accuracy ({n_windows} windows):")
    print(f"  FBCCA:  {fbcca_win_correct}/{n_windows} = {fbcca_win_correct/n_windows:.1%}")
    print(f"  MI:     {mi_win_correct}/{n_windows} = {mi_win_correct/n_windows:.1%}")
    print(f"  Fusion: {fusion_win_correct}/{n_windows} = {fusion_win_correct/n_windows:.1%}")

    # Confusion matrix for FBCCA
    print(f"\nFBCCA confusion matrix:")
    print(f"  {'':>10} {'Pred L':>8} {'Pred R':>8}")
    for true_cond in ["left", "right"]:
        cond_labels = [all_fbcca[i] for i in range(n_windows) if all_conditions[i] == true_cond]
        n_l = sum(1 for l in cond_labels if l == "left")
        n_r = sum(1 for l in cond_labels if l == "right")
        print(f"  True {true_cond:>5}: {n_l:>8} {n_r:>8}")

    # Confusion matrix for MI
    if mi_engine.available:
        print(f"\nMI confusion matrix:")
        print(f"  {'':>10} {'Pred L':>8} {'Pred R':>8} {'Pred ?':>8}")
        for true_cond in ["left", "right"]:
            cond_labels = [all_mi[i] for i in range(n_windows) if all_conditions[i] == true_cond]
            n_l = sum(1 for l in cond_labels if l == "left")
            n_r = sum(1 for l in cond_labels if l == "right")
            n_u = sum(1 for l in cond_labels if l not in ("left", "right"))
            print(f"  True {true_cond:>5}: {n_l:>8} {n_r:>8} {n_u:>8}")

    print(f"\n{'='*70}")
    print("Done.")


def replay_sim_rt(
    xdf_path: str,
    window_size_s: float = 1.5,
    stride_s: float = 0.25,
    mi_checkpoint: str = "",
    both_sides: bool = False,
    sample_rate: float = 250.0,
    marker_delay_s: float = 0.0,
) -> None:
    """Simulate real-time RingBuffer pipeline on XDF data.

    This diagnostic mode processes XDF data through the SAME RingBuffer +
    buffer.clear() + get_window() pipeline that the real-time classifier uses.
    This isolates whether accuracy loss is from buffer logic or LSL data delivery.

    Args:
        xdf_path: Path to XDF file
        window_size_s: Classification window size in seconds
        stride_s: Sliding window stride in seconds
        mi_checkpoint: Path to MI model checkpoint
        both_sides: Enable both-sides flicker mode
        sample_rate: Sample rate in Hz
        marker_delay_s: Simulated marker delay in seconds (0.0 = clear at exact marker,
                        1.5 = clear 1.5s after marker, simulating UDP→OpenBCI→LSL delay)
    """
    eeg_data, eeg_ts, marker_vals, marker_ts = load_xdf(xdf_path)

    # Config
    config = ClassifierConfig(
        window_size_s=window_size_s,
        stride_s=stride_s,
        mi_checkpoint_path=mi_checkpoint or ClassifierConfig.mi_checkpoint_path,
        ssvep_both_sides_mode=both_sides,
        sample_rate=sample_rate,
    )

    # Initialize engines
    cca_engine = CCAEngine(config)
    mi_engine = MIEngine(config)
    fusion = DecisionFusion(config, cca_engine, mi_engine)

    print(f"\n--- Simulated Real-Time Pipeline ---")
    print(f"FBCCA: {len(cca_engine._fb_sos)} filter-bank sub-bands")
    print(f"MI model available: {mi_engine.available}")
    if mi_engine.available:
        print(f"MI model: {type(mi_engine._model).__name__}")
    print(f"Window: {window_size_s}s ({int(window_size_s * sample_rate)} samples)")
    print(f"Stride: {stride_s}s")
    print(f"Marker delay: {marker_delay_s}s")

    # Build trial timeline from markers
    trials: list[dict] = []
    current_trial = None
    for i, val in enumerate(marker_vals):
        if val in _LEFT_START_MARKERS:
            current_trial = {"condition": "left", "start_ts": marker_ts[i],
                             "end_ts": None, "marker": val}
        elif val in _RIGHT_START_MARKERS:
            current_trial = {"condition": "right", "start_ts": marker_ts[i],
                             "end_ts": None, "marker": val}
        elif val in _ALL_END_MARKERS and current_trial is not None:
            current_trial["end_ts"] = marker_ts[i]
            trials.append(current_trial)
            current_trial = None

    print(f"\n--- Trial Timeline ---")
    print(f"Total trials: {len(trials)}")
    n_left = sum(1 for t in trials if t["condition"] == "left")
    n_right = sum(1 for t in trials if t["condition"] == "right")
    print(f"Left trials: {n_left}, Right trials: {n_right}")

    # Parameters
    window_samples = int(window_size_s * sample_rate)
    stride_samples = int(stride_s * sample_rate)
    chunk_size = stride_samples * 2  # Same as pull_eeg_chunk(max_samples=)

    # Results storage
    trial_results: list[dict] = []

    print(f"\n--- Per-Trial Classification (Simulated RT) ---")
    print(f"{'Trial':>5} {'Cond':>6} {'FBCCA':>6} {'MI':>6} {'Fusion':>7} {'Conf':>6} "
          f"{'FBCCA_sc':>9} {'MI_sc':>9} {'N_win':>6}")
    print("-" * 78)

    for trial_idx, trial in enumerate(trials):
        condition = trial["condition"]
        start_ts = trial["start_ts"]
        end_ts = trial["end_ts"]
        if end_ts is None:
            end_ts = eeg_ts[-1]

        # Find sample indices for this trial
        start_idx = np.searchsorted(eeg_ts, start_ts)
        end_idx = np.searchsorted(eeg_ts, end_ts)

        # Apply marker delay: clear happens delay seconds AFTER the marker
        # In real-time, marker arrives late, so buffer.clear() wipes early trial data
        clear_ts = start_ts + marker_delay_s
        clear_idx = np.searchsorted(eeg_ts, clear_ts)

        # Ensure clear_idx is within bounds
        clear_idx = max(start_idx, min(clear_idx, end_idx))

        # Create RingBuffer (same as real-time: 5 second buffer, 8 channels)
        buffer = RingBuffer(int(5 * sample_rate), 8)

        # Clear the buffer at the simulated marker arrival time
        buffer.clear()

        # Feed data from clear_idx to end_idx in chunks
        # Simulate the real-time loop: append chunks, classify when ready
        fbcca_labels = []
        mi_labels = []
        fusion_labels = []
        fusion_confs = []
        fbcca_scores = []
        mi_scores = []

        last_classify_idx = clear_idx  # Track when we last classified

        for chunk_start in range(clear_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)
            chunk_data = eeg_data[chunk_start:chunk_end]

            if len(chunk_data) == 0:
                continue

            # Append chunk to buffer (simulating LSL delivery)
            buffer.append(chunk_data)

            # Check if we should classify
            # In real-time: classify when buffer.available >= window_samples
            # AND we've advanced by stride_samples since last classification
            samples_since_last = buffer.available - (last_classify_idx - clear_idx)
            # More accurate: track position in data stream
            samples_since_last = chunk_end - last_classify_idx

            if buffer.available >= window_samples and samples_since_last >= stride_samples:
                window = buffer.get_window(window_samples)
                if window is not None:
                    # FBCCA
                    fbcca_label, fbcca_conf = cca_engine.classify(window)
                    fbcca_labels.append(fbcca_label)
                    fbcca_scores.append(fbcca_conf)

                    # MI
                    if mi_engine.available:
                        mi_label, mi_conf = mi_engine.classify(window)
                    else:
                        mi_label, mi_conf = "unknown", 0.0
                    mi_labels.append(mi_label)
                    mi_scores.append(mi_conf)

                    # Fusion
                    result = fusion.classify(window, timestamp=0.0, trial_id=trial_idx + 1)
                    fusion_labels.append(result.label)
                    fusion_confs.append(result.confidence)

                    last_classify_idx = chunk_end

        # Use TrialTracker for majority vote (same as real-time)
        tracker = TrialTracker()
        tracker._condition = condition
        tracker._trial_id = trial_idx + 1
        tracker._state = TrialState.ACTIVE
        for label, conf in zip(fusion_labels, fusion_confs):
            tracker.add_classification(label, conf)

        fbcca_majority, _ = tracker.get_cumulative_decision()
        # For FBCCA and MI, use simple Counter-based majority
        from collections import Counter
        fbcca_majority = Counter(fbcca_labels).most_common(1)[0][0] if fbcca_labels else "unknown"
        mi_majority = Counter(mi_labels).most_common(1)[0][0] if mi_labels else "unknown"
        fusion_majority, fusion_avg_conf = tracker.get_cumulative_decision()

        avg_fbcca = np.mean(fbcca_scores) if fbcca_scores else 0.0
        avg_mi = np.mean(mi_scores) if mi_scores else 0.0
        n_windows = len(fbcca_labels)

        trial_results.append({
            "condition": condition,
            "fbcca_majority": fbcca_majority,
            "mi_majority": mi_majority,
            "fusion_majority": fusion_majority,
            "fbcca_correct": fbcca_majority == condition,
            "mi_correct": mi_majority == condition,
            "fusion_correct": fusion_majority == condition,
            "n_windows": n_windows,
            "fbcca_labels": fbcca_labels,
            "mi_labels": mi_labels,
            "fusion_labels": fusion_labels,
            "avg_fbcca_score": avg_fbcca,
            "avg_mi_score": avg_mi,
            "avg_fusion_conf": fusion_avg_conf,
            "clear_idx": clear_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
        })

        # Per-trial detail line
        fbcca_mark = "V" if fbcca_majority == condition else "X"
        mi_mark = "V" if mi_majority == condition else "X"
        fusion_mark = "V" if fusion_majority == condition else "X"
        print(f"{trial_idx+1:>5} {condition:>6} {fbcca_majority:>5}{fbcca_mark} "
              f"{mi_majority:>5}{mi_mark} {fusion_majority:>6}{fusion_mark} "
              f"{fusion_avg_conf:>6.3f} {avg_fbcca:>9.4f} {avg_mi:>9.4f} {n_windows:>6}")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print(f"\n{'='*78}")
    print(f"SUMMARY REPORT (Simulated Real-Time Pipeline)")
    print(f"{'='*78}")

    n_total = len(trial_results)
    if n_total == 0:
        print("No trials classified.")
        return

    fbcca_correct = sum(1 for r in trial_results if r["fbcca_correct"])
    mi_correct = sum(1 for r in trial_results if r["mi_correct"])
    fusion_correct = sum(1 for r in trial_results if r["fusion_correct"])

    print(f"\nTrial-level accuracy (majority vote):")
    print(f"  FBCCA:  {fbcca_correct}/{n_total} = {fbcca_correct/n_total:.1%}")
    print(f"  MI:     {mi_correct}/{n_total} = {mi_correct/n_total:.1%}")
    print(f"  Fusion: {fusion_correct}/{n_total} = {fusion_correct/n_total:.1%}")

    # Per-condition breakdown
    for cond in ["left", "right"]:
        cond_results = [r for r in trial_results if r["condition"] == cond]
        if not cond_results:
            continue
        n = len(cond_results)
        fbcca_c = sum(1 for r in cond_results if r["fbcca_correct"])
        mi_c = sum(1 for r in cond_results if r["mi_correct"])
        fusion_c = sum(1 for r in cond_results if r["fusion_correct"])
        print(f"\n  {cond.upper()} trials ({n}):")
        print(f"    FBCCA:  {fbcca_c}/{n} = {fbcca_c/n:.1%}")
        print(f"    MI:     {mi_c}/{n} = {mi_c/n:.1%}")
        print(f"    Fusion: {fusion_c}/{n} = {fusion_c/n:.1%}")

    # Window-level accuracy
    all_fbcca = []
    all_mi = []
    all_fusion = []
    all_conditions = []
    for r in trial_results:
        all_fbcca.extend(r["fbcca_labels"])
        all_mi.extend(r["mi_labels"])
        all_fusion.extend(r["fusion_labels"])
        all_conditions.extend([r["condition"]] * r["n_windows"])

    n_windows = len(all_fbcca)
    if n_windows > 0:
        fbcca_win_correct = sum(1 for lbl, cond in zip(all_fbcca, all_conditions) if lbl == cond)
        mi_win_correct = sum(1 for lbl, cond in zip(all_mi, all_conditions) if lbl == cond)
        fusion_win_correct = sum(1 for lbl, cond in zip(all_fusion, all_conditions) if lbl == cond)

        print(f"\nWindow-level accuracy ({n_windows} windows):")
        print(f"  FBCCA:  {fbcca_win_correct}/{n_windows} = {fbcca_win_correct/n_windows:.1%}")
        print(f"  MI:     {mi_win_correct}/{n_windows} = {mi_win_correct/n_windows:.1%}")
        print(f"  Fusion: {fusion_win_correct}/{n_windows} = {fusion_win_correct/n_windows:.1%}")

    # Confusion matrix for FBCCA
    print(f"\nFBCCA confusion matrix:")
    print(f"  {'':>10} {'Pred L':>8} {'Pred R':>8}")
    for true_cond in ["left", "right"]:
        cond_labels = [all_fbcca[i] for i in range(n_windows) if all_conditions[i] == true_cond]
        n_l = sum(1 for l in cond_labels if l == "left")
        n_r = sum(1 for l in cond_labels if l == "right")
        print(f"  True {true_cond:>5}: {n_l:>8} {n_r:>8}")

    # Confusion matrix for MI
    if mi_engine.available:
        print(f"\nMI confusion matrix:")
        print(f"  {'':>10} {'Pred L':>8} {'Pred R':>8} {'Pred ?':>8}")
        for true_cond in ["left", "right"]:
            cond_labels = [all_mi[i] for i in range(n_windows) if all_conditions[i] == true_cond]
            n_l = sum(1 for l in cond_labels if l == "left")
            n_r = sum(1 for l in cond_labels if l == "right")
            n_u = sum(1 for l in cond_labels if l not in ("left", "right"))
            print(f"  True {true_cond:>5}: {n_l:>8} {n_r:>8} {n_u:>8}")

    # Diagnostic info: show data loss from marker delay
    if marker_delay_s > 0:
        print(f"\n--- Marker Delay Impact ---")
        print(f"Marker delay: {marker_delay_s}s")
        for r in trial_results[:5]:  # Show first 5 trials
            samples_lost = r["clear_idx"] - r["start_idx"]
            time_lost = samples_lost / sample_rate
            print(f"  Trial {r.get('condition', '?')}: lost {time_lost:.2f}s ({samples_lost} samples) of early trial data")

    print(f"\n{'='*78}")
    print("Done.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline XDF replay classifier")
    parser.add_argument("xdf_path", nargs="?", default=None,
                        help="Path to XDF file (default: auto-find in datasets)")
    parser.add_argument("--window", type=float, default=1.5,
                        help="Classification window size in seconds (default: 1.5)")
    parser.add_argument("--stride", type=float, default=0.25,
                        help="Sliding window stride in seconds (default: 0.25)")
    parser.add_argument("--mi-checkpoint", type=str, default="",
                        help="Path to MI model checkpoint (default: use config default)")
    parser.add_argument("--no-mi", action="store_true",
                        help="Disable MI model (FBCCA only)")
    parser.add_argument("--both-sides", action="store_true",
                        help="Enable both-sides flicker mode")
    parser.add_argument("--sim-rt", action="store_true",
                        help="Simulate real-time RingBuffer pipeline on XDF data (diagnostic)")
    parser.add_argument("--sim-marker-delay", type=float, default=0.0,
                        help="Simulated marker delay in seconds (default: 0.0, try 1.5 for real-time behavior)")
    args = parser.parse_args()

    # Default XDF path
    if args.xdf_path is None:
        default_path = Path(r"D:\CSDIY\EEG\datasets\customs\P001\S001\binary\mi_ssvep_rt\0424\run-1.xdf")
        if default_path.exists():
            args.xdf_path = str(default_path)
        else:
            sys.exit(f"ERROR: No XDF path specified and default not found: {default_path}")

    mi_ckpt = "__NO_MI__" if args.no_mi else args.mi_checkpoint

    if args.sim_rt:
        replay_sim_rt(
            xdf_path=args.xdf_path,
            window_size_s=args.window,
            stride_s=args.stride,
            mi_checkpoint=mi_ckpt,
            both_sides=args.both_sides,
            marker_delay_s=args.sim_marker_delay,
        )
    else:
        replay(
            xdf_path=args.xdf_path,
            window_size_s=args.window,
            stride_s=args.stride,
            mi_checkpoint=mi_ckpt,
            both_sides=args.both_sides,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
