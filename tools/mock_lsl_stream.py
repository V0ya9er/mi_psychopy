#!/usr/bin/env python
"""Mock LSL stream simulator for MI-SSVEP classifier testing.

Publishes two LSL streams:
  1. EEG stream  — 8 channels (C3,Cz,C4,P3,Pz,P4,O1,O2), 250 Hz, float32
  2. Marker stream — 1 channel, irregular rate, float32

Simulates the mi_ssvep trial flow:
  marker 5 (fixation_on) → 1.5s → marker 61/62 (ssvep_left/right) → 4.5s
  → marker 29 (task_off) → 3.0s ITI

During SSVEP phase, injects sinusoidal signal at the target frequency on O1/O2.
"""

import argparse
import math
import random
import time
import sys

try:
    import pylsl
except ImportError:
    print("ERROR: pylsl not installed. Run: pip install pylsl")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNELS = ["C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
N_CHANNELS = len(CHANNELS)
CH_O1 = 6  # O1 index
CH_O2 = 7  # O2 index

MARKER_FIXATION_ON = 5
MARKER_SSVEP_LEFT = 61
MARKER_SSVEP_RIGHT = 62
MARKER_TASK_OFF = 29

SSVEP_FREQ_LEFT = 10.0   # Hz
SSVEP_FREQ_RIGHT = 15.0  # Hz

# Timing (seconds) — matches config_default.yaml mi_ssvep paradigm
DURATION_FIXATION = 1.5
DURATION_SSVEP = 4.5
DURATION_ITI = 3.0

SSVEP_AMPLITUDE = 50.0  # µV-scale amplitude for injected sinusoid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def push_marker(outlet: pylsl.StreamOutlet, value: float) -> None:
    """Push a single-sample marker and print it."""
    outlet.push_sample([value])
    print(f"  [Marker] {value:.0f}")


def run(n_trials: int, sample_rate: float, eeg_name: str, marker_name: str) -> None:
    # --- Create EEG stream info ---
    eeg_info = pylsl.StreamInfo(
        name=eeg_name,
        type="EEG",
        channel_count=N_CHANNELS,
        nominal_srate=sample_rate,
        channel_format="float32",
        source_id="mock_eeg_001",
    )
    # Add channel labels
    chns = eeg_info.desc().append_child("channels")
    for ch in CHANNELS:
        chns.append_child("channel").append_child_value("label", ch)

    eeg_outlet = pylsl.StreamOutlet(eeg_info)

    # --- Create Marker stream info ---
    marker_info = pylsl.StreamInfo(
        name=marker_name,
        type="Markers",
        channel_count=1,
        nominal_srate=0,  # irregular
        channel_format="float32",
        source_id="mock_marker_001",
    )
    marker_outlet = pylsl.StreamOutlet(marker_info)

    print(f"Publishing EEG stream:   name={eeg_name!r}, type='EEG', "
          f"ch={N_CHANNELS}, rate={sample_rate} Hz")
    print(f"Publishing Marker stream: name={marker_name!r}, type='Markers'")
    print(f"Trials: {n_trials}")
    print("=" * 60)

    sample_interval = 1.0 / sample_rate
    sample_idx = 0  # global sample counter for sinusoid phase

    for trial_i in range(1, n_trials + 1):
        condition = random.choice(["left", "right"])
        ssvep_freq = SSVEP_FREQ_LEFT if condition == "left" else SSVEP_FREQ_RIGHT
        ssvep_marker = MARKER_SSVEP_LEFT if condition == "left" else MARKER_SSVEP_RIGHT

        print(f"\nTrial {trial_i}/{n_trials} — condition: {condition} "
              f"(SSVEP {ssvep_freq} Hz)")

        # ---- Phase 1: Fixation ----
        push_marker(marker_outlet, MARKER_FIXATION_ON)
        fixation_samples = int(DURATION_FIXATION * sample_rate)
        for _ in range(fixation_samples):
            sample = [0.0] * N_CHANNELS
            eeg_outlet.push_sample(sample)
            sample_idx += 1
            time.sleep(sample_interval)

        # ---- Phase 2: SSVEP (MI+SSVEP task) ----
        push_marker(marker_outlet, ssvep_marker)
        ssvep_samples = int(DURATION_SSVEP * sample_rate)
        for _ in range(ssvep_samples):
            sample = [0.0] * N_CHANNELS
            t = sample_idx / sample_rate
            sig = SSVEP_AMPLITUDE * math.sin(2.0 * math.pi * ssvep_freq * t)
            sample[CH_O1] = sig
            sample[CH_O2] = sig
            eeg_outlet.push_sample(sample)
            sample_idx += 1
            time.sleep(sample_interval)

        # ---- Phase 3: Task off ----
        push_marker(marker_outlet, MARKER_TASK_OFF)
        iti_samples = int(DURATION_ITI * sample_rate)
        for _ in range(iti_samples):
            sample = [0.0] * N_CHANNELS
            eeg_outlet.push_sample(sample)
            sample_idx += 1
            time.sleep(sample_interval)

    print("\n" + "=" * 60)
    print(f"All {n_trials} trial(s) completed. Streams remain open — "
          "press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down mock LSL streams.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mock LSL stream simulator for MI-SSVEP classifier testing"
    )
    parser.add_argument(
        "--n-trials", type=int, default=10,
        help="Number of trials to simulate (default: 10)",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=250.0,
        help="EEG sample rate in Hz (default: 250)",
    )
    parser.add_argument(
        "--eeg-stream-name", type=str, default="mock_eeg",
        help="LSL stream name for EEG (default: 'mock_eeg')",
    )
    parser.add_argument(
        "--marker-stream-name", type=str, default="mock_markers",
        help="LSL stream name for Markers (default: 'mock_markers')",
    )
    args = parser.parse_args()

    try:
        run(
            n_trials=args.n_trials,
            sample_rate=args.sample_rate,
            eeg_name=args.eeg_stream_name,
            marker_name=args.marker_stream_name,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
