#!/usr/bin/env python
"""Real-time SSVEP+MI classifier — standalone process entry point.

Connects to LSL EEG + Marker streams, classifies using CCA (SSVEP) + MI
(pre-trained model), and publishes results via LSL classification_result stream.

Usage:
    python realtime_classifier.py --timeout 30 --window-size-s 1.0
    python realtime_classifier.py --mi-checkpoint path/to/model.pth
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from rt_classifier.config import ClassifierConfig, ClassificationResult, TrialState
from rt_classifier.buffer import RingBuffer
from rt_classifier.cca_engine import CCAEngine
from rt_classifier.mi_engine import MIEngine
from rt_classifier.lsl_io import LSLSubscriber, LSLPublisher
from rt_classifier.decision import DecisionFusion, TrialTracker, MARKER_SSVEP_LEFT, MARKER_SSVEP_RIGHT, MARKER_TASK_OFF, MARKER_RT_SSVEP_LEFT, MARKER_RT_SSVEP_RIGHT, MARKER_RT_TASK_OFF
from rt_classifier.feedback import CompositeFeedbackSink
from rt_classifier.decision_log import CSVFeedbackSink

logger = logging.getLogger("realtime_classifier")

# Global flag for graceful shutdown
_running = True


def _signal_handler(signum, frame):
    global _running
    logger.info("Received interrupt signal, shutting down...")
    _running = False


def run_classifier(config: ClassifierConfig) -> None:
    """Main classifier loop."""
    global _running
    _running = True

    # 1. Initialize components
    subscriber = LSLSubscriber(config)
    cca_engine = CCAEngine(config)
    mi_engine = MIEngine(config)
    fusion = DecisionFusion(config, cca_engine, mi_engine)
    tracker = TrialTracker()
    publisher = LSLPublisher(config)

    # Feedback: CSV only (LSL is handled by publisher separately)
    csv_sink = CSVFeedbackSink(Path(config.log_output_dir), config.run_name)
    feedback = CompositeFeedbackSink()
    feedback.add(csv_sink)

    # Buffer: ~5 seconds of data at 250Hz
    sample_rate = config.sample_rate
    window_samples = int(config.window_size_s * sample_rate)
    stride_samples = int(config.stride_s * sample_rate)
    buffer = RingBuffer(int(5 * sample_rate), 8)  # 5s buffer, 8 channels

    last_classify_time = 0.0

    # 2. Connect to LSL streams
    logger.info(f"Connecting to LSL streams (timeout={config.lsl_connect_timeout_s}s)...")
    if not subscriber.connect(timeout_s=config.lsl_connect_timeout_s):
        logger.error("Failed to connect to LSL streams. Exiting.")
        return

    eeg_info = subscriber.eeg_info
    if eeg_info:
        logger.info(f"EEG stream: {eeg_info.n_channels}ch @ {eeg_info.sample_rate}Hz")
        sample_rate = eeg_info.sample_rate
        window_samples = int(config.window_size_s * sample_rate)
        stride_samples = int(config.stride_s * sample_rate)

    logger.info(f"MI model available: {mi_engine.available}")
    logger.info(f"Classification window: {window_samples} samples ({config.window_size_s}s)")
    logger.info("Classifier running. Press Ctrl+C to stop.")

    # 3. Main loop
    try:
        while _running:
            # 3a. Pull EEG data → RingBuffer
            eeg_result = subscriber.pull_eeg_chunk(max_samples=stride_samples * 2)
            if eeg_result is not None:
                samples, timestamps = eeg_result
                buffer.append(samples)

            # 3b. Pull markers → update TrialTracker
            marker_result = subscriber.pull_marker(timeout=0.0)
            if marker_result is not None:
                marker_ts, marker_val = marker_result
                tracker.on_marker(marker_val)

                if marker_val in (MARKER_SSVEP_LEFT, MARKER_SSVEP_RIGHT, MARKER_RT_SSVEP_LEFT, MARKER_RT_SSVEP_RIGHT):  # Trial start
                    buffer.clear()
                    logger.info(f"Trial {tracker.trial_id} started (marker={marker_val})")
                elif marker_val in (MARKER_TASK_OFF, MARKER_RT_TASK_OFF):  # Trial end
                    # Publish cumulative decision
                    final_label, final_conf = tracker.get_cumulative_decision()
                    if final_label != "unknown":
                        result = ClassificationResult(
                            label=final_label, confidence=final_conf,
                            cca_score=0.0, mi_score=0.0,
                            timestamp=time.time(), trial_id=tracker.trial_id,
                            cca_label=final_label, mi_label="cumulative",
                        )
                        publisher.publish(result)
                        feedback.push_result(result)
                        logger.info(f"Trial {tracker.trial_id} final: {final_label} (conf={final_conf:.3f})")
                    tracker.reset()

            # 3c. If trial ACTIVE and buffer has enough data → sliding window classification
            now = time.time()
            if (tracker.state == TrialState.ACTIVE
                    and buffer.available >= window_samples
                    and now - last_classify_time >= config.stride_s):
                window = buffer.get_window(window_samples)
                if window is not None:
                    result = fusion.classify(window, timestamp=now, trial_id=tracker.trial_id)
                    tracker.add_classification(result.label, result.confidence)

                    # Push intermediate result
                    publisher.publish(result)
                    feedback.push_result(result)

                    # Log per-window classification for real-time console output
                    logger.info(
                        f"[Trial {tracker.trial_id}] window: "
                        f"label={result.label} conf={result.confidence:.3f} "
                        f"(cca={result.cca_label}/{result.cca_score:.3f} "
                        f"mi={result.mi_label}/{result.mi_score:.3f})"
                    )

                    last_classify_time = now

            # 3d. Short sleep to avoid busy-waiting
            time.sleep(0.005)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # 4. Cleanup
        logger.info("Shutting down classifier...")
        subscriber.disconnect()
        publisher.close()
        feedback.close()
        logger.info("Classifier stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time SSVEP+MI classifier")
    parser.add_argument("--mi-checkpoint", type=str, default="", help="Path to MI model checkpoint (.pth)")
    parser.add_argument("--timeout", type=float, default=30.0, help="LSL connection timeout (seconds)")
    parser.add_argument("--window-size-s", type=float, default=1.0, help="Classification window size (seconds)")
    parser.add_argument("--stride-s", type=float, default=0.25, help="Sliding window stride (seconds)")
    parser.add_argument("--confidence-threshold", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--both-sides", action="store_true", help="Enable both-sides flicker mode")
    parser.add_argument("--output-dir", type=str, default="logs", help="Decision log output directory")
    parser.add_argument("--run-name", type=str, default="", help="Run name for logging")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Suppress noisy pylsl logs
    logging.getLogger("pylsl").setLevel(logging.WARNING)

    run_name = args.run_name or f"rt_{time.strftime('%Y%m%d-%H%M%S')}"

    config = ClassifierConfig(
        mi_checkpoint_path=args.mi_checkpoint,
        lsl_connect_timeout_s=args.timeout,
        window_size_s=args.window_size_s,
        stride_s=args.stride_s,
        confidence_threshold=args.confidence_threshold,
        log_output_dir=args.output_dir,
        run_name=run_name,
        ssvep_both_sides_mode=args.both_sides,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    run_classifier(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
