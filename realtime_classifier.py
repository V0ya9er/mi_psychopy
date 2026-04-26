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
    _trial_win_count = 0   # per-trial window counter (resets on trial start)

    # LSL Diagnostics (optional)
    diag = None
    if config.enable_diag:
        from rt_classifier.lsl_diag import LSLDiagnostics
        diag = LSLDiagnostics(sample_rate, save_dir=config.diag_output_dir)
        logger.info("LSL diagnostics ENABLED — windows will be saved to %s", config.diag_output_dir)

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
                if diag is not None:
                    diag.on_eeg_chunk(samples, timestamps)

            # 3b. Pull markers → update TrialTracker
            marker_result = subscriber.pull_marker(timeout=0.0)
            if marker_result is not None:
                marker_ts, marker_val = marker_result
                tracker.on_marker(marker_val)

                if marker_val in (MARKER_SSVEP_LEFT, MARKER_SSVEP_RIGHT, MARKER_RT_SSVEP_LEFT, MARKER_RT_SSVEP_RIGHT):  # Trial start
                    buffer.clear()
                    _trial_win_count = 0
                    if diag is not None:
                        diag.on_trial_start(tracker.trial_id)
                    logger.info(f"Trial {tracker.trial_id} started (marker={marker_val})")
                elif marker_val in (MARKER_TASK_OFF, MARKER_RT_TASK_OFF):  # Trial end
                    # Publish cumulative decision
                    final_label, final_conf = tracker.get_cumulative_decision()
                    trial_duration_s = time.time() - tracker.start_time if tracker.start_time > 0 else 0.0
                    expected_windows = max(1, int((trial_duration_s - config.window_size_s) / config.stride_s) + 1) if trial_duration_s > config.window_size_s else 0
                    if final_label != "unknown":
                        result = ClassificationResult(
                            label=final_label, confidence=final_conf,
                            cca_score=0.0, mi_score=0.0,
                            timestamp=time.time(), trial_id=tracker.trial_id,
                            cca_label=final_label, mi_label="cumulative",
                        )
                        publisher.publish(result)
                        feedback.push_result(result)
                        logger.info(
                            f"Trial {tracker.trial_id} final: {final_label} (conf={final_conf:.3f}) "
                            f"[windows={tracker.window_count} expected~{expected_windows} "
                            f"duration={trial_duration_s:.1f}s "
                            f"buffer={buffer.available}/{window_samples}]"
                        )
                    # Log trial diagnostics
                    if diag is not None:
                        trial_diag = diag.on_trial_end(tracker.trial_id)
                        logger.info(
                            f"Trial {tracker.trial_id} LSL diag: "
                            f"gaps={trial_diag['gap_count']} max_gap={trial_diag['max_gap_s']*1000:.1f}ms "
                            f"samples={trial_diag['total_samples_received']} "
                            f"loss_rate={trial_diag['sample_loss_rate']:.1%}"
                        )
                    tracker.reset()

            # 3c. If trial ACTIVE and buffer has enough data → sliding window classification
            now = time.time()
            _max_ok = (config.max_windows_per_trial <= 0
                       or _trial_win_count < config.max_windows_per_trial)
            if (tracker.state == TrialState.ACTIVE
                    and buffer.available >= window_samples
                    and now - last_classify_time >= config.stride_s
                    and _max_ok):
                window = buffer.get_window(window_samples)
                if window is not None:
                    result = fusion.classify(window, timestamp=now, trial_id=tracker.trial_id)
                    _trial_win_count += 1

                    # Save window for diagnostics
                    if diag is not None:
                        diag.on_window_classified(window, result, tracker.trial_id, _trial_win_count)

                    # Skip initial lock-in windows from majority vote
                    if _trial_win_count > config.skip_initial_windows:
                        tracker.add_classification(result.label, result.confidence)

                    # Push intermediate result (publish even skipped windows for feedback)
                    publisher.publish(result)
                    feedback.push_result(result)

                    # Log per-window classification for real-time console output
                    logger.info(
                        f"[Trial {tracker.trial_id} w{tracker.window_count}] "
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
        if diag is not None:
            summary = diag.get_summary()
            logger.info("=== LSL Diagnostics Summary ===")
            logger.info(f"Total samples received: {summary['total_samples_received']}")
            logger.info(f"Total gaps: {summary['total_gaps']}")
            logger.info(f"Max gap: {summary['max_gap_s']*1000:.1f}ms")
            logger.info(f"Avg jitter: {summary['avg_jitter_ms']:.2f}ms")
            logger.info(f"Max jitter: {summary['max_jitter_ms']:.2f}ms")
            logger.info(f"Estimated sample loss: {summary['sample_loss_rate']:.1%}")
        logger.info("Classifier stopped.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time SSVEP+MI classifier")
    parser.add_argument("--mi-checkpoint", type=str, default="", help="Path to MI model checkpoint (.pth)")
    parser.add_argument("--no-mi", action="store_true", help="Disable MI classification (FBCCA only)")
    parser.add_argument("--timeout", type=float, default=30.0, help="LSL connection timeout (seconds)")
    parser.add_argument("--window-size-s", type=float, default=1.0, help="Classification window size (seconds)")
    parser.add_argument("--stride-s", type=float, default=0.25, help="Sliding window stride (seconds)")
    parser.add_argument("--confidence-threshold", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--both-sides", action="store_true", help="Enable both-sides flicker mode")
    parser.add_argument("--max-windows", type=int, default=0, help="Max windows per trial (0=unlimited, default 13 for 4.5s flicker)")
    parser.add_argument("--skip-initial", type=int, default=0, help="Skip first N windows per trial (SSVEP lock-in, default 2)")
    parser.add_argument("--output-dir", type=str, default="logs", help="Decision log output directory")
    parser.add_argument("--run-name", type=str, default="", help="Run name for logging")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--diag", action="store_true",
                        help="Enable LSL diagnostics (gap detection, jitter stats, window saving)")
    parser.add_argument("--diag-dir", type=str, default="diag",
                        help="Directory for diagnostic output files")
    args = parser.parse_args()

    # --no-mi overrides any --mi-checkpoint to force FBCCA-only mode
    mi_checkpoint = "" if args.no_mi else args.mi_checkpoint

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
        mi_checkpoint_path=mi_checkpoint,
        lsl_connect_timeout_s=args.timeout,
        window_size_s=args.window_size_s,
        stride_s=args.stride_s,
        confidence_threshold=args.confidence_threshold,
        max_windows_per_trial=args.max_windows,
        skip_initial_windows=args.skip_initial,
        log_output_dir=args.output_dir,
        run_name=run_name,
        ssvep_both_sides_mode=args.both_sides,
        enable_diag=args.diag,
        diag_output_dir=args.diag_dir,
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    run_classifier(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
