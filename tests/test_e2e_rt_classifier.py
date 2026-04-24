#!/usr/bin/env python
"""End-to-end integration tests for the real-time SSVEP+MI classifier.

Run from project root: python tests/test_e2e_rt_classifier.py
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on sys.path so rt_classifier/ is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


class TestModuleImports(unittest.TestCase):
    """Test that all rt_classifier modules can be imported."""

    def test_config_imports(self):
        from rt_classifier.config import ClassifierConfig, ClassificationResult, TrialState
        self.assertIsNotNone(ClassifierConfig)
        self.assertIsNotNone(ClassificationResult)
        self.assertIsNotNone(TrialState)

    def test_buffer_import(self):
        from rt_classifier.buffer import RingBuffer
        self.assertIsNotNone(RingBuffer)

    def test_cca_engine_import(self):
        from rt_classifier.cca_engine import CCAEngine
        self.assertIsNotNone(CCAEngine)

    def test_mi_engine_import(self):
        from rt_classifier.mi_engine import MIEngine
        self.assertIsNotNone(MIEngine)

    def test_feedback_import(self):
        from rt_classifier.feedback import FeedbackSink, LSLFeedbackSink, AudioFeedbackSink, CompositeFeedbackSink
        self.assertIsNotNone(FeedbackSink)
        self.assertIsNotNone(AudioFeedbackSink)

    def test_decision_import(self):
        from rt_classifier.decision import DecisionFusion, TrialTracker
        self.assertIsNotNone(DecisionFusion)
        self.assertIsNotNone(TrialTracker)

    def test_lsl_io_import(self):
        from rt_classifier.lsl_io import LSLSubscriber, LSLPublisher
        self.assertIsNotNone(LSLSubscriber)

    def test_decision_log_import(self):
        from rt_classifier.decision_log import DecisionLogger, CSVFeedbackSink
        self.assertIsNotNone(DecisionLogger)


class TestRingBuffer(unittest.TestCase):
    """Test RingBuffer append/read/wraparound."""

    def test_basic_append_read(self):
        from rt_classifier.buffer import RingBuffer
        buf = RingBuffer(100, 8)
        data = np.random.randn(50, 8)
        buf.append(data)
        self.assertEqual(buf.available, 50)
        w = buf.get_window(30)
        assert w is not None
        self.assertEqual(w.shape, (30, 8))
        np.testing.assert_array_equal(w, data[20:50])

    def test_wraparound(self):
        from rt_classifier.buffer import RingBuffer
        buf = RingBuffer(100, 8)
        for i in range(15):
            buf.append(np.full((10, 8), i, dtype=np.float64))
        self.assertEqual(buf.available, 100)
        w = buf.get_window(10)
        self.assertTrue(np.all(w == 14))
        buf.clear()
        self.assertEqual(buf.available, 0)
        self.assertIsNone(buf.get_window(10))


class TestCCAEngine(unittest.TestCase):
    """Test CCA-based SSVEP frequency detection."""

    def test_10hz_classified_as_left(self):
        from rt_classifier.cca_engine import CCAEngine
        from rt_classifier.config import ClassifierConfig
        engine = CCAEngine(ClassifierConfig())
        sr = 250
        t = np.arange(250) / sr
        signal = np.column_stack([np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(250) for _ in range(8)])
        label, conf = engine.classify(signal)
        self.assertEqual(label, "left")
        self.assertGreater(conf, 0.5)

    def test_15hz_classified_as_right(self):
        from rt_classifier.cca_engine import CCAEngine
        from rt_classifier.config import ClassifierConfig
        engine = CCAEngine(ClassifierConfig())
        sr = 250
        t = np.arange(250) / sr
        signal = np.column_stack([np.sin(2 * np.pi * 15 * t) + 0.1 * np.random.randn(250) for _ in range(8)])
        label, conf = engine.classify(signal)
        self.assertEqual(label, "right")
        self.assertGreater(conf, 0.5)


class TestMIEngine(unittest.TestCase):
    """Test MI engine graceful degradation and preprocessing."""

    def test_no_checkpoint_graceful(self):
        from rt_classifier.mi_engine import MIEngine
        from rt_classifier.config import ClassifierConfig
        engine = MIEngine(ClassifierConfig(mi_checkpoint_path="/nonexistent/path.pth"))
        self.assertFalse(engine.available)
        label, conf = engine.classify(np.random.randn(250, 8))
        self.assertEqual(label, "unknown")
        self.assertEqual(conf, 0.0)

    def test_zscore_preprocessing(self):
        from rt_classifier.mi_engine import _zscore
        raw = np.random.randn(250, 8) * 100
        zscored = _zscore(raw)
        self.assertLess(abs(zscored.mean()), 0.1)


class TestDecisionFusion(unittest.TestCase):
    """Test CCA+MI decision fusion."""

    def test_cca_only_fusion(self):
        from rt_classifier.decision import DecisionFusion
        from rt_classifier.cca_engine import CCAEngine
        from rt_classifier.mi_engine import MIEngine
        from rt_classifier.config import ClassifierConfig
        cfg = ClassifierConfig()
        fusion = DecisionFusion(cfg, CCAEngine(cfg), MIEngine(cfg))
        sr = 250
        t = np.arange(250) / sr
        signal = np.column_stack([np.sin(2 * np.pi * 10 * t) + 0.05 * np.random.randn(250) for _ in range(8)])
        result = fusion.classify(signal)
        self.assertEqual(result.label, "left")
        self.assertEqual(result.cca_label, "left")
        self.assertEqual(result.mi_label, "unknown")


class TestTrialTracker(unittest.TestCase):
    """Test trial state transitions and marker handling."""

    def test_state_transitions(self):
        from rt_classifier.decision import TrialTracker
        from rt_classifier.config import TrialState
        tracker = TrialTracker()
        self.assertEqual(tracker.state, TrialState.WAITING)
        tracker.on_marker(61)
        self.assertEqual(tracker.state, TrialState.ACTIVE)
        self.assertEqual(tracker.condition, "left")
        self.assertEqual(tracker.trial_id, 1)
        tracker.on_marker(29)
        self.assertEqual(tracker.state, TrialState.ENDED)
        tracker.reset()
        self.assertEqual(tracker.state, TrialState.WAITING)


class TestDecisionLogger(unittest.TestCase):
    """Test CSV file creation and content."""

    def test_csv_creation(self):
        from rt_classifier.decision_log import DecisionLogger
        from rt_classifier.config import ClassificationResult
        with tempfile.TemporaryDirectory() as tmp:
            logger = DecisionLogger(Path(tmp), "test_run")
            result = ClassificationResult(
                label="left", confidence=0.85, cca_score=0.92,
                mi_score=0.0, timestamp=1.234, trial_id=1,
            )
            logger.log_decision(result, cca_label="left", mi_label="unknown", window_samples=250)
            logger.close()
            csv_path = Path(tmp) / "test_run" / "decisions.csv"
            self.assertTrue(csv_path.exists())
            content = csv_path.read_text()
            self.assertIn("left", content)


class TestFeedbackSink(unittest.TestCase):
    """Test FeedbackSink protocol — all sink classes instantiate."""

    def test_audio_feedback_sink(self):
        from rt_classifier.feedback import AudioFeedbackSink
        sink = AudioFeedbackSink()
        # Should accept push_result without error
        from rt_classifier.config import ClassificationResult
        result = ClassificationResult(
            label="left", confidence=0.9, cca_score=0.9,
            mi_score=0.0, timestamp=1.0, trial_id=1,
        )
        sink.push_result(result)
        sink.close()

    def test_composite_feedback_sink(self):
        from rt_classifier.feedback import CompositeFeedbackSink, AudioFeedbackSink
        composite = CompositeFeedbackSink()
        composite.add(AudioFeedbackSink())
        from rt_classifier.config import ClassificationResult
        result = ClassificationResult(
            label="right", confidence=0.8, cca_score=0.8,
            mi_score=0.0, timestamp=2.0, trial_id=2,
        )
        composite.push_result(result)
        composite.close()

    def test_lsl_feedback_sink_import(self):
        """LSLFeedbackSink is importable; instantiation requires pylsl."""
        from rt_classifier.feedback import LSLFeedbackSink
        self.assertIsNotNone(LSLFeedbackSink)
        # Attempt instantiation — skip if pylsl not installed
        try:
            import pylsl  # noqa: F401
            sink = LSLFeedbackSink()
            self.assertIsNotNone(sink)
            sink.close()
        except ImportError:
            self.skipTest("pylsl not installed")


class TestClassifierSmoke(unittest.TestCase):
    """Smoke test — classifier exits cleanly with timeout when no LSL streams."""

    def test_timeout_exit(self):
        proc = subprocess.run(
            [sys.executable, str(Path(_PROJECT_ROOT) / "realtime_classifier.py"), "--timeout", "3"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(proc.returncode, 0, f"Exit code != 0: {proc.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
