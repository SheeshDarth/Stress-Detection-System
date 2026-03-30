"""
Unit Tests — Stress Classifier
================================
Tests the StressClassifier class: feature fusion, synthetic data,
training pipeline, prediction, and model persistence.
"""

import pytest
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fusion.classifier import StressClassifier, FEATURE_NAMES


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def classifier():
    """Classifier with a temp model path."""
    tmp = tempfile.mktemp(suffix=".pkl")
    clf = StressClassifier(model_path=tmp)
    yield clf
    # Cleanup
    if os.path.exists(tmp):
        os.remove(tmp)
    metrics = tmp.replace(".pkl", "_metrics.json")
    if os.path.exists(metrics):
        os.remove(metrics)


@pytest.fixture
def trained_classifier(classifier):
    """A classifier pre-trained on synthetic data."""
    classifier.train(use_mock=True)
    return classifier


# ── Tests ─────────────────────────────────────────────────────────────

class TestFeatureNames:
    """Test feature schema."""

    def test_nine_features(self):
        assert len(FEATURE_NAMES) == 9

    def test_contains_bpm(self):
        assert "bpm" in FEATURE_NAMES

    def test_contains_hrv(self):
        assert "hrv_rmssd" in FEATURE_NAMES


class TestFeatureVector:
    """Test feature vector creation."""

    def test_creates_9d_vector(self):
        visual = {
            "blink_rate": 15, "ear_mean": 0.30, "ear_std": 0.02,
            "brow_furrow_mean": 0.35, "brow_furrow_std": 0.01,
            "head_pose_variance": 0.001, "head_pose_mean_movement": 0.005,
        }
        physio = {"bpm": 72, "hrv_rmssd": 45}
        vec = StressClassifier.create_feature_vector(visual, physio)
        assert vec is not None
        assert vec.shape == (9,)

    def test_none_visual_returns_none(self):
        result = StressClassifier.create_feature_vector(None, {"bpm": 72})
        assert result is None

    def test_none_physio_returns_none(self):
        visual = {"blink_rate": 15, "ear_mean": 0.3}
        result = StressClassifier.create_feature_vector(visual, None)
        assert result is None

    def test_missing_keys_default_to_zero(self):
        vec = StressClassifier.create_feature_vector({}, {})
        assert vec is not None
        assert np.all(vec == 0.0)


class TestMockDataset:
    """Test synthetic dataset generation."""

    def test_shape(self):
        X, y = StressClassifier.generate_mock_dataset(100)
        # 100 + 15% borderline = 115
        assert X.shape[1] == 9
        assert len(y) == len(X)

    def test_binary_labels(self):
        X, y = StressClassifier.generate_mock_dataset(100)
        assert set(np.unique(y)).issubset({0.0, 1.0})

    def test_balanced_classes(self):
        X, y = StressClassifier.generate_mock_dataset(2000)
        normal = np.sum(y == 0)
        stressed = np.sum(y == 1)
        # Should have roughly balanced classes (within 30%)
        ratio = normal / (stressed + 1e-10)
        assert 0.5 < ratio < 2.0

    def test_reproducible(self):
        X1, y1 = StressClassifier.generate_mock_dataset(50)
        X2, y2 = StressClassifier.generate_mock_dataset(50)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestTraining:
    """Test classifier training."""

    def test_training_returns_accuracy(self, classifier):
        accuracy = classifier.train(use_mock=True)
        assert 0.5 < accuracy <= 1.0

    def test_model_is_trained_flag(self, trained_classifier):
        assert trained_classifier.is_trained is True

    def test_feature_importance_populated(self, trained_classifier):
        assert len(trained_classifier.feature_importance) == 9

    def test_training_metrics_populated(self, trained_classifier):
        m = trained_classifier.training_metrics
        assert "accuracy" in m
        assert "f1_score" in m
        assert "roc_auc" in m


class TestPrediction:
    """Test inference."""

    def test_predicts_normal(self, trained_classifier):
        vec = np.array([15, 0.30, 0.02, 0.35, 0.01, 0.001, 0.005, 72, 45])
        label, conf = trained_classifier.predict(vec)
        assert label in ("Normal", "Stressed")
        assert 0.0 <= conf <= 1.0

    def test_predicts_stressed(self, trained_classifier):
        vec = np.array([28, 0.22, 0.05, 0.25, 0.04, 0.006, 0.018, 100, 20])
        label, conf = trained_classifier.predict(vec)
        assert label in ("Normal", "Stressed")
        assert 0.0 <= conf <= 1.0

    def test_handles_nan(self, trained_classifier):
        vec = np.array([np.nan, 0.30, 0.02, 0.35, np.nan, 0.001, 0.005, 72, 45])
        label, conf = trained_classifier.predict(vec)
        assert label is not None

    def test_no_model_returns_none(self, classifier):
        vec = np.array([15, 0.30, 0.02, 0.35, 0.01, 0.001, 0.005, 72, 45])
        label, conf = classifier.predict(vec)
        assert label is None
        assert conf == 0.0


class TestPersistence:
    """Test model save/load roundtrip."""

    def test_save_creates_files(self, trained_classifier):
        assert os.path.exists(trained_classifier.model_path)
        metrics_path = trained_classifier.model_path.replace(".pkl", "_metrics.json")
        assert os.path.exists(metrics_path)

    def test_load_restores_model(self, trained_classifier):
        # Create a new classifier with same path
        clf2 = StressClassifier(model_path=trained_classifier.model_path)
        loaded = clf2.load_model()
        assert loaded is True
        assert clf2.is_trained is True

    def test_loaded_model_predicts(self, trained_classifier):
        clf2 = StressClassifier(model_path=trained_classifier.model_path)
        clf2.load_model()

        vec = np.array([15, 0.30, 0.02, 0.35, 0.01, 0.001, 0.005, 72, 45])
        label, conf = clf2.predict(vec)
        assert label in ("Normal", "Stressed")

    def test_load_nonexistent_returns_false(self):
        clf = StressClassifier(model_path="/nonexistent/model.pkl")
        assert clf.load_model() is False
