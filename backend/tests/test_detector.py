"""
Tests for the anomaly detection module.
"""

import pytest
import numpy as np
from app.engine.detector import MahalanobisDetector, AnomalyResult, ModelAnomalyResult
from app.engine.features import FEATURE_NAMES


def make_clean_features(n=50, seed=42):
    """Generate synthetic clean feature vectors."""
    rng = np.random.RandomState(seed)
    features_list = []
    for i in range(n):
        feat = {}
        for name in FEATURE_NAMES:
            feat[name] = float(rng.randn() * 0.1 + 1.0)
        features_list.append(feat)
    return features_list


def make_anomalous_features(n=10, seed=99):
    """Generate synthetic anomalous feature vectors (shifted distribution)."""
    rng = np.random.RandomState(seed)
    features_list = []
    for i in range(n):
        feat = {}
        for name in FEATURE_NAMES:
            feat[name] = float(rng.randn() * 0.1 + 10.0)  # Far from baseline
        features_list.append(feat)
    return features_list


class TestMahalanobisDetector:
    def test_fit_basic(self):
        features = make_clean_features(20)
        detector = MahalanobisDetector()
        detector.fit(features)
        assert detector.is_fitted_
        assert detector.mean_ is not None
        assert detector.precision_matrix_ is not None

    def test_fit_too_few_samples(self):
        with pytest.raises(ValueError):
            detector = MahalanobisDetector()
            detector.fit([{"f01_entropy_deficit": 1.0}])

    def test_score_clean_normal(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        test_feat = make_clean_features(1, seed=100)[0]
        result = detector.score_tensor(test_feat, "test_clean")
        assert result.risk_classification in ("normal", "suspicious")

    def test_score_anomalous_high(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        test_feat = make_anomalous_features(1, seed=200)[0]
        result = detector.score_tensor(test_feat, "test_anomalous")
        assert result.anomaly_score > 0

    def test_calibrate(self):
        train = make_clean_features(30)
        calib = make_clean_features(10, seed=55)
        detector = MahalanobisDetector()
        detector.fit(train)
        thresholds = detector.calibrate(calib)
        assert "suspicious_threshold" in thresholds
        assert "high_anomaly_threshold" in thresholds

    def test_leave_one_out_contributions(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        test_feat = make_anomalous_features(1)[0]
        result = detector.score_tensor(test_feat, "test")
        assert len(result.feature_contributions) > 0

    def test_model_aggregation(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)

        tensor_results = []
        for i, feat in enumerate(make_clean_features(5, seed=80)):
            r = detector.score_tensor(feat, f"tensor_{i}")
            tensor_results.append(r)

        model_result = detector.score_model(tensor_results)
        assert model_result.total_tensors == 5
        assert model_result.model_score >= 0

    def test_serialization_roundtrip(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        
        state = detector.to_dict()
        restored = MahalanobisDetector.from_dict(state)
        
        assert restored.is_fitted_
        np.testing.assert_array_almost_equal(detector.mean_, restored.mean_)
        np.testing.assert_array_almost_equal(
            detector.precision_matrix_, restored.precision_matrix_
        )

    def test_insufficient_features(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        sparse_feat = {"f01_entropy_deficit": 1.0}  # Only 1 feature
        result = detector.score_tensor(sparse_feat, "sparse")
        assert result.risk_classification == "insufficient_evidence"

    def test_unfitted_detector(self):
        detector = MahalanobisDetector()
        feat = make_clean_features(1)[0]
        result = detector.score_tensor(feat, "test")
        assert result.risk_classification == "insufficient_evidence"

    def test_explanation_generated(self):
        train = make_clean_features(30)
        detector = MahalanobisDetector()
        detector.fit(train)
        test_feat = make_anomalous_features(1)[0]
        result = detector.score_tensor(test_feat, "test_tensor")
        assert len(result.explanation) > 0
        assert "test_tensor" in result.explanation
