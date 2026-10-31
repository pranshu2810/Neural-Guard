"""
Anomaly Detection Module — Shrinkage-Regularized Mahalanobis Distance Detector.

Implements the multi-feature anomaly detection pipeline described in the research paper:
1. Fit baseline statistics from trusted reference model tensors.
2. Standardize feature vectors.
3. Compute regularized covariance matrix (Ledoit-Wolf shrinkage).
4. Calculate Mahalanobis distance for incoming tensors.
5. Calibrate thresholds on a held-out calibration set.
6. Classify tensors and aggregate to model-level scores.
7. Leave-one-feature-out contribution analysis for explainability.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from dataclasses import dataclass, field, asdict
import json
import logging

from app.engine.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single tensor."""
    tensor_name: str
    anomaly_score: float
    risk_classification: str  # normal, suspicious, high_anomaly, insufficient_evidence
    feature_values: Dict[str, Optional[float]]
    feature_deviations: Dict[str, Optional[float]]  # Z-scores from baseline
    feature_contributions: Dict[str, float]  # Leave-one-out contribution
    explanation: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class ModelAnomalyResult:
    """Aggregated anomaly result for an entire model."""
    model_score: float
    risk_classification: str
    total_tensors: int
    analyzed_tensors: int
    normal_count: int
    suspicious_count: int
    high_anomaly_count: int
    insufficient_count: int
    tensor_results: List[AnomalyResult]
    warnings: List[str] = field(default_factory=list)


class MahalanobisDetector:
    """
    Shrinkage-regularized Mahalanobis distance anomaly detector.
    
    Implementation Notes:
    - Uses Ledoit-Wolf shrinkage estimator for covariance regularization.
    - Features with None values are imputed with the baseline mean for that
      feature during scoring (not during fitting).
    - Thresholds default to 3.0 (suspicious) and 5.0 (high anomaly) in
      Mahalanobis distance units, configurable and overridable via calibration.
    """
    
    def __init__(
        self,
        suspicious_threshold: float = 3.0,
        high_anomaly_threshold: float = 5.0,
    ):
        self.suspicious_threshold = suspicious_threshold
        self.high_anomaly_threshold = high_anomaly_threshold
        
        # Fitted parameters
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.precision_matrix_: Optional[np.ndarray] = None
        self.covariance_matrix_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False
        self.feature_names_: List[str] = FEATURE_NAMES
        self.n_features_: int = len(FEATURE_NAMES)
        self.n_training_samples_: int = 0
    
    def _features_to_vector(self, features: Dict[str, Optional[float]]) -> Tuple[np.ndarray, List[bool]]:
        """
        Convert feature dict to numpy vector.
        Returns (vector, mask) where mask[i] is True if feature i was available.
        """
        vec = np.zeros(self.n_features_)
        mask = []
        for i, name in enumerate(self.feature_names_):
            val = features.get(name)
            if val is not None and np.isfinite(val):
                vec[i] = val
                mask.append(True)
            else:
                mask.append(False)
        return vec, mask
    
    def fit(self, training_features: List[Dict[str, Optional[float]]]) -> None:
        """
        Fit baseline statistics from trusted reference tensor features.
        
        Args:
            training_features: List of feature dicts from trusted tensors.
        """
        if len(training_features) < 2:
            raise ValueError("Need at least 2 training samples to fit baseline.")
        
        # Build feature matrix, imputing None with column means
        raw_matrix = []
        for features in training_features:
            vec, _ = self._features_to_vector(features)
            raw_matrix.append(vec)
        
        X = np.array(raw_matrix)
        self.n_training_samples_ = X.shape[0]
        
        # Handle columns with all-None by setting them to 0
        col_means = np.nanmean(X, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        
        # Replace NaN/zero values with column means
        for i in range(X.shape[1]):
            nan_mask = ~np.isfinite(X[:, i])
            X[nan_mask, i] = col_means[i]
        
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
        
        # Standardize
        X_std = (X - self.mean_) / self.std_
        
        # Ledoit-Wolf shrinkage covariance estimation
        try:
            lw = LedoitWolf()
            lw.fit(X_std)
            self.covariance_matrix_ = lw.covariance_
            self.precision_matrix_ = lw.precision_
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed, falling back to diagonal: {e}")
            # Fallback: diagonal covariance (features treated as independent)
            self.covariance_matrix_ = np.eye(self.n_features_)
            self.precision_matrix_ = np.eye(self.n_features_)
        
        self.is_fitted_ = True
        logger.info(
            f"Detector fitted on {self.n_training_samples_} samples, "
            f"{self.n_features_} features."
        )
    
    def calibrate(
        self,
        calibration_features: List[Dict[str, Optional[float]]],
        false_positive_rate: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calibrate anomaly thresholds using a held-out calibration dataset
        of known-clean tensors.
        
        Sets thresholds at percentiles corresponding to desired FPR.
        
        Args:
            calibration_features: Feature dicts from known-clean tensors.
            false_positive_rate: Desired false-positive rate for 'suspicious' threshold.
            
        Returns:
            Dict with calibrated thresholds.
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before calibration.")
        
        if len(calibration_features) < 5:
            logger.warning("Too few calibration samples. Using default thresholds.")
            return {
                "suspicious_threshold": self.suspicious_threshold,
                "high_anomaly_threshold": self.high_anomaly_threshold,
            }
        
        # Score all calibration samples
        scores = []
        for features in calibration_features:
            score = self._compute_mahalanobis(features)
            if score is not None:
                scores.append(score)
        
        if len(scores) < 5:
            logger.warning("Too few valid calibration scores. Using defaults.")
            return {
                "suspicious_threshold": self.suspicious_threshold,
                "high_anomaly_threshold": self.high_anomaly_threshold,
            }
        
        scores = np.array(scores)
        
        # Suspicious: (1 - FPR) percentile
        self.suspicious_threshold = float(np.percentile(scores, 100 * (1 - false_positive_rate)))
        # High anomaly: at a stricter threshold (FPR / 10)
        self.high_anomaly_threshold = float(np.percentile(scores, 100 * (1 - false_positive_rate / 10)))
        
        logger.info(
            f"Calibrated thresholds: suspicious={self.suspicious_threshold:.3f}, "
            f"high_anomaly={self.high_anomaly_threshold:.3f}"
        )
        
        return {
            "suspicious_threshold": self.suspicious_threshold,
            "high_anomaly_threshold": self.high_anomaly_threshold,
        }
    
    def _compute_mahalanobis(self, features: Dict[str, Optional[float]]) -> Optional[float]:
        """Compute Mahalanobis distance for a single feature vector."""
        if not self.is_fitted_:
            return None
        
        vec, mask = self._features_to_vector(features)
        
        # Impute missing features with baseline mean
        for i, available in enumerate(mask):
            if not available:
                vec[i] = self.mean_[i]
        
        # Standardize
        x_std = (vec - self.mean_) / self.std_
        
        try:
            dist = mahalanobis(x_std, np.zeros(self.n_features_), self.precision_matrix_)
            return float(dist)
        except Exception:
            # Fallback to Euclidean
            return float(np.sqrt(np.sum(x_std ** 2)))
    
    def _leave_one_out_contributions(
        self, features: Dict[str, Optional[float]], full_score: float
    ) -> Dict[str, float]:
        """
        Leave-one-feature-out contribution analysis.
        
        For each feature, compute the Mahalanobis distance without that feature
        and report the difference from the full score.
        """
        contributions = {}
        
        vec, mask = self._features_to_vector(features)
        for i, available in enumerate(mask):
            if not available:
                vec[i] = self.mean_[i]
        
        x_std = (vec - self.mean_) / self.std_
        
        for i, name in enumerate(self.feature_names_):
            # Zero out this feature's contribution
            x_modified = x_std.copy()
            x_modified[i] = 0.0
            
            try:
                reduced_score = mahalanobis(
                    x_modified, np.zeros(self.n_features_), self.precision_matrix_
                )
            except Exception:
                reduced_score = float(np.sqrt(np.sum(x_modified ** 2)))
            
            # Contribution = how much the score drops when this feature is removed
            contributions[name] = max(0.0, full_score - reduced_score)
        
        return contributions
    
    def _classify(self, score: float) -> str:
        """Classify a Mahalanobis distance score."""
        if score >= self.high_anomaly_threshold:
            return "high_anomaly"
        elif score >= self.suspicious_threshold:
            return "suspicious"
        else:
            return "normal"
    
    def _generate_explanation(
        self,
        tensor_name: str,
        score: float,
        classification: str,
        contributions: Dict[str, float],
        deviations: Dict[str, Optional[float]],
    ) -> str:
        """Generate human-readable explanation for the anomaly result."""
        from app.engine.features import FEATURE_DESCRIPTIONS
        
        if classification == "normal":
            return (
                f"Tensor '{tensor_name}' has a Mahalanobis distance of {score:.2f}, "
                f"which is within the normal range. No statistical anomalies detected."
            )
        
        # Find top contributing features
        sorted_contributions = sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        )
        top_features = sorted_contributions[:3]
        
        explanation_parts = [
            f"Tensor '{tensor_name}' has a Mahalanobis distance of {score:.2f}, "
            f"which exceeds the {'high anomaly' if classification == 'high_anomaly' else 'suspicious'} threshold."
        ]
        
        explanation_parts.append("\nKey contributing features:")
        for fname, contrib in top_features:
            if contrib > 0.01:
                desc = FEATURE_DESCRIPTIONS.get(fname, fname)
                dev = deviations.get(fname)
                dev_str = f" (deviation: {dev:+.2f}σ)" if dev is not None else ""
                explanation_parts.append(f"  • {desc}{dev_str}")
        
        explanation_parts.append(
            "\nIMPORTANT: This indicates a statistical anomaly, not definitive proof "
            "of malicious manipulation. Training procedures, quantization, and "
            "serialization can affect feature distributions."
        )
        
        return "\n".join(explanation_parts)
    
    def score_tensor(self, features: Dict[str, Optional[float]], tensor_name: str = "unknown") -> AnomalyResult:
        """
        Score a single tensor's features and produce an anomaly result.
        """
        warnings = []
        
        # Check if enough features are available
        available_count = sum(1 for v in features.values() if v is not None and np.isfinite(v))
        if available_count < 3:
            return AnomalyResult(
                tensor_name=tensor_name,
                anomaly_score=0.0,
                risk_classification="insufficient_evidence",
                feature_values=features,
                feature_deviations={},
                feature_contributions={},
                explanation=f"Tensor '{tensor_name}' has only {available_count} computable features. "
                           "Insufficient evidence for reliable anomaly detection.",
                warnings=[f"Only {available_count}/10 features available."],
            )
        
        if not self.is_fitted_:
            warnings.append("Detector not fitted — using raw feature magnitudes.")
            return AnomalyResult(
                tensor_name=tensor_name,
                anomaly_score=0.0,
                risk_classification="insufficient_evidence",
                feature_values=features,
                feature_deviations={},
                feature_contributions={},
                explanation="No baseline available. Cannot compute anomaly score.",
                warnings=warnings,
            )
        
        # Compute score
        score = self._compute_mahalanobis(features)
        if score is None or not np.isfinite(score):
            return AnomalyResult(
                tensor_name=tensor_name,
                anomaly_score=0.0,
                risk_classification="insufficient_evidence",
                feature_values=features,
                feature_deviations={},
                feature_contributions={},
                explanation="Could not compute Mahalanobis distance.",
                warnings=["Score computation failed."],
            )
        
        classification = self._classify(score)
        
        # Feature deviations (Z-scores from baseline)
        deviations = {}
        for i, name in enumerate(self.feature_names_):
            val = features.get(name)
            if val is not None and np.isfinite(val):
                deviations[name] = float((val - self.mean_[i]) / self.std_[i])
            else:
                deviations[name] = None
        
        # Leave-one-out contributions
        contributions = self._leave_one_out_contributions(features, score)
        
        # Explanation
        explanation = self._generate_explanation(
            tensor_name, score, classification, contributions, deviations
        )
        
        return AnomalyResult(
            tensor_name=tensor_name,
            anomaly_score=score,
            risk_classification=classification,
            feature_values=features,
            feature_deviations=deviations,
            feature_contributions=contributions,
            explanation=explanation,
            warnings=warnings,
        )
    
    def score_model(self, tensor_results: List[AnomalyResult]) -> ModelAnomalyResult:
        """
        Aggregate tensor-level results into a model-level assessment.
        
        Model score = max tensor score (conservative approach).
        Model classification = worst tensor classification.
        """
        if not tensor_results:
            return ModelAnomalyResult(
                model_score=0.0,
                risk_classification="insufficient_evidence",
                total_tensors=0,
                analyzed_tensors=0,
                normal_count=0,
                suspicious_count=0,
                high_anomaly_count=0,
                insufficient_count=0,
                tensor_results=[],
                warnings=["No tensors to analyze."],
            )
        
        scores = [r.anomaly_score for r in tensor_results if r.risk_classification != "insufficient_evidence"]
        
        normal_count = sum(1 for r in tensor_results if r.risk_classification == "normal")
        suspicious_count = sum(1 for r in tensor_results if r.risk_classification == "suspicious")
        high_anomaly_count = sum(1 for r in tensor_results if r.risk_classification == "high_anomaly")
        insufficient_count = sum(1 for r in tensor_results if r.risk_classification == "insufficient_evidence")
        
        if not scores:
            model_score = 0.0
            model_class = "insufficient_evidence"
        else:
            model_score = float(max(scores))
            if high_anomaly_count > 0:
                model_class = "high_anomaly"
            elif suspicious_count > 0:
                model_class = "suspicious"
            else:
                model_class = "normal"
        
        return ModelAnomalyResult(
            model_score=model_score,
            risk_classification=model_class,
            total_tensors=len(tensor_results),
            analyzed_tensors=len(scores),
            normal_count=normal_count,
            suspicious_count=suspicious_count,
            high_anomaly_count=high_anomaly_count,
            insufficient_count=insufficient_count,
            tensor_results=tensor_results,
        )
    
    def to_dict(self) -> Dict:
        """Serialize detector state for storage."""
        return {
            "suspicious_threshold": self.suspicious_threshold,
            "high_anomaly_threshold": self.high_anomaly_threshold,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "covariance": self.covariance_matrix_.tolist() if self.covariance_matrix_ is not None else None,
            "precision": self.precision_matrix_.tolist() if self.precision_matrix_ is not None else None,
            "is_fitted": self.is_fitted_,
            "n_training_samples": self.n_training_samples_,
            "feature_names": self.feature_names_,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "MahalanobisDetector":
        """Deserialize detector state."""
        det = cls(
            suspicious_threshold=data.get("suspicious_threshold", 3.0),
            high_anomaly_threshold=data.get("high_anomaly_threshold", 5.0),
        )
        if data.get("is_fitted"):
            det.mean_ = np.array(data["mean"])
            det.std_ = np.array(data["std"])
            det.covariance_matrix_ = np.array(data["covariance"])
            det.precision_matrix_ = np.array(data["precision"])
            det.is_fitted_ = True
            det.n_training_samples_ = data.get("n_training_samples", 0)
        return det
