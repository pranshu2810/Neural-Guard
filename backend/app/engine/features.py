"""
Ten Statistical Feature Extractors for Neural Network Weight Steganography Detection.

Each function takes a numpy array of weight values and returns a scalar feature value
or None if the tensor is too small / contains invalid data.

Implementation Assumptions (documented per the project requirements):
- Where the research paper does not specify exact formulas, we use standard
  statistical-test implementations from the literature (NIST SP 800-22, chi-square
  goodness of fit, etc.).
- Features operate on the low-order bits of the IEEE 754 mantissa by default.
- All features return None rather than arbitrary fill values when computation fails.
"""

import numpy as np
from scipy import stats
from scipy.special import rel_entr
from typing import Optional, Dict, Any
from app.engine.bit_utils import (
    float32_to_bits, float16_to_bits, extract_lsb, extract_bit_plane,
    get_mantissa_bits, filter_valid_floats, detect_fp16_origin
)

# Minimum elements required for reliable statistics
MIN_ELEMENTS = 256


def _prepare_bits(weights: np.ndarray, num_lsb: int = 1) -> Optional[np.ndarray]:
    """Convert weight tensor to LSB array for analysis."""
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask]
    if len(flat) < MIN_ELEMENTS:
        return None
    # Use float32 bit patterns
    fp32 = flat.astype(np.float32)
    bit_patterns = fp32.view(np.uint32)
    lsb = extract_lsb(bit_patterns, num_lsb)
    return lsb


# ─────────────────────────────────────────────────────────────────────
# Feature 1: G-Test / Entropy Deficit
# ─────────────────────────────────────────────────────────────────────

def f01_entropy_deficit(weights: np.ndarray) -> Optional[float]:
    """
    Measures departure of the LSB distribution from uniform using a G-test
    (log-likelihood ratio test).
    
    For clean weights, the LSB of the mantissa should be approximately
    uniformly distributed. Steganographic embedding may create a bias.
    
    Returns the G-statistic (higher = more anomalous).
    Returns None if tensor is too small.
    """
    lsb = _prepare_bits(weights, num_lsb=1)
    if lsb is None:
        return None
    
    n = len(lsb)
    observed_0 = np.sum(lsb == 0)
    observed_1 = n - observed_0
    expected = n / 2.0
    
    # Avoid log(0) — if all bits are the same, that itself is maximally anomalous
    if observed_0 == 0 or observed_1 == 0:
        return float(n)  # Maximum G-statistic for this size
    
    # G = 2 * sum(O_i * ln(O_i / E_i))
    g_stat = 2.0 * (
        observed_0 * np.log(observed_0 / expected) +
        observed_1 * np.log(observed_1 / expected)
    )
    return float(g_stat)


# ─────────────────────────────────────────────────────────────────────
# Feature 2: Pairs-of-Values Test
# ─────────────────────────────────────────────────────────────────────

def f02_pairs_of_values(weights: np.ndarray) -> Optional[float]:
    """
    Examines adjacent-value relationships in the mantissa LSBs.
    
    For each consecutive pair of LSBs (b_i, b_{i+1}), counts the four
    possible transitions: (0,0), (0,1), (1,0), (1,1).
    Under uniformity, each should appear with probability 1/4.
    
    Returns chi-squared statistic against uniform distribution.
    """
    lsb = _prepare_bits(weights, num_lsb=1)
    if lsb is None or len(lsb) < 4:
        return None
    
    n = len(lsb) - 1  # number of pairs
    if n < MIN_ELEMENTS:
        return None
    
    # Encode pairs as 2-bit values: b_i * 2 + b_{i+1}
    pairs = lsb[:-1] * 2 + lsb[1:]
    observed = np.bincount(pairs, minlength=4).astype(np.float64)
    expected = np.full(4, n / 4.0)
    
    chi2, _ = stats.chisquare(observed, expected)
    return float(chi2)


# ─────────────────────────────────────────────────────────────────────
# Feature 3: Bit-Plane Bias
# ─────────────────────────────────────────────────────────────────────

def f03_bit_plane_bias(weights: np.ndarray) -> Optional[float]:
    """
    Measures bias across multiple low-order bit planes of the mantissa.
    
    For each of the lowest 4 bit planes, compute the fraction of 1-bits.
    The overall bias is the maximum absolute deviation from 0.5 across planes.
    
    Implementation assumption: We examine bit planes 0-3 (4 lowest mantissa bits).
    """
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask]
    if len(flat) < MIN_ELEMENTS:
        return None
    
    fp32 = flat.astype(np.float32)
    bit_patterns = fp32.view(np.uint32)
    
    max_bias = 0.0
    for plane in range(4):
        bits = (bit_patterns >> plane) & 1
        fraction_ones = np.mean(bits)
        bias = abs(fraction_ones - 0.5)
        max_bias = max(max_bias, bias)
    
    return float(max_bias)


# ─────────────────────────────────────────────────────────────────────
# Feature 4: Serial Correlation
# ─────────────────────────────────────────────────────────────────────

def f04_serial_correlation(weights: np.ndarray) -> Optional[float]:
    """
    Lag-1 autocorrelation of the LSB sequence.
    
    For truly random LSBs, the autocorrelation should be near zero.
    Embedded structured data may introduce serial dependencies.
    
    Returns the absolute value of lag-1 autocorrelation.
    """
    lsb = _prepare_bits(weights, num_lsb=1)
    if lsb is None or len(lsb) < MIN_ELEMENTS + 1:
        return None
    
    lsb_float = lsb.astype(np.float64)
    mean = np.mean(lsb_float)
    var = np.var(lsb_float)
    
    if var < 1e-15:
        # Constant sequence — maximally anomalous for autocorrelation context
        return 1.0
    
    n = len(lsb_float)
    shifted = lsb_float[1:] - mean
    original = lsb_float[:-1] - mean
    autocorr = np.sum(shifted * original) / ((n - 1) * var)
    
    return float(abs(autocorr))


# ─────────────────────────────────────────────────────────────────────
# Feature 5: Runs Test
# ─────────────────────────────────────────────────────────────────────

def f05_runs_test(weights: np.ndarray) -> Optional[float]:
    """
    Wald-Wolfowitz runs test on the LSB sequence.
    
    Counts the number of runs (consecutive sequences of same bit value).
    Under randomness, the number of runs follows a known distribution.
    
    Returns the absolute Z-score (higher = more anomalous).
    """
    lsb = _prepare_bits(weights, num_lsb=1)
    if lsb is None:
        return None
    
    n = len(lsb)
    n1 = np.sum(lsb == 1)
    n0 = n - n1
    
    if n1 == 0 or n0 == 0:
        return float(n)  # Degenerate case — all same value
    
    # Count runs
    runs = 1 + np.sum(lsb[1:] != lsb[:-1])
    
    # Expected runs and variance under H0
    expected_runs = 1.0 + (2.0 * n0 * n1) / n
    var_runs = (2.0 * n0 * n1 * (2.0 * n0 * n1 - n)) / (n * n * (n - 1.0))
    
    if var_runs < 1e-15:
        return 0.0
    
    z = abs(runs - expected_runs) / np.sqrt(var_runs)
    return float(z)


# ─────────────────────────────────────────────────────────────────────
# Feature 6: Bit Mutual Information
# ─────────────────────────────────────────────────────────────────────

def f06_bit_mutual_info(weights: np.ndarray) -> Optional[float]:
    """
    Mutual information between adjacent LSBs.
    
    MI(b_i, b_{i+1}) measures how much knowing one LSB tells you about the next.
    For independent random bits, MI should be near zero.
    Structured embedded data increases MI.
    
    Returns MI in nats (natural log base).
    """
    lsb = _prepare_bits(weights, num_lsb=1)
    if lsb is None or len(lsb) < MIN_ELEMENTS + 1:
        return None
    
    n = len(lsb) - 1
    
    # Joint distribution of (b_i, b_{i+1})
    pairs = lsb[:-1] * 2 + lsb[1:]
    joint_counts = np.bincount(pairs, minlength=4).astype(np.float64)
    joint_prob = joint_counts / n
    
    # Marginal distributions
    p_x = np.array([np.sum(lsb[:-1] == 0), np.sum(lsb[:-1] == 1)], dtype=np.float64) / n
    p_y = np.array([np.sum(lsb[1:] == 0), np.sum(lsb[1:] == 1)], dtype=np.float64) / n
    
    # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
    mi = 0.0
    for i in range(2):
        for j in range(2):
            pxy = joint_prob[i * 2 + j]
            px = p_x[i]
            py = p_y[j]
            if pxy > 1e-15 and px > 1e-15 and py > 1e-15:
                mi += pxy * np.log(pxy / (px * py))
    
    return float(max(mi, 0.0))  # MI is non-negative


# ─────────────────────────────────────────────────────────────────────
# Feature 7: Trailing-Zero Count
# ─────────────────────────────────────────────────────────────────────

def f07_trailing_zero_count(weights: np.ndarray) -> Optional[float]:
    """
    Average number of trailing zero bits in the mantissa.
    
    Clean trained weights typically have few trailing zeros.
    Quantized, pruned, or manipulated weights may have anomalous
    trailing-zero distributions.
    
    Returns mean trailing zero count across all elements.
    """
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask]
    if len(flat) < MIN_ELEMENTS:
        return None
    
    fp32 = flat.astype(np.float32)
    bit_patterns = fp32.view(np.uint32)
    
    # Extract mantissa (lower 23 bits)
    mantissa = bit_patterns & 0x007FFFFF
    
    # Count trailing zeros for each mantissa
    # For zero mantissa, all 23 mantissa bits are zero
    trailing_zeros = np.zeros(len(mantissa), dtype=np.float64)
    for bit_pos in range(23):
        is_zero_at_pos = ((mantissa >> bit_pos) & 1) == 0
        trailing_zeros += is_zero_at_pos.astype(np.float64)
        # Stop counting for values that have a 1 at this position
        mantissa = np.where(is_zero_at_pos, mantissa, mantissa | (1 << bit_pos))
    
    # More efficient: use vectorized bit tricks
    # Number of trailing zeros = position of lowest set bit
    # For mantissa == 0, trailing zeros = 23
    non_zero = mantissa != 0
    tz = np.full(len(mantissa), 23.0)
    
    if np.any(non_zero):
        # Recompute from original mantissa
        orig_mantissa = (fp32.view(np.uint32) & 0x007FFFFF)
        for i in range(23):
            has_bit = ((orig_mantissa >> i) & 1).astype(bool)
            # First set bit position is the trailing zero count
            first_time = has_bit & (tz == 23.0)
            tz = np.where(first_time, float(i), tz)
    
    return float(np.mean(tz))


# ─────────────────────────────────────────────────────────────────────
# Feature 8: Zero-Field Fraction
# ─────────────────────────────────────────────────────────────────────

def f08_zero_field_fraction(weights: np.ndarray) -> Optional[float]:
    """
    Fraction of weight values that are exactly zero (bit pattern 0x00000000
    or 0x80000000 for signed zero).
    
    Pruned/sparse models have high zero fractions. Steganographic embedding
    in supposedly pruned weights is detectable if zero-field fraction deviates
    from expected sparsity patterns.
    
    Returns fraction in [0, 1].
    """
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask]
    if len(flat) < MIN_ELEMENTS:
        return None
    
    fp32 = flat.astype(np.float32)
    bit_patterns = fp32.view(np.uint32)
    
    # Exactly zero: bit pattern 0x00000000 (+0) or 0x80000000 (-0)
    is_zero = (bit_patterns == 0x00000000) | (bit_patterns == 0x80000000)
    fraction = np.sum(is_zero) / len(bit_patterns)
    
    return float(fraction)


# ─────────────────────────────────────────────────────────────────────
# Feature 9: FP16-Lattice Fraction / Precision-History Indicator
# ─────────────────────────────────────────────────────────────────────

def f09_fp16_lattice_fraction(weights: np.ndarray) -> Optional[float]:
    """
    Fraction of FP32 values whose bit patterns are consistent with
    FP16→FP32 conversion (lower 13 mantissa bits all zero).
    
    If a model was trained in FP16 and saved as FP32, this fraction
    will be very high (~1.0). This is NOT proof of manipulation but
    indicates precision history that affects other feature baselines.
    
    Implementation assumption: We check lower 13 bits of FP32 mantissa.
    An FP16 value promoted to FP32 will have these bits as zero.
    """
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask]
    if len(flat) < MIN_ELEMENTS:
        return None
    
    fp32 = flat.astype(np.float32)
    bit_patterns = fp32.view(np.uint32)
    
    # Lower 13 mantissa bits
    lower_13_mask = 0x00001FFF
    on_lattice = np.sum((bit_patterns & lower_13_mask) == 0)
    fraction = on_lattice / len(bit_patterns)
    
    return float(fraction)


# ─────────────────────────────────────────────────────────────────────
# Feature 10: Excess Kurtosis
# ─────────────────────────────────────────────────────────────────────

def f10_excess_kurtosis(weights: np.ndarray) -> Optional[float]:
    """
    Excess kurtosis of the weight distribution.
    
    Neural network weights typically follow distributions close to
    Gaussian (excess kurtosis ≈ 0). Steganographic manipulation or
    unusual training procedures may produce heavy or light tails.
    
    Returns Fisher's excess kurtosis (kurtosis - 3).
    """
    flat = weights.flatten()
    valid_mask = np.isfinite(flat)
    flat = flat[valid_mask].astype(np.float64)
    if len(flat) < MIN_ELEMENTS:
        return None
    
    std = np.std(flat)
    if std < 1e-15:
        return 0.0  # Constant distribution has zero kurtosis
    
    kurt = float(stats.kurtosis(flat, fisher=True))
    
    # Guard against extreme values from numerical instability
    if not np.isfinite(kurt):
        return None
    
    return kurt


# ─────────────────────────────────────────────────────────────────────
# Feature Extraction Orchestrator
# ─────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "f01_entropy_deficit",
    "f02_pairs_of_values",
    "f03_bit_plane_bias",
    "f04_serial_correlation",
    "f05_runs_test",
    "f06_bit_mutual_info",
    "f07_trailing_zero_count",
    "f08_zero_field_fraction",
    "f09_fp16_lattice_fraction",
    "f10_excess_kurtosis",
]

FEATURE_FUNCTIONS = {
    "f01_entropy_deficit": f01_entropy_deficit,
    "f02_pairs_of_values": f02_pairs_of_values,
    "f03_bit_plane_bias": f03_bit_plane_bias,
    "f04_serial_correlation": f04_serial_correlation,
    "f05_runs_test": f05_runs_test,
    "f06_bit_mutual_info": f06_bit_mutual_info,
    "f07_trailing_zero_count": f07_trailing_zero_count,
    "f08_zero_field_fraction": f08_zero_field_fraction,
    "f09_fp16_lattice_fraction": f09_fp16_lattice_fraction,
    "f10_excess_kurtosis": f10_excess_kurtosis,
}

FEATURE_DESCRIPTIONS = {
    "f01_entropy_deficit": "G-test statistic measuring LSB distribution departure from uniform",
    "f02_pairs_of_values": "Chi-squared statistic for adjacent LSB pair distribution",
    "f03_bit_plane_bias": "Maximum bit-plane bias across lowest 4 mantissa bit planes",
    "f04_serial_correlation": "Absolute lag-1 autocorrelation of LSB sequence",
    "f05_runs_test": "Wald-Wolfowitz runs test Z-score on LSB sequence",
    "f06_bit_mutual_info": "Mutual information between adjacent LSBs",
    "f07_trailing_zero_count": "Mean trailing zero count in mantissa bits",
    "f08_zero_field_fraction": "Fraction of exactly-zero weight values",
    "f09_fp16_lattice_fraction": "Fraction of values consistent with FP16→FP32 conversion",
    "f10_excess_kurtosis": "Fisher's excess kurtosis of weight distribution",
}


def extract_all_features(
    weights: np.ndarray,
    tensor_name: str = "unknown",
) -> Dict[str, Any]:
    """
    Extract all ten statistical features from a weight tensor.
    
    Returns a dict with:
      - feature values (None if not computable)
      - tensor metadata (shape, dtype, num_elements)
      - warnings (list of strings)
      - fp16_origin_detected (bool)
    """
    warnings = []
    flat = weights.flatten()
    num_elements = len(flat)
    
    # Check for minimum size
    if num_elements < MIN_ELEMENTS:
        warnings.append(
            f"Tensor '{tensor_name}' has {num_elements} elements, below minimum {MIN_ELEMENTS}. "
            "Feature extraction may be unreliable or impossible."
        )
    
    # Check for invalid values
    num_invalid = np.sum(~np.isfinite(flat.astype(np.float64)))
    if num_invalid > 0:
        warnings.append(
            f"Tensor '{tensor_name}' contains {num_invalid} NaN/Inf values "
            f"({100*num_invalid/num_elements:.1f}%). These are excluded from analysis."
        )
    
    # Detect FP16 origin
    valid = flat[np.isfinite(flat.astype(np.float64))].astype(np.float64)
    fp16_origin = False
    if len(valid) >= MIN_ELEMENTS:
        fp16_origin = detect_fp16_origin(valid)
        if fp16_origin:
            warnings.append(
                f"Tensor '{tensor_name}' appears to have been converted from FP16 to FP32. "
                "This affects feature baselines and is recorded as precision-history metadata."
            )
    
    # Extract each feature
    features = {}
    for name, func in FEATURE_FUNCTIONS.items():
        try:
            val = func(weights)
            features[name] = val
            if val is None:
                warnings.append(f"Feature '{name}' returned None for tensor '{tensor_name}'.")
        except Exception as e:
            features[name] = None
            warnings.append(f"Feature '{name}' failed for tensor '{tensor_name}': {str(e)}")
    
    return {
        "features": features,
        "tensor_name": tensor_name,
        "tensor_shape": list(weights.shape),
        "tensor_dtype": str(weights.dtype),
        "num_elements": num_elements,
        "fp16_origin_detected": fp16_origin,
        "warnings": warnings,
    }
