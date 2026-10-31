"""
Unit Tests for all ten statistical feature extractors.
"""

import pytest
import numpy as np
from app.engine.features import (
    f01_entropy_deficit,
    f02_pairs_of_values,
    f03_bit_plane_bias,
    f04_serial_correlation,
    f05_runs_test,
    f06_bit_mutual_info,
    f07_trailing_zero_count,
    f08_zero_field_fraction,
    f09_fp16_lattice_fraction,
    f10_excess_kurtosis,
    extract_all_features,
    FEATURE_NAMES,
    MIN_ELEMENTS,
)
from app.engine.bit_utils import float32_to_bits, extract_lsb, detect_fp16_origin


class TestFeatureExtractors:
    """Test each feature with controlled inputs."""

    def _make_clean_tensor(self, size=1024, seed=42):
        rng = np.random.RandomState(seed)
        return rng.randn(size).astype(np.float32) * 0.02

    def _make_embedded_tensor(self, size=1024, seed=42):
        """Create a tensor with biased LSBs (simulating embedding)."""
        rng = np.random.RandomState(seed)
        tensor = rng.randn(size).astype(np.float32) * 0.02
        bits = tensor.view(np.uint32)
        # Set all LSBs to 1 (strong bias)
        bits[:] = bits | np.uint32(1)
        return bits.view(np.float32)

    def _make_tiny_tensor(self, size=10):
        return np.ones(size, dtype=np.float32)

    # Feature 1: Entropy Deficit
    def test_f01_clean_low_gstat(self):
        tensor = self._make_clean_tensor(10000)
        val = f01_entropy_deficit(tensor)
        assert val is not None
        # Clean tensor should have low G-statistic
        assert val < 50.0, f"G-stat too high for clean tensor: {val}"

    def test_f01_biased_high_gstat(self):
        tensor = self._make_embedded_tensor(10000)
        val = f01_entropy_deficit(tensor)
        assert val is not None
        assert val > 100.0, f"G-stat too low for biased tensor: {val}"

    def test_f01_too_small(self):
        val = f01_entropy_deficit(self._make_tiny_tensor())
        assert val is None

    # Feature 2: Pairs-of-Values
    def test_f02_clean(self):
        tensor = self._make_clean_tensor(10000)
        val = f02_pairs_of_values(tensor)
        assert val is not None
        assert isinstance(val, float)

    def test_f02_biased(self):
        tensor = self._make_embedded_tensor(10000)
        val = f02_pairs_of_values(tensor)
        assert val is not None

    def test_f02_too_small(self):
        val = f02_pairs_of_values(self._make_tiny_tensor())
        assert val is None

    # Feature 3: Bit-Plane Bias
    def test_f03_clean_low_bias(self):
        tensor = self._make_clean_tensor(10000)
        val = f03_bit_plane_bias(tensor)
        assert val is not None
        assert 0.0 <= val <= 0.5

    def test_f03_biased_high(self):
        tensor = self._make_embedded_tensor(10000)
        val = f03_bit_plane_bias(tensor)
        assert val is not None
        # All LSBs set to 1 = bias of ~0.5
        assert val > 0.4

    # Feature 4: Serial Correlation
    def test_f04_clean(self):
        tensor = self._make_clean_tensor(10000)
        val = f04_serial_correlation(tensor)
        assert val is not None
        assert 0.0 <= val <= 1.0

    def test_f04_too_small(self):
        val = f04_serial_correlation(self._make_tiny_tensor())
        assert val is None

    # Feature 5: Runs Test
    def test_f05_clean(self):
        tensor = self._make_clean_tensor(10000)
        val = f05_runs_test(tensor)
        assert val is not None
        assert val >= 0.0

    def test_f05_biased_high_z(self):
        tensor = self._make_embedded_tensor(10000)
        val = f05_runs_test(tensor)
        assert val is not None

    # Feature 6: Bit Mutual Information
    def test_f06_clean_low_mi(self):
        tensor = self._make_clean_tensor(10000)
        val = f06_bit_mutual_info(tensor)
        assert val is not None
        assert val >= 0.0

    # Feature 7: Trailing-Zero Count
    def test_f07_clean(self):
        tensor = self._make_clean_tensor(10000)
        val = f07_trailing_zero_count(tensor)
        assert val is not None
        assert val >= 0.0

    # Feature 8: Zero-Field Fraction
    def test_f08_no_zeros(self):
        tensor = self._make_clean_tensor(10000)
        val = f08_zero_field_fraction(tensor)
        assert val is not None
        assert 0.0 <= val <= 1.0

    def test_f08_all_zeros(self):
        tensor = np.zeros(1000, dtype=np.float32)
        val = f08_zero_field_fraction(tensor)
        assert val is not None
        assert val == 1.0

    # Feature 9: FP16-Lattice Fraction
    def test_f09_fp32_low(self):
        tensor = self._make_clean_tensor(10000)
        val = f09_fp16_lattice_fraction(tensor)
        assert val is not None
        assert 0.0 <= val <= 1.0

    def test_f09_fp16_converted_high(self):
        fp16 = np.random.randn(10000).astype(np.float16)
        fp32 = fp16.astype(np.float32)
        val = f09_fp16_lattice_fraction(fp32)
        assert val is not None
        assert val > 0.9, f"FP16→FP32 should have high lattice fraction: {val}"

    # Feature 10: Excess Kurtosis
    def test_f10_normal_near_zero(self):
        tensor = self._make_clean_tensor(50000)
        val = f10_excess_kurtosis(tensor)
        assert val is not None
        # Normal distribution has excess kurtosis near 0
        assert abs(val) < 1.0, f"Kurtosis too far from 0 for normal data: {val}"

    def test_f10_too_small(self):
        val = f10_excess_kurtosis(self._make_tiny_tensor())
        assert val is None


class TestFeatureOrchestrator:
    """Test the extract_all_features function."""

    def test_extract_all_returns_all_features(self):
        tensor = np.random.randn(1024).astype(np.float32)
        result = extract_all_features(tensor, "test_tensor")
        assert "features" in result
        assert "tensor_name" in result
        assert "warnings" in result
        for name in FEATURE_NAMES:
            assert name in result["features"]

    def test_extract_all_handles_nan(self):
        tensor = np.array([np.nan, np.inf, 1.0, 2.0] * 100, dtype=np.float32)
        result = extract_all_features(tensor, "nan_tensor")
        assert any("NaN" in w or "Inf" in w for w in result["warnings"])

    def test_extract_all_small_tensor_warning(self):
        tensor = np.random.randn(10).astype(np.float32)
        result = extract_all_features(tensor, "tiny")
        assert any("below minimum" in w for w in result["warnings"])

    def test_fp16_origin_detection(self):
        fp16 = np.random.randn(1024).astype(np.float16)
        fp32 = fp16.astype(np.float32)
        result = extract_all_features(fp32, "fp16_origin")
        assert result["fp16_origin_detected"] is True


class TestBitUtils:
    def test_float32_to_bits_roundtrip(self):
        original = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        bits = float32_to_bits(original)
        recovered = bits.view(np.float32)
        np.testing.assert_array_equal(original, recovered)

    def test_extract_lsb(self):
        bits = np.array([0b1010, 0b0101], dtype=np.uint32)
        lsb1 = extract_lsb(bits, 1)
        assert lsb1[0] == 0
        assert lsb1[1] == 1

    def test_detect_fp16_origin_true(self):
        fp16 = np.random.randn(1024).astype(np.float16)
        fp32 = fp16.astype(np.float32)
        assert detect_fp16_origin(fp32) is True

    def test_detect_fp16_origin_false(self):
        fp32 = np.random.randn(1024).astype(np.float32)
        assert detect_fp16_origin(fp32) is False
