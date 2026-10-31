"""
Bit-level utility functions for extracting binary representations of floating-point
weight tensors. These preserve the original IEEE 754 bit patterns.

Implementation Notes:
- FP32 values use numpy's float32 → uint32 view to get exact bit patterns.
- FP16 values use float16 → uint16 view.
- We extract the low-order bits (LSBs) of the mantissa, which is where
  steganographic payloads are most likely embedded.
"""

import numpy as np
from typing import Optional


def float32_to_bits(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to uint32 bit pattern array."""
    return arr.astype(np.float32).view(np.uint32)


def float16_to_bits(arr: np.ndarray) -> np.ndarray:
    """Convert float16 array to uint16 bit pattern array."""
    return arr.astype(np.float16).view(np.uint16)


def extract_lsb(bit_arr: np.ndarray, num_bits: int = 1) -> np.ndarray:
    """Extract the lowest `num_bits` bits from an integer bit-pattern array."""
    mask = (1 << num_bits) - 1
    return (bit_arr & mask).astype(np.uint32)


def extract_bit_plane(bit_arr: np.ndarray, bit_index: int) -> np.ndarray:
    """Extract a specific bit plane (0 = LSB)."""
    return ((bit_arr >> bit_index) & 1).astype(np.uint8)


def get_mantissa_bits(bit_arr: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """
    Extract mantissa bits from IEEE 754 bit patterns.
    FP32: bits [0..22] (23 mantissa bits)
    FP16: bits [0..9]  (10 mantissa bits)
    """
    if dtype == "float32":
        return (bit_arr & 0x007FFFFF).astype(np.uint32)
    elif dtype == "float16":
        return (bit_arr & 0x03FF).astype(np.uint16)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_exponent_bits(bit_arr: np.ndarray, dtype: str = "float32") -> np.ndarray:
    """Extract exponent bits."""
    if dtype == "float32":
        return ((bit_arr >> 23) & 0xFF).astype(np.uint8)
    elif dtype == "float16":
        return ((bit_arr >> 10) & 0x1F).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def filter_valid_floats(arr: np.ndarray) -> np.ndarray:
    """Remove NaN and Inf values from a float array."""
    mask = np.isfinite(arr)
    return arr[mask]


def detect_fp16_origin(fp32_arr: np.ndarray, threshold: float = 0.9) -> bool:
    """
    Heuristic to detect if an FP32 tensor was likely converted from FP16.
    
    FP16→FP32 conversion produces values that, when viewed as uint32, have
    their lower 13 mantissa bits all zero (since FP16 has only 10 mantissa
    bits vs FP32's 23).
    
    Implementation assumption: If > threshold fraction of values have
    lower 13 bits zero, we classify as likely FP16-origin.
    """
    bit_patterns = float32_to_bits(fp32_arr)
    lower_13_mask = 0x00001FFF
    zeros_in_lower = np.sum((bit_patterns & lower_13_mask) == 0)
    fraction = zeros_in_lower / len(bit_patterns)
    return bool(fraction >= threshold)


def safe_array_prep(arr: np.ndarray, min_elements: int = 256) -> Optional[np.ndarray]:
    """
    Prepare array for analysis: flatten, filter invalid values.
    Returns None if the resulting array is too small for reliable statistics.
    """
    flat = arr.flatten().astype(np.float64)
    valid = filter_valid_floats(flat)
    if len(valid) < min_elements:
        return None
    return valid
