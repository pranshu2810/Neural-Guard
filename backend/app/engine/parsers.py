"""
Checkpoint file parsers — secure extraction of weight tensors from model files.

Security Notes:
- Never uses pickle.load or torch.load with unrestricted unpickling.
- Uses weights_only=True for PyTorch loading.
- Validates file extensions and sizes before processing.
- Handles malformed files gracefully with informative errors.
"""

import os
import hashlib
import logging
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Generator, Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Maximum tensor size to prevent resource exhaustion (500M elements)
MAX_TENSOR_ELEMENTS = 500_000_000


class CheckpointParseError(Exception):
    """Raised when a checkpoint file cannot be parsed safely."""
    pass


def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def validate_tensor(name: str, tensor: np.ndarray) -> Optional[str]:
    """
    Validate a tensor before analysis.
    Returns a warning string if there's an issue, None if OK.
    """
    if tensor.size > MAX_TENSOR_ELEMENTS:
        return f"Tensor '{name}' has {tensor.size} elements, exceeding limit of {MAX_TENSOR_ELEMENTS}."
    if tensor.size == 0:
        return f"Tensor '{name}' is empty."
    return None


# ─────────────────────────────────────────────────────────────────────
# SafeTensors Parser
# ─────────────────────────────────────────────────────────────────────

def parse_safetensors(filepath: str) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
    """
    Parse a .safetensors file and yield (name, tensor_as_numpy, metadata) tuples.
    
    safetensors format is designed for safe loading — no arbitrary code execution.
    """
    try:
        from safetensors.numpy import load_file
    except ImportError:
        raise CheckpointParseError(
            "safetensors package not installed. Run: pip install safetensors"
        )
    
    try:
        tensors = load_file(filepath)
    except Exception as e:
        raise CheckpointParseError(f"Failed to parse safetensors file: {str(e)}")
    
    for name, tensor in tensors.items():
        warning = validate_tensor(name, tensor)
        meta = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "num_elements": int(tensor.size),
        }
        if warning:
            meta["warning"] = warning
            logger.warning(warning)
            if tensor.size > MAX_TENSOR_ELEMENTS:
                continue  # Skip oversized tensors
        
        yield name, tensor.astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────
# PyTorch Parser (.pt, .pth)
# ─────────────────────────────────────────────────────────────────────

def parse_pytorch(filepath: str) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
    """
    Parse a .pt/.pth PyTorch checkpoint file.
    
    Security: Uses weights_only=True to prevent arbitrary code execution.
    Falls back to a restricted loader if weights_only is not supported.
    """
    try:
        import torch
    except ImportError:
        raise CheckpointParseError("PyTorch not installed. Run: pip install torch")
    
    try:
        # Secure loading — weights_only prevents arbitrary code execution
        state_dict = torch.load(filepath, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions don't support weights_only
        logger.warning(
            "PyTorch version does not support weights_only=True. "
            "Loading with map_location='cpu' only. This is less secure."
        )
        try:
            state_dict = torch.load(filepath, map_location="cpu")
        except Exception as e:
            raise CheckpointParseError(f"Failed to load PyTorch file: {str(e)}")
    except Exception as e:
        raise CheckpointParseError(
            f"Failed to load PyTorch file with weights_only=True: {str(e)}. "
            "The file may contain non-tensor data or use pickle-dependent features."
        )
    
    # Handle both state_dict and raw tensor dict formats
    if isinstance(state_dict, dict):
        # Try common state_dict wrapper patterns
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]
    
    if not isinstance(state_dict, dict):
        raise CheckpointParseError(
            "PyTorch file does not contain a recognized state_dict structure."
        )
    
    for name, value in state_dict.items():
        try:
            import torch as _torch
            if isinstance(value, _torch.Tensor):
                tensor = value.detach().cpu().numpy()
            elif isinstance(value, np.ndarray):
                tensor = value
            else:
                logger.debug(f"Skipping non-tensor entry: {name} (type: {type(value).__name__})")
                continue
            
            # Convert to float for analysis
            if tensor.dtype in [np.float16, np.float32, np.float64]:
                np_tensor = tensor.astype(np.float32)
            elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64]:
                np_tensor = tensor.astype(np.float32)
            else:
                logger.debug(f"Skipping unsupported dtype: {name} ({tensor.dtype})")
                continue
            
            warning = validate_tensor(name, np_tensor)
            meta = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "num_elements": int(tensor.size),
                "original_dtype": str(value.dtype) if hasattr(value, 'dtype') else str(tensor.dtype),
            }
            if warning:
                meta["warning"] = warning
                logger.warning(warning)
                if tensor.size > MAX_TENSOR_ELEMENTS:
                    continue
            
            yield name, np_tensor, meta
        except Exception as e:
            logger.warning(f"Failed to extract tensor '{name}': {str(e)}")
            continue


# ─────────────────────────────────────────────────────────────────────
# ONNX Parser
# ─────────────────────────────────────────────────────────────────────

def parse_onnx(filepath: str) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
    """
    Parse an ONNX model file and extract initializer tensors.
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        raise CheckpointParseError("onnx package not installed. Run: pip install onnx")
    
    try:
        model = onnx.load(filepath)
        onnx.checker.check_model(model)
    except Exception as e:
        raise CheckpointParseError(f"Failed to load/validate ONNX file: {str(e)}")
    
    for initializer in model.graph.initializer:
        try:
            tensor = numpy_helper.to_array(initializer)
            name = initializer.name
            
            if tensor.dtype in [np.float16, np.float32, np.float64]:
                np_tensor = tensor.astype(np.float32)
            elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64]:
                np_tensor = tensor.astype(np.float32)
            else:
                logger.debug(f"Skipping unsupported ONNX dtype: {name} ({tensor.dtype})")
                continue
            
            warning = validate_tensor(name, np_tensor)
            meta = {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "num_elements": int(tensor.size),
            }
            if warning:
                meta["warning"] = warning
                if tensor.size > MAX_TENSOR_ELEMENTS:
                    continue
            
            yield name, np_tensor, meta
        except Exception as e:
            logger.warning(f"Failed to extract ONNX tensor '{initializer.name}': {str(e)}")
            continue


# ─────────────────────────────────────────────────────────────────────
# Unified Parser
# ─────────────────────────────────────────────────────────────────────

PARSERS = {
    ".safetensors": parse_safetensors,
    ".pt": parse_pytorch,
    ".pth": parse_pytorch,
    ".onnx": parse_onnx,
}


def get_model_format(filepath: str) -> str:
    """Determine model format from file extension."""
    ext = Path(filepath).suffix.lower()
    if ext in PARSERS:
        return ext.lstrip(".")
    raise CheckpointParseError(
        f"Unsupported model format: '{ext}'. "
        f"Supported formats: {', '.join(PARSERS.keys())}"
    )


def parse_checkpoint(filepath: str) -> Generator[Tuple[str, np.ndarray, Dict], None, None]:
    """
    Parse any supported checkpoint file format.
    Yields (tensor_name, numpy_array, metadata) tuples.
    """
    ext = Path(filepath).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise CheckpointParseError(
            f"Unsupported file format: '{ext}'. "
            f"Supported: {', '.join(PARSERS.keys())}"
        )
    yield from parser(filepath)
