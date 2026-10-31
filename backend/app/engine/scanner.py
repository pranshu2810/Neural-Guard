"""
Scanner — Orchestrates the full model scanning pipeline.

1. Parse checkpoint file (secure extraction)
2. Extract features from each tensor
3. Score each tensor against the baseline detector
4. Aggregate model-level results
5. Store results in the database
"""

import logging
import datetime
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.features import extract_all_features, FEATURE_NAMES
from app.engine.detector import MahalanobisDetector, AnomalyResult, ModelAnomalyResult
from app.engine.parsers import (
    parse_checkpoint, compute_file_hash, get_file_size,
    get_model_format, CheckpointParseError
)
from app.models import ScanRecord, TensorResult
from app.config import settings

logger = logging.getLogger(__name__)


class ScanError(Exception):
    """Raised when a scan fails."""
    pass


async def run_scan(
    filepath: str,
    filename: str,
    session: AsyncSession,
    detector: Optional[MahalanobisDetector] = None,
    scan_id: Optional[str] = None,
    progress_callback=None,
) -> ScanRecord:
    """
    Run a complete model scan.
    
    Args:
        filepath: Path to the uploaded model file on disk.
        filename: Original uploaded filename.
        session: Database session.
        detector: Optional fitted MahalanobisDetector for anomaly scoring.
        scan_id: Optional pre-assigned scan ID.
        progress_callback: Optional async callback(stage, progress, message).
    
    Returns:
        Completed ScanRecord with all results.
    """
    from app.models import generate_uuid
    
    # Create scan record
    scan = ScanRecord(
        id=scan_id or generate_uuid(),
        filename=filename,
        status="processing",
    )
    
    try:
        # Stage 1: File validation
        if progress_callback:
            await progress_callback("validation", 0.05, "Validating file...")
        
        model_format = get_model_format(filepath)
        scan.model_format = model_format
        scan.file_size_bytes = get_file_size(filepath)
        scan.file_hash_sha256 = compute_file_hash(filepath)
        
        # Check file size limit
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if scan.file_size_bytes > max_bytes:
            raise ScanError(
                f"File size ({scan.file_size_bytes / 1024 / 1024:.1f} MB) "
                f"exceeds limit ({settings.MAX_UPLOAD_SIZE_MB} MB)."
            )
        
        # Stage 2: Parse checkpoint
        if progress_callback:
            await progress_callback("parsing", 0.10, "Parsing model checkpoint...")
        
        tensors = []
        dtype_counts = {}
        for name, tensor, meta in parse_checkpoint(filepath):
            tensors.append((name, tensor, meta))
            dtype = meta.get("dtype", "unknown")
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        scan.total_tensors = len(tensors)
        scan.total_parameters = sum(t[1].size for t in tensors)
        scan.dtype_distribution = dtype_counts
        
        if not tensors:
            raise ScanError("No valid tensors found in the model file.")
        
        # Stage 3: Feature extraction
        if progress_callback:
            await progress_callback("extraction", 0.20, f"Extracting features from {len(tensors)} tensors...")
        
        all_warnings = []
        tensor_features_list = []
        
        for i, (name, tensor, meta) in enumerate(tensors):
            progress = 0.20 + 0.50 * (i / len(tensors))
            if progress_callback:
                await progress_callback(
                    "extraction", progress,
                    f"Analyzing tensor {i+1}/{len(tensors)}: {name}"
                )
            
            result = extract_all_features(tensor, tensor_name=name)
            tensor_features_list.append({
                "name": name,
                "meta": meta,
                "extraction": result,
            })
            all_warnings.extend(result["warnings"])
        
        # Stage 4: Anomaly scoring
        if progress_callback:
            await progress_callback("scoring", 0.75, "Computing anomaly scores...")
        
        anomaly_results = []
        for tf in tensor_features_list:
            if detector and detector.is_fitted_:
                ar = detector.score_tensor(
                    tf["extraction"]["features"],
                    tensor_name=tf["name"]
                )
            else:
                # No baseline — report features but mark as insufficient evidence
                ar = AnomalyResult(
                    tensor_name=tf["name"],
                    anomaly_score=0.0,
                    risk_classification="insufficient_evidence",
                    feature_values=tf["extraction"]["features"],
                    feature_deviations={},
                    feature_contributions={},
                    explanation=(
                        f"No trusted baseline available for comparison. "
                        f"Features extracted but anomaly scoring requires a fitted detector. "
                        f"Register trusted models first."
                    ),
                    warnings=["No baseline detector available."],
                )
            anomaly_results.append(ar)
        
        # Stage 5: Model-level aggregation
        if detector and detector.is_fitted_:
            model_result = detector.score_model(anomaly_results)
        else:
            model_result = ModelAnomalyResult(
                model_score=0.0,
                risk_classification="insufficient_evidence",
                total_tensors=len(anomaly_results),
                analyzed_tensors=0,
                normal_count=0,
                suspicious_count=0,
                high_anomaly_count=0,
                insufficient_count=len(anomaly_results),
                tensor_results=anomaly_results,
                warnings=["No baseline detector — all tensors marked as insufficient evidence."],
            )
        
        scan.anomaly_score = model_result.model_score
        scan.risk_classification = model_result.risk_classification
        scan.warnings = all_warnings[:50]  # Limit stored warnings
        
        # Stage 6: Store tensor results
        if progress_callback:
            await progress_callback("storing", 0.90, "Saving results...")
        
        for ar, tf in zip(anomaly_results, tensor_features_list):
            tr = TensorResult(
                scan_id=scan.id,
                tensor_name=ar.tensor_name,
                tensor_shape=tf["meta"].get("shape"),
                tensor_dtype=tf["meta"].get("dtype"),
                num_elements=tf["meta"].get("num_elements"),
                f01_entropy_deficit=ar.feature_values.get("f01_entropy_deficit"),
                f02_pairs_of_values=ar.feature_values.get("f02_pairs_of_values"),
                f03_bit_plane_bias=ar.feature_values.get("f03_bit_plane_bias"),
                f04_serial_correlation=ar.feature_values.get("f04_serial_correlation"),
                f05_runs_test=ar.feature_values.get("f05_runs_test"),
                f06_bit_mutual_info=ar.feature_values.get("f06_bit_mutual_info"),
                f07_trailing_zero_count=ar.feature_values.get("f07_trailing_zero_count"),
                f08_zero_field_fraction=ar.feature_values.get("f08_zero_field_fraction"),
                f09_fp16_lattice_fraction=ar.feature_values.get("f09_fp16_lattice_fraction"),
                f10_excess_kurtosis=ar.feature_values.get("f10_excess_kurtosis"),
                anomaly_score=ar.anomaly_score,
                risk_classification=ar.risk_classification,
                feature_contributions=ar.feature_contributions,
                explanation=ar.explanation,
                warnings=ar.warnings,
            )
            session.add(tr)
        
        scan.status = "completed"
        scan.completed_at = datetime.datetime.utcnow()
        
    except CheckpointParseError as e:
        scan.status = "failed"
        scan.error_message = str(e)
        logger.error(f"Checkpoint parse error: {e}")
    except ScanError as e:
        scan.status = "failed"
        scan.error_message = str(e)
        logger.error(f"Scan error: {e}")
    except Exception as e:
        scan.status = "failed"
        scan.error_message = f"Unexpected error: {str(e)}"
        logger.exception(f"Unexpected scan error")
    
    session.add(scan)
    await session.commit()
    await session.refresh(scan)
    
    if progress_callback:
        await progress_callback("complete", 1.0, "Scan complete.")
    
    return scan
