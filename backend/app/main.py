"""
Neural Guard — FastAPI Main Application.

Provides REST API endpoints for model scanning, anomaly detection,
baseline management, reporting, and dashboard statistics.
"""

import os
import shutil
import json
import logging
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.config import settings
from app.models import (
    ScanRecord, TensorResult, TrustedBaseline,
    init_db, get_session, generate_uuid,
)
from app.schemas import (
    ScanResponse, TensorResultResponse, BaselineCreate, BaselineResponse,
    DashboardStats, CompareRequest, CompareResponse, HealthResponse,
    ExperimentRequest, ExperimentResponse,
)
from app.engine.scanner import run_scan
from app.engine.detector import MahalanobisDetector
from app.engine.parsers import compute_file_hash, get_model_format, CheckpointParseError
from app.engine.features import extract_all_features, FEATURE_NAMES
from app.engine.reports import generate_json_report, generate_pdf_report
from app.engine.parsers import parse_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Application Setup
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Neural Guard API",
    description=(
        "Explainable Neural Network Weight Integrity & Steganography Detection Platform. "
        "Analyzes model checkpoint files for statistical anomalies that may indicate "
        "hidden data embedding or weight manipulation."
    ),
    version=settings.APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance (loaded from baseline if available)
_detector: Optional[MahalanobisDetector] = None


def get_detector() -> Optional[MahalanobisDetector]:
    global _detector
    if _detector is None:
        # Try loading from saved state
        state_path = Path(settings.BASELINE_DIR) / "detector_state.json"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    state = json.load(f)
                _detector = MahalanobisDetector.from_dict(state)
                logger.info("Loaded detector state from disk.")
            except Exception as e:
                logger.warning(f"Failed to load detector state: {e}")
        
        # If still None, initialize a default baseline detector
        if _detector is None:
            try:
                from app.engine.experiments import generate_clean_fp32
                clean_features = []
                for i in range(15):
                    t = generate_clean_fp32((128, 128), seed=200 + i)
                    ext = extract_all_features(t, f"default_baseline_{i}")
                    clean_features.append(ext["features"])
                det = MahalanobisDetector()
                det.fit(clean_features)
                Path(settings.BASELINE_DIR).mkdir(parents=True, exist_ok=True)
                save_detector(det)
                _detector = det
                logger.info("Initialized and saved default baseline detector.")
            except Exception as e:
                logger.warning(f"Could not initialize default baseline detector: {e}")

    return _detector


def save_detector(detector: MahalanobisDetector):
    global _detector
    _detector = detector
    state_path = Path(settings.BASELINE_DIR) / "detector_state.json"
    with open(state_path, "w") as f:
        json.dump(detector.to_dict(), f)
    logger.info("Detector state saved to disk.")


@app.on_event("startup")
async def startup():
    await init_db()
    get_detector()  # Try loading existing detector
    logger.info("Neural Guard API started.")


# ─────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        detection_engine="operational",
        database="connected",
    )


# ─────────────────────────────────────────────────────────────────────
# Scan Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/scan/upload", response_model=ScanResponse, tags=["Scanning"])
async def upload_and_scan(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a model file and initiate a security scan.
    
    Supported formats: .safetensors, .pt, .pth, .onnx
    
    Security: Files are parsed using safe loading mechanisms.
    PyTorch files use weights_only=True. No arbitrary code is executed.
    """
    # Validate extension
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{ext}'. Supported: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    # Save uploaded file
    scan_id = generate_uuid()
    upload_dir = Path(settings.UPLOAD_DIR) / scan_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / filename
    
    try:
        with open(filepath, "wb") as f:
            content = await file.read()
            # Check size limit
            if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
                )
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Run scan synchronously for immediate client feedback in MVP
    try:
        detector = get_detector()
        scan = await run_scan(
            filepath=str(filepath),
            filename=filename,
            session=session,
            detector=detector,
            scan_id=scan_id,
        )
    except Exception as e:
        logger.exception(f"Scan failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")
    
    return ScanResponse.model_validate(scan)


@app.get("/api/v1/scan/{scan_id}", response_model=ScanResponse, tags=["Scanning"])
async def get_scan(scan_id: str, session: AsyncSession = Depends(get_session)):
    """Get scan results by ID."""
    result = await session.execute(select(ScanRecord).where(ScanRecord.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    return ScanResponse.model_validate(scan)


@app.get("/api/v1/scan/{scan_id}/tensors", response_model=list[TensorResultResponse], tags=["Scanning"])
async def get_scan_tensors(
    scan_id: str,
    sort_by: str = Query("anomaly_score", description="Sort field"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    """Get tensor-level results for a scan."""
    # Verify scan exists
    result = await session.execute(select(ScanRecord).where(ScanRecord.id == scan_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Scan not found.")
    
    query = select(TensorResult).where(TensorResult.scan_id == scan_id)
    
    # Sorting
    sort_col = getattr(TensorResult, sort_by, TensorResult.anomaly_score)
    if order == "desc":
        query = query.order_by(desc(sort_col))
    else:
        query = query.order_by(sort_col)
    
    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    tensors = result.scalars().all()
    
    return [TensorResultResponse.model_validate(t) for t in tensors]


@app.get("/api/v1/scan/{scan_id}/explanations", tags=["Explainability"])
async def get_scan_explanations(
    scan_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get explainability data for suspicious tensors in a scan."""
    result = await session.execute(
        select(TensorResult)
        .where(TensorResult.scan_id == scan_id)
        .where(TensorResult.risk_classification.in_(["suspicious", "high_anomaly"]))
        .order_by(desc(TensorResult.anomaly_score))
    )
    tensors = result.scalars().all()
    
    explanations = []
    for t in tensors:
        explanations.append({
            "tensor_name": t.tensor_name,
            "tensor_shape": t.tensor_shape,
            "tensor_dtype": t.tensor_dtype,
            "anomaly_score": t.anomaly_score,
            "risk_classification": t.risk_classification,
            "explanation": t.explanation,
            "feature_contributions": t.feature_contributions,
            "feature_values": {
                "f01_entropy_deficit": t.f01_entropy_deficit,
                "f02_pairs_of_values": t.f02_pairs_of_values,
                "f03_bit_plane_bias": t.f03_bit_plane_bias,
                "f04_serial_correlation": t.f04_serial_correlation,
                "f05_runs_test": t.f05_runs_test,
                "f06_bit_mutual_info": t.f06_bit_mutual_info,
                "f07_trailing_zero_count": t.f07_trailing_zero_count,
                "f08_zero_field_fraction": t.f08_zero_field_fraction,
                "f09_fp16_lattice_fraction": t.f09_fp16_lattice_fraction,
                "f10_excess_kurtosis": t.f10_excess_kurtosis,
            },
            "warnings": t.warnings,
        })
    
    return {"scan_id": scan_id, "explanations": explanations}


# ─────────────────────────────────────────────────────────────────────
# Report Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/scan/{scan_id}/report", tags=["Reports"])
async def get_scan_report(
    scan_id: str,
    format: str = Query("json", description="Report format: json or pdf"),
    session: AsyncSession = Depends(get_session),
):
    """Generate and download a forensic report."""
    # Get scan
    result = await session.execute(select(ScanRecord).where(ScanRecord.id == scan_id))
    scan = result.scalar_one_or_none()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    
    if scan.status != "completed":
        raise HTTPException(status_code=400, detail="Scan is not completed yet.")
    
    # Get tensors
    result = await session.execute(
        select(TensorResult).where(TensorResult.scan_id == scan_id)
    )
    tensors = result.scalars().all()
    
    scan_data = {
        "id": scan.id,
        "filename": scan.filename,
        "file_hash_sha256": scan.file_hash_sha256,
        "file_size_bytes": scan.file_size_bytes,
        "model_format": scan.model_format,
        "created_at": str(scan.created_at) if scan.created_at else None,
        "completed_at": str(scan.completed_at) if scan.completed_at else None,
        "anomaly_score": scan.anomaly_score,
        "risk_classification": scan.risk_classification,
        "total_tensors": scan.total_tensors,
        "total_parameters": scan.total_parameters,
        "dtype_distribution": scan.dtype_distribution,
    }
    
    tensor_data = []
    for t in tensors:
        tensor_data.append({
            "tensor_name": t.tensor_name,
            "tensor_shape": t.tensor_shape,
            "tensor_dtype": t.tensor_dtype,
            "num_elements": t.num_elements,
            "anomaly_score": t.anomaly_score,
            "risk_classification": t.risk_classification,
            "explanation": t.explanation,
            "feature_contributions": t.feature_contributions,
        })
    
    report_dir = Path(settings.REPORTS_DIR) / scan_id
    report_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "pdf":
        report_path = str(report_dir / f"neural_guard_report_{scan_id[:8]}.pdf")
        generate_pdf_report(scan_data, tensor_data, report_path)
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"neural_guard_report_{scan_id[:8]}.pdf",
        )
    else:
        report_path = str(report_dir / f"neural_guard_report_{scan_id[:8]}.json")
        generate_json_report(scan_data, tensor_data, report_path)
        return FileResponse(
            report_path,
            media_type="application/json",
            filename=f"neural_guard_report_{scan_id[:8]}.json",
        )


# ─────────────────────────────────────────────────────────────────────
# Baseline Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/baselines", response_model=BaselineResponse, tags=["Baselines"])
async def create_baseline(
    name: str = Query(..., description="Baseline name"),
    description: str = Query("", description="Description"),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Register a trusted model as a baseline reference.
    
    Extracts features from all tensors and stores them for future comparison.
    The model is NOT automatically marked as trusted — the user must explicitly
    register it through this endpoint.
    """
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: '{ext}'."
        )
    
    baseline_id = generate_uuid()
    upload_dir = Path(settings.BASELINE_DIR) / baseline_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filepath = upload_dir / filename
    
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)
    
    file_hash = compute_file_hash(str(filepath))
    model_format = ext.lstrip(".")
    
    # Extract features from all tensors
    all_features = []
    try:
        for tname, tensor, meta in parse_checkpoint(str(filepath)):
            result = extract_all_features(tensor, tensor_name=tname)
            all_features.append(result["features"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse model: {str(e)}")
    
    if not all_features:
        raise HTTPException(status_code=400, detail="No valid tensors found.")
    
    # Compute feature statistics
    import numpy as np
    feature_matrix = []
    for feat in all_features:
        vec = [feat.get(fn) for fn in FEATURE_NAMES]
        feature_matrix.append(vec)
    
    feature_matrix = np.array(feature_matrix, dtype=np.float64)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    means = np.mean(feature_matrix, axis=0).tolist()
    stds = np.std(feature_matrix, axis=0).tolist()
    
    baseline = TrustedBaseline(
        id=baseline_id,
        name=name,
        description=description,
        file_hash_sha256=file_hash,
        model_format=model_format,
        num_tensors=len(all_features),
        feature_means={fn: m for fn, m in zip(FEATURE_NAMES, means)},
        feature_stds={fn: s for fn, s in zip(FEATURE_NAMES, stds)},
        tensor_features=[f for f in all_features],
        metadata={"filename": filename, "format": model_format},
    )
    
    session.add(baseline)
    await session.commit()
    await session.refresh(baseline)
    
    # Re-fit detector with all baselines
    await _refit_detector(session)
    
    return BaselineResponse.model_validate(baseline)


@app.get("/api/v1/baselines", response_model=list[BaselineResponse], tags=["Baselines"])
async def list_baselines(session: AsyncSession = Depends(get_session)):
    """List all registered trusted baselines."""
    result = await session.execute(
        select(TrustedBaseline).order_by(desc(TrustedBaseline.created_at))
    )
    baselines = result.scalars().all()
    return [BaselineResponse.model_validate(b) for b in baselines]


async def _refit_detector(session: AsyncSession):
    """Re-fit the global detector using all registered baselines."""
    result = await session.execute(select(TrustedBaseline))
    baselines = result.scalars().all()
    
    all_features = []
    for baseline in baselines:
        if baseline.tensor_features:
            all_features.extend(baseline.tensor_features)
    
    if len(all_features) < 2:
        logger.info("Not enough baseline tensors to fit detector.")
        return
    
    detector = MahalanobisDetector(
        suspicious_threshold=settings.ANOMALY_THRESHOLD_SUSPICIOUS,
        high_anomaly_threshold=settings.ANOMALY_THRESHOLD_HIGH,
    )
    detector.fit(all_features)
    
    # Use first 20% as calibration if we have enough
    n_calib = max(5, len(all_features) // 5)
    if len(all_features) > n_calib + 2:
        calib_features = all_features[:n_calib]
        detector.calibrate(calib_features)
    
    save_detector(detector)
    logger.info(f"Detector re-fitted with {len(all_features)} baseline tensors.")


# ─────────────────────────────────────────────────────────────────────
# Compare Endpoint
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/compare", response_model=CompareResponse, tags=["Analysis"])
async def compare_with_baseline(
    request: CompareRequest,
    session: AsyncSession = Depends(get_session),
):
    """Compare a scanned model against a specific trusted baseline."""
    # Get scan tensors
    result = await session.execute(
        select(TensorResult).where(TensorResult.scan_id == request.scan_id)
    )
    scan_tensors = result.scalars().all()
    if not scan_tensors:
        raise HTTPException(status_code=404, detail="Scan not found or has no tensors.")
    
    # Get baseline
    result = await session.execute(
        select(TrustedBaseline).where(TrustedBaseline.id == request.baseline_id)
    )
    baseline = result.scalar_one_or_none()
    if not baseline:
        raise HTTPException(status_code=404, detail="Baseline not found.")
    
    if not baseline.tensor_features:
        raise HTTPException(status_code=400, detail="Baseline has no stored features.")
    
    # Fit detector on this specific baseline
    detector = MahalanobisDetector()
    detector.fit(baseline.tensor_features)
    
    # Score each scan tensor
    tensor_results = []
    for t in scan_tensors:
        features = {
            "f01_entropy_deficit": t.f01_entropy_deficit,
            "f02_pairs_of_values": t.f02_pairs_of_values,
            "f03_bit_plane_bias": t.f03_bit_plane_bias,
            "f04_serial_correlation": t.f04_serial_correlation,
            "f05_runs_test": t.f05_runs_test,
            "f06_bit_mutual_info": t.f06_bit_mutual_info,
            "f07_trailing_zero_count": t.f07_trailing_zero_count,
            "f08_zero_field_fraction": t.f08_zero_field_fraction,
            "f09_fp16_lattice_fraction": t.f09_fp16_lattice_fraction,
            "f10_excess_kurtosis": t.f10_excess_kurtosis,
        }
        ar = detector.score_tensor(features, t.tensor_name)
        tensor_results.append({
            "tensor_name": ar.tensor_name,
            "anomaly_score": ar.anomaly_score,
            "risk_classification": ar.risk_classification,
            "explanation": ar.explanation,
            "feature_contributions": ar.feature_contributions,
        })
    
    scores = [r["anomaly_score"] for r in tensor_results]
    model_score = max(scores) if scores else 0.0
    
    if any(r["risk_classification"] == "high_anomaly" for r in tensor_results):
        model_class = "high_anomaly"
    elif any(r["risk_classification"] == "suspicious" for r in tensor_results):
        model_class = "suspicious"
    else:
        model_class = "normal"
    
    return CompareResponse(
        scan_id=request.scan_id,
        baseline_id=request.baseline_id,
        model_score=model_score,
        risk_classification=model_class,
        tensor_results=tensor_results,
    )


# ─────────────────────────────────────────────────────────────────────
# Dashboard Stats
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/dashboard/stats", response_model=DashboardStats, tags=["Dashboard"])
async def dashboard_stats(session: AsyncSession = Depends(get_session)):
    """Get dashboard statistics."""
    # Total scans
    result = await session.execute(select(func.count(ScanRecord.id)))
    total_scans = result.scalar() or 0
    
    # By classification
    for classification in ["normal", "suspicious", "high_anomaly", "insufficient_evidence"]:
        result = await session.execute(
            select(func.count(ScanRecord.id)).where(ScanRecord.risk_classification == classification)
        )
    
    result = await session.execute(
        select(func.count(ScanRecord.id)).where(ScanRecord.risk_classification == "normal")
    )
    normal_count = result.scalar() or 0
    
    result = await session.execute(
        select(func.count(ScanRecord.id)).where(ScanRecord.risk_classification == "suspicious")
    )
    suspicious_count = result.scalar() or 0
    
    result = await session.execute(
        select(func.count(ScanRecord.id)).where(ScanRecord.risk_classification == "high_anomaly")
    )
    high_anomaly_count = result.scalar() or 0
    
    result = await session.execute(
        select(func.count(ScanRecord.id)).where(
            ScanRecord.risk_classification == "insufficient_evidence"
        )
    )
    insufficient_count = result.scalar() or 0
    
    result = await session.execute(
        select(func.count(ScanRecord.id)).where(ScanRecord.status == "failed")
    )
    failed_count = result.scalar() or 0
    
    # Average anomaly score
    result = await session.execute(
        select(func.avg(ScanRecord.anomaly_score)).where(ScanRecord.anomaly_score.isnot(None))
    )
    avg_score = result.scalar() or 0.0
    
    # Total parameters
    result = await session.execute(
        select(func.sum(ScanRecord.total_parameters))
    )
    total_params = result.scalar() or 0
    
    # Recent scans
    result = await session.execute(
        select(ScanRecord).order_by(desc(ScanRecord.created_at)).limit(10)
    )
    recent = result.scalars().all()
    
    return DashboardStats(
        total_scans=total_scans,
        normal_count=normal_count,
        suspicious_count=suspicious_count,
        high_anomaly_count=high_anomaly_count,
        insufficient_count=insufficient_count,
        failed_count=failed_count,
        avg_anomaly_score=float(avg_score),
        total_parameters_analyzed=total_params,
        recent_scans=[ScanResponse.model_validate(s) for s in recent],
        risk_distribution={
            "normal": normal_count,
            "suspicious": suspicious_count,
            "high_anomaly": high_anomaly_count,
            "insufficient_evidence": insufficient_count,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Scan History
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/scans", response_model=list[ScanResponse], tags=["Scanning"])
async def list_scans(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
):
    """List all scans with optional filtering."""
    query = select(ScanRecord).order_by(desc(ScanRecord.created_at))
    if status:
        query = query.where(ScanRecord.status == status)
    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    scans = result.scalars().all()
    return [ScanResponse.model_validate(s) for s in scans]


# ─────────────────────────────────────────────────────────────────────
# Experiment Endpoint
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/experiments/run", tags=["Experiments"])
async def run_experiment_endpoint(request: ExperimentRequest):
    """
    Run a synthetic detection experiment.
    
    WARNING: This uses synthetic data only. Results do not represent
    real-world detection performance.
    """
    from app.engine.experiments import run_experiment
    
    try:
        results = run_experiment(
            n_clean=request.n_clean,
            n_embedded=request.n_embedded,
            tensor_shape=tuple(request.tensor_shape),
            seed=request.seed,
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}")
