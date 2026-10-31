"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ─── Scan Schemas ────────────────────────────────────────────────────

class ScanResponse(BaseModel):
    id: str
    filename: str
    file_hash_sha256: Optional[str] = None
    file_size_bytes: Optional[int] = None
    model_format: str
    status: str
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tensors: int = 0
    total_parameters: int = 0
    anomaly_score: Optional[float] = None
    risk_classification: Optional[str] = None
    dtype_distribution: Optional[Dict[str, int]] = None
    warnings: Optional[List[str]] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True


class TensorResultResponse(BaseModel):
    id: str
    scan_id: str
    tensor_name: str
    tensor_shape: Optional[List[int]] = None
    tensor_dtype: Optional[str] = None
    num_elements: Optional[int] = None
    f01_entropy_deficit: Optional[float] = None
    f02_pairs_of_values: Optional[float] = None
    f03_bit_plane_bias: Optional[float] = None
    f04_serial_correlation: Optional[float] = None
    f05_runs_test: Optional[float] = None
    f06_bit_mutual_info: Optional[float] = None
    f07_trailing_zero_count: Optional[float] = None
    f08_zero_field_fraction: Optional[float] = None
    f09_fp16_lattice_fraction: Optional[float] = None
    f10_excess_kurtosis: Optional[float] = None
    anomaly_score: Optional[float] = None
    risk_classification: Optional[str] = None
    feature_contributions: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    warnings: Optional[List[str]] = None

    class Config:
        from_attributes = True


# ─── Baseline Schemas ────────────────────────────────────────────────

class BaselineCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class BaselineResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    file_hash_sha256: Optional[str] = None
    model_format: Optional[str] = None
    created_at: Optional[datetime] = None
    num_tensors: Optional[int] = None
    metadata: Optional[Dict] = None

    class Config:
        from_attributes = True


# ─── Dashboard Schemas ───────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_scans: int = 0
    normal_count: int = 0
    suspicious_count: int = 0
    high_anomaly_count: int = 0
    insufficient_count: int = 0
    failed_count: int = 0
    avg_anomaly_score: float = 0.0
    total_parameters_analyzed: int = 0
    recent_scans: List[ScanResponse] = []
    risk_distribution: Dict[str, int] = {}


# ─── Compare Schemas ─────────────────────────────────────────────────

class CompareRequest(BaseModel):
    scan_id: str
    baseline_id: str


class CompareResponse(BaseModel):
    scan_id: str
    baseline_id: str
    model_score: float
    risk_classification: str
    tensor_results: List[Dict[str, Any]] = []
    warnings: List[str] = []


# ─── Health Schemas ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    detection_engine: str = "operational"
    database: str = "connected"


# ─── Experiment Schemas ──────────────────────────────────────────────

class ExperimentRequest(BaseModel):
    n_clean: int = Field(default=30, ge=10, le=200)
    n_embedded: int = Field(default=30, ge=10, le=200)
    tensor_shape: List[int] = Field(default=[256, 256])
    seed: int = Field(default=42)


class ExperimentResponse(BaseModel):
    experiment_type: str
    n_clean: int
    n_embedded: int
    tensor_shape: List[int]
    train_size: int
    calibration_size: int
    test_size: int
    thresholds: Dict[str, float]
    metrics: Dict[str, Any]
    disclaimer: str
