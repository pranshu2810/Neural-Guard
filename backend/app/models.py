"""
SQLAlchemy database models for Neural Guard.
"""

import datetime
import uuid
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, JSON, Boolean,
    ForeignKey, Enum as SAEnum, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.config import settings

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ScanRecord(Base):
    """A single model scan operation."""
    __tablename__ = "scan_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    filename = Column(String, nullable=False)
    file_hash_sha256 = Column(String(64), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    model_format = Column(String(20), nullable=False)  # safetensors, pt, pth, onnx
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Model-level aggregated results
    total_tensors = Column(Integer, default=0)
    total_parameters = Column(Integer, default=0)
    anomaly_score = Column(Float, nullable=True)
    risk_classification = Column(String(20), nullable=True)  # normal, suspicious, high_anomaly, insufficient_evidence
    dtype_distribution = Column(JSON, nullable=True)

    # Metadata
    scan_config = Column(JSON, nullable=True)
    warnings = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    tensor_results = relationship("TensorResult", back_populates="scan", cascade="all, delete-orphan")


class TensorResult(Base):
    """Per-tensor feature extraction and anomaly detection results."""
    __tablename__ = "tensor_results"

    id = Column(String, primary_key=True, default=generate_uuid)
    scan_id = Column(String, ForeignKey("scan_records.id"), nullable=False)

    tensor_name = Column(String, nullable=False)
    tensor_shape = Column(JSON, nullable=True)
    tensor_dtype = Column(String(20), nullable=True)
    num_elements = Column(Integer, nullable=True)

    # Ten statistical features (nullable for insufficient evidence)
    f01_entropy_deficit = Column(Float, nullable=True)
    f02_pairs_of_values = Column(Float, nullable=True)
    f03_bit_plane_bias = Column(Float, nullable=True)
    f04_serial_correlation = Column(Float, nullable=True)
    f05_runs_test = Column(Float, nullable=True)
    f06_bit_mutual_info = Column(Float, nullable=True)
    f07_trailing_zero_count = Column(Float, nullable=True)
    f08_zero_field_fraction = Column(Float, nullable=True)
    f09_fp16_lattice_fraction = Column(Float, nullable=True)
    f10_excess_kurtosis = Column(Float, nullable=True)

    # Anomaly scoring
    anomaly_score = Column(Float, nullable=True)
    risk_classification = Column(String(20), nullable=True)
    feature_contributions = Column(JSON, nullable=True)  # Leave-one-out contributions
    explanation = Column(Text, nullable=True)
    warnings = Column(JSON, nullable=True)

    # Relationships
    scan = relationship("ScanRecord", back_populates="tensor_results")


class TrustedBaseline(Base):
    """Registered trusted model baseline for comparison."""
    __tablename__ = "trusted_baselines"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    file_hash_sha256 = Column(String(64), nullable=True)
    model_format = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    # Aggregated feature statistics from this baseline
    feature_means = Column(JSON, nullable=True)      # {feature_name: mean}
    feature_stds = Column(JSON, nullable=True)        # {feature_name: std}
    feature_covariance = Column(JSON, nullable=True)  # Serialized covariance matrix
    num_tensors = Column(Integer, nullable=True)
    tensor_features = Column(JSON, nullable=True)     # Raw feature vectors for recalculation
    metadata = Column(JSON, nullable=True)


class ScanHistory(Base):
    """Lightweight scan history for dashboard display."""
    __tablename__ = "scan_history"

    id = Column(String, primary_key=True, default=generate_uuid)
    scan_id = Column(String, ForeignKey("scan_records.id"), nullable=False)
    action = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    details = Column(JSON, nullable=True)


# Database engine and session
engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)


async def get_session():
    """Dependency for FastAPI endpoints."""
    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


async def init_db():
    """Create all tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
