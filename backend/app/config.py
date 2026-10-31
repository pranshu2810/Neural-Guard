"""
Application configuration loaded from environment variables.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with sensible defaults for local dev."""

    APP_NAME: str = "Neural Guard"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./neural_guard.db"

    # File upload
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: list[str] = [".safetensors", ".pt", ".pth", ".onnx"]

    # Detection thresholds (default calibration)
    ANOMALY_THRESHOLD_SUSPICIOUS: float = 3.0
    ANOMALY_THRESHOLD_HIGH: float = 5.0
    MIN_TENSOR_ELEMENTS: int = 256

    # Trusted baseline storage
    BASELINE_DIR: str = str(Path(__file__).resolve().parent.parent / "baselines")

    # Reports
    REPORTS_DIR: str = str(Path(__file__).resolve().parent.parent / "reports")

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Security
    SECRET_KEY: str = "neural-guard-dev-secret-change-in-production"

    class Config:
        env_file = ".env"


settings = Settings()

# Create directories
for d in [settings.UPLOAD_DIR, settings.BASELINE_DIR, settings.REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)
