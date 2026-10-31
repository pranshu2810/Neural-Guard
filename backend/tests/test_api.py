import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models import init_db

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["detection_engine"] == "operational"

@pytest.mark.asyncio
async def test_dashboard_stats():
    await init_db()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.get("/api/v1/dashboard/stats")
        assert res.status_code == 200
        data = res.json()
        assert "total_scans" in data
        assert "risk_distribution" in data

@pytest.mark.asyncio
async def test_synthetic_experiment():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.post(
            "/api/v1/experiments/run",
            json={
                "n_clean": 10,
                "n_embedded": 10,
                "tensor_shape": [32, 32],
                "seed": 42
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert "metrics" in data
        assert "accuracy" in data["metrics"]
        assert "roc_auc" in data["metrics"]
