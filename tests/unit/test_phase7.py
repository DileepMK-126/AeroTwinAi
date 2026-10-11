"""
AEROTWIN — Phase 7 Unit Tests: API, WebSocket Hub, and End-to-End Pipeline

Tests:
1. AerotwinPipeline processes nominal record into full PipelineOutput
2. AerotwinPipeline serializes WebSocket payload matching dashboard specification
3. AerotwinPipeline detects and flags engine fault through full pipeline
4. FastAPI health check endpoint returns 200 OK
5. FastAPI ingest telemetry endpoint processes record and returns 200 OK
6. FastAPI recent telemetry and alerts retrieval endpoints
"""

from __future__ import annotations

import datetime
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.pipeline import AerotwinPipeline, PipelineOutput
from configs.config_loader import load_settings
from data.schemas import DataSource, FlightPhase, TelemetryRecord


@pytest.fixture
def nominal_record():
    return TelemetryRecord(
        timestamp=datetime.datetime(2026, 10, 2, 12, 0, 0, tzinfo=datetime.timezone.utc),
        engine_id="rotax_pipeline_test",
        source=DataSource.simulated,
        flight_phase=FlightPhase.cruise,
        altitude_m=2000.0,
        oat_c=12.0,
        rpm=4800.0,
        throttle_pct=72.0,
        fuel_flow_lph=22.0,
        cht_c=[115.0, 114.0, 116.0, 115.0],
        egt_c=[760.0, 758.0, 762.0, 759.0],
        oil_temp_c=92.0,
        oil_press_kpa=380.0,
        coolant_temp_c=96.0,
        bus_voltage_v=14.0,
        vib_rms_g=0.03,
    )


class TestPipeline:
    def test_pipeline_nominal_process(self, nominal_record):
        pipeline = AerotwinPipeline(dt=1.0, persist=False)
        output = pipeline.process(nominal_record)

        assert isinstance(output, PipelineOutput)
        assert output.step == 1
        assert output.record.engine_id == nominal_record.engine_id
        assert output.twin_state.overall_health_score > 0.60
        assert output.decision.decision in ("CONTINUE", "CONTINUE_DEGRADED")

    def test_pipeline_to_ws_message(self, nominal_record):
        pipeline = AerotwinPipeline(dt=1.0, persist=False)
        output = pipeline.process(nominal_record)
        msg = output.to_ws_message()

        assert isinstance(msg, dict)
        assert msg["type"] == "telemetry_update"
        assert "twin" in msg
        assert "residuals" in msg
        assert "alerts" in msg
        assert "diagnosis" in msg
        assert "prognosis" in msg
        assert "mission" in msg
        assert "decision" in msg["mission"]

    def test_pipeline_anomaly_escalation(self, nominal_record):
        pipeline = AerotwinPipeline(dt=1.0, persist=False)
        # Warmup
        for _ in range(5):
            pipeline.process(nominal_record)

        # Inject severe coolant loss
        faulty = nominal_record.model_copy()
        faulty.cht_c = [240.0, 245.0, 242.0, 244.0]
        faulty.coolant_temp_c = 135.0

        for _ in range(10):
            out = pipeline.process(faulty)

        # Should trigger alerts and diagnosis
        assert len(out.alerts) > 0
        assert any(a.severity in ("warning", "critical") for a in out.alerts)
        assert out.diagnosis.primary_fault == "coolant_loss"


class TestAPIEndpoints:
    @pytest.fixture
    def api_key(self):
        settings = load_settings()
        return settings.ingest_api_key

    @pytest.mark.asyncio
    async def test_health_check(self):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["service"] == "aerotwin-backend"

    @pytest.mark.asyncio
    async def test_ingest_telemetry_endpoint(self, api_key, nominal_record):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = nominal_record.model_dump(mode="json")
            response = await client.post(
                "/api/v1/telemetry",
                json=payload,
                headers={"X-API-Key": api_key},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["engine_id"] == nominal_record.engine_id
            assert "decision" in data

    @pytest.mark.asyncio
    async def test_ingest_unauthorized(self, nominal_record):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = nominal_record.model_dump(mode="json")
            response = await client.post(
                "/api/v1/telemetry",
                json=payload,
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_recent_telemetry(self, api_key, nominal_record):
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            payload = nominal_record.model_dump(mode="json")
            await client.post("/api/v1/telemetry", json=payload, headers={"X-API-Key": api_key})

            response = await client.get(f"/api/v1/telemetry/{nominal_record.engine_id}/recent")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 1
