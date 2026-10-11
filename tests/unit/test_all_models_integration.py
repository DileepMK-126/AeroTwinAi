"""
AEROTWIN — Comprehensive All Models Integration Test Suite
Verifies that all 7 models (6 trained ML pipelines + Physical EKF Twin)
are loaded, integrated, execute live inference, and pass benchmark verification.
"""

from __future__ import annotations

import pytest
from core.real_engine import get_real_manager
from core.registry import get_model_registry


class TestAllModelsIntegration:
    """Verifies end-to-end integration of all ML models and datasets."""

    def test_model_registry_loaded_and_active(self):
        reg = get_model_registry()
        models = reg.list_models()
        assert len(models) >= 6

        # Check key trained models
        assert "alfa_engine_failure_classifier" in models
        assert "best_bearing_fault_classifier" in models
        assert "bearing_isolation_forest" in models
        assert "cmapss_rul_ensemble" in models
        assert "aerotwin_diesel_piston_classifier" in models
        assert "best_propeller_anomaly_classifier" in models

        # Check files exist
        for m_name in [
            "alfa_engine_failure_classifier",
            "best_bearing_fault_classifier",
            "bearing_isolation_forest",
            "cmapss_rul_ensemble",
            "aerotwin_diesel_piston_classifier",
            "best_propeller_anomaly_classifier",
        ]:
            assert models[m_name]["file_exists"] is True
            assert reg.is_trained(m_name) is True

    def test_real_engine_manager_models_loaded(self):
        mgr = get_real_manager()
        assert mgr.engine_model is not None, "ALFA engine model should be loaded"
        assert mgr.bearing_model is not None, "Bearing fault classifier should be loaded"
        assert mgr.bearing_anomaly_detector is not None, "Bearing anomaly detector should be loaded"
        assert mgr.diesel_model is not None, "Diesel piston model should be loaded"
        assert mgr.propeller_model is not None, "Propeller anomaly model should be loaded"
        assert mgr.cmapss_df is not None, "C-MAPSS multi-dataset should be loaded"
        assert len(mgr.cmapss_df) > 500, "C-MAPSS multi-dataset should have > 500 engines"

    def test_dataset_switching_and_streaming(self):
        mgr = get_real_manager()
        for ds in ["alfa_uav", "bearing", "cmapss", "propeller", "diesel", "rotax"]:
            assert mgr.set_dataset(ds) is True
            assert mgr.current_dataset == ds
            sample = mgr.next_sample()
            assert "source" in sample
            assert "rpm" in sample
            assert "model_inference" in sample
            inf = sample["model_inference"]
            assert "model_used" in inf
            assert "prediction" in inf
            assert "health_score" in inf
            assert "latency_ms" in inf

    def test_full_system_benchmark_all_passed(self):
        mgr = get_real_manager()
        res = mgr.run_full_benchmark()
        assert res["status"] == "success"
        assert res["all_passed"] is True
        assert res["total_models_evaluated"] == 7
        assert len(res["benchmarks"]) == 7

        for b in res["benchmarks"]:
            assert b["status"] == "PASSED"
            assert b["confidence"] > 0.0
            assert b["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_api_endpoints_for_ui_features(self):
        import httpx
        from api.main import app

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            # 1. Models inventory
            res = await client.get("/api/v1/models/inventory")
            assert res.status_code == 200
            inventory = res.json()
            assert len(inventory) >= 6

            # 2. Live benchmark
            res = await client.get("/api/v1/models/benchmark")
            assert res.status_code == 200
            bench = res.json()
            assert bench["all_passed"] is True
            assert bench["total_models_evaluated"] == 7

            # 3. Eval report
            res = await client.get("/api/v1/eval/report")
            assert res.status_code == 200
            assert "report_markdown" in res.json()

            # 4. Dataset switching
            for ds in ["diesel", "propeller", "cmapss", "bearing", "alfa_uav", "rotax"]:
                res = await client.post(f"/api/v1/sim/dataset?name={ds}")
                assert res.status_code == 200
                assert res.json()["current_dataset"] == ds

            # 5. Fault injection
            for fid in ["coolant_loss", "oil_starvation", "bearing_wear", "sensor_bias"]:
                res = await client.post(
                    "/api/v1/sim/fault",
                    json={
                        "fault_id": fid,
                        "severity": 1.0,
                        "ramp_type": "abrupt",
                        "sensor_channel": "egt_c_1" if fid == "sensor_bias" else None,
                        "sensor_bias_value": 50.0,
                    },
                )
                assert res.status_code == 200
                assert res.json()["status"] == "injected"

            # 6. Sim controls: start, stop, reset
            res = await client.post("/api/v1/sim/start?speed=2.0")
            assert res.status_code == 200
            assert res.json()["status"] == "running"

            res = await client.post("/api/v1/sim/stop")
            assert res.status_code == 200
            assert res.json()["status"] == "stopped"

            res = await client.post("/api/v1/sim/reset")
            assert res.status_code == 200
            assert res.json()["status"] == "reset"
