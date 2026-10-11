"""
AEROTWIN — Phase 3 Unit Tests: Digital Twin & State Estimation (EKF)

Tests:
1. EKF initialization and state dimension
2. EKF step produces bounded state estimates
3. EKF handles missing sensor measurements gracefully
4. EKF tracks health parameters
5. DigitalTwin step processes TelemetryRecord
6. DigitalTwin produces valid residuals and normalized residuals
7. DigitalTwin isolates sensor anomalies
8. DigitalTwin health score is in [0, 1] range
9. DigitalTwin reset restores initial conditions
"""

from __future__ import annotations

import datetime
import pytest
import numpy as np

from configs.config_loader import load_engine_config
from core.physics.engine_model import EngineState, PistonEngineModel
from core.twin.ekf import ExtendedKalmanFilter, TwinEstimate, N_STATES
from core.twin.twin import DigitalTwin, TwinState
from data.schemas import TelemetryRecord, DataSource, FlightPhase


@pytest.fixture
def engine_cfg():
    return load_engine_config()


@pytest.fixture
def sample_record(engine_cfg):
    model = PistonEngineModel(engine_cfg)
    s = EngineState()
    for _ in range(10):
        s = model.step(s, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=1000.0)

    return TelemetryRecord(
        timestamp=datetime.datetime(2026, 10, 2, 12, 0, 0, tzinfo=datetime.timezone.utc),
        engine_id="rotax_914_test",
        source=DataSource.simulated,
        flight_phase=FlightPhase.cruise,
        altitude_m=1000.0,
        oat_c=15.0,
        rpm=4500.0,
        throttle_pct=70.0,
        fuel_flow_lph=float(s.fuel_flow_lph),
        cht_c=[float(c) for c in s.cht_c],
        egt_c=[float(e) for e in s.egt_c],
        oil_temp_c=float(s.oil_temp_c),
        oil_press_kpa=float(s.oil_press_kpa),
        coolant_temp_c=float(s.coolant_temp_c),
        bus_voltage_v=14.0,
        vib_rms_g=0.03,
    )


class TestEKF:
    def test_ekf_init(self, engine_cfg):
        ekf = ExtendedKalmanFilter(engine_cfg)
        assert len(ekf.x) == N_STATES
        assert ekf.P.shape == (N_STATES, N_STATES)
        # Check initial covariance is positive definite
        assert np.all(np.diag(ekf.P) > 0)

    def test_ekf_step_normal_measurements(self, engine_cfg):
        ekf = ExtendedKalmanFilter(engine_cfg)
        model = PistonEngineModel(engine_cfg)
        state = model.step(EngineState(), dt=1.0, throttle=0.7, rpm=4500.0)

        measurements = {
            "oil_temp_c": state.oil_temp_c,
            "coolant_temp_c": state.coolant_temp_c,
            "oil_press_kpa": state.oil_press_kpa,
            "egt_c": state.egt_c,
            "cht_c": state.cht_c,
        }

        est = ekf.step(measurements, state, dt=1.0)
        assert isinstance(est, TwinEstimate)
        assert est.oil_temp_est > 0
        assert est.coolant_temp_est > 0
        assert len(est.egt_est) == 4
        assert len(est.cht_est) == 4
        assert 0.0 <= est.hp_ve_factor <= 2.0
        assert 0.0 <= est.hp_cooling_effectiveness <= 2.0

    def test_ekf_handles_missing_measurements(self, engine_cfg):
        ekf = ExtendedKalmanFilter(engine_cfg)
        model = PistonEngineModel(engine_cfg)
        state = model.step(EngineState(), dt=1.0, throttle=0.7, rpm=4500.0)

        # Some measurements are missing (None)
        measurements = {
            "oil_temp_c": None,
            "coolant_temp_c": state.coolant_temp_c,
            "oil_press_kpa": None,
            "egt_c": None,
            "cht_c": state.cht_c,
        }

        est = ekf.step(measurements, state, dt=1.0)
        assert isinstance(est, TwinEstimate)
        # State should still be valid numbers
        assert not np.isnan(est.oil_temp_est)
        assert not np.isnan(est.oil_press_est)

    def test_ekf_reset(self, engine_cfg):
        ekf = ExtendedKalmanFilter(engine_cfg)
        model = PistonEngineModel(engine_cfg)
        state = model.step(EngineState(), dt=1.0, throttle=0.9, rpm=5500.0)
        measurements = {
            "oil_temp_c": 120.0,
            "coolant_temp_c": 115.0,
            "oil_press_kpa": 450.0,
            "egt_c": [850.0, 850.0, 850.0, 850.0],
            "cht_c": [140.0, 140.0, 140.0, 140.0],
        }
        for _ in range(5):
            ekf.step(measurements, state, dt=1.0)

        ekf.reset()
        assert len(ekf.x) == N_STATES
        assert ekf.x[0] == 90.0


class TestDigitalTwin:
    def test_twin_update_returns_twin_state(self, sample_record):
        twin = DigitalTwin()
        result = twin.update(sample_record)

        assert isinstance(result, TwinState)
        assert isinstance(result.estimate, TwinEstimate)
        assert isinstance(result.physics_state, EngineState)
        assert 0.0 <= result.overall_health_score <= 1.0

    def test_twin_residual_generation(self, sample_record):
        twin = DigitalTwin()
        result = twin.update(sample_record)

        assert "oil_temp_c" in result.residuals
        assert "oil_temp_c" in result.normalised_residuals
        assert "oil_temp_c" in result.confidence

    def test_twin_healthy_score_near_one(self, sample_record):
        twin = DigitalTwin()
        # Feed 10 nominal records
        for _ in range(10):
            res = twin.update(sample_record)

        assert res.overall_health_score > 0.70
        assert not res.engine_fault_suspected

    def test_twin_detects_severe_sensor_bias(self, sample_record):
        twin = DigitalTwin()
        # Feed normal records first
        for _ in range(5):
            twin.update(sample_record)

        # Inject extreme bias into single sensor (oil_temp = 190 °C)
        biased_rec = sample_record.model_copy()
        biased_rec.oil_temp_c = 190.0

        for _ in range(10):
            res = twin.update(biased_rec)

        assert abs(res.normalised_residuals.get("oil_temp_c", 0.0)) > 2.0

    def test_twin_reset(self, sample_record):
        twin = DigitalTwin()
        twin.update(sample_record)
        assert twin._step_count == 1
        twin.reset()
        assert twin._step_count == 0
        assert len(twin._residual_history) == 0
