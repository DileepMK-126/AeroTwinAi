"""
AEROTWIN — Phase 4 Unit Tests: Anomaly Detection & Fault Diagnosis

Tests:
1. ThresholdDetector triggers warning and critical alerts on residual excursions
2. EWMADetector detects gradual sustained parameter drift
3. CUSUMDetector detects step changes and shifts
4. DetectionEngine merges multi-detector outputs
5. DiagnosisEngine correctly classifies healthy conditions
6. DiagnosisEngine identifies coolant loss signature
7. DiagnosisEngine identifies oil starvation signature
8. DiagnosisEngine identifies bearing wear signature
9. DiagnosisEngine isolates sensor-only faults from actual engine faults
"""

from __future__ import annotations

import pytest
from core.detection.detection_engine import (
    DetectionEngine,
    DetectionAlert,
    ThresholdDetector,
    EWMADetector,
    CUSUMDetector,
)
from core.diagnosis.diagnosis_engine import DiagnosisEngine, DiagnosisResult


class TestDetectionLayerA:
    def test_threshold_detector_normal(self):
        detector = ThresholdDetector(warning_sigma=2.5, critical_sigma=4.0)
        residuals = {"oil_temp_c": 0.5, "coolant_temp_c": -1.0}
        alerts = detector.update(residuals)
        assert len(alerts) == 0

    def test_threshold_detector_warning(self):
        detector = ThresholdDetector(warning_sigma=2.5, critical_sigma=4.0)
        residuals = {"oil_temp_c": 3.0}
        alerts = detector.update(residuals)
        assert len(alerts) == 1
        assert alerts[0].severity == "warning"
        assert alerts[0].score == 3.0

    def test_threshold_detector_critical(self):
        detector = ThresholdDetector(warning_sigma=2.5, critical_sigma=4.0)
        residuals = {"oil_press_kpa": -5.2}
        alerts = detector.update(residuals)
        assert len(alerts) == 1
        assert alerts[0].severity == "critical"
        assert alerts[0].score == 5.2

    def test_ewma_detector_drift(self):
        ewma = EWMADetector(lambda_=0.2, L=3.0)
        # Feed nominal data
        for _ in range(10):
            alerts = ewma.update({"cht_c_1": 0.1})
            assert len(alerts) == 0

        # Feed sustained high residual
        drift_alerts = []
        for _ in range(15):
            alerts = ewma.update({"cht_c_1": 3.5})
            drift_alerts.extend(alerts)

        assert len(drift_alerts) > 0
        assert drift_alerts[-1].detector == "ewma"

    def test_cusum_detector_shift(self):
        cusum = CUSUMDetector(k_slack=0.5, h=4.0)
        for _ in range(5):
            alerts = cusum.update({"vib_rms_g": 0.0})
            assert len(alerts) == 0

        # Persistent step shift
        shift_alerts = []
        for _ in range(10):
            alerts = cusum.update({"vib_rms_g": 2.0})
            shift_alerts.extend(alerts)

        assert len(shift_alerts) > 0
        assert shift_alerts[-1].detector == "cusum"


class TestDetectionEngine:
    def test_detection_engine_nominal(self):
        engine = DetectionEngine()
        residuals = {
            "oil_temp_c": 0.2,
            "coolant_temp_c": -0.1,
            "oil_press_kpa": 0.3,
            "cht_c_1": 0.0,
        }
        alerts = engine.update(residuals)
        assert len(alerts) == 0

    def test_detection_engine_multi_alert_on_anomaly(self):
        engine = DetectionEngine()
        residuals = {
            "oil_press_kpa": -4.5,  # severe drop
        }
        alerts = engine.update(residuals)
        assert len(alerts) >= 1
        assert any(a.severity == "critical" for a in alerts)


class TestDiagnosisEngine:
    def test_diagnose_healthy_engine(self):
        diag = DiagnosisEngine()
        residuals = {
            "oil_temp_c": 0.1,
            "coolant_temp_c": -0.2,
            "oil_press_kpa": 0.0,
            "vib_rms_g": 0.1,
        }
        result = diag.diagnose(residuals, sensor_fault_channels=[])
        assert isinstance(result, DiagnosisResult)
        assert not result.is_sensor_fault
        assert result.primary_fault in (None, "no_fault")

    def test_diagnose_coolant_loss(self):
        diag = DiagnosisEngine()
        # Elevated CHT across cylinders and elevated coolant temp
        residuals = {
            "cht_c_1": 3.8,
            "cht_c_2": 4.1,
            "cht_c_3": 3.5,
            "cht_c_4": 3.9,
            "coolant_temp_c": 3.2,
            "oil_press_kpa": 0.1,
        }
        result = diag.diagnose(residuals, sensor_fault_channels=[])
        assert result.primary_fault == "coolant_loss"
        assert result.primary_confidence > 0.5
        assert "Coolant Loss" in result.explanation

    def test_diagnose_oil_starvation(self):
        diag = DiagnosisEngine()
        # Severely depressed oil pressure, strongly elevated oil temp, secondary vib
        residuals = {
            "oil_press_kpa": -5.5,
            "oil_temp_c": 3.8,
            "vib_rms_g": 1.8,
        }
        result = diag.diagnose(residuals, sensor_fault_channels=[])
        assert result.primary_fault == "oil_starvation"
        assert result.primary_confidence > 0.35
        assert "Oil Starvation" in result.explanation

    def test_diagnose_bearing_wear(self):
        diag = DiagnosisEngine()
        # High vibration and reduced oil pressure
        residuals = {
            "vib_rms_g": 3.5,
            "oil_press_kpa": -2.2,
            "oil_temp_c": 2.0,
        }
        result = diag.diagnose(residuals, sensor_fault_channels=[])
        assert result.primary_fault == "bearing_wear"
        assert result.primary_confidence > 0.4

    def test_diagnose_sensor_fault_isolated(self):
        diag = DiagnosisEngine()
        # Single sensor glitch on oil_temp flagged by twin
        residuals = {
            "oil_temp_c": 6.0,  # anomalous single channel
            "oil_press_kpa": 0.0,
            "coolant_temp_c": 0.0,
        }
        result = diag.diagnose(residuals, sensor_fault_channels=["oil_temp_c"])
        assert result.is_sensor_fault
        assert "oil_temp_c" in result.sensor_fault_channels
        assert result.primary_fault == "sensor_fault"
