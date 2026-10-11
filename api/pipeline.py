"""
AEROTWIN — Complete Pipeline

Ties together all phases:
  TelemetryRecord → DigitalTwin → DetectionEngine → DiagnosisEngine → PrognosisEngine → MissionEngine

Can be used standalone or via the API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.detection.detection_engine import DetectionAlert, DetectionEngine
from core.diagnosis.diagnosis_engine import DiagnosisResult, DiagnosisEngine
from core.mission.mission_engine import FlightState, MissionDecision, MissionReliabilityEngine
from core.prognosis.prognosis_engine import PrognosisEngine, RULPrediction
from core.twin.twin import DigitalTwin, TwinState
from data.schemas import TelemetryRecord
from storage.storage import get_sqlite

logger = logging.getLogger("aerotwin.pipeline")


@dataclass
class PipelineOutput:
    """Full pipeline output for one telemetry record."""
    record: TelemetryRecord
    twin_state: TwinState
    alerts: List[DetectionAlert]
    diagnosis: DiagnosisResult
    rul: RULPrediction
    decision: MissionDecision
    step: int = 0

    def to_ws_message(self) -> Dict[str, Any]:
        """Serialise for WebSocket broadcast."""
        est = self.twin_state.estimate
        return {
            "type": "telemetry_update",
            "step": self.step,
            "timestamp": self.record.timestamp.isoformat(),
            "engine_id": self.record.engine_id,
            "source": self.record.source if isinstance(self.record.source, str) else self.record.source.value,
            "flight_phase": self.record.flight_phase,
            "altitude_m": self.record.altitude_m,
            "rpm": self.record.rpm,
            "throttle_pct": self.record.throttle_pct,
            "map_kpa": self.record.map_kpa,
            "boost_kpa": self.record.boost_kpa,
            "fuel_flow_lph": self.record.fuel_flow_lph,
            "cht_c": self.record.cht_c,
            "egt_c": self.record.egt_c,
            "oil_temp_c": self.record.oil_temp_c,
            "oil_press_kpa": self.record.oil_press_kpa,
            "coolant_temp_c": self.record.coolant_temp_c,
            "vib_rms_g": self.record.vib_rms_g,
            # Twin estimates
            "twin": {
                "egt_est": est.egt_est,
                "cht_est": est.cht_est,
                "oil_temp_est": est.oil_temp_est,
                "oil_press_est": est.oil_press_est,
                "hp_ve_factor": est.hp_ve_factor,
                "hp_friction_factor": est.hp_friction_factor,
                "hp_cooling_effectiveness": est.hp_cooling_effectiveness,
                "hp_injector_flow": est.hp_injector_flow,
                "hp_boost_efficiency": est.hp_boost_efficiency,
                "hp_combustion_eff_avg": est.hp_combustion_eff_avg,
                "health_score": self.twin_state.overall_health_score,
            },
            "residuals": self.twin_state.normalised_residuals,
            # Detection
            "alerts": [
                {
                    "detector": a.detector,
                    "level": a.level,
                    "severity": a.severity,
                    "channels": a.channels,
                    "message": a.message,
                    "score": a.score,
                }
                for a in self.alerts
            ],
            # Diagnosis
            "diagnosis": {
                "primary_fault": self.diagnosis.primary_fault,
                "confidence": self.diagnosis.primary_confidence,
                "explanation": self.diagnosis.explanation,
                "recommended_action": self.diagnosis.recommended_action,
                "is_sensor_fault": self.diagnosis.is_sensor_fault,
                "fault_probabilities": dict(list(
                    sorted(self.diagnosis.fault_probabilities.items(), key=lambda x: -x[1])
                )[:5]),  # top 5
            },
            # Prognosis
            "prognosis": {
                "rul_median_h": self.rul.rul_median_h,
                "rul_q10_h": self.rul.rul_q10_h,
                "rul_q90_h": self.rul.rul_q90_h,
                "critical_parameter": self.rul.critical_parameter,
                "confidence": self.rul.confidence,
            },
            # Mission
            "mission": {
                "decision": self.decision.decision,
                "p_mission_complete": self.decision.p_mission_complete,
                "rule_fired": self.decision.rule_fired,
                "readiness_score": self.decision.readiness_score,
                "recommended_actions": self.decision.recommended_actions[:2],
            },
        }


class AerotwinPipeline:
    """
    Full end-to-end AEROTWIN pipeline.
    Stateful — call process(record) at each timestep.
    """

    def __init__(self, dt: float = 1.0, persist: bool = True) -> None:
        self.dt = dt
        self.persist = persist
        self.twin = DigitalTwin(dt=dt)
        self.detection = DetectionEngine()
        self.diagnosis = DiagnosisEngine()
        self.prognosis = PrognosisEngine()
        self.mission = MissionReliabilityEngine()
        self._step = 0
        self._flight_state = FlightState(remaining_mission_time_h=8.0)

    def reset(self) -> None:
        """Reset internal pipeline state."""
        self._step = 0
        self.twin = DigitalTwin(dt=self.dt)
        self.detection = DetectionEngine()
        self.diagnosis = DiagnosisEngine()
        self.prognosis = PrognosisEngine()
        self.mission = MissionReliabilityEngine()
        self._flight_state = FlightState(remaining_mission_time_h=8.0)


    def process(self, record: TelemetryRecord) -> PipelineOutput:
        """Process one telemetry record through the full pipeline."""
        self._step += 1

        # 1. Twin state estimation
        twin_state = self.twin.update(record)

        # 2. Detection
        alerts = self.detection.update(twin_state.normalised_residuals)

        # 3. Diagnosis
        diagnosis = self.diagnosis.diagnose(
            twin_state.normalised_residuals,
            twin_state.sensor_fault_channels,
        )

        # 4. Prognosis
        hp_dict = self.prognosis.hp_from_twin_estimate(twin_state.estimate)
        rul = self.prognosis.predict(hp_dict, dt=self.dt)

        # 5. Mission reliability
        if record.flight_phase:
            phase_val = record.flight_phase if isinstance(record.flight_phase, str) else record.flight_phase.value
            self._flight_state.phase = phase_val
        if record.altitude_m:
            self._flight_state.altitude_m = record.altitude_m
        if record.airspeed_mps:
            self._flight_state.airspeed_mps = record.airspeed_mps

        decision = self.mission.assess(rul, self._flight_state, twin_state.overall_health_score)

        output = PipelineOutput(
            record=record,
            twin_state=twin_state,
            alerts=alerts,
            diagnosis=diagnosis,
            rul=rul,
            decision=decision,
            step=self._step,
        )

        # 6. Persist critical alerts to SQLite
        if self.persist and alerts:
            db = get_sqlite()
            for alert in alerts:
                if alert.severity in ("warning", "critical"):
                    db.insert_alert(
                        engine_id=record.engine_id,
                        alert_type=f"detection_{alert.level}",
                        severity=alert.severity,
                        message=alert.message,
                        fault_id=diagnosis.primary_fault,
                        confidence=diagnosis.primary_confidence,
                        details={"channels": alert.channels, "score": alert.score},
                    )

        return output
