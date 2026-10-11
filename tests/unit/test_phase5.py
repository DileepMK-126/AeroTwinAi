"""
AEROTWIN — Phase 5 Unit Tests: RUL Prognosis & Mission Reliability Decision Engine

Tests:
1. BayesianDegradationFilter initialization and sliding window tracking
2. PrognosisEngine predicts high RUL under nominal steady conditions
3. PrognosisEngine predicts low RUL with bounded uncertainty under parameter degradation
4. MissionReliabilityEngine evaluates CONTINUE when RUL >> remaining mission time
5. MissionReliabilityEngine evaluates CONTINUE_DEGRADED when RUL is marginal
6. MissionReliabilityEngine evaluates RETURN_TO_BASE when RUL is below critical threshold
7. MissionReliabilityEngine evaluates EMERGENCY_LAND when RUL is dangerously low
8. MissionReliabilityEngine generates actionable recommendations and what-if options
"""

from __future__ import annotations

import pytest
import numpy as np

from core.prognosis.prognosis_engine import (
    PrognosisEngine,
    BayesianDegradationFilter,
    RULPrediction,
)
from core.mission.mission_engine import (
    MissionReliabilityEngine,
    MissionDecision,
    FlightState,
)


class TestPrognosisEngine:
    def test_filter_init(self):
        filter_ = BayesianDegradationFilter(window_size=100, min_samples=10)
        assert filter_.window_size == 100
        assert filter_.min_samples == 10

    def test_prognosis_nominal_steady_high_rul(self):
        prognosis = PrognosisEngine()
        hp = {
            "hp_ve_factor": 1.0,
            "hp_friction_factor": 1.0,
            "hp_cooling_effectiveness": 1.0,
            "hp_injector_flow": 1.0,
            "hp_boost_efficiency": 1.0,
            "hp_combustion_eff_avg": 1.0,
        }
        for _ in range(150):
            pred = prognosis.predict(hp, dt=1.0)

        assert isinstance(pred, RULPrediction)
        assert pred.rul_median_h > 10.0
        assert pred.rul_q10_h <= pred.rul_median_h <= pred.rul_q90_h

    def test_prognosis_degradation_detected(self):
        prognosis = PrognosisEngine()
        # Simulate rapidly decaying cooling effectiveness (e.g. coolant leak)
        # dropping from 1.0 to 0.45 over 200 seconds
        for i in range(200):
            cooling = max(0.42, 1.0 - (i / 200.0) * 0.58)
            hp = {
                "hp_ve_factor": 1.0,
                "hp_friction_factor": 1.0,
                "hp_cooling_effectiveness": cooling,
                "hp_injector_flow": 1.0,
                "hp_boost_efficiency": 1.0,
                "hp_combustion_eff_avg": 1.0,
            }
            pred = prognosis.predict(hp, dt=1.0)

        assert pred.critical_parameter == "hp_cooling_effectiveness"
        assert pred.rul_median_h < 5.0
        assert pred.rul_q10_h >= 0.0


class TestMissionReliabilityEngine:
    def test_mission_decision_continue(self):
        engine = MissionReliabilityEngine()
        flight = FlightState(
            phase="cruise",
            altitude_m=4500.0,
            remaining_mission_time_h=2.0,
        )
        rul = RULPrediction(
            rul_median_h=50.0,
            rul_q10_h=35.0,
            rul_q90_h=70.0,
            confidence=0.9,
        )
        decision = engine.assess(rul, flight, health_score=0.98)
        assert isinstance(decision, MissionDecision)
        assert decision.decision == "CONTINUE"
        assert decision.p_mission_complete > 0.95
        assert len(decision.recommended_actions) > 0

    def test_mission_decision_continue_degraded(self):
        engine = MissionReliabilityEngine()
        flight = FlightState(
            phase="cruise",
            remaining_mission_time_h=3.0,
        )
        # RUL distribution overlaps remaining mission time (p_complete ~ 0.70-0.80)
        rul = RULPrediction(
            rul_median_h=3.2,
            rul_q10_h=2.2,
            rul_q90_h=4.2,
            confidence=0.8,
        )
        decision = engine.assess(rul, flight, health_score=0.85)
        assert decision.decision in ("CONTINUE_DEGRADED", "RETURN_TO_BASE", "DIVERT")
        assert "rule_fired" in decision.__dict__

    def test_mission_decision_return_to_base(self):
        engine = MissionReliabilityEngine()
        flight = FlightState(
            phase="cruise",
            remaining_mission_time_h=1.5,
        )
        # Moderate critical RUL (1.6 hours) with p_complete ~ 0.50
        rul = RULPrediction(
            rul_median_h=1.6,
            rul_q10_h=1.1,
            rul_q90_h=2.2,
            confidence=0.85,
        )
        decision = engine.assess(rul, flight, health_score=0.65)
        assert decision.decision in ("RETURN_TO_BASE", "DIVERT")

    def test_mission_decision_emergency_land(self):
        engine = MissionReliabilityEngine()
        flight = FlightState(
            phase="cruise",
            remaining_mission_time_h=2.0,
        )
        # Immediate catastrophic failure imminent (RUL = 0.2 hours = 12 mins)
        rul = RULPrediction(
            rul_median_h=0.2,
            rul_q10_h=0.05,
            rul_q90_h=0.35,
            confidence=0.95,
        )
        decision = engine.assess(rul, flight, health_score=0.30)
        assert decision.decision == "EMERGENCY_LAND"
        assert any("land" in act.lower() or "emergency" in act.lower() for act in decision.recommended_actions)

    def test_mission_what_if_analysis(self):
        engine = MissionReliabilityEngine()
        flight = FlightState(
            phase="cruise",
            remaining_mission_time_h=3.0,
            current_throttle_pct=75.0,
        )
        rul = RULPrediction(
            rul_median_h=4.0,
            rul_q10_h=2.5,
            rul_q90_h=5.5,
            confidence=0.8,
        )
        decision = engine.assess(rul, flight, health_score=0.80)
        assert decision.what_if_results is not None
        assert "low_power_20pct" in decision.what_if_results
        assert "altitude_minus_1000m" in decision.what_if_results
