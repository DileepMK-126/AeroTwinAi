"""
AEROTWIN — Phase 5: Mission Reliability Engine

Converts health states + RUL distributions + flight state into:
1. Probability of mission completion
2. Time-to-failure hazard
3. Decision: CONTINUE | CONTINUE_DEGRADED | DIVERT | RETURN_TO_BASE | EMERGENCY_LAND

Uses Monte Carlo over RUL/degradation uncertainty.
Every recommendation shows its inputs, probabilities, and the rule that fired.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.config_loader import load_mission_config
from core.prognosis.prognosis_engine import RULPrediction

logger = logging.getLogger("aerotwin.mission")


# ---------------------------------------------------------------------------
# Decision Enum
# ---------------------------------------------------------------------------

DECISIONS = [
    "CONTINUE",
    "CONTINUE_DEGRADED",
    "DIVERT",
    "RETURN_TO_BASE",
    "EMERGENCY_LAND",
]


# ---------------------------------------------------------------------------
# Mission Decision Result
# ---------------------------------------------------------------------------

@dataclass
class MissionDecision:
    """Output of the mission reliability engine."""
    decision: str                           # one of DECISIONS
    p_mission_complete: float               # probability of completing mission
    time_to_failure_h: float               # expected time to failure
    rul_median_h: float
    rul_interval_80: Tuple[float, float]
    remaining_mission_time_h: float        # planned time left to complete mission

    # Explanation
    rule_fired: str = ""
    inputs_summary: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)

    # What-if mode outputs
    what_if_results: Optional[Dict[str, "MissionDecision"]] = None

    # Fleet readiness (pre-flight go/no-go)
    pre_flight_ready: bool = True
    readiness_score: float = 1.0


@dataclass
class FlightState:
    """Current flight state for mission planning."""
    phase: str = "cruise"
    altitude_m: float = 4500.0
    airspeed_mps: float = 55.0
    fuel_remaining_kg: float = 60.0
    remaining_mission_distance_km: float = 0.0
    remaining_mission_time_h: float = 4.0
    current_throttle_pct: float = 70.0


# ---------------------------------------------------------------------------
# Mission Reliability Engine
# ---------------------------------------------------------------------------

class MissionReliabilityEngine:
    """
    Monte Carlo mission reliability engine.
    """

    def __init__(
        self,
        mission_config: Optional[Dict[str, Any]] = None,
        n_mc_samples: int = 1000,
        seed: int = 42,
    ) -> None:
        raw = mission_config or load_mission_config()
        self._config = raw
        self._risk = raw.get("risk_thresholds", {})
        self._decisions_cfg = raw.get("decisions", {})
        self._alternates = raw.get("alternates", [])
        self._glide = raw.get("engine_out", {})
        self.n_mc = n_mc_samples
        self._rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def assess(
        self,
        rul_prediction: RULPrediction,
        flight_state: FlightState,
        health_score: float = 1.0,
        compute_what_if: bool = True,
    ) -> MissionDecision:
        """
        Assess mission reliability and produce a decision.

        Args:
            rul_prediction: RUL estimate with uncertainty from prognosis engine.
            flight_state: Current flight state.
            health_score: Overall engine health score (0–1).
            compute_what_if: Whether to recursively compute what-if variations.

        Returns:
            MissionDecision with probability, decision, and rule trace.
        """
        rul_h = rul_prediction.rul_median_h
        rul_std = (rul_prediction.rul_q90_h - rul_prediction.rul_q10_h) / 2.56  # approx std

        # --- Monte Carlo over RUL uncertainty ---
        if rul_std > 0:
            rul_samples = self._rng.normal(
                loc=rul_h, scale=rul_std, size=self.n_mc
            )
            rul_samples = np.clip(rul_samples, 0.0, 10000.0)
        else:
            rul_samples = np.full(self.n_mc, rul_h)

        remaining_h = flight_state.remaining_mission_time_h

        # P(complete mission) = P(RUL > remaining_mission_time)
        p_complete = float(np.mean(rul_samples > remaining_h))

        # --- Select decision based on thresholds ---
        decision, rule = self._select_decision(
            p_complete=p_complete,
            rul_h=rul_h,
            health_score=health_score,
            flight_state=flight_state,
        )

        # --- Recommended actions ---
        actions = self._get_recommended_actions(decision, rul_h, flight_state)

        # --- What-if analysis (reduced power / altitude) ---
        what_if = self._compute_what_if(rul_prediction, flight_state, health_score) if compute_what_if else None

        # --- Pre-flight readiness (for fleet view) ---
        readiness = self._compute_readiness(rul_h, health_score)

        return MissionDecision(
            decision=decision,
            p_mission_complete=p_complete,
            time_to_failure_h=rul_h,
            rul_median_h=rul_prediction.rul_median_h,
            rul_interval_80=rul_prediction.rul_interval_80,
            remaining_mission_time_h=remaining_h,
            rule_fired=rule,
            inputs_summary={
                "rul_median_h": rul_h,
                "rul_q10_h": rul_prediction.rul_q10_h,
                "rul_q90_h": rul_prediction.rul_q90_h,
                "p_complete": p_complete,
                "health_score": health_score,
                "remaining_mission_h": remaining_h,
                "phase": flight_state.phase,
                "altitude_m": flight_state.altitude_m,
                "critical_parameter": rul_prediction.critical_parameter,
            },
            recommended_actions=actions,
            what_if_results=what_if,
            pre_flight_ready=readiness >= 0.90,
            readiness_score=readiness,
        )

    def _select_decision(
        self,
        p_complete: float,
        rul_h: float,
        health_score: float,
        flight_state: FlightState,
    ) -> Tuple[str, str]:
        """Apply decision rules and return (decision, rule_description)."""
        risk = self._risk
        p_crit = risk.get("p_mission_complete_critical", 0.6)
        p_min = risk.get("p_mission_complete_min", 0.85)
        p_emergency = risk.get("p_mission_complete_emergency", 0.3)
        rul_emergency = risk.get("rul_hours_emergency", 0.5)
        rul_critical = risk.get("rul_hours_critical", 2.0)
        # Acute safety risks from low engine health score or short RUL
        if rul_h <= rul_emergency or p_complete <= p_emergency or health_score <= 0.35:
            return (
                "EMERGENCY_LAND",
                f"RUL <= {rul_emergency}h ({rul_h:.2f}h) or P(complete) <= {p_emergency} ({p_complete:.2f}) or health_score <= 0.35 ({health_score:.2f})"
            )

        if rul_h <= rul_critical or p_complete <= p_crit or health_score <= 0.65:
            return (
                "RETURN_TO_BASE",
                f"RUL <= {rul_critical}h ({rul_h:.2f}h) or P(complete) <= {p_crit} ({p_complete:.2f}) or health_score <= 0.65 ({health_score:.2f})"
            )

        if p_complete < p_min or health_score < 0.85:
            return (
                "CONTINUE_DEGRADED",
                f"P(complete)={p_complete:.2f} < threshold {p_min} or health_score={health_score:.2f}: reduce power 15%, altitude -500m"
            )

        if p_complete < 0.92:
            return (
                "DIVERT",
                f"P(complete)={p_complete:.2f} < 0.92: proceed to nearest alternate"
            )

        return (
            "CONTINUE",
            f"P(complete)={p_complete:.2f} >= {p_min}, RUL={rul_h:.1f}h, health={health_score:.2f}. All parameters within limits."
        )

    def _get_recommended_actions(
        self, decision: str, rul_h: float, flight_state: FlightState
    ) -> List[str]:
        actions = {
            "CONTINUE": [
                "Monitor all engine parameters.",
                f"Schedule maintenance if RUL drops below {self._risk.get('rul_hours_warning', 5)}h.",
            ],
            "CONTINUE_DEGRADED": [
                "Reduce throttle by 15%.",
                "Descend 500m below current altitude.",
                "Alert ground control of degraded status.",
                "Prepare alternate landing plan.",
            ],
            "DIVERT": [
                "Set course for nearest alternate immediately.",
                "Alert ground control of divert.",
                "Monitor engine parameters closely.",
            ],
            "RETURN_TO_BASE": [
                "ABORT MISSION. Return to base immediately.",
                "Alert ground control of RTB.",
                "Reduce to minimum safe power setting.",
                "Prepare for engine-out procedure.",
            ],
            "EMERGENCY_LAND": [
                "DECLARE EMERGENCY.",
                "Land at nearest available site IMMEDIATELY.",
                "Alert ATC/ground control of emergency.",
                "Prepare for possible engine failure.",
            ],
        }
        return actions.get(decision, ["Follow standard operating procedures."])

    def _compute_what_if(
        self,
        rul: RULPrediction,
        fs: FlightState,
        health: float,
    ) -> Dict[str, "MissionDecision"]:
        """Compute what-if scenarios (reduced power, reduced altitude)."""
        results = {}

        # What-if: reduce throttle by 20%
        fs_low_power = FlightState(
            phase=fs.phase,
            altitude_m=fs.altitude_m,
            airspeed_mps=fs.airspeed_mps,
            fuel_remaining_kg=fs.fuel_remaining_kg,
            remaining_mission_time_h=fs.remaining_mission_time_h * 1.1,  # slower = more time
            current_throttle_pct=fs.current_throttle_pct * 0.80,
        )
        results["low_power_20pct"] = self.assess(rul, fs_low_power, health, compute_what_if=False)

        # What-if: reduce altitude by 1000m
        fs_low_alt = FlightState(
            phase=fs.phase,
            altitude_m=max(500.0, fs.altitude_m - 1000.0),
            airspeed_mps=fs.airspeed_mps,
            fuel_remaining_kg=fs.fuel_remaining_kg,
            remaining_mission_time_h=fs.remaining_mission_time_h,
            current_throttle_pct=fs.current_throttle_pct,
        )
        results["altitude_minus_1000m"] = self.assess(rul, fs_low_alt, health, compute_what_if=False)

        return results

    def _compute_readiness(self, rul_h: float, health_score: float) -> float:
        """Pre-flight go/no-go readiness score (0–1)."""
        # RUL must cover at least 1 full mission
        mission_h = self._config.get("mission", {}).get("default_profile", {}).get("max_duration_h", 12.0)
        rul_factor = min(1.0, rul_h / (mission_h * 2))  # comfortable margin
        health_factor = health_score
        return 0.5 * rul_factor + 0.5 * health_factor
