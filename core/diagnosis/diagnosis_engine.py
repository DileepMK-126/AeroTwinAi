"""
AEROTWIN — Phase 4: Fault Diagnosis Engine

Physics-based fault tree (Level A — no training required).
Produces fault probability distribution and natural-language explanations.

Future: learned 1D-CNN classifier loaded from registry (Level B).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aerotwin.diagnosis")


# ---------------------------------------------------------------------------
# Diagnosis Result
# ---------------------------------------------------------------------------

@dataclass
class DiagnosisResult:
    """Output of the diagnosis engine."""
    # Probability distribution over fault classes
    fault_probabilities: Dict[str, float] = field(default_factory=dict)
    # Top fault candidate
    primary_fault: Optional[str] = None
    primary_confidence: float = 0.0
    # Whether this might be a novel/unknown fault
    ood_score: float = 0.0
    # Explainability
    contributing_channels: List[str] = field(default_factory=list)
    explanation: str = ""
    recommended_action: str = ""
    # Sensor vs engine fault verdict
    is_sensor_fault: bool = False
    sensor_fault_channels: List[str] = field(default_factory=list)
    # Detection source
    diagnosis_method: str = "physics_fault_tree"


# ---------------------------------------------------------------------------
# Physics-Based Fault Tree
# ---------------------------------------------------------------------------

FAULT_SIGNATURES = {
    "bearing_wear": {
        "primary": ["vib_rms_g", "oil_press_kpa"],
        "secondary": ["oil_temp_c"],
        "signs": {
            "vib_rms_g": +1,      # +1 = should increase, -1 = should decrease
            "oil_press_kpa": -1,
            "oil_temp_c": +1,
        },
        "multi_channel_threshold": 2,   # need N channels to be consistent
        "maintenance": "Inspect and replace main/rod bearings. Check oil quality.",
    },
    "coolant_loss": {
        "primary": ["cht_c_1", "cht_c_2", "cht_c_3", "cht_c_4", "coolant_temp_c"],
        "secondary": [],
        "signs": {
            "cht_c_1": +1, "cht_c_2": +1, "cht_c_3": +1, "cht_c_4": +1,
            "coolant_temp_c": +1,
        },
        "multi_channel_threshold": 3,
        "maintenance": "Immediate power reduction if CHT > 150°C. Inspect coolant system.",
    },
    "oil_starvation": {
        "primary": ["oil_press_kpa", "oil_temp_c"],
        "secondary": ["vib_rms_g"],
        "signs": {
            "oil_press_kpa": -1,
            "oil_temp_c": +1,
            "vib_rms_g": +1,
        },
        "multi_channel_threshold": 1,
        "maintenance": "IMMEDIATE: Reduce power. Land ASAP. Engine failure imminent.",
    },
    "spark_plug_degradation": {
        "primary": ["egt_c_1", "egt_c_2", "egt_c_3", "egt_c_4"],
        "secondary": ["cht_c_1", "cht_c_2", "cht_c_3", "cht_c_4"],
        "signs": {
            "egt_c_1": +1, "egt_c_2": +1, "egt_c_3": +1, "egt_c_4": +1,
        },
        "multi_channel_threshold": 1,
        "cylinder_specific": True,
        "maintenance": "Replace spark plug in affected cylinder(s).",
    },
    "turbocharger_degradation": {
        "primary": ["boost_kpa", "map_kpa"],
        "secondary": ["egt_c_1", "egt_c_2", "egt_c_3", "egt_c_4"],
        "signs": {
            "boost_kpa": -1,
            "map_kpa": -1,
            "egt_c_1": +1, "egt_c_2": +1, "egt_c_3": +1, "egt_c_4": +1,
        },
        "multi_channel_threshold": 2,
        "maintenance": "Inspect turbocharger. Check for oil leaks in turbo bearing.",
    },
    "injector_clogging": {
        "primary": ["egt_c_1", "egt_c_2", "egt_c_3", "egt_c_4", "fuel_flow_lph"],
        "secondary": [],
        "signs": {
            "egt_c_1": -1, "egt_c_2": -1, "egt_c_3": -1, "egt_c_4": -1,  # lean → lower EGT
            "fuel_flow_lph": -1,
        },
        "multi_channel_threshold": 2,
        "cylinder_specific": True,
        "maintenance": "Clean or replace injector in affected cylinder.",
    },
    "propeller_imbalance": {
        "primary": ["vib_rms_g"],
        "secondary": [],
        "signs": {"vib_rms_g": +1},
        "multi_channel_threshold": 1,
        "maintenance": "Inspect propeller for damage. Balance or replace.",
    },
}


class PhysicsFaultTree:
    """
    Rule-based fault tree using residual patterns.
    Each fault has expected signatures (which channels should be anomalous
    and in which direction). Probability is computed as pattern match score.
    """

    def __init__(self, faults_config: Optional[Dict[str, Any]] = None) -> None:
        self._faults_config = faults_config
        self._fault_signatures = FAULT_SIGNATURES

    def diagnose(
        self,
        normalised_residuals: Dict[str, float],
        sensor_fault_channels: List[str],
    ) -> DiagnosisResult:
        """
        Match normalised residuals against fault signatures.
        Returns fault probability distribution.
        """
        if not normalised_residuals:
            return DiagnosisResult(explanation="No residual data available")

        # Filter out sensor-faulty channels from diagnosis
        clean_residuals = {
            ch: nr for ch, nr in normalised_residuals.items()
            if ch not in sensor_fault_channels
        }

        # If mostly sensor faults or only sensor-fault channels anomalous — flag as sensor fault
        if sensor_fault_channels and (
            len(sensor_fault_channels) >= len(normalised_residuals) * 0.6
            or (
                all(abs(clean_residuals.get(ch, 0.0)) <= 1.5 for ch in clean_residuals)
                and any(abs(normalised_residuals.get(ch, 0.0)) > 2.0 for ch in sensor_fault_channels)
            )
        ):
            return DiagnosisResult(
                is_sensor_fault=True,
                sensor_fault_channels=sensor_fault_channels,
                primary_fault="sensor_fault",
                primary_confidence=0.95,
                explanation=f"Sensor fault suspected on channels: {', '.join(sensor_fault_channels)}",
                recommended_action="Check sensor wiring and calibration. Do not act on engine maintenance until sensor is verified.",
                diagnosis_method="physics_fault_tree",
            )

        # Score each fault
        scores: Dict[str, float] = {}
        for fault_id, sig in self._fault_signatures.items():
            score = self._score_fault(fault_id, sig, clean_residuals)
            scores[fault_id] = score

        # Normalise scores to probabilities
        scores["no_fault"] = max(0.0, 1.0 - max(scores.values(), default=0.0))
        total = sum(scores.values()) + 1e-9
        probs = {k: v / total for k, v in scores.items()}

        # Primary fault
        primary_fault = max(probs, key=lambda k: probs[k])
        primary_confidence = probs[primary_fault]

        # OOD score: residuals that don't match any known fault
        max_residual = max(abs(nr) for nr in clean_residuals.values()) if clean_residuals else 0.0
        ood_score = max(0.0, max_residual / 5.0 - max(scores.values()))

        # Contributing channels
        contributing = [
            ch for ch, nr in clean_residuals.items() if abs(nr) > 2.0
        ]

        # Natural language explanation
        explanation, action = self._explain(
            primary_fault if primary_confidence > 0.3 else "unknown",
            clean_residuals,
            contributing,
        )

        return DiagnosisResult(
            fault_probabilities=probs,
            primary_fault=primary_fault if primary_confidence > 0.25 else None,
            primary_confidence=primary_confidence,
            ood_score=min(1.0, ood_score),
            contributing_channels=contributing,
            explanation=explanation,
            recommended_action=action,
            is_sensor_fault=False,
            sensor_fault_channels=sensor_fault_channels,
            diagnosis_method="physics_fault_tree",
        )

    def _score_fault(
        self, fault_id: str, sig: dict, residuals: Dict[str, float]
    ) -> float:
        """Score how well a fault signature matches current residuals."""
        primary_channels = sig["primary"]
        secondary_channels = sig.get("secondary", [])
        signs = sig["signs"]
        threshold = sig.get("multi_channel_threshold", 1)

        matches = 0
        score = 0.0
        for ch in primary_channels:
            nr = residuals.get(ch, 0.0)
            expected_sign = signs.get(ch, 0)
            if expected_sign != 0 and abs(nr) > 1.5:
                if (nr > 0) == (expected_sign > 0):
                    matches += 1
                    score += abs(nr)

        if matches < threshold:
            return 0.0

        # Secondary channel support
        for ch in secondary_channels:
            nr = residuals.get(ch, 0.0)
            expected_sign = signs.get(ch, 0)
            if expected_sign != 0 and abs(nr) > 1.5 and ((nr > 0) == (expected_sign > 0)):
                score += 0.5 * abs(nr)
                matches += 1

        # Penalise single-channel faults if other unrelated engine channels are anomalous
        if len(primary_channels) == 1:
            other_anomalies = sum(
                1 for ch, nr in residuals.items() if ch not in primary_channels and abs(nr) > 2.0
            )
            if other_anomalies >= 1:
                score *= 0.4

        corroboration = 1.0 + 0.4 * (matches - 1)
        base = score / max(len(primary_channels), 1)
        return base * corroboration

    def _explain(
        self, fault_id: str, residuals: Dict[str, float], channels: List[str]
    ) -> Tuple[str, str]:
        """Generate natural-language explanation."""
        if fault_id == "no_fault" or fault_id == "unknown":
            if channels:
                return (
                    f"Anomalous readings on {', '.join(channels[:3])} but no clear fault pattern matched. "
                    "Monitor closely and investigate if condition persists.",
                    "Monitor all parameters. Run diagnostics at next maintenance.",
                )
            return ("No anomalies detected. Engine operating normally.", "Continue normal operation.")

        sig = self._fault_signatures.get(fault_id, {})
        primary = sig.get("primary", [])
        maintenance = sig.get("maintenance", "Follow standard maintenance procedure.")

        # Format channel deviations
        channel_descriptions = []
        for ch in channels[:4]:
            nr = residuals.get(ch, 0.0)
            direction = "elevated" if nr > 0 else "depressed"
            channel_descriptions.append(f"{ch} {direction} by {abs(nr):.1f}σ vs twin prediction")

        channel_text = "; ".join(channel_descriptions) if channel_descriptions else "multiple channels deviated"
        explanation = (
            f"Fault signature consistent with '{fault_id.replace('_', ' ').title()}'. "
            f"{channel_text}. "
            f"Confidence: {len(channels)}/{len(primary)} primary channels anomalous."
        )
        return explanation, maintenance


# ---------------------------------------------------------------------------
# Diagnosis Engine (with registry integration)
# ---------------------------------------------------------------------------

class DiagnosisEngine:
    """
    Diagnosis engine that uses the fault tree as primary detector,
    with optional learned classifier from registry.
    """

    def __init__(self, faults_config: Optional[Dict[str, Any]] = None) -> None:
        self.fault_tree = PhysicsFaultTree(faults_config)
        self._learned_classifier = None
        try:
            from core.registry import get_model_registry
            self._learned_classifier = get_model_registry().load_model("alfa_engine_failure_classifier")
        except Exception as e:
            logger.debug("Registry classifier not loaded: %s", e)

    def diagnose(
        self,
        normalised_residuals: Dict[str, float],
        sensor_fault_channels: List[str],
    ) -> DiagnosisResult:
        """Run diagnosis and return result."""
        # Primary: physics fault tree
        result = self.fault_tree.diagnose(normalised_residuals, sensor_fault_channels)
        return result
