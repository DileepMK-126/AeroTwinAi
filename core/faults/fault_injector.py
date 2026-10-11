"""
AEROTWIN — Phase 2: Fault Injector

Drives faults into the engine state based on configs/faults.yaml.
Each fault has an onset time, severity ramp type, and affected health parameters.

Severity ramp types:
  abrupt     — step change at onset
  linear     — linear ramp from 0 to 1 over duration
  exponential — exponential ramp (fast initial, slow later)
  intermittent — random on/off with increasing duty cycle
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from configs.config_loader import load_faults_config
from core.physics.engine_model import EngineState


# ---------------------------------------------------------------------------
# Active Fault Descriptor
# ---------------------------------------------------------------------------

@dataclass
class ActiveFault:
    """Describes a single active fault instance."""
    fault_id: str
    onset_time_s: float             # simulation time when fault begins
    failure_time_s: float           # when severity reaches 1.0 (functional failure)
    ramp_type: str = "linear"       # abrupt | linear | exponential | intermittent
    cylinder_index: Optional[int] = None   # None = all cylinders
    sensor_channel: Optional[str] = None  # for sensor faults
    sensor_bias_value: float = 0.0  # for sensor_bias fault
    intermittent_period_s: float = 30.0
    intermittent_duty: float = 0.5  # 0–1

    def severity(self, t_s: float) -> float:
        """Return current fault severity [0, 1] at simulation time t_s."""
        if t_s < self.onset_time_s:
            return 0.0
        duration = max(self.failure_time_s - self.onset_time_s, 1.0)
        elapsed = t_s - self.onset_time_s
        progress = min(elapsed / duration, 1.0)

        if self.ramp_type == "abrupt":
            return 1.0 if progress > 0 else 0.0
        elif self.ramp_type == "linear":
            return progress
        elif self.ramp_type == "exponential":
            return 1.0 - math.exp(-5.0 * progress)
        elif self.ramp_type == "intermittent":
            # Intermittent: duty cycle increases with progress
            duty = self.intermittent_duty * (0.5 + 0.5 * progress)
            phase = (t_s % self.intermittent_period_s) / self.intermittent_period_s
            return progress if phase < duty else 0.0
        return progress

    def true_rul_hours(self, t_s: float) -> Optional[float]:
        """True remaining useful life in hours (None for sensor-only faults)."""
        if self.failure_time_s is None:
            return None
        remaining_s = max(0.0, self.failure_time_s - t_s)
        return remaining_s / 3600.0


# ---------------------------------------------------------------------------
# Fault Injector
# ---------------------------------------------------------------------------

class FaultInjector:
    """
    Manages active faults and applies perturbations to the engine state.

    Usage:
        injector = FaultInjector.from_config()
        injector.add_fault(ActiveFault("bearing_wear", onset_time_s=3600, ...))

        # In simulation loop:
        result = injector.step(t_s, state, step_data)
        perturbations = result["perturbations"]
        fault_id = result["fault_id"]
    """

    def __init__(self, faults_config: Optional[Dict[str, Any]] = None) -> None:
        self._config = faults_config or load_faults_config()
        self._fault_defs = self._config.get("faults", {})
        self._active_faults: List[ActiveFault] = []
        self._rng = random.Random(42)

    @classmethod
    def from_config(cls) -> "FaultInjector":
        return cls(load_faults_config())

    def add_fault(self, fault: ActiveFault) -> None:
        self._active_faults.append(fault)

    def clear_faults(self) -> None:
        self._active_faults.clear()

    def active_fault_ids(self, t_s: float) -> List[str]:
        return [
            f.fault_id for f in self._active_faults
            if f.severity(t_s) > 0.0
        ]

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------

    def step(
        self,
        t_s: float,
        state: EngineState,
        step_data: dict,
    ) -> dict:
        """
        Apply all active faults for time t_s.

        Returns:
            {
                "perturbations": dict of health param overrides,
                "fault_id": str | None (primary active fault),
                "severity": float,
                "true_rul_hours": float | None,
                "sensor_faults": dict of sensor channel perturbations,
            }
        """
        perturbations: Dict[str, Any] = {}
        sensor_faults: Dict[str, Any] = {}
        active_ids = []
        primary_fault_id = None
        primary_severity = 0.0
        min_rul: Optional[float] = None

        for fault in self._active_faults:
            sev = fault.severity(t_s)
            if sev <= 0.0:
                continue

            active_ids.append(fault.fault_id)

            # Apply health parameter perturbations from fault definition
            fault_def = self._fault_defs.get(fault.fault_id, {})
            hp = fault_def.get("health_params", {})

            for hp_key, hp_target in hp.items():
                if hp_key == "hp_combustion_eff_per_cyl":
                    # Per-cylinder combustion efficiency
                    new_val = 1.0 - (1.0 - hp_target) * sev
                    if fault.cylinder_index is not None:
                        idx = fault.cylinder_index
                        if 0 <= idx < 4:
                            current = list(state.hp_combustion_eff)
                            current[idx] = min(current[idx], new_val)
                            perturbations["hp_combustion_eff"] = current
                    else:
                        perturbations["hp_combustion_eff"] = [
                            1.0 - (1.0 - hp_target) * sev
                        ] * 4
                else:
                    # Scalar health params — interpolate toward degraded value
                    if hp_target < 1.0:
                        new_val = 1.0 - (1.0 - hp_target) * sev
                    else:
                        # hp_target > 1.0 means increases (e.g. friction factor)
                        new_val = 1.0 + (hp_target - 1.0) * sev
                    # Keep the worst (most degraded) value if multiple faults
                    current = perturbations.get(hp_key, getattr(state, hp_key, 1.0))
                    if hp_target < 1.0:
                        perturbations[hp_key] = min(current, new_val)
                    else:
                        perturbations[hp_key] = max(current, new_val)

            # Sensor fault perturbations
            cat = fault_def.get("category", "")
            if cat == "sensor" and fault.sensor_channel:
                sensor_faults[fault.sensor_channel] = {
                    "type": fault.fault_id,
                    "severity": sev,
                    "bias": fault.sensor_bias_value * sev,
                }

            # Track primary fault (highest severity)
            if sev > primary_severity:
                primary_severity = sev
                primary_fault_id = fault.fault_id

            # RUL
            rul = fault.true_rul_hours(t_s)
            if rul is not None:
                if min_rul is None or rul < min_rul:
                    min_rul = rul

        return {
            "perturbations": perturbations,
            "fault_id": primary_fault_id,
            "severity": primary_severity,
            "true_rul_hours": min_rul,
            "sensor_faults": sensor_faults,
            "active_fault_ids": active_ids,
        }

    # -------------------------------------------------------------------------
    # Factory methods for common scenarios
    # -------------------------------------------------------------------------

    @classmethod
    def gradual_bearing_wear(
        cls,
        onset_time_s: float = 3600.0,
        failure_time_s: float = 36000.0,
    ) -> "FaultInjector":
        injector = cls.from_config()
        injector.add_fault(ActiveFault(
            fault_id="bearing_wear",
            onset_time_s=onset_time_s,
            failure_time_s=failure_time_s,
            ramp_type="exponential",
        ))
        return injector

    @classmethod
    def sudden_oil_starvation(
        cls,
        onset_time_s: float = 10800.0,
    ) -> "FaultInjector":
        injector = cls.from_config()
        injector.add_fault(ActiveFault(
            fault_id="oil_starvation",
            onset_time_s=onset_time_s,
            failure_time_s=onset_time_s + 900.0,  # 15 minutes to failure
            ramp_type="abrupt",
        ))
        return injector

    @classmethod
    def sensor_only_egt_bias(
        cls,
        channel: str = "egt_c",
        bias_value: float = 50.0,
        onset_time_s: float = 1800.0,
    ) -> "FaultInjector":
        injector = cls.from_config()
        injector.add_fault(ActiveFault(
            fault_id="sensor_bias",
            onset_time_s=onset_time_s,
            failure_time_s=onset_time_s + 36000.0,
            ramp_type="abrupt",
            sensor_channel=channel,
            sensor_bias_value=bias_value,
        ))
        return injector
