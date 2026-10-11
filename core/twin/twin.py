"""
AEROTWIN — Phase 3: Digital Twin

The main twin class that ties together:
1. Physics engine model (the forward model)
2. EKF state estimator (fuses sensors with physics)
3. Residual generator (expected vs measured)
4. Sensor-fault isolation (consistency check)
5. Missing data handling

This is what makes AEROTWIN a *twin* rather than a dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from configs.config_loader import EngineConfig, load_engine_config
from core.physics.engine_model import EngineState, PistonEngineModel
from core.twin.ekf import ExtendedKalmanFilter, TwinEstimate
from data.schemas import TelemetryRecord

logger = logging.getLogger("aerotwin.twin")


@dataclass
class TwinState:
    """Combined output from the digital twin at each timestep."""
    estimate: TwinEstimate
    physics_state: EngineState
    residuals: Dict[str, float] = field(default_factory=dict)
    normalised_residuals: Dict[str, float] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    # Flags
    sensor_fault_channels: List[str] = field(default_factory=list)
    engine_fault_suspected: bool = False
    # Health summary (0=healthy, 1=failed)
    overall_health_score: float = 1.0


class DigitalTwin:
    """
    AEROTWIN Digital Twin.

    Processes a stream of TelemetryRecord objects, updating the physics model
    and EKF, and returning TwinState at each step.
    """

    def __init__(
        self,
        engine_config: Optional[EngineConfig] = None,
        dt: float = 1.0,
    ) -> None:
        self.cfg = engine_config or load_engine_config()
        self.dt = dt
        self.physics = PistonEngineModel(self.cfg)
        self.ekf = ExtendedKalmanFilter(self.cfg)
        self._physics_state = EngineState()
        self._last_rpm: float = self.cfg.power.idle_rpm
        self._last_throttle: float = 0.12
        self._last_altitude: float = 0.0
        self._step_count: int = 0

        # Sensor fault isolation: track residual history per channel
        self._residual_history: Dict[str, List[float]] = {}
        self._residual_history_len = 30  # seconds

    # -------------------------------------------------------------------------
    # Main update
    # -------------------------------------------------------------------------

    def update(self, record: TelemetryRecord) -> TwinState:
        """
        Process one telemetry record.
        1. Update physics twin from control inputs.
        2. Run EKF with measurements.
        3. Apply health params back to physics.
        4. Compute residuals and fault isolation.
        5. Return TwinState.
        """
        self._step_count += 1

        # --- Extract control inputs from record ---
        throttle = (record.throttle_pct or 50.0) / 100.0
        rpm = record.rpm or self._last_rpm
        altitude_m = record.altitude_m or self._last_altitude
        isa_std_temp_c = 15.0 - 0.0065 * min(altitude_m, 11000.0)
        delta_isa = (record.oat_c - isa_std_temp_c) if record.oat_c is not None else 0.0

        # Update health params in physics state from EKF estimate
        est = self.ekf.x  # current EKF state vector
        from core.twin.ekf import IDX_HP_VE, IDX_HP_FRICTION, IDX_HP_COOLING, IDX_HP_INJECTOR, IDX_HP_BOOST, IDX_HP_COMB_AVG
        self._physics_state.hp_ve_factor = float(np.clip(est[IDX_HP_VE], 0.3, 1.2))
        self._physics_state.hp_friction_factor = float(np.clip(est[IDX_HP_FRICTION], 0.8, 2.0))
        self._physics_state.hp_cooling_effectiveness = float(np.clip(est[IDX_HP_COOLING], 0.1, 1.5))
        self._physics_state.hp_injector_flow = float(np.clip(est[IDX_HP_INJECTOR], 0.3, 1.3))
        self._physics_state.hp_boost_efficiency = float(np.clip(est[IDX_HP_BOOST], 0.3, 1.2))
        self._physics_state.hp_combustion_eff = [
            float(np.clip(est[IDX_HP_COMB_AVG], 0.5, 1.0))
        ] * 4

        # --- Step physics model ---
        self._physics_state = self.physics.step(
            state=self._physics_state,
            dt=self.dt,
            throttle=throttle,
            rpm=rpm,
            altitude_m=altitude_m,
            delta_isa_c=delta_isa,
        )

        # --- Build measurement dict from record ---
        measurements = self._record_to_measurements(record)

        # --- Run EKF ---
        twin_estimate = self.ekf.step(
            measurements=measurements,
            engine_state_estimate=self._physics_state,
            dt=self.dt,
        )

        # --- Update residual history ---
        for ch, nr in twin_estimate.normalised_residuals.items():
            if ch not in self._residual_history:
                self._residual_history[ch] = []
            self._residual_history[ch].append(nr)
            if len(self._residual_history[ch]) > self._residual_history_len:
                self._residual_history[ch].pop(0)

        # --- Sensor fault isolation vs Engine fault ---
        # Single channel deviation = isolated sensor fault
        # Multi-channel corroboration (>= 2 channels) = system/engine fault
        large_residual_channels = [
            ch for ch, nr in twin_estimate.normalised_residuals.items()
            if abs(nr) > 2.5
        ]

        if len(large_residual_channels) >= 2:
            sensor_fault_channels = []
            engine_fault_suspected = True
        elif len(large_residual_channels) == 1:
            sensor_fault_channels = large_residual_channels
            engine_fault_suspected = False
        else:
            sensor_fault_channels = []
            engine_fault_suspected = False

        # --- Overall health score ---
        health_score = self._compute_health_score(twin_estimate)

        # --- Update cached values ---
        self._last_rpm = rpm
        self._last_throttle = throttle
        self._last_altitude = altitude_m

        return TwinState(
            estimate=twin_estimate,
            physics_state=self._physics_state,
            residuals=twin_estimate.residuals,
            normalised_residuals=twin_estimate.normalised_residuals,
            confidence=twin_estimate.confidence,
            sensor_fault_channels=sensor_fault_channels,
            engine_fault_suspected=engine_fault_suspected,
            overall_health_score=health_score,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _record_to_measurements(self, record: TelemetryRecord) -> Dict[str, Any]:
        """Convert TelemetryRecord to measurement dict for EKF."""
        return {
            "oil_temp_c": record.oil_temp_c,
            "coolant_temp_c": record.coolant_temp_c,
            "oil_press_kpa": record.oil_press_kpa,
            "egt_c": record.egt_c,
            "cht_c": record.cht_c,
        }

    def _assess_engine_fault(
        self,
        estimate: TwinEstimate,
        sensor_fault_channels: List[str],
    ) -> bool:
        """
        Assess whether residuals are consistent with an engine fault
        (as opposed to a sensor fault).

        Simple logic: if multiple non-sensor-faulty channels have large residuals,
        it's likely an engine fault. Sensor faults typically affect only one channel.
        """
        non_sensor_fault_channels = [
            ch for ch, nr in estimate.normalised_residuals.items()
            if abs(nr) > 2.5 and ch not in sensor_fault_channels
        ]
        return len(non_sensor_fault_channels) >= 2

    def _compute_health_score(self, estimate: TwinEstimate) -> float:
        """
        Compute overall engine health score from estimated health parameters
        and measurement residuals.
        1.0 = fully healthy, 0.0 = complete failure.
        """
        hp_scores = [
            max(0.0, min(1.0, estimate.hp_ve_factor)),
            max(0.0, min(1.0, 2.0 - estimate.hp_friction_factor)),  # invert: higher friction = lower health
            max(0.0, min(1.0, estimate.hp_cooling_effectiveness)),
            max(0.0, min(1.0, estimate.hp_injector_flow)),
            max(0.0, min(1.0, estimate.hp_boost_efficiency)),
            max(0.0, min(1.0, estimate.hp_combustion_eff_avg)),
        ]
        base_score = float(np.mean(hp_scores))

        # Check for multi-channel or severe residual deviations indicating engine failure
        # Only apply after filter initial convergence (warmup > 3 steps)
        if self._step_count > 3:
            large_residuals = [
                abs(nr) for nr in estimate.normalised_residuals.values()
                if abs(nr) > 3.0
            ]
            if len(large_residuals) >= 2 or any(abs(nr) > 8.0 for nr in estimate.normalised_residuals.values()):
                max_dev = max((abs(nr) for nr in estimate.normalised_residuals.values()), default=0.0)
                penalty = min(0.85, (max_dev - 3.0) * 0.12)
                base_score = max(0.1, base_score - penalty)

        return base_score

    def reset(self) -> None:
        """Reset twin to initial state."""
        self._physics_state = EngineState()
        self.ekf.reset()
        self._residual_history.clear()
        self._step_count = 0
