"""
AEROTWIN — Phase 1: Closed-Loop Simulator

Drives the engine physics model through a flight profile,
applies sensor models, injects faults (Phase 2+), and produces
a stream of TelemetryRecord objects with ground-truth labels.

CLI entry point: `python -m core.simulator.run`
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional

from configs.config_loader import load_engine_config, load_settings
from core.physics.engine_model import EngineState, PistonEngineModel
from core.simulator.flight_profile import FlightProfileGenerator
from data.schemas import DataSource, FlightPhase, TelemetryRecord

logger = logging.getLogger("aerotwin.simulator")


class Simulator:
    """
    Closed-loop AEROTWIN simulator.

    Drives the engine physics model through a FlightProfile,
    applies sensor models, and yields TelemetryRecord objects.

    Fault injection is applied by fault_injector (Phase 2).
    """

    def __init__(
        self,
        seed: int = 42,
        dt: float = 1.0,
        engine_config_path: Optional[str] = None,
        fault_injector=None,   # Optional FaultInjector (Phase 2)
        sensor_model=None,     # Optional SensorModel (Phase 2)
    ) -> None:
        self.seed = seed
        self.dt = dt
        self.rng = random.Random(seed)
        self.engine_cfg = load_engine_config(engine_config_path)
        self.engine_model = PistonEngineModel(self.engine_cfg)
        self.profile_gen = FlightProfileGenerator(rng=self.rng, dt=dt)
        self.fault_injector = fault_injector
        self.sensor_model = sensor_model

    # -------------------------------------------------------------------------
    # Mission simulation
    # -------------------------------------------------------------------------

    def run_mission(
        self,
        mission_id: str = "sim_001",
        engine_id: str = "rotax_914_like_001",
        profile_type: str = "standard",
        cruise_altitude_m: float = 4500.0,
        loiter_duration_s: float = 28800.0,
        delta_isa_c: float = 0.0,
        engine_age_factor: float = 0.0,
        gust_enabled: bool = True,
        start_time: Optional[datetime] = None,
    ) -> Iterator[TelemetryRecord]:
        """
        Run a complete simulated mission and yield TelemetryRecord per timestep.

        Args:
            mission_id:       Unique mission identifier.
            engine_id:        Engine instance identifier.
            profile_type:     'standard' | 'short'.
            cruise_altitude_m: Target cruise altitude in metres.
            loiter_duration_s: Loiter phase duration in seconds.
            delta_isa_c:      ISA temperature deviation.
            engine_age_factor: 0=new engine, 1=end of life.
            gust_enabled:     Add random temperature/gust disturbances.
            start_time:       Mission start time (UTC, defaults to now).
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        # Generate flight profile
        if profile_type == "short":
            profile = self.profile_gen.generate_short_mission(
                cruise_altitude_m=cruise_altitude_m,
                delta_isa_c=delta_isa_c,
            )
        else:
            profile = self.profile_gen.generate_standard_male_mission(
                cruise_altitude_m=cruise_altitude_m,
                loiter_duration_s=loiter_duration_s,
                delta_isa_c=delta_isa_c,
                gust_enabled=gust_enabled,
                engine_age_factor=engine_age_factor,
            )

        # Initial engine state
        state = EngineState()
        # Apply age factor to health parameters
        state.hp_ve_factor = 1.0 - engine_age_factor * 0.15
        state.hp_friction_factor = 1.0 + engine_age_factor * 0.30
        state.hp_cooling_effectiveness = 1.0 - engine_age_factor * 0.10

        logger.info(
            "Simulation start: mission=%s engine=%s profile=%s duration=%.0f s",
            mission_id, engine_id, profile_type, profile.total_duration_s,
        )

        rpm = self.engine_cfg.power.idle_rpm

        for step_data in profile.iterate():
            t_s = step_data["time_s"]
            throttle = step_data["throttle"]
            altitude_m = step_data["altitude_m"]
            airspeed_mps = step_data["airspeed_mps"]
            phase = step_data["phase"]
            delta_isa = step_data["delta_isa_c"]

            # Compute target RPM from throttle and phase
            rpm = self._throttle_to_rpm(throttle, phase)

            # Apply fault injector (Phase 2 — no-op if not provided)
            fault_perturbations = {}
            fault_id = None
            fault_severity = 0.0
            true_rul_hours = None
            if self.fault_injector is not None:
                result = self.fault_injector.step(t_s, state, step_data)
                fault_perturbations = result.get("perturbations", {})
                fault_id = result.get("fault_id")
                fault_severity = result.get("severity", 0.0)
                true_rul_hours = result.get("true_rul_hours")
                # Apply health-parameter perturbations from fault
                for hp_key, hp_val in fault_perturbations.items():
                    if hasattr(state, hp_key):
                        setattr(state, hp_key, hp_val)

            # Step the physics engine
            state = self.engine_model.step(
                state=state,
                dt=self.dt,
                throttle=throttle,
                rpm=rpm,
                altitude_m=altitude_m,
                delta_isa_c=delta_isa,
            )

            # Build true (noise-free) telemetry
            timestamp = start_time + timedelta(seconds=t_s)

            # Apply sensor model (Phase 2 — pass-through if not provided)
            sensed_state = self._apply_sensor_model(state, t_s)

            record = TelemetryRecord(
                timestamp=timestamp,
                engine_id=engine_id,
                source=DataSource.simulated,
                flight_phase=phase,
                altitude_m=altitude_m,
                oat_c=state.atm.oat_c if state.atm else 15.0,
                airspeed_mps=airspeed_mps,
                rpm=sensed_state["rpm"],
                throttle_pct=throttle * 100.0,
                map_kpa=sensed_state["map_kpa"],
                boost_kpa=sensed_state["boost_kpa"],
                fuel_flow_lph=sensed_state["fuel_flow_lph"],
                cht_c=sensed_state["cht_c"],
                egt_c=sensed_state["egt_c"],
                oil_temp_c=sensed_state["oil_temp_c"],
                oil_press_kpa=sensed_state["oil_press_kpa"],
                coolant_temp_c=sensed_state["coolant_temp_c"],
                fuel_press_kpa=sensed_state.get("fuel_press_kpa", 150.0),
                bus_voltage_v=state.bus_voltage_v,
                vib_rms_g=sensed_state["vib_rms_g"],
                # Ground-truth labels
                fault_id=fault_id,
                fault_severity=fault_severity if fault_id else None,
                true_rul_hours=true_rul_hours,
            )
            yield record

        logger.info("Simulation complete: mission=%s", mission_id)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _throttle_to_rpm(self, throttle: float, phase: FlightPhase) -> float:
        """Convert throttle position to RPM based on propeller load model."""
        idle = self.engine_cfg.power.idle_rpm
        max_cont = self.engine_cfg.power.max_continuous_rpm
        if phase == FlightPhase.ground:
            return idle + throttle * (idle * 0.5)
        return idle + throttle * (max_cont - idle)

    def _apply_sensor_model(self, state: EngineState, t_s: float) -> dict:
        """
        Apply sensor model (noise, lag, dropout).
        If sensor_model is not provided (Phase 2), returns clean values.
        """
        if self.sensor_model is not None:
            return self.sensor_model.apply(state, t_s)
        # Phase 0/1: pass-through with tiny noise
        noise = lambda std: self.rng.gauss(0, std)
        return {
            "rpm": max(0.0, state.rpm + noise(20.0)),
            "map_kpa": max(0.0, state.map_kpa + noise(0.5)),
            "boost_kpa": max(0.0, state.boost_kpa + noise(1.0)),
            "fuel_flow_lph": max(0.0, state.fuel_flow_lph + noise(0.2)),
            "cht_c": [c + noise(2.0) for c in state.cht_c],
            "egt_c": [e + noise(5.0) for e in state.egt_c],
            "oil_temp_c": state.oil_temp_c + noise(1.0),
            "oil_press_kpa": max(0.0, state.oil_press_kpa + noise(5.0)),
            "coolant_temp_c": state.coolant_temp_c + noise(1.0),
            "fuel_press_kpa": max(0.0, 150.0 + noise(2.0)),
            "vib_rms_g": max(0.0, state.vib_rms_g + abs(noise(0.005))),
        }
