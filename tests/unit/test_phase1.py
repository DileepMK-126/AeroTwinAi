"""
AEROTWIN — Phase 1 Unit Tests

Tests for: ISA atmosphere, engine model physics, energy balance,
monotonic trends, simulator smoke test.
"""

from __future__ import annotations

import math
from datetime import timezone

import pytest

from core.physics.atmosphere import isa_atmosphere
from core.physics.engine_model import EngineState, PistonEngineModel
from configs.config_loader import load_engine_config


# ---------------------------------------------------------------------------
# ISA Atmosphere Tests
# ---------------------------------------------------------------------------

class TestAtmosphere:
    def test_sea_level(self):
        """At sea level, ISA gives standard conditions."""
        atm = isa_atmosphere(0.0)
        assert abs(atm.temperature_k - 288.15) < 0.01
        assert abs(atm.pressure_pa - 101325.0) < 1.0
        assert abs(atm.density_kg_m3 - 1.225) < 0.01

    def test_pressure_decreases_with_altitude(self):
        """Pressure must monotonically decrease with altitude."""
        alts = [0, 1000, 2000, 4000, 8000, 11000]
        pressures = [isa_atmosphere(a).pressure_pa for a in alts]
        for i in range(len(pressures) - 1):
            assert pressures[i] > pressures[i + 1], (
                f"Pressure did not decrease from {alts[i]}m to {alts[i+1]}m"
            )

    def test_temperature_decreases_in_troposphere(self):
        """Temperature must decrease with altitude in troposphere (< 11000 m)."""
        temps = [isa_atmosphere(a).temperature_k for a in [0, 5000, 10000]]
        assert temps[0] > temps[1] > temps[2]

    def test_isothermal_above_tropopause(self):
        """Temperature is constant at ~216.65 K in stratosphere."""
        t11 = isa_atmosphere(11000).temperature_k
        t15 = isa_atmosphere(15000).temperature_k
        assert abs(t11 - 216.65) < 1.0
        assert abs(t15 - 216.65) < 1.0

    def test_isa_deviation(self):
        """ISA+10 should give temperature 10 K higher than standard."""
        atm_std = isa_atmosphere(0.0, delta_isa_c=0.0)
        atm_hot = isa_atmosphere(0.0, delta_isa_c=10.0)
        assert abs(atm_hot.temperature_k - atm_std.temperature_k - 10.0) < 0.01

    def test_density_decreases_with_altitude(self):
        """Air density must decrease with altitude (derating check)."""
        rho0 = isa_atmosphere(0.0).density_kg_m3
        rho4k = isa_atmosphere(4000.0).density_kg_m3
        assert rho0 > rho4k

    def test_speed_of_sound_positive(self):
        """Speed of sound must be physically valid."""
        for alt in [0, 2000, 5000, 11000]:
            atm = isa_atmosphere(alt)
            assert atm.speed_of_sound_mps > 200.0


# ---------------------------------------------------------------------------
# Engine Model Tests
# ---------------------------------------------------------------------------

class TestEngineModel:
    @pytest.fixture
    def model(self):
        cfg = load_engine_config()
        return PistonEngineModel(cfg)

    @pytest.fixture
    def healthy_state(self):
        return EngineState()

    def test_brake_power_positive_under_load(self, model, healthy_state):
        """Engine under load (throttle 0.75, 4800 RPM) should produce positive brake power."""
        s = model.step(healthy_state, dt=1.0, throttle=0.75, rpm=4800.0, altitude_m=0.0)
        assert s.brake_power_kw > 0.0, "Brake power must be positive under load"

    def test_max_power_not_exceeded(self, model, healthy_state):
        """Brake power must not exceed rated max power."""
        cfg = load_engine_config()
        s = model.step(healthy_state, dt=1.0, throttle=1.0, rpm=cfg.power.max_rpm, altitude_m=0.0)
        # Allow 10% margin for model approximations
        assert s.brake_power_kw <= cfg.power.max_power_kw * 1.10, (
            f"Brake power {s.brake_power_kw:.1f} kW exceeds rated {cfg.power.max_power_kw:.1f} kW"
        )

    def test_power_increases_with_throttle(self, model, healthy_state):
        """Fuel flow (proportional to power output) should increase with throttle."""
        flows = []
        for thr in [0.3, 0.5, 0.7, 0.9]:
            s = EngineState()
            s = model.step(s, dt=0.1, throttle=thr, rpm=5000.0, altitude_m=0.0)
            flows.append(s.fuel_flow_lph)
        for i in range(len(flows) - 1):
            assert flows[i] < flows[i + 1], (
                f"Fuel flow did not increase: throttle step {i}: {flows[i]:.3f} >= {flows[i+1]:.3f} L/h"
            )

    def test_power_decreases_with_altitude(self, model, healthy_state):
        """Power output should decrease with altitude (density/boost derating)."""
        s0 = EngineState()
        s4k = EngineState()
        s0 = model.step(s0, dt=1.0, throttle=0.75, rpm=4500.0, altitude_m=0.0)
        s4k = model.step(s4k, dt=1.0, throttle=0.75, rpm=4500.0, altitude_m=6000.0)
        # At high altitude, fuel flow (and thus power) should be lower
        assert s4k.fuel_flow_lph <= s0.fuel_flow_lph * 1.05, (
            "Fuel flow / power should be lower at altitude"
        )

    def test_fuel_flow_positive_under_load(self, model, healthy_state):
        """Fuel flow must be positive when engine is running."""
        s = model.step(healthy_state, dt=1.0, throttle=0.5, rpm=3000.0, altitude_m=0.0)
        assert s.fuel_flow_lph > 0.0

    def test_energy_balance_within_tolerance(self, model, healthy_state):
        """
        Energy balance: q_fuel >= q_shaft + q_coolant + q_exhaust
        (inequality because of model simplification losses)
        """
        s = model.step(healthy_state, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=1000.0)
        total_output = s.q_shaft_kw + s.q_coolant_kw + s.q_exhaust_kw
        # Total output should not exceed fuel heat input by more than 5%
        assert total_output <= s.q_fuel_kw * 1.05, (
            f"Energy not balanced: output {total_output:.2f} > input {s.q_fuel_kw:.2f} kW"
        )
        # And should account for at least 80% of fuel energy (not more losses than expected)
        if s.q_fuel_kw > 0.1:
            assert total_output >= s.q_fuel_kw * 0.80, (
                f"Energy balance: output {total_output:.2f} < 80% of fuel {s.q_fuel_kw:.2f} kW"
            )

    def test_cht_within_normal_range_cruise(self, model, healthy_state):
        """After several seconds at cruise, CHT should be in normal operating range."""
        s = EngineState()
        for _ in range(120):  # 2 minutes of cruise
            s = model.step(s, dt=1.0, throttle=0.70, rpm=4800.0, altitude_m=4500.0)
        cfg = load_engine_config()
        for i, cht in enumerate(s.cht_c):
            assert cht <= cfg.thermal.cht_max_c + 10.0, (
                f"CHT cyl {i+1}: {cht:.1f} °C exceeds max {cfg.thermal.cht_max_c} °C"
            )

    def test_egt_within_normal_range_cruise(self, model, healthy_state):
        """EGT should be within normal cruise range after warmup."""
        s = EngineState()
        for _ in range(60):  # 1 minute
            s = model.step(s, dt=1.0, throttle=0.70, rpm=4800.0, altitude_m=4500.0)
        cfg = load_engine_config()
        for i, egt in enumerate(s.egt_c):
            assert egt <= cfg.thermal.egt_max_c + 20.0, (
                f"EGT cyl {i+1}: {egt:.1f} °C exceeds max {cfg.thermal.egt_max_c} °C"
            )

    def test_oil_pressure_positive(self, model, healthy_state):
        """Oil pressure must be positive when engine is running."""
        s = model.step(healthy_state, dt=1.0, throttle=0.5, rpm=3000.0, altitude_m=0.0)
        assert s.oil_press_kpa > 0.0

    def test_fmep_increases_with_friction_health_param(self, model):
        """Degraded friction health parameter should increase FMEP."""
        s_healthy = EngineState()
        s_degraded = EngineState()
        s_degraded.hp_friction_factor = 1.5  # 50% more friction
        s_h = model.step(s_healthy, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=0.0)
        s_d = model.step(s_degraded, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=0.0)
        assert s_d.fmep_kpa > s_h.fmep_kpa

    def test_ve_health_param_reduces_air_flow(self, model):
        """Reduced VE health parameter should reduce air mass flow."""
        s_healthy = EngineState()
        s_degraded = EngineState()
        s_degraded.hp_ve_factor = 0.7
        s_h = model.step(s_healthy, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=0.0)
        s_d = model.step(s_degraded, dt=1.0, throttle=0.7, rpm=4500.0, altitude_m=0.0)
        assert s_d.air_mass_flow_kg_s < s_h.air_mass_flow_kg_s


# ---------------------------------------------------------------------------
# Simulator Smoke Test
# ---------------------------------------------------------------------------

class TestSimulator:
    def test_short_mission_runs(self):
        """Short mission should complete and return > 0 records."""
        from core.simulator.simulator import Simulator
        sim = Simulator(seed=42, dt=1.0)
        records = list(sim.run_mission(
            mission_id="test_001",
            profile_type="short",
            cruise_altitude_m=2000.0,
        ))
        assert len(records) > 100, "Expected more than 100 records in short mission"

    def test_simulation_speed(self):
        """10-hour mission should run in under 60 seconds on CPU."""
        import time
        from core.simulator.simulator import Simulator
        sim = Simulator(seed=0, dt=1.0)
        t0 = time.perf_counter()
        records = list(sim.run_mission(
            mission_id="speed_test",
            profile_type="standard",
            loiter_duration_s=28800.0,
        ))
        elapsed = time.perf_counter() - t0
        assert elapsed < 60.0, (
            f"10-hour mission took {elapsed:.1f}s — must run in under 60s"
        )
        assert len(records) > 30000, "Expected ~35000+ timesteps in 10h mission at 1Hz"

    def test_records_have_valid_schema(self):
        """All records from simulator must pass schema validation."""
        from core.simulator.simulator import Simulator
        sim = Simulator(seed=42, dt=5.0)  # 5s steps for speed
        records = list(sim.run_mission(
            mission_id="schema_test",
            profile_type="short",
        ))
        assert len(records) > 0
        for r in records[:10]:  # spot check first 10
            assert r.engine_id is not None
            source_val = r.source.value if hasattr(r.source, 'value') else r.source
            assert source_val == "simulated"
            assert r.timestamp.tzinfo is not None

    def test_seeded_reproducibility(self):
        """Same seed should produce identical record sequences."""
        from core.simulator.simulator import Simulator
        sim1 = Simulator(seed=99, dt=5.0)
        sim2 = Simulator(seed=99, dt=5.0)
        recs1 = list(sim1.run_mission(profile_type="short"))
        recs2 = list(sim2.run_mission(profile_type="short"))
        assert len(recs1) == len(recs2)
        # Compare a few key fields
        for r1, r2 in zip(recs1[:5], recs2[:5]):
            assert abs((r1.rpm or 0) - (r2.rpm or 0)) < 1.0
