"""
AEROTWIN — Phase 1: Flight Profile Generator

Generates realistic MALE UAV flight profiles (throttle, altitude, airspeed
as functions of time) for use by the closed-loop simulator.

Phases: ground → takeoff → climb → cruise → loiter → descent → landing
With gust and temperature disturbances.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from data.schemas import FlightPhase


# ---------------------------------------------------------------------------
# Flight Profile Segment
# ---------------------------------------------------------------------------

@dataclass
class FlightSegment:
    """A single phase of flight with target states."""
    phase: FlightPhase
    duration_s: float
    target_altitude_m: float
    target_airspeed_mps: float
    target_throttle: float    # 0–1
    delta_isa_c: float = 0.0  # temperature deviation


@dataclass
class FlightProfile:
    """
    A complete mission flight profile.
    Provides an iterator of (time_s, throttle, altitude_m, airspeed_mps, phase, delta_isa_c)
    tuples at a given sample rate.
    """
    segments: List[FlightSegment]
    dt: float = 1.0    # timestep in seconds

    @property
    def total_duration_s(self) -> float:
        return sum(seg.duration_s for seg in self.segments)

    def iterate(self) -> Iterator[dict]:
        """
        Yield one dict per timestep across all segments.
        Linearly interpolates throttle, altitude, and airspeed between segment boundaries.
        """
        t = 0.0
        for i, seg in enumerate(self.segments):
            n_steps = max(1, int(seg.duration_s / self.dt))
            prev = self.segments[i - 1] if i > 0 else seg
            for step in range(n_steps):
                alpha = step / max(n_steps - 1, 1)   # 0.0 → 1.0
                throttle = prev.target_throttle + alpha * (seg.target_throttle - prev.target_throttle)
                altitude = prev.target_altitude_m + alpha * (seg.target_altitude_m - prev.target_altitude_m)
                airspeed = prev.target_airspeed_mps + alpha * (seg.target_airspeed_mps - prev.target_airspeed_mps)
                yield {
                    "time_s": round(t, 3),
                    "throttle": max(0.0, min(1.0, throttle)),
                    "altitude_m": max(0.0, altitude),
                    "airspeed_mps": max(0.0, airspeed),
                    "phase": seg.phase,
                    "delta_isa_c": seg.delta_isa_c,
                }
                t += self.dt


# ---------------------------------------------------------------------------
# Profile Factory
# ---------------------------------------------------------------------------

class FlightProfileGenerator:
    """
    Generates standard MALE UAV mission profiles with optional disturbances.
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        dt: float = 1.0,
    ) -> None:
        self.rng = rng or random.Random(42)
        self.dt = dt

    def generate_standard_male_mission(
        self,
        cruise_altitude_m: float = 4500.0,
        cruise_airspeed_mps: float = 55.0,
        loiter_airspeed_mps: float = 40.0,
        loiter_duration_s: float = 28800.0,   # 8 hours
        delta_isa_c: float = 0.0,
        gust_enabled: bool = True,
        engine_age_factor: float = 0.0,        # 0 = new, 1 = end of life
    ) -> FlightProfile:
        """
        Standard 10-hour MALE mission profile.

        ground → takeoff → climb → cruise → loiter → descent → landing
        """
        # Add temperature variations for realism
        if gust_enabled:
            temp_variance = self.rng.uniform(-5.0, 5.0)
        else:
            temp_variance = 0.0
        isa = delta_isa_c + temp_variance

        segments = [
            FlightSegment(
                phase=FlightPhase.ground,
                duration_s=300.0,
                target_altitude_m=0.0,
                target_airspeed_mps=0.0,
                target_throttle=0.12,   # idle
                delta_isa_c=isa,
            ),
            FlightSegment(
                phase=FlightPhase.takeoff,
                duration_s=120.0,
                target_altitude_m=50.0,
                target_airspeed_mps=25.0,
                target_throttle=1.0,    # full power
                delta_isa_c=isa,
            ),
            FlightSegment(
                phase=FlightPhase.climb,
                duration_s=600.0,
                target_altitude_m=cruise_altitude_m,
                target_airspeed_mps=cruise_airspeed_mps,
                target_throttle=0.85,
                delta_isa_c=isa,
            ),
            FlightSegment(
                phase=FlightPhase.cruise,
                duration_s=1800.0,
                target_altitude_m=cruise_altitude_m,
                target_airspeed_mps=cruise_airspeed_mps,
                target_throttle=0.70,
                delta_isa_c=isa,
            ),
            FlightSegment(
                phase=FlightPhase.loiter,
                duration_s=loiter_duration_s,
                target_altitude_m=cruise_altitude_m,
                target_airspeed_mps=loiter_airspeed_mps,
                target_throttle=0.55,
                delta_isa_c=isa + self.rng.uniform(-3.0, 3.0),
            ),
            FlightSegment(
                phase=FlightPhase.descent,
                duration_s=600.0,
                target_altitude_m=200.0,
                target_airspeed_mps=45.0,
                target_throttle=0.25,
                delta_isa_c=isa,
            ),
            FlightSegment(
                phase=FlightPhase.landing,
                duration_s=180.0,
                target_altitude_m=0.0,
                target_airspeed_mps=0.0,
                target_throttle=0.10,
                delta_isa_c=isa,
            ),
        ]

        return FlightProfile(segments=segments, dt=self.dt)

    def generate_short_mission(
        self,
        cruise_altitude_m: float = 2000.0,
        duration_s: float = 3600.0,
        delta_isa_c: float = 0.0,
    ) -> FlightProfile:
        """Short test mission (1 hour) for smoke tests and quick CI checks."""
        isa = delta_isa_c
        segments = [
            FlightSegment(FlightPhase.ground, 60.0, 0.0, 0.0, 0.12, isa),
            FlightSegment(FlightPhase.takeoff, 60.0, 50.0, 20.0, 1.0, isa),
            FlightSegment(FlightPhase.climb, 300.0, cruise_altitude_m, 50.0, 0.85, isa),
            FlightSegment(FlightPhase.cruise, duration_s - 1020.0, cruise_altitude_m, 50.0, 0.70, isa),
            FlightSegment(FlightPhase.descent, 300.0, 100.0, 40.0, 0.25, isa),
            FlightSegment(FlightPhase.landing, 120.0, 0.0, 0.0, 0.10, isa),
        ]
        return FlightProfile(segments=segments, dt=self.dt)
