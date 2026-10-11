"""
AEROTWIN — Phase 1: Physics Core

ISA Atmosphere model with temperature offset and altitude-dependent properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# ISA Atmosphere Constants
# ---------------------------------------------------------------------------

SEA_LEVEL_TEMP_K = 288.15       # K (15 °C)
SEA_LEVEL_PRESSURE_PA = 101325.0  # Pa
SEA_LEVEL_DENSITY = 1.225        # kg/m³
LAPSE_RATE = 0.0065              # K/m (troposphere)
GRAVITY = 9.80665               # m/s²
GAS_CONSTANT_AIR = 287.05       # J/(kg·K)
GAMMA_AIR = 1.4                 # ratio of specific heats
TROPOPAUSE_ALT_M = 11000.0      # metres — above this: isothermal stratosphere
TROPOPAUSE_TEMP_K = 216.65      # K at tropopause


@dataclass
class AtmosphereState:
    """ISA atmosphere state at a given altitude."""
    altitude_m: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    speed_of_sound_mps: float
    oat_c: float           # outside air temperature in °C (includes offset)

    @property
    def temperature_c(self) -> float:
        return self.temperature_k - 273.15


def isa_atmosphere(
    altitude_m: float,
    delta_isa_c: float = 0.0,
) -> AtmosphereState:
    """
    Compute ISA atmosphere at the given altitude.

    Args:
        altitude_m: Geopotential altitude in metres (0–20 000 m).
        delta_isa_c: ISA temperature deviation in °C (e.g., +10 for hot day).

    Returns:
        AtmosphereState with temperature, pressure, density, speed of sound.

    Reference:
        ICAO Doc 7488 / ISO 2533 International Standard Atmosphere.
    """
    alt = max(0.0, float(altitude_m))

    if alt <= TROPOPAUSE_ALT_M:
        # Troposphere: linear temperature lapse
        T_isa_k = SEA_LEVEL_TEMP_K - LAPSE_RATE * alt
        P_pa = SEA_LEVEL_PRESSURE_PA * (T_isa_k / SEA_LEVEL_TEMP_K) ** (
            GRAVITY / (LAPSE_RATE * GAS_CONSTANT_AIR)
        )
    else:
        # Lower stratosphere: isothermal
        # First compute pressure at tropopause
        T_isa_k = TROPOPAUSE_TEMP_K
        P_tropo = SEA_LEVEL_PRESSURE_PA * (TROPOPAUSE_TEMP_K / SEA_LEVEL_TEMP_K) ** (
            GRAVITY / (LAPSE_RATE * GAS_CONSTANT_AIR)
        )
        P_pa = P_tropo * math.exp(
            -GRAVITY * (alt - TROPOPAUSE_ALT_M) / (GAS_CONSTANT_AIR * TROPOPAUSE_TEMP_K)
        )

    # Apply ISA deviation (only to temperature, not pressure)
    T_total_k = T_isa_k + delta_isa_c
    rho = P_pa / (GAS_CONSTANT_AIR * T_total_k)
    a = math.sqrt(GAMMA_AIR * GAS_CONSTANT_AIR * T_total_k)

    return AtmosphereState(
        altitude_m=alt,
        temperature_k=T_total_k,
        pressure_pa=P_pa,
        density_kg_m3=rho,
        speed_of_sound_mps=a,
        oat_c=T_total_k - 273.15,
    )
