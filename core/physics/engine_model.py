"""
AEROTWIN — Phase 1: Engine Physics Model

Mean-value engine model with Otto-cycle losses for a Rotax 914-class
turbocharged 4-cylinder 4-stroke spark-ignition aero piston engine.

Models implemented:
  - Volumetric efficiency (VE) curve
  - Air mass flow
  - Fuel flow and air-fuel ratio (AFR)
  - Indicated mean effective pressure (IMEP)
  - Friction mean effective pressure (FMEP)
  - Brake mean effective pressure (BMEP) and brake power
  - Heat release split: shaft work / coolant / exhaust
  - EGT thermal dynamics (lumped first-order)
  - CHT thermal dynamics (lumped first-order, per cylinder)
  - Oil thermal model
  - Turbocharger boost model with altitude correction

All parameters loaded from configs/engine_reference.yaml.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from configs.config_loader import EngineConfig
from core.physics.atmosphere import AtmosphereState, isa_atmosphere


# ---------------------------------------------------------------------------
# Engine State Dataclass
# ---------------------------------------------------------------------------

@dataclass
class EngineState:
    """Complete instantaneous engine state vector."""
    # Inputs / controls
    rpm: float = 0.0             # engine speed [RPM]
    throttle: float = 0.0        # throttle 0–1
    altitude_m: float = 0.0
    delta_isa_c: float = 0.0

    # Atmosphere
    atm: Optional[AtmosphereState] = None

    # Air/fuel
    air_mass_flow_kg_s: float = 0.0
    fuel_flow_kg_s: float = 0.0
    fuel_flow_lph: float = 0.0
    afr: float = 15.0            # actual air-fuel ratio
    lambda_: float = 1.0         # excess air ratio (lambda = AFR / AFR_stoich)

    # Pressure / boost
    map_kpa: float = 101.325     # manifold absolute pressure [kPa]
    boost_kpa: float = 101.325   # turbo boost pressure [kPa absolute]
    boost_ratio: float = 1.0     # P_boost / P_ambient

    # Torque / power
    imep_kpa: float = 0.0        # indicated mean effective pressure
    fmep_kpa: float = 0.0        # friction mep
    bmep_kpa: float = 0.0        # brake mep
    brake_torque_nm: float = 0.0
    brake_power_kw: float = 0.0
    thermal_efficiency: float = 0.35  # overall

    # Heat release
    q_fuel_kw: float = 0.0       # total fuel heat release rate
    q_shaft_kw: float = 0.0      # shaft work
    q_coolant_kw: float = 0.0    # heat to coolant
    q_exhaust_kw: float = 0.0    # heat to exhaust gases

    # Thermal states (persistent — updated each timestep)
    egt_c: List[float] = field(default_factory=lambda: [750.0] * 4)   # per cylinder
    cht_c: List[float] = field(default_factory=lambda: [110.0] * 4)   # per cylinder
    oil_temp_c: float = 90.0
    coolant_temp_c: float = 95.0

    # Lubrication
    oil_press_kpa: float = 400.0

    # Electrical
    bus_voltage_v: float = 14.0

    # Vibration (base level — faults add to this)
    vib_rms_g: float = 0.02

    # Health parameters (multipliers on key subsystem quantities)
    # 1.0 = healthy, degrades toward 0 or increases with wear
    hp_ve_factor: float = 1.0          # volumetric efficiency health
    hp_friction_factor: float = 1.0    # friction (>1 = more friction)
    hp_cooling_effectiveness: float = 1.0  # cooling efficiency
    hp_injector_flow: float = 1.0      # fuel delivery health
    hp_boost_efficiency: float = 1.0   # turbocharger health
    hp_combustion_eff: List[float] = field(default_factory=lambda: [1.0] * 4)


# ---------------------------------------------------------------------------
# Engine Model
# ---------------------------------------------------------------------------

class PistonEngineModel:
    """
    Mean-value piston engine model for AEROTWIN.
    Computes the full engine state given RPM, throttle, altitude, and ISA deviation.
    Uses health parameters from the digital twin for degraded operation modelling.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.cfg = config
        # Pre-compute derived geometry
        self.displacement_m3 = config.meta.displacement_cc * 1e-6
        self.n_cylinders = config.meta.cylinders
        self.bore_m = config.meta.bore_mm * 1e-3
        self.stroke_m = config.meta.stroke_mm * 1e-3
        self.cyl_displacement_m3 = self.displacement_m3 / self.n_cylinders

    # -------------------------------------------------------------------------
    # Step function (called at each timestep)
    # -------------------------------------------------------------------------

    def step(
        self,
        state: EngineState,
        dt: float,
        throttle: float,
        rpm: float,
        altitude_m: float = 0.0,
        delta_isa_c: float = 0.0,
    ) -> EngineState:
        """
        Advance the engine state by dt seconds.

        Args:
            state:      Current engine state (with thermal states and health params).
            dt:         Timestep in seconds.
            throttle:   Throttle position 0–1.
            rpm:        Engine speed in RPM.
            altitude_m: Current altitude in metres.
            delta_isa_c: ISA temperature deviation in °C.

        Returns:
            Updated EngineState.
        """
        s = state
        s.throttle = max(0.0, min(1.0, throttle))
        s.rpm = max(s.rpm * 0.0, rpm)  # accept externally set RPM
        s.altitude_m = altitude_m
        s.delta_isa_c = delta_isa_c

        # 1. Atmosphere
        atm = isa_atmosphere(altitude_m, delta_isa_c)
        s.atm = atm

        # 2. Boost / MAP
        boost_kpa, map_kpa = self._compute_boost(s, atm)
        s.boost_kpa = boost_kpa
        s.map_kpa = map_kpa

        # 3. Air mass flow
        air_mass_flow = self._compute_air_mass_flow(s, atm, map_kpa)
        s.air_mass_flow_kg_s = air_mass_flow

        # 4. Fuel flow and AFR
        fuel_kg_s, afr, lambda_ = self._compute_fuel_flow(s, air_mass_flow)
        s.fuel_flow_kg_s = fuel_kg_s
        s.fuel_flow_lph = (
            fuel_kg_s / self.cfg.fuel.fuel_density_kg_per_l * 3600.0
        )
        s.afr = afr
        s.lambda_ = lambda_

        # 5. Heat release
        lhv_j_per_kg = self.cfg.fuel.lhv_mj_per_kg * 1e6
        q_fuel_w = fuel_kg_s * lhv_j_per_kg
        s.q_fuel_kw = q_fuel_w * 1e-3

        # 6. IMEP, FMEP, BMEP, brake power
        imep, fmep, bmep, torque, power = self._compute_power(s, q_fuel_w)
        s.imep_kpa = imep
        s.fmep_kpa = fmep
        s.bmep_kpa = bmep
        s.brake_torque_nm = torque
        s.brake_power_kw = power * 1e-3
        s.q_shaft_kw = max(0.0, power * 1e-3)

        # 7. Heat split
        q_coolant_w, q_exhaust_w = self._heat_split(q_fuel_w, power)
        s.q_coolant_kw = q_coolant_w * 1e-3
        s.q_exhaust_kw = q_exhaust_w * 1e-3

        # 8. Thermal dynamics
        self._update_thermal(s, dt, q_exhaust_w, q_coolant_w, atm)

        # 9. Oil pressure
        s.oil_press_kpa = self._oil_pressure(s)

        # 10. Vibration (base level without faults)
        s.vib_rms_g = self._base_vibration(s)

        return s

    # -------------------------------------------------------------------------
    # Sub-model: Boost / Turbocharger
    # -------------------------------------------------------------------------

    def _compute_boost(
        self, s: EngineState, atm: AtmosphereState
    ) -> tuple[float, float]:
        """
        Compute turbocharger boost and MAP.

        Below critical altitude: boost maintains sea-level-equivalent MAP.
        Above critical altitude: boost capacity falls off.
        Wastegate is pneumatic/automatic — limits boost to max_boost_kpa.

        Returns:
            (boost_kpa, map_kpa) both in kPa absolute
        """
        P_amb_kpa = atm.pressure_pa * 1e-3
        crit_alt = self.cfg.turbocharger.critical_altitude_m
        max_boost = self.cfg.turbocharger.max_boost_kpa  # absolute kPa

        # Throttle + health factor modulates MAP
        throttle_factor = s.throttle
        boost_health = s.hp_boost_efficiency

        if s.altitude_m <= crit_alt:
            # Turbocharger maintains approximately sea-level MAP
            # Max MAP = sea-level pressure * throttle
            sea_level_kpa = 101.325
            target_map = sea_level_kpa * throttle_factor
            # Boost needed to achieve this MAP
            boost_kpa = min(target_map / max(boost_health, 0.1), max_boost)
            map_kpa = min(boost_kpa * boost_health, max_boost) * throttle_factor
        else:
            # Above critical altitude: boost falls linearly
            excess = s.altitude_m - crit_alt
            falloff = math.exp(-excess / 3000.0)   # exponential falloff
            boost_kpa = min(P_amb_kpa + (max_boost - 101.325) * falloff * boost_health, max_boost)
            map_kpa = boost_kpa * throttle_factor

        return boost_kpa, map_kpa

    # -------------------------------------------------------------------------
    # Sub-model: Air Mass Flow
    # -------------------------------------------------------------------------

    def _compute_air_mass_flow(
        self,
        s: EngineState,
        atm: AtmosphereState,
        map_kpa: float,
    ) -> float:
        """
        Compute inducted air mass flow using volumetric efficiency.

        For a 4-stroke engine:
            m_air = (N/2) * Ve * VE * rho_inlet

        where rho_inlet = MAP / (R_air * T_inlet)
        """
        N_rps = s.rpm / 60.0
        strokes_per_rev = 2  # 4-stroke = fires every 2 revolutions

        # VE curve: bell-shaped around ve_rpm_peak
        ve_cfg = self.cfg.volumetric_efficiency
        rpm_ratio = s.rpm / max(ve_cfg.ve_rpm_peak, 1.0)
        ve = ve_cfg.ve_ref * (
            1.0 - ve_cfg.ve_rpm_exponent * (rpm_ratio - 1.0) ** 2
        )
        ve = max(0.3, min(1.0, ve)) * s.hp_ve_factor

        # Inlet air density from MAP (absolute) and approximate inlet temperature
        T_inlet_k = atm.temperature_k + 30.0  # intercooler/charge heating
        rho_inlet = (map_kpa * 1e3) / (287.05 * T_inlet_k)

        air_mass_flow = (N_rps / strokes_per_rev) * self.displacement_m3 * ve * rho_inlet
        return max(0.0, air_mass_flow)

    # -------------------------------------------------------------------------
    # Sub-model: Fuel Flow and AFR
    # -------------------------------------------------------------------------

    def _compute_fuel_flow(
        self, s: EngineState, air_kg_s: float
    ) -> tuple[float, float, float]:
        """
        Compute fuel mass flow, AFR, and lambda.

        Target lambda is from config (cruise/takeoff) modulated by throttle.
        Injector health factor scales the actual fuel delivery.
        """
        stoich_afr = self.cfg.fuel.stoich_afr
        lambda_target = (
            self.cfg.fuel.lambda_target_takeoff
            if s.throttle > 0.9
            else self.cfg.fuel.lambda_target_cruise
        )
        target_afr = stoich_afr * lambda_target
        # Actual injector flow limited by health factor
        fuel_kg_s = (air_kg_s / target_afr) * s.hp_injector_flow
        fuel_kg_s = max(0.0, fuel_kg_s)
        # Actual AFR and lambda
        if fuel_kg_s > 0:
            afr = air_kg_s / fuel_kg_s
            lambda_ = afr / stoich_afr
        else:
            afr = 999.0
            lambda_ = 999.0
        return fuel_kg_s, afr, lambda_

    # -------------------------------------------------------------------------
    # Sub-model: Power (IMEP → FMEP → BMEP → Brake Power)
    # -------------------------------------------------------------------------

    def _compute_power(
        self, s: EngineState, q_fuel_w: float
    ) -> tuple[float, float, float, float, float]:
        """
        Compute IMEP, FMEP, BMEP, brake torque, and brake power.

        Indicated thermal efficiency uses an idealised Otto-cycle efficiency
        with combustion losses.
        """
        # Otto-cycle ideal thermal efficiency (air-standard)
        cr = self.cfg.meta.compression_ratio
        eta_otto = 1.0 - 1.0 / (cr ** (GAMMA_MINUS_1))

        # Apply combustion efficiency per cylinder (average)
        eta_comb = sum(s.hp_combustion_eff) / len(s.hp_combustion_eff)

        # Friction mep (bar → kPa)
        fmep_cfg = self.cfg.friction
        fmep_bar = fmep_cfg.fmep_bar_a + fmep_cfg.fmep_bar_b * s.rpm
        fmep_kpa = fmep_bar * 100.0 * s.hp_friction_factor

        # Indicated power → indicated mep
        if self.displacement_m3 > 0 and s.rpm > 100:
            ind_power_w = q_fuel_w * eta_otto * eta_comb
            # P_i = IMEP * V_d * (N/2) for 4-stroke
            N_rps = s.rpm / 60.0
            imep_pa = ind_power_w / (self.displacement_m3 * N_rps / 2.0)
            imep_kpa = imep_pa * 1e-3
        else:
            imep_kpa = 0.0
            ind_power_w = 0.0

        # Brake mep = IMEP - FMEP
        bmep_kpa = max(0.0, imep_kpa - fmep_kpa)

        # Brake torque and power
        N_rps = s.rpm / 60.0
        if N_rps > 0:
            brake_power_w = bmep_kpa * 1e3 * self.displacement_m3 * N_rps / 2.0
            brake_torque_nm = brake_power_w / (2.0 * math.pi * N_rps)
        else:
            brake_power_w = 0.0
            brake_torque_nm = 0.0

        return imep_kpa, fmep_kpa, bmep_kpa, brake_torque_nm, brake_power_w

    # -------------------------------------------------------------------------
    # Sub-model: Heat Split
    # -------------------------------------------------------------------------

    def _heat_split(self, q_fuel_w: float, shaft_w: float) -> tuple[float, float]:
        """
        Split total fuel heat between coolant and exhaust.

        Typical aero piston engine:
          ~30% shaft work, ~35% coolant, ~35% exhaust (varies)
        """
        q_remaining = max(0.0, q_fuel_w - shaft_w)
        # Coolant fraction depends on cooling effectiveness health param
        # (reduce effectiveness → more heat to metal → CHT rises)
        # We model a fixed split for now; twin health params adjust
        coolant_fraction = 0.50   # of remaining heat
        q_coolant = q_remaining * coolant_fraction
        q_exhaust = q_remaining * (1.0 - coolant_fraction)
        return q_coolant, q_exhaust

    # -------------------------------------------------------------------------
    # Sub-model: Thermal Dynamics
    # -------------------------------------------------------------------------

    def _update_thermal(
        self,
        s: EngineState,
        dt: float,
        q_exhaust_w: float,
        q_coolant_w: float,
        atm: AtmosphereState,
    ) -> None:
        """
        Update thermal states using lumped first-order RC models.
        Modifies state in-place.
        """
        cfg = self.cfg.thermal

        # --- EGT (per cylinder, driven by exhaust enthalpy and mass flow) ---
        m_dot_exhaust = max(1e-4, s.air_mass_flow_kg_s + s.fuel_flow_kg_s)
        cp_exhaust = 1150.0  # J / (kg * K) for combustion gases
        delta_t_exhaust = (q_exhaust_w / (m_dot_exhaust * cp_exhaust))
        for i in range(self.n_cylinders):
            eta_comb_i = s.hp_combustion_eff[i]
            # Poor combustion → higher EGT (unburnt fuel burns in exhaust manifold)
            egt_driving = atm.oat_c + delta_t_exhaust * (2.0 - eta_comb_i)
            egt_driving = max(0.0, min(egt_driving, 1100.0))
            # First-order lag
            tau = cfg.egt_time_constant_s
            s.egt_c[i] += (egt_driving - s.egt_c[i]) * (dt / (tau + dt))

        # --- CHT (per cylinder, driven by coolant heat and ambient) ---
        q_per_cyl_coolant_w = q_coolant_w / self.n_cylinders
        cooling_eff = max(0.05, s.hp_cooling_effectiveness)
        # Cooling conductance scales with engine speed (coolant pump + fan/airflow)
        rpm_ratio = max(0.2, s.rpm / max(1.0, float(self.cfg.power.max_rpm)))
        cht_conductance = (35.0 + 30.0 * rpm_ratio) * cooling_eff
        for i in range(self.n_cylinders):
            cht_driving = atm.oat_c + (q_per_cyl_coolant_w / cht_conductance)
            cht_driving = max(-40.0, min(cht_driving, 290.0))
            tau = cfg.cht_time_constant_s
            s.cht_c[i] += (cht_driving - s.cht_c[i]) * (dt / (tau + dt))

        # --- Oil temperature (driven by friction heat and cooling) ---
        q_friction_w = (s.fmep_kpa * 1e3 * self.displacement_m3 * s.rpm / 120.0)
        oil_conductance = 55.0 + 35.0 * rpm_ratio
        oil_driving = atm.oat_c + (q_friction_w / oil_conductance)
        oil_driving = max(-40.0, min(oil_driving, 180.0))
        tau_oil = cfg.oil_time_constant_s
        s.oil_temp_c += (oil_driving - s.oil_temp_c) * (dt / (tau_oil + dt))

        # --- Coolant temperature ---
        coolant_conductance = (110.0 + 90.0 * rpm_ratio) * cooling_eff
        coolant_driving = atm.oat_c + (q_coolant_w / coolant_conductance)
        coolant_driving = max(-40.0, min(coolant_driving, 180.0))
        s.coolant_temp_c += (coolant_driving - s.coolant_temp_c) * (dt / (60.0 + dt))

    # -------------------------------------------------------------------------
    # Sub-model: Oil Pressure
    # -------------------------------------------------------------------------

    def _oil_pressure(self, s: EngineState) -> float:
        """
        Oil pressure model: increases with RPM, decreases with oil temperature.
        Severely drops when friction health degrades (oil starvation / pump loss).
        """
        cfg = self.cfg.lubrication
        rpm_factor = min(s.rpm / self.cfg.power.max_rpm, 1.0)
        temp_factor = max(0.4, 1.0 - (s.oil_temp_c - 90.0) / 150.0)
        # Severe friction increase directly impairs lubrication film and indicates loss of oil
        friction_factor = max(0.1, 1.0 - max(0.0, s.hp_friction_factor - 1.0) * 0.6)
        p = cfg.oil_press_normal_kpa * rpm_factor * temp_factor * friction_factor
        return max(10.0, min(p, cfg.oil_press_max_kpa))

    # -------------------------------------------------------------------------
    # Sub-model: Base Vibration
    # -------------------------------------------------------------------------

    def _base_vibration(self, s: EngineState) -> float:
        """
        Base vibration RMS in g — proportional to RPM (combustion excitation).
        Fault injection adds imbalance/bearing components on top.
        """
        return 0.005 + (s.rpm / self.cfg.power.max_rpm) * 0.03


# ---------------------------------------------------------------------------
# Module-level constant needed in power calculation
# ---------------------------------------------------------------------------

GAMMA_MINUS_1 = 1.4 - 1.0   # = 0.4 for air
