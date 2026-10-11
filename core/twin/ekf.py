"""
AEROTWIN — Phase 3: Extended Kalman Filter (EKF) State Estimator

Fuses noisy sensor measurements with the physics model to estimate:
1. Current engine state (RPM, MAP, temperatures, pressures)
2. Latent health parameters (VE factor, friction factor, etc.)

The EKF runs in parallel with the physics twin:
  - Prediction step: use physics model to propagate state
  - Update step: correct using sensor measurements

Health parameter identification:
  The state vector is augmented with the 6 health parameters,
  which are estimated as slowly-drifting random walks.

Residual generation:
  After each update: residual = measurement - expected_from_twin
  Normalised residuals are the primary input to detection algorithms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from configs.config_loader import load_engine_config, EngineConfig
from core.physics.engine_model import EngineState, PistonEngineModel

logger = logging.getLogger("aerotwin.twin.ekf")

# ---------------------------------------------------------------------------
# State vector indices (must match _state_to_vec and _vec_to_state)
# ---------------------------------------------------------------------------

# Physical states
IDX_OIL_TEMP = 0
IDX_COOLANT_TEMP = 1
IDX_OIL_PRESS = 2
IDX_EGT_0 = 3
IDX_EGT_1 = 4
IDX_EGT_2 = 5
IDX_EGT_3 = 6
IDX_CHT_0 = 7
IDX_CHT_1 = 8
IDX_CHT_2 = 9
IDX_CHT_3 = 10

# Health parameters (augmented state — evolve as random walks)
IDX_HP_VE = 11           # volumetric efficiency factor
IDX_HP_FRICTION = 12     # friction factor
IDX_HP_COOLING = 13      # cooling effectiveness
IDX_HP_INJECTOR = 14     # injector flow factor
IDX_HP_BOOST = 15        # boost efficiency
IDX_HP_COMB_AVG = 16     # average combustion efficiency

N_STATES = 17

# Measurement vector indices
M_OIL_TEMP = 0
M_COOLANT_TEMP = 1
M_OIL_PRESS = 2
M_EGT_0 = 3
M_EGT_1 = 4
M_EGT_2 = 5
M_EGT_3 = 6
M_CHT_0 = 7
M_CHT_1 = 8
M_CHT_2 = 9
M_CHT_3 = 10
N_MEASUREMENTS = 11


@dataclass
class TwinEstimate:
    """Output of the EKF estimator at each timestep."""
    # Estimated health parameters
    hp_ve_factor: float = 1.0
    hp_friction_factor: float = 1.0
    hp_cooling_effectiveness: float = 1.0
    hp_injector_flow: float = 1.0
    hp_boost_efficiency: float = 1.0
    hp_combustion_eff_avg: float = 1.0

    # Residuals (measurement - expected from twin)
    residuals: Dict[str, float] = field(default_factory=dict)
    # Normalised residuals (by expected noise std)
    normalised_residuals: Dict[str, float] = field(default_factory=dict)
    # Per-channel confidence (1 = full confidence, 0 = no confidence)
    confidence: Dict[str, float] = field(default_factory=dict)

    # EKF covariance trace (state uncertainty)
    state_uncertainty: float = 1.0

    # Estimated twin state
    egt_est: List[float] = field(default_factory=lambda: [750.0] * 4)
    cht_est: List[float] = field(default_factory=lambda: [110.0] * 4)
    oil_temp_est: float = 90.0
    coolant_temp_est: float = 95.0
    oil_press_est: float = 400.0

    # Sensor fault flag per channel
    sensor_fault_flags: Dict[str, bool] = field(default_factory=dict)


class ExtendedKalmanFilter:
    """
    EKF for engine state estimation and health parameter identification.

    State vector x (N_STATES=17):
      [oil_temp, coolant_temp, oil_press,
       egt_0..3, cht_0..3,
       hp_ve, hp_friction, hp_cooling, hp_injector, hp_boost, hp_comb_avg]

    Measurement vector z (N_MEASUREMENTS=11):
      [oil_temp, coolant_temp, oil_press, egt_0..3, cht_0..3]
    """

    def __init__(self, engine_config: Optional[EngineConfig] = None) -> None:
        self.cfg = engine_config or load_engine_config()
        self.engine_model = PistonEngineModel(self.cfg)

        # EKF state vector — initialise at nominal healthy values
        self.x = np.array([
            90.0,   # oil_temp
            95.0,   # coolant_temp
            400.0,  # oil_press
            750.0, 750.0, 750.0, 750.0,  # egt per cyl
            110.0, 110.0, 110.0, 110.0,  # cht per cyl
            1.0,    # hp_ve
            1.0,    # hp_friction
            1.0,    # hp_cooling
            1.0,    # hp_injector
            1.0,    # hp_boost
            1.0,    # hp_comb_avg
        ], dtype=np.float64)

        # Covariance matrix — initial uncertainty
        self.P = np.eye(N_STATES, dtype=np.float64)
        self.P[IDX_OIL_TEMP, IDX_OIL_TEMP] = 25.0
        self.P[IDX_COOLANT_TEMP, IDX_COOLANT_TEMP] = 25.0
        self.P[IDX_OIL_PRESS, IDX_OIL_PRESS] = 400.0
        for i in range(4):
            self.P[IDX_EGT_0 + i, IDX_EGT_0 + i] = 400.0  # egt
            self.P[IDX_CHT_0 + i, IDX_CHT_0 + i] = 100.0  # cht
        # Health params: high initial uncertainty
        for i in range(IDX_HP_VE, N_STATES):
            self.P[i, i] = 0.1

        # Process noise Q — how much we expect state to change per step
        self._Q = np.zeros((N_STATES, N_STATES))
        # Physical states: driven by engine conditions (moderate noise)
        self._Q[IDX_OIL_TEMP, IDX_OIL_TEMP] = 0.1
        self._Q[IDX_COOLANT_TEMP, IDX_COOLANT_TEMP] = 0.1
        self._Q[IDX_OIL_PRESS, IDX_OIL_PRESS] = 1.0
        for i in range(4):
            self._Q[IDX_EGT_0 + i, IDX_EGT_0 + i] = 2.0
            self._Q[IDX_CHT_0 + i, IDX_CHT_0 + i] = 0.5
        # Health params: slow random walk (small process noise)
        for i in range(IDX_HP_VE, N_STATES):
            self._Q[i, i] = 1e-5

        # Measurement noise R (from sensor config)
        self._R = np.diag([
            1.0,      # oil_temp (°C²)
            1.0,      # coolant_temp
            25.0,     # oil_press (kPa²)
            25.0, 25.0, 25.0, 25.0,   # egt per cyl
            4.0, 4.0, 4.0, 4.0,       # cht per cyl
        ])

        # Noise stds for normalisation (sqrt of R diagonal)
        self._meas_stds = np.sqrt(np.diag(self._R))

        # Measurement channel names (for residual dict)
        self._meas_names = [
            "oil_temp_c", "coolant_temp_c", "oil_press_kpa",
            "egt_c_1", "egt_c_2", "egt_c_3", "egt_c_4",
            "cht_c_1", "cht_c_2", "cht_c_3", "cht_c_4",
        ]

        # Sensor fault detection threshold (normalised residual)
        self._sensor_fault_threshold = 5.0  # sigma

        # History for smoothing
        self._step_count = 0

    # -------------------------------------------------------------------------
    # Main EKF step
    # -------------------------------------------------------------------------

    def step(
        self,
        measurements: Dict[str, Optional[float]],
        engine_state_estimate: EngineState,
        dt: float = 1.0,
    ) -> TwinEstimate:
        """
        Run one EKF prediction-update cycle.

        Args:
            measurements: Observed sensor values (keyed by field name).
            engine_state_estimate: Current physics model state (for prediction).
            dt: Timestep in seconds.

        Returns:
            TwinEstimate with health params, residuals, and confidence.
        """
        self._step_count += 1

        # ---- Prediction Step ----
        x_pred = self._predict_state(engine_state_estimate, dt)
        P_pred = self.P + self._Q * dt

        # ---- Build measurement vector ----
        z, z_mask = self._build_measurement_vector(measurements)

        if z_mask.sum() > 0:
            # ---- Compute expected measurement from predicted state ----
            z_exp = self._expected_measurement(x_pred)

            # ---- Innovation (residual) ----
            innov = z - z_exp

            # ---- Linearised measurement Jacobian (H) ----
            H = self._measurement_jacobian(x_pred)

            # ---- Only update with valid (non-missing) measurements ----
            valid_idx = np.where(z_mask)[0]
            H_v = H[valid_idx, :]
            innov_v = innov[valid_idx]
            R_v = self._R[np.ix_(valid_idx, valid_idx)]

            # ---- Innovation covariance ----
            S = H_v @ P_pred @ H_v.T + R_v

            # ---- Kalman gain ----
            K = P_pred @ H_v.T @ np.linalg.inv(S + 1e-8 * np.eye(S.shape[0]))

            # ---- State update ----
            x_upd = x_pred + K @ innov_v
            P_upd = (np.eye(N_STATES) - K @ H_v) @ P_pred

            # Clamp health parameters to physically valid range
            x_upd[IDX_HP_VE:] = np.clip(x_upd[IDX_HP_VE:], 0.3, 1.5)

            self.x = x_upd
            self.P = P_upd

            # ---- Normalised residuals ----
            meas_stds_v = self._meas_stds[valid_idx]
            norm_innov = innov_v / np.maximum(meas_stds_v, 1e-6)

            # ---- Compute full residual dict ----
            residuals = {}
            norm_residuals = {}
            confidence = {}
            sensor_faults = {}

            for local_idx, global_idx in enumerate(valid_idx):
                name = self._meas_names[global_idx]
                residuals[name] = float(innov[global_idx])
                nr = float(norm_innov[local_idx])
                norm_residuals[name] = nr
                # Confidence: high if residual is small
                confidence[name] = float(np.exp(-0.5 * nr**2))
                sensor_faults[name] = abs(nr) > self._sensor_fault_threshold

        else:
            # No valid measurements — propagate only
            self.x = x_pred
            self.P = P_pred
            residuals = {}
            norm_residuals = {}
            confidence = {}
            sensor_faults = {}

        # ---- Build output estimate ----
        return TwinEstimate(
            hp_ve_factor=float(self.x[IDX_HP_VE]),
            hp_friction_factor=float(self.x[IDX_HP_FRICTION]),
            hp_cooling_effectiveness=float(self.x[IDX_HP_COOLING]),
            hp_injector_flow=float(self.x[IDX_HP_INJECTOR]),
            hp_boost_efficiency=float(self.x[IDX_HP_BOOST]),
            hp_combustion_eff_avg=float(self.x[IDX_HP_COMB_AVG]),
            residuals=residuals,
            normalised_residuals=norm_residuals,
            confidence=confidence,
            state_uncertainty=float(np.trace(self.P)),
            egt_est=[float(self.x[IDX_EGT_0 + i]) for i in range(4)],
            cht_est=[float(self.x[IDX_CHT_0 + i]) for i in range(4)],
            oil_temp_est=float(self.x[IDX_OIL_TEMP]),
            coolant_temp_est=float(self.x[IDX_COOLANT_TEMP]),
            oil_press_est=float(self.x[IDX_OIL_PRESS]),
            sensor_fault_flags=sensor_faults,
        )

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def _predict_state(self, engine_state: EngineState, dt: float) -> np.ndarray:
        """
        Predict next state from physics model.
        Health parameters evolve as random walks (no deterministic change).
        """
        x = self.x.copy()

        # Update physical states from engine model
        x[IDX_OIL_TEMP] = engine_state.oil_temp_c
        x[IDX_COOLANT_TEMP] = engine_state.coolant_temp_c
        x[IDX_OIL_PRESS] = engine_state.oil_press_kpa
        for i in range(4):
            x[IDX_EGT_0 + i] = engine_state.egt_c[i]
            x[IDX_CHT_0 + i] = engine_state.cht_c[i]

        # Health params persist (random walk — Q handles uncertainty)
        return x

    # -------------------------------------------------------------------------
    # Measurement model
    # -------------------------------------------------------------------------

    def _expected_measurement(self, x: np.ndarray) -> np.ndarray:
        """Compute expected measurement vector from state."""
        return np.array([
            x[IDX_OIL_TEMP],
            x[IDX_COOLANT_TEMP],
            x[IDX_OIL_PRESS],
            x[IDX_EGT_0], x[IDX_EGT_1], x[IDX_EGT_2], x[IDX_EGT_3],
            x[IDX_CHT_0], x[IDX_CHT_1], x[IDX_CHT_2], x[IDX_CHT_3],
        ])

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Linearised measurement Jacobian H = d(h)/d(x).
        For our linear measurement model, H is constant.
        """
        H = np.zeros((N_MEASUREMENTS, N_STATES))
        # Physical states map directly to measurements
        H[M_OIL_TEMP, IDX_OIL_TEMP] = 1.0
        H[M_COOLANT_TEMP, IDX_COOLANT_TEMP] = 1.0
        H[M_OIL_PRESS, IDX_OIL_PRESS] = 1.0
        for i in range(4):
            H[M_EGT_0 + i, IDX_EGT_0 + i] = 1.0
            H[M_CHT_0 + i, IDX_CHT_0 + i] = 1.0
        return H

    def _build_measurement_vector(
        self, measurements: Dict[str, Optional[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build measurement vector and validity mask from sensor dict.
        Missing (None) measurements are masked out.
        """
        z = np.zeros(N_MEASUREMENTS)
        z_mask = np.zeros(N_MEASUREMENTS, dtype=bool)

        mapping = {
            "oil_temp_c": M_OIL_TEMP,
            "coolant_temp_c": M_COOLANT_TEMP,
            "oil_press_kpa": M_OIL_PRESS,
        }

        for key, idx in mapping.items():
            val = measurements.get(key)
            if val is not None:
                z[idx] = float(val)
                z_mask[idx] = True

        # Per-cylinder channels
        egt = measurements.get("egt_c")
        cht = measurements.get("cht_c")
        if isinstance(egt, list):
            for i in range(min(4, len(egt))):
                if egt[i] is not None:
                    z[M_EGT_0 + i] = float(egt[i])
                    z_mask[M_EGT_0 + i] = True
        if isinstance(cht, list):
            for i in range(min(4, len(cht))):
                if cht[i] is not None:
                    z[M_CHT_0 + i] = float(cht[i])
                    z_mask[M_CHT_0 + i] = True

        return z, z_mask

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """Reset EKF state to healthy defaults."""
        self.__init__(self.cfg)
