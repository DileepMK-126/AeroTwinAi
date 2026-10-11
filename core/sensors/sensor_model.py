"""
AEROTWIN — Phase 2: Sensor Model

Applies noise, drift, dropout, saturation, and quantization to engine state
to simulate realistic sensor readings from a UAV engine monitoring system.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from configs.config_loader import load_sensors_config
from core.physics.engine_model import EngineState


class SensorModel:
    """
    Applies realistic sensor imperfections per channel:
      - Gaussian noise (std from config)
      - Drift (linear drift over time)
      - Dropout (NaN with given probability)
      - Saturation (clamp to min/max)
      - Quantization (round to step)
      - First-order lag (low-pass filter)

    For per-cylinder channels (cht_c, egt_c): applies independently per cylinder.
    """

    def __init__(
        self,
        sensor_config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        cfg_root = sensor_config or load_sensors_config()
        self._cfg = cfg_root.get("sensors", {})
        self._rng = random.Random(seed)

        # Per-channel lag state: previous filtered value
        self._lag_state: Dict[str, Any] = {}
        # Per-channel cumulative drift state
        self._drift_state: Dict[str, float] = {}

    # -------------------------------------------------------------------------
    # Main apply function
    # -------------------------------------------------------------------------

    def apply(self, state: EngineState, t_s: float) -> Dict[str, Any]:
        """
        Apply sensor model to engine state at time t_s.
        Returns a dict of sensed values (matching TelemetryRecord fields).
        """
        out: Dict[str, Any] = {}

        # Single-value channels
        scalar_map = {
            "rpm":           state.rpm,
            "throttle_pct":  state.throttle * 100.0,
            "map_kpa":       state.map_kpa,
            "boost_kpa":     state.boost_kpa,
            "fuel_flow_lph": state.fuel_flow_lph,
            "oil_temp_c":    state.oil_temp_c,
            "oil_press_kpa": state.oil_press_kpa,
            "coolant_temp_c": state.coolant_temp_c,
            "fuel_press_kpa": 150.0,   # modelled value
            "bus_voltage_v": state.bus_voltage_v,
            "vib_rms_g":     state.vib_rms_g,
        }

        for ch, true_val in scalar_map.items():
            out[ch] = self._process_channel(ch, true_val, t_s)

        # Per-cylinder channels
        cht_sensed = []
        egt_sensed = []
        for i in range(4):
            cht_ch = f"cht_c_cyl{i+1}"
            egt_ch = f"egt_c_cyl{i+1}"
            cht_sensed.append(self._process_channel("cht_c", state.cht_c[i], t_s, sub_key=cht_ch))
            egt_sensed.append(self._process_channel("egt_c", state.egt_c[i], t_s, sub_key=egt_ch))

        out["cht_c"] = cht_sensed
        out["egt_c"] = egt_sensed

        return out

    # -------------------------------------------------------------------------
    # Per-channel processing pipeline
    # -------------------------------------------------------------------------

    def _process_channel(
        self,
        channel: str,
        true_val: float,
        t_s: float,
        sub_key: Optional[str] = None,
    ) -> Optional[float]:
        """Apply sensor pipeline for a single channel value."""
        key = sub_key or channel
        cfg = self._cfg.get(channel, {})

        # 1. Dropout
        dropout_prob = cfg.get("dropout_prob", 0.0)
        if self._rng.random() < dropout_prob:
            return None  # sensor reading lost

        # 2. Drift (accumulates over time)
        drift_rate = cfg.get("drift_rate", 0.0)
        if drift_rate > 0:
            if key not in self._drift_state:
                self._drift_state[key] = 0.0
            self._drift_state[key] += drift_rate * (1.0 / 3600.0)  # per timestep at 1Hz
            val = true_val + self._drift_state[key]
        else:
            val = true_val

        # 3. Gaussian noise
        noise_std = cfg.get("noise_std", 0.0)
        if noise_std > 0:
            val += self._rng.gauss(0.0, noise_std)

        # 4. First-order lag (low-pass filter)
        tau = cfg.get("lag_tau_s", 0.0)
        if tau > 0:
            if key not in self._lag_state:
                self._lag_state[key] = val  # initialise to first value
            prev = self._lag_state[key]
            alpha = 1.0 / (tau + 1.0)   # approximate discrete-time
            val = prev + alpha * (val - prev)
            self._lag_state[key] = val

        # 5. Quantization
        quant = cfg.get("quantization_step", 0.0)
        if quant > 0:
            val = round(val / quant) * quant

        # 6. Saturation
        sat_min = cfg.get("saturation_min", None)
        sat_max = cfg.get("saturation_max", None)
        if sat_min is not None:
            val = max(sat_min, val)
        if sat_max is not None:
            val = min(sat_max, val)

        return val

    # -------------------------------------------------------------------------
    # Sensor fault injection (called by FaultInjector)
    # -------------------------------------------------------------------------

    def apply_sensor_fault(
        self,
        sensed_values: Dict[str, Any],
        sensor_faults: Dict[str, Any],
        t_s: float,
    ) -> Dict[str, Any]:
        """
        Apply sensor-specific faults (bias, drift, stuck-at, spikes, dropout)
        on top of already-sensed values.
        """
        result = dict(sensed_values)
        for channel, fault_info in sensor_faults.items():
            fault_type = fault_info.get("type", "")
            sev = fault_info.get("severity", 0.0)

            if channel not in result:
                continue

            if fault_type == "sensor_bias":
                bias = fault_info.get("bias", 0.0)
                val = result[channel]
                if isinstance(val, list):
                    result[channel] = [v + bias if v is not None else None for v in val]
                elif val is not None:
                    result[channel] = val + bias

            elif fault_type == "sensor_drift":
                drift = fault_info.get("bias", 0.0)  # cumulative drift stored as bias
                val = result[channel]
                if isinstance(val, list):
                    result[channel] = [v + drift if v is not None else None for v in val]
                elif val is not None:
                    result[channel] = val + drift

            elif fault_type == "sensor_stuck":
                # Stuck at value recorded at fault onset
                stuck_key = f"_stuck_{channel}"
                if stuck_key not in self._lag_state:
                    self._lag_state[stuck_key] = result[channel]
                result[channel] = self._lag_state[stuck_key]

            elif fault_type == "sensor_spikes":
                if self._rng.random() < sev * 0.1:  # 10% chance of spike at max severity
                    val = result[channel]
                    spike_mag = self._rng.uniform(5, 50) * (2 * self._rng.randint(0, 1) - 1)
                    if isinstance(val, list):
                        idx = self._rng.randint(0, len(val) - 1)
                        val_list = list(val)
                        if val_list[idx] is not None:
                            val_list[idx] += spike_mag
                        result[channel] = val_list
                    elif val is not None:
                        result[channel] = val + spike_mag

            elif fault_type == "sensor_dropout":
                if self._rng.random() < sev * 0.5:
                    result[channel] = None

            elif fault_type == "sensor_noise_burst":
                burst_std = sev * 20.0
                val = result[channel]
                if isinstance(val, list):
                    result[channel] = [
                        v + self._rng.gauss(0, burst_std) if v is not None else None
                        for v in val
                    ]
                elif val is not None:
                    result[channel] = val + self._rng.gauss(0, burst_std)

        return result
