"""
AEROTWIN — Configuration Loader
Loads and validates all YAML configs from the configs/ directory.
Engine configuration is swappable — point to a different YAML with the same keys.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


# ---------------------------------------------------------------------------
# Engine config models
# ---------------------------------------------------------------------------

class EngineMeta(BaseModel):
    engine_id: str
    description: str
    displacement_cc: float
    cylinders: int
    strokes: int
    compression_ratio: float
    bore_mm: float
    stroke_mm: float
    firing_order: List[int]


class EnginePower(BaseModel):
    max_power_kw: float
    max_rpm: float
    idle_rpm: float
    max_continuous_rpm: float
    max_torque_nm: float
    specific_fuel_consumption_g_per_kwh: float


class EngineTurbocharger(BaseModel):
    type: str
    max_boost_kpa: float
    critical_altitude_m: float
    wastegate: str
    turbo_efficiency_compressor: float
    turbo_efficiency_turbine: float
    turbo_inertia_kgm2: float


class EngineThermal(BaseModel):
    cht_max_c: float
    cht_normal_cruise_c: float
    egt_max_c: float
    egt_normal_cruise_c: float
    oil_temp_max_c: float
    oil_temp_normal_c: float
    coolant_temp_max_c: float
    cht_time_constant_s: float
    egt_time_constant_s: float
    oil_time_constant_s: float


class EngineLubrication(BaseModel):
    oil_press_min_kpa: float
    oil_press_normal_kpa: float
    oil_press_max_kpa: float
    oil_volume_liters: float


class EngineFuel(BaseModel):
    fuel_type: str
    lhv_mj_per_kg: float
    stoich_afr: float
    lambda_target_cruise: float
    lambda_target_takeoff: float
    fuel_density_kg_per_l: float


class EngineFriction(BaseModel):
    fmep_bar_a: float
    fmep_bar_b: float


class EngineVolumetricEfficiency(BaseModel):
    ve_ref: float
    ve_rpm_peak: float
    ve_rpm_exponent: float


class EnginePropeller(BaseModel):
    type: str
    diameter_m: float
    pitch_m: float
    blade_count: int
    cp_ref: float
    ct_ref: float
    efficiency_max: float


class EngineElectrical(BaseModel):
    alternator_max_a: float
    bus_voltage_nominal_v: float
    battery_capacity_ah: float


class EngineConfig(BaseModel):
    meta: EngineMeta
    power: EnginePower
    turbocharger: EngineTurbocharger
    thermal: EngineThermal
    lubrication: EngineLubrication
    fuel: EngineFuel
    friction: EngineFriction
    volumetric_efficiency: EngineVolumetricEfficiency
    propeller: EnginePropeller
    electrical: EngineElectrical


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    return data


def load_engine_config(
    yaml_path: Optional[Union[str, Path]] = None,
) -> EngineConfig:
    """
    Load and validate the engine reference configuration.
    Default: configs/engine_reference.yaml
    Pass yaml_path to load a different engine config (must have same keys).
    """
    path = Path(yaml_path) if yaml_path else CONFIGS_DIR / "engine_reference.yaml"
    raw = load_yaml(path)
    config = EngineConfig.model_validate(raw)
    logger.info("Loaded engine config: %s from %s", config.meta.engine_id, path)
    return config


def load_sensors_config() -> Dict[str, Any]:
    path = CONFIGS_DIR / "sensors.yaml"
    return load_yaml(path)


def load_faults_config() -> Dict[str, Any]:
    path = CONFIGS_DIR / "faults.yaml"
    return load_yaml(path)


def load_mission_config() -> Dict[str, Any]:
    path = CONFIGS_DIR / "mission.yaml"
    return load_yaml(path)


def load_models_config() -> Dict[str, Any]:
    path = CONFIGS_DIR / "models.yaml"
    return load_yaml(path)


# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Application-level settings loaded from environment variables."""
    env: str = Field(default="development")
    log_level: str = Field(default="INFO")
    secret_key: str = Field(default="change-me-in-production")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    mqtt_host: str = Field(default="localhost")
    mqtt_port: int = Field(default=1883)
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    sqlite_path: str = Field(default="./data/aerotwin.db")
    dataset_dir: str = Field(default="./data/datasets")
    model_registry_dir: str = Field(default="./data/models")
    sim_seed: int = Field(default=42)
    sim_dt: float = Field(default=1.0)
    sim_realtime_factor: float = Field(default=600.0)
    next_public_api_url: str = Field(default="http://localhost:8000")
    next_public_ws_url: str = Field(default="ws://localhost:8000/ws")
    ingest_api_key: str = Field(default="change-me-api-key")
    ingest_rate_limit: int = Field(default=1000)


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        env=os.getenv("AEROTWIN_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        mqtt_host=os.getenv("MQTT_HOST", "localhost"),
        mqtt_port=int(os.getenv("MQTT_PORT", "1883")),
        mqtt_username=os.getenv("MQTT_USERNAME"),
        mqtt_password=os.getenv("MQTT_PASSWORD"),
        sqlite_path=os.getenv("SQLITE_PATH", "./data/aerotwin.db"),
        dataset_dir=os.getenv("DATASET_DIR", "./data/datasets"),
        model_registry_dir=os.getenv("MODEL_REGISTRY_DIR", "./data/models"),
        sim_seed=int(os.getenv("SIM_SEED", "42")),
        sim_dt=float(os.getenv("SIM_DT", "1.0")),
        sim_realtime_factor=float(os.getenv("SIM_REALTIME_FACTOR", "600.0")),
        ingest_api_key=os.getenv("INGEST_API_KEY", "change-me-api-key"),
        ingest_rate_limit=int(os.getenv("INGEST_RATE_LIMIT", "1000")),
    )
