"""
AEROTWIN — Phase 2: Dataset Generator CLI

Generates labelled multi-mission Parquet datasets with:
  - No-fault missions
  - Single-fault missions (per fault type)
  - Multi-fault missions
  - Varied altitude, temperature, engine age conditions
  - Train/val/test split by mission (never by row)

Usage:
    python -m core.faults.dataset_generator --missions 500 --seed 1
    make sim ARGS="--missions 500 --seed 1"
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from configs.config_loader import load_faults_config
from core.faults.fault_injector import ActiveFault, FaultInjector
from core.sensors.sensor_model import SensorModel
from core.simulator.simulator import Simulator
from core.logging_config import configure_logging
from storage.storage import ParquetDatasetWriter

logger = logging.getLogger("aerotwin.dataset_gen")

FAULT_SCENARIOS = [
    "no_fault",
    "spark_plug_degradation",
    "injector_clogging",
    "bearing_wear",
    "piston_ring_wear",
    "coolant_loss",
    "oil_starvation",
    "turbocharger_degradation",
    "wastegate_stuck",
    "fuel_pump_degradation",
    "alternator_fault",
    "propeller_imbalance",
    "lean_drift",
    "sensor_bias_egt",
    "multi_fault_bearing_turbo",
]


def assign_split(idx: int, n: int) -> str:
    if idx < int(0.70 * n):
        return "train"
    elif idx < int(0.85 * n):
        return "val"
    return "test"


def build_fault_injector(scenario: str, mission_duration_s: float, rng: random.Random) -> FaultInjector:
    """Build a FaultInjector for the given scenario."""
    injector = FaultInjector.from_config()
    if scenario == "no_fault":
        return injector

    onset_time = rng.uniform(0.1, 0.6) * mission_duration_s

    if scenario == "bearing_wear":
        duration = rng.uniform(0.2, 0.7) * (mission_duration_s - onset_time)
        injector.add_fault(ActiveFault("bearing_wear", onset_time, onset_time + duration, "exponential"))

    elif scenario == "oil_starvation":
        injector.add_fault(ActiveFault("oil_starvation", onset_time, onset_time + 900.0, "abrupt"))

    elif scenario == "coolant_loss":
        duration = rng.uniform(1800.0, 7200.0)
        injector.add_fault(ActiveFault("coolant_loss", onset_time, onset_time + duration, "linear"))

    elif scenario == "spark_plug_degradation":
        cyl = rng.randint(0, 3)
        dur = rng.uniform(0.3, 0.8) * (mission_duration_s - onset_time)
        injector.add_fault(ActiveFault("spark_plug_degradation", onset_time, onset_time + dur, "linear", cylinder_index=cyl))

    elif scenario == "injector_clogging":
        cyl = rng.randint(0, 3)
        dur = rng.uniform(3600.0, 18000.0)
        injector.add_fault(ActiveFault("injector_clogging", onset_time, onset_time + dur, "linear", cylinder_index=cyl))

    elif scenario == "turbocharger_degradation":
        dur = rng.uniform(0.3, 0.9) * (mission_duration_s - onset_time)
        injector.add_fault(ActiveFault("turbocharger_degradation", onset_time, onset_time + dur, "linear"))

    elif scenario == "wastegate_stuck":
        injector.add_fault(ActiveFault("wastegate_stuck", onset_time, mission_duration_s, "abrupt"))

    elif scenario == "fuel_pump_degradation":
        dur = rng.uniform(3600.0, 21600.0)
        injector.add_fault(ActiveFault("fuel_pump_degradation", onset_time, onset_time + dur, "exponential"))

    elif scenario == "alternator_fault":
        injector.add_fault(ActiveFault("alternator_fault", onset_time, mission_duration_s, "abrupt",
                                       sensor_channel="bus_voltage_v", sensor_bias_value=-2.0))

    elif scenario == "propeller_imbalance":
        injector.add_fault(ActiveFault("propeller_imbalance", onset_time, mission_duration_s, "abrupt"))

    elif scenario == "lean_drift":
        dur = rng.uniform(3600.0, 18000.0)
        injector.add_fault(ActiveFault("lean_drift", onset_time, onset_time + dur, "linear"))

    elif scenario == "piston_ring_wear":
        cyl = rng.randint(0, 3)
        dur = rng.uniform(0.3, 0.9) * (mission_duration_s - onset_time)
        injector.add_fault(ActiveFault("piston_ring_wear", onset_time, onset_time + dur, "linear", cylinder_index=cyl))

    elif scenario == "sensor_bias_egt":
        bias_val = rng.choice([-50.0, -30.0, 30.0, 50.0])
        injector.add_fault(ActiveFault("sensor_bias", onset_time, mission_duration_s, "abrupt",
                                       sensor_channel="egt_c", sensor_bias_value=bias_val))

    elif scenario == "multi_fault_bearing_turbo":
        dur1 = rng.uniform(3600.0, 14400.0)
        dur2 = rng.uniform(3600.0, 14400.0)
        injector.add_fault(ActiveFault("bearing_wear", onset_time, onset_time + dur1, "exponential"))
        onset2 = onset_time + rng.uniform(1800.0, 7200.0)
        injector.add_fault(ActiveFault("turbocharger_degradation", onset2, onset2 + dur2, "linear"))

    return injector


def main() -> None:
    configure_logging()
    p = argparse.ArgumentParser(description="AEROTWIN Dataset Generator")
    p.add_argument("--missions", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--profile", choices=["short", "standard"], default="standard")
    p.add_argument("--output-dir", type=str, default="./data/datasets")
    args = p.parse_args()

    writer = ParquetDatasetWriter(Path(args.output_dir))
    rng_master = random.Random(args.seed)

    logger.info("Generating %d missions, seed=%d, profile=%s", args.missions, args.seed, args.profile)
    t0 = time.perf_counter()

    for i in range(args.missions):
        mission_seed = rng_master.randint(0, 999999)
        rng = random.Random(mission_seed)

        scenario = rng.choice(FAULT_SCENARIOS)
        split = assign_split(i, args.missions)
        engine_id = f"engine_{rng.randint(1, 8):03d}"
        delta_isa = rng.uniform(-15.0, 15.0)
        age_factor = rng.uniform(0.0, 1.0)
        loiter_s = rng.uniform(14400.0, 36000.0) if args.profile == "standard" else 1800.0
        mission_duration_s = 300 + 120 + 600 + 1800 + loiter_s + 600 + 180

        fault_injector = build_fault_injector(scenario, mission_duration_s, rng)
        sensor_model = SensorModel(seed=mission_seed)

        sim = Simulator(
            seed=mission_seed,
            dt=args.dt,
            fault_injector=fault_injector,
            sensor_model=sensor_model,
        )

        mission_id = f"ds_{args.seed:04d}_{i:06d}"
        records = list(sim.run_mission(
            mission_id=mission_id,
            engine_id=engine_id,
            profile_type=args.profile,
            loiter_duration_s=loiter_s,
            delta_isa_c=delta_isa,
            engine_age_factor=age_factor,
        ))

        writer.write_mission(mission_id, records, split=split)

        if (i + 1) % max(1, args.missions // 20) == 0 or i == args.missions - 1:
            elapsed = time.perf_counter() - t0
            logger.info(
                "[%d/%d] scenario=%s split=%s records=%d elapsed=%.1fs",
                i + 1, args.missions, scenario, split, len(records), elapsed,
            )

    elapsed = time.perf_counter() - t0
    summary = writer.split_summary()
    logger.info("Dataset generation complete in %.1f s | %s", elapsed, summary)


if __name__ == "__main__":
    main()
