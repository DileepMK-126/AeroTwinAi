"""
AEROTWIN — Simulator CLI Entry Point

Usage:
    python -m core.simulator.run
    python -m core.simulator.run --missions 500 --seed 1
    make sim
    make sim ARGS="--missions 500 --seed 1 --profile short"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure aerotwin root is on path when run as module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.logging_config import configure_logging
from core.simulator.simulator import Simulator
from storage.storage import ParquetDatasetWriter

logger = logging.getLogger("aerotwin.simulator.run")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AEROTWIN Simulator CLI")
    p.add_argument("--missions", type=int, default=1, help="Number of missions to simulate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--dt", type=float, default=1.0, help="Timestep (seconds)")
    p.add_argument("--profile", choices=["standard", "short"], default="short",
                   help="Flight profile: 'short' (1h, fast CI) or 'standard' (10h)")
    p.add_argument("--cruise-alt", type=float, default=4500.0,
                   help="Cruise altitude in metres")
    p.add_argument("--save-parquet", action="store_true", default=True,
                   help="Save dataset to Parquet")
    p.add_argument("--output-dir", type=str, default="./data/datasets",
                   help="Parquet output directory")
    p.add_argument("--save-plots", action="store_true", default=False,
                   help="Save mission plots (requires matplotlib)")
    return p.parse_args()


def assign_split(mission_idx: int, n_missions: int) -> str:
    """Assign train/val/test split based on mission index (70/15/15)."""
    if mission_idx < int(0.70 * n_missions):
        return "train"
    elif mission_idx < int(0.85 * n_missions):
        return "val"
    else:
        return "test"


def main() -> None:
    configure_logging()
    args = parse_args()

    sim = Simulator(seed=args.seed, dt=args.dt)
    writer = ParquetDatasetWriter(dataset_dir=Path(args.output_dir))

    logger.info(
        "Running %d mission(s), profile=%s, seed=%d, dt=%.1f s",
        args.missions, args.profile, args.seed, args.dt,
    )

    t0 = time.perf_counter()

    for i in range(args.missions):
        import random as _r
        rng = _r.Random(args.seed + i)
        mission_id = f"sim_{args.seed:04d}_{i:06d}"
        engine_id = f"rotax_914_like_{rng.randint(1, 4):03d}"
        split = assign_split(i, args.missions)
        delta_isa = rng.uniform(-15.0, 15.0)
        age_factor = rng.uniform(0.0, 1.0)
        loiter_s = rng.uniform(14400.0, 36000.0) if args.profile == "standard" else 1800.0

        records = list(sim.run_mission(
            mission_id=mission_id,
            engine_id=engine_id,
            profile_type=args.profile,
            cruise_altitude_m=args.cruise_alt,
            loiter_duration_s=loiter_s,
            delta_isa_c=delta_isa,
            engine_age_factor=age_factor,
        ))

        if args.save_parquet:
            writer.write_mission(mission_id, records, split=split)

        if (i + 1) % max(1, args.missions // 10) == 0 or i == args.missions - 1:
            elapsed = time.perf_counter() - t0
            logger.info(
                "Mission %d/%d complete | records=%d | elapsed=%.1f s | split=%s",
                i + 1, args.missions, len(records), elapsed, split,
            )

    elapsed = time.perf_counter() - t0
    logger.info("All missions complete in %.1f s", elapsed)

    if args.save_parquet:
        summary = writer.split_summary()
        logger.info("Dataset summary: %s", summary)

    if args.save_plots:
        _save_last_mission_plots(records, mission_id)


def _save_last_mission_plots(records, mission_id: str) -> None:
    """Save diagnostic plots of the last simulated mission."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots")
        return

    rows = [r.to_flat_dict() for r in records]
    df = pd.DataFrame(rows)
    df["time_s"] = range(len(df))

    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle(f"Mission: {mission_id}", fontsize=13)

    plots = [
        ("rpm", "RPM", "RPM"),
        ("throttle_pct", "Throttle", "%"),
        ("map_kpa", "MAP", "kPa"),
        ("boost_kpa", "Boost", "kPa"),
        ("cht_c_1", "CHT Cyl 1", "°C"),
        ("egt_c_1", "EGT Cyl 1", "°C"),
        ("oil_temp_c", "Oil Temp", "°C"),
        ("oil_press_kpa", "Oil Press", "kPa"),
    ]

    for ax, (col, label, unit) in zip(axes.flat, plots):
        if col in df.columns:
            ax.plot(df["time_s"], df[col], linewidth=0.8)
        ax.set_title(f"{label} [{unit}]")
        ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path("eval") / f"{mission_id}_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot: %s", out_path)
    plt.close()


if __name__ == "__main__":
    main()
