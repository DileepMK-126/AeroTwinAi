"""
AEROTWIN — Phase 6: CSV/Parquet Replay Adapter

Replays historical data from CSV or Parquet files at configurable speed.
Supports channel mapping via a YAML mapping file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from data.schemas import DataSource, TelemetryRecord

logger = logging.getLogger("aerotwin.ingest.replay")


class ReplayAdapter:
    """
    Replays a CSV or Parquet file at configurable speed.
    Supports channel remapping via a mapping dict.
    """

    def __init__(
        self,
        file_path: str,
        speed_factor: float = 1.0,   # 1x = real-time, 600x = 600x faster
        channel_mapping: Optional[Dict[str, str]] = None,  # {file_col: schema_field}
        source: DataSource = DataSource.replay,
        engine_id: str = "replay_engine_001",
    ) -> None:
        self.file_path = Path(file_path)
        self.speed_factor = max(0.001, speed_factor)
        self.channel_mapping = channel_mapping or {}
        self.source = source
        self.engine_id = engine_id
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> "ReplayAdapter":
        """Load file into memory."""
        suffix = self.file_path.suffix.lower()
        if suffix == ".parquet":
            self._df = pd.read_parquet(self.file_path)
        elif suffix in (".csv", ".txt"):
            self._df = pd.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        # Apply channel mapping
        if self.channel_mapping:
            self._df = self._df.rename(columns=self.channel_mapping)
        logger.info("Loaded replay file: %s (%d rows)", self.file_path, len(self._df))
        return self

    def stream(self, dt: float = 1.0) -> Iterator[TelemetryRecord]:
        """
        Yield TelemetryRecord from each row, sleeping to maintain speed_factor.
        dt: timestep in seconds (used for real-time pacing).
        """
        if self._df is None:
            self.load()
        assert self._df is not None

        sleep_time = dt / self.speed_factor

        for _, row in self._df.iterrows():
            data = row.to_dict()
            data["engine_id"] = data.get("engine_id", self.engine_id)
            data["source"] = self.source.value

            # Map CHT/EGT columns if present as separate columns
            for key in ["cht_c", "egt_c"]:
                if key not in data:
                    cols = [f"{key}_{i+1}" for i in range(4)]
                    vals = [data.pop(c, None) for c in cols]
                    if any(v is not None for v in vals):
                        data[key] = [v or 0.0 for v in vals]

            # Remove NaN values
            data = {k: (None if (v != v) else v) for k, v in data.items()}

            try:
                record = TelemetryRecord.model_validate(data)
                yield record
            except Exception as e:
                logger.debug("Skipping row: %s", e)

            if sleep_time > 0:
                time.sleep(sleep_time)

    def validate_mapping(self) -> Dict[str, Any]:
        """Validate channel mapping and return a coverage report."""
        if self._df is None:
            self.load()
        assert self._df is not None

        from data.schemas import TelemetryRecord
        schema_fields = set(TelemetryRecord.model_fields.keys())
        file_cols = set(self._df.columns)
        mapped_cols = set(self.channel_mapping.values())

        matched = schema_fields & (file_cols | mapped_cols)
        missing = schema_fields - matched - {"record_id", "sequence_number", "fault_id", "fault_severity", "true_rul_hours", "accel_raw", "spectral_features"}

        return {
            "file_columns": list(file_cols),
            "schema_fields": list(schema_fields),
            "mapped": list(matched),
            "missing_from_file": list(missing),
            "coverage_pct": len(matched) / len(schema_fields) * 100,
        }
