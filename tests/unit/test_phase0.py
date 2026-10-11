"""
AEROTWIN — Phase 0 Unit Tests
Tests for: schema validation, config loading, storage layer.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from data.schemas import (
    DataSource,
    FlightPhase,
    MissionMeta,
    SensorQuality,
    TelemetryBatch,
    TelemetryRecord,
)


# ---------------------------------------------------------------------------
# TelemetryRecord schema tests
# ---------------------------------------------------------------------------

class TestTelemetrySchema:
    def _minimal_record(self) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "engine_id": "test_engine_001",
            "source": "simulated",
        }

    def test_minimal_record_valid(self):
        """A record with only mandatory fields should be valid."""
        rec = TelemetryRecord(**self._minimal_record())
        assert rec.engine_id == "test_engine_001"
        assert rec.source == DataSource.simulated

    def test_full_record_valid(self):
        """A fully populated record should validate successfully."""
        rec = TelemetryRecord(
            timestamp=datetime.now(timezone.utc),
            engine_id="test_engine_001",
            source=DataSource.simulated,
            flight_phase=FlightPhase.cruise,
            altitude_m=4500.0,
            oat_c=-10.0,
            airspeed_mps=55.0,
            rpm=4800.0,
            throttle_pct=75.0,
            map_kpa=95.0,
            boost_kpa=110.0,
            fuel_flow_lph=18.0,
            cht_c=[110.0, 112.0, 108.0, 111.0],
            egt_c=[750.0, 755.0, 748.0, 752.0],
            oil_temp_c=88.0,
            oil_press_kpa=410.0,
            coolant_temp_c=95.0,
            fuel_press_kpa=150.0,
            bus_voltage_v=14.2,
            vib_rms_g=0.05,
        )
        assert rec.cht_c is not None
        assert len(rec.cht_c) == 4
        assert rec.egt_c is not None
        assert len(rec.egt_c) == 4

    def test_optional_fields_none(self):
        """Optional fields default to None."""
        rec = TelemetryRecord(**self._minimal_record())
        assert rec.rpm is None
        assert rec.cht_c is None
        assert rec.fault_id is None

    def test_cht_wrong_length_raises(self):
        """CHT with != 4 values should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="4 values"):
            TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="eng",
                source="simulated",
                cht_c=[100.0, 110.0],  # wrong length
            )

    def test_egt_wrong_length_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="4 values"):
            TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="eng",
                source="simulated",
                egt_c=[750.0, 760.0, 740.0],  # wrong length
            )

    def test_rpm_out_of_range_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="eng",
                source="simulated",
                rpm=99999.0,  # way too high
            )

    def test_source_enum_validation(self):
        """Invalid source string should raise."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="eng",
                source="invented_source",
            )

    def test_naive_timestamp_becomes_utc(self):
        """Naive timestamp should be treated as UTC."""
        rec = TelemetryRecord(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),  # no tz
            engine_id="eng",
            source="simulated",
        )
        assert rec.timestamp.tzinfo is not None

    def test_to_flat_dict(self):
        """to_flat_dict should expand cht_c and egt_c into named columns."""
        rec = TelemetryRecord(
            timestamp=datetime.now(timezone.utc),
            engine_id="eng",
            source="simulated",
            cht_c=[100.0, 105.0, 102.0, 103.0],
            egt_c=[740.0, 745.0, 742.0, 743.0],
        )
        flat = rec.to_flat_dict()
        assert "cht_c_1" in flat
        assert "cht_c_4" in flat
        assert "egt_c_1" in flat
        assert "cht_c" not in flat

    def test_record_id_is_uuid(self):
        """record_id should be auto-generated UUID string."""
        import uuid
        rec = TelemetryRecord(**self._minimal_record())
        uuid.UUID(rec.record_id)  # should not raise

    def test_sensor_quality_defaults(self):
        """SensorQuality should default all channels to True."""
        sq = SensorQuality()
        assert sq.rpm is True
        assert all(sq.cht_c)


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------

class TestConfigLoader:
    def test_load_engine_config(self):
        """Engine config should load and validate correctly."""
        from configs.config_loader import load_engine_config
        cfg = load_engine_config()
        assert cfg.meta.cylinders == 4
        assert cfg.power.max_power_kw > 0
        assert 0 < cfg.turbocharger.turbo_efficiency_compressor < 1

    def test_engine_config_swappable(self):
        """Passing same path should produce same result (idempotent)."""
        from configs.config_loader import load_engine_config
        from pathlib import Path
        cfg1 = load_engine_config()
        cfg2 = load_engine_config(Path("configs/engine_reference.yaml"))
        assert cfg1.meta.engine_id == cfg2.meta.engine_id

    def test_load_settings_defaults(self):
        """Settings should load defaults when env vars are absent."""
        from configs.config_loader import load_settings
        s = load_settings()
        assert s.api_port == 8000
        assert s.sim_seed == 42


# ---------------------------------------------------------------------------
# Storage layer tests
# ---------------------------------------------------------------------------

class TestSQLiteStorage:
    def test_insert_and_retrieve_telemetry(self):
        """Insert a record, retrieve it, verify round-trip."""
        import os
        from storage.storage import SQLiteStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            storage = SQLiteStorage(db_path=db_path)
            storage.connect()

            rec = TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="test_engine",
                source=DataSource.simulated,
                rpm=4800.0,
                throttle_pct=75.0,
            )
            storage.insert_telemetry(rec)

            rows = storage.get_recent_telemetry("test_engine", n=10)
            assert len(rows) == 1
            assert rows[0]["rpm"] == 4800.0
            storage.close()

    def test_insert_alert(self):
        """Insert an alert and verify it appears in unacknowledged list."""
        from storage.storage import SQLiteStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_alert.db"
            storage = SQLiteStorage(db_path=db_path)
            storage.connect()

            alert_id = storage.insert_alert(
                engine_id="test_engine",
                alert_type="detection",
                severity="warning",
                message="EGT cylinder 3 elevated",
                confidence=0.85,
            )
            assert alert_id is not None

            alerts = storage.get_unacknowledged_alerts("test_engine")
            assert len(alerts) == 1
            assert alerts[0]["message"] == "EGT cylinder 3 elevated"
            storage.close()

    def test_acknowledge_alert(self):
        """Acknowledged alert should not appear in unacknowledged list."""
        from storage.storage import SQLiteStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(db_path=Path(tmpdir) / "test.db")
            storage.connect()

            alert_id = storage.insert_alert(
                engine_id="eng",
                alert_type="diagnosis",
                severity="critical",
                message="Bearing wear detected",
            )
            storage.acknowledge_alert(alert_id)
            alerts = storage.get_unacknowledged_alerts("eng")
            assert len(alerts) == 0
            storage.close()

    def test_duplicate_record_id_ignored(self):
        """Inserting the same record twice should not raise (INSERT OR IGNORE)."""
        from storage.storage import SQLiteStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = SQLiteStorage(db_path=Path(tmpdir) / "test.db")
            storage.connect()

            rec = TelemetryRecord(
                timestamp=datetime.now(timezone.utc),
                engine_id="eng",
                source=DataSource.simulated,
            )
            storage.insert_telemetry(rec)
            storage.insert_telemetry(rec)  # duplicate — should not raise

            rows = storage.get_recent_telemetry("eng", n=10)
            assert len(rows) == 1  # only one row
            storage.close()


class TestParquetDataset:
    def test_write_and_read_mission(self):
        """Write a mission to Parquet and read it back."""
        from storage.storage import ParquetDatasetWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetDatasetWriter(dataset_dir=Path(tmpdir))

            records = [
                TelemetryRecord(
                    timestamp=datetime(2025, 1, 1, 0, i, 0, tzinfo=timezone.utc),
                    engine_id="eng",
                    source=DataSource.simulated,
                    rpm=float(4800 + i),
                )
                for i in range(10)
            ]

            path = writer.write_mission("test_mission_001", records, split="train")
            assert path.exists()

            df = writer.load_split("train")
            assert len(df) == 10

    def test_split_summary(self):
        """split_summary should return counts per split."""
        from storage.storage import ParquetDatasetWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ParquetDatasetWriter(dataset_dir=Path(tmpdir))
            summary = writer.split_summary()
            assert summary["train"] == 0
            assert summary["val"] == 0
            assert summary["test"] == 0
