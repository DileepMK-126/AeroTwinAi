"""
AEROTWIN — Phase 6 Unit Tests: Ingestion Layer (MQTT & Replay Adapters)

Tests:
1. MQTTIngestAdapter payload parsing into TelemetryRecord
2. MQTTIngestAdapter defaults missing engine_id, source, timestamp
3. MQTTIngestAdapter handles malformed payloads gracefully
4. ReplayAdapter loads CSV and applies column mapping
5. ReplayAdapter streams valid TelemetryRecord sequence
6. ReplayAdapter validates schema coverage report
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import pytest
import pandas as pd

from data.schemas import DataSource, TelemetryRecord
from ingest.mqtt_adapter import MQTTIngestAdapter
from ingest.replay_adapter import ReplayAdapter


class TestMQTTIngestAdapter:
    def test_parse_valid_payload(self):
        adapter = MQTTIngestAdapter()
        payload = json.dumps({
            "timestamp": "2026-10-02T12:00:00Z",
            "engine_id": "test_eng_01",
            "source": "real_user",
            "rpm": 4500.0,
            "oil_press_kpa": 410.0,
            "cht_c": [112.0, 110.0, 114.0, 111.0],
            "egt_c": [760.0, 755.0, 762.0, 758.0],
        }).encode("utf-8")

        rec = adapter._parse_payload("test_eng_01", payload)
        assert isinstance(rec, TelemetryRecord)
        assert rec.engine_id == "test_eng_01"
        assert rec.rpm == 4500.0
        assert rec.oil_press_kpa == 410.0

    def test_parse_payload_auto_defaults(self):
        adapter = MQTTIngestAdapter()
        # Minimal payload missing timestamp, engine_id, source
        payload = json.dumps({
            "rpm": 5000.0,
            "fuel_flow_lph": 22.5,
        }).encode("utf-8")

        rec = adapter._parse_payload("injected_eng_02", payload)
        assert isinstance(rec, TelemetryRecord)
        assert rec.engine_id == "injected_eng_02"
        assert rec.source == DataSource.real_user
        assert rec.timestamp is not None

    def test_parse_corrupt_payload_returns_none(self):
        adapter = MQTTIngestAdapter()
        payload = b"not a json string at all {"
        rec = adapter._parse_payload("test_eng_03", payload)
        assert rec is None


class TestReplayAdapter:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        csv_file = tmp_path / "flight_data.csv"
        df = pd.DataFrame({
            "time_utc": ["2026-10-02T12:00:00Z", "2026-10-02T12:00:01Z", "2026-10-02T12:00:02Z"],
            "ENG_RPM": [4500.0, 4520.0, 4510.0],
            "OIL_P": [400.0, 402.0, 398.0],
            "OIL_T": [90.0, 91.0, 91.5],
            "cht_c_1": [110.0, 110.5, 111.0],
            "cht_c_2": [112.0, 112.5, 113.0],
            "cht_c_3": [109.0, 109.5, 110.0],
            "cht_c_4": [111.0, 111.5, 112.0],
            "egt_c_1": [750.0, 752.0, 751.0],
            "egt_c_2": [755.0, 756.0, 754.0],
            "egt_c_3": [748.0, 749.0, 747.0],
            "egt_c_4": [752.0, 753.0, 751.0],
        })
        df.to_csv(csv_file, index=False)
        return str(csv_file)

    def test_replay_load_and_mapping(self, sample_csv):
        mapping = {
            "time_utc": "timestamp",
            "ENG_RPM": "rpm",
            "OIL_P": "oil_press_kpa",
            "OIL_T": "oil_temp_c",
        }
        adapter = ReplayAdapter(
            file_path=sample_csv,
            speed_factor=1000.0,
            channel_mapping=mapping,
        )
        adapter.load()
        assert adapter._df is not None
        assert "rpm" in adapter._df.columns
        assert "oil_press_kpa" in adapter._df.columns

    def test_replay_streaming(self, sample_csv):
        mapping = {
            "time_utc": "timestamp",
            "ENG_RPM": "rpm",
            "OIL_P": "oil_press_kpa",
            "OIL_T": "oil_temp_c",
        }
        adapter = ReplayAdapter(
            file_path=sample_csv,
            speed_factor=10000.0,  # virtually instant
            channel_mapping=mapping,
            engine_id="replay_uav_01",
        )
        records = list(adapter.stream(dt=0.001))
        assert len(records) == 3
        assert all(isinstance(r, TelemetryRecord) for r in records)
        assert records[0].engine_id == "replay_uav_01"
        assert records[0].rpm == 4500.0
        assert len(records[0].cht_c) == 4
        assert len(records[0].egt_c) == 4

    def test_replay_coverage_report(self, sample_csv):
        mapping = {
            "time_utc": "timestamp",
            "ENG_RPM": "rpm",
            "OIL_P": "oil_press_kpa",
        }
        adapter = ReplayAdapter(sample_csv, channel_mapping=mapping)
        report = adapter.validate_mapping()
        assert "coverage_pct" in report
        assert report["coverage_pct"] > 0
        assert "rpm" in report["mapped"]
