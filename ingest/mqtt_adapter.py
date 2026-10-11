"""
AEROTWIN — Phase 6: MQTT Ingest Adapter

Subscribes to MQTT topics, parses telemetry, validates against unified schema,
and feeds into the twin pipeline.

Topics: aerotwin/{engine_id}/telemetry
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from data.schemas import DataSource, TelemetryRecord

logger = logging.getLogger("aerotwin.ingest.mqtt")


class MQTTIngestAdapter:
    """
    MQTT ingest adapter.
    Subscribes to aerotwin/{engine_id}/telemetry topics.
    Validates and converts incoming JSON payloads to TelemetryRecord.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        topic_prefix: str = "aerotwin",
        on_record: Optional[Callable[[TelemetryRecord], None]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.topic_prefix = topic_prefix
        self.on_record = on_record
        self.username = username
        self.password = password
        self._client = None
        self._connected = False

    def _parse_payload(self, engine_id: str, payload: bytes) -> Optional[TelemetryRecord]:
        """Parse JSON MQTT payload into TelemetryRecord."""
        try:
            data = json.loads(payload.decode("utf-8"))
            # Inject engine_id and source if not present
            data.setdefault("engine_id", engine_id)
            data.setdefault("source", DataSource.real_user.value)
            if "timestamp" not in data:
                data["timestamp"] = datetime.now(timezone.utc).isoformat()

            record = TelemetryRecord.model_validate(data)
            return record
        except Exception as e:
            logger.warning("Failed to parse MQTT payload: %s", e)
            return None

    def start(self) -> None:
        """Start MQTT client and subscribe."""
        try:
            import paho.mqtt.client as mqtt

            client = mqtt.Client()
            if self.username:
                client.username_pw_set(self.username, self.password)

            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self._connected = True
                    topic = f"{self.topic_prefix}/+/telemetry"
                    client.subscribe(topic)
                    logger.info("MQTT connected to %s:%d, subscribed to %s", self.host, self.port, topic)
                else:
                    logger.error("MQTT connection failed: rc=%d", rc)

            def on_message(client, userdata, msg):
                parts = msg.topic.split("/")
                engine_id = parts[1] if len(parts) >= 3 else "unknown"
                record = self._parse_payload(engine_id, msg.payload)
                if record and self.on_record:
                    self.on_record(record)

            client.on_connect = on_connect
            client.on_message = on_message
            client.connect(self.host, self.port, keepalive=60)
            self._client = client
            client.loop_start()

        except ImportError:
            logger.warning("paho-mqtt not installed — MQTT ingest unavailable")

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    def publish(self, engine_id: str, record: TelemetryRecord) -> None:
        """Publish a TelemetryRecord to MQTT (for simulator→backend bridge)."""
        if not self._client or not self._connected:
            return
        topic = f"{self.topic_prefix}/{engine_id}/telemetry"
        payload = record.model_dump_json()
        self._client.publish(topic, payload, qos=0)
