"""
AEROTWIN — Telemetry and WebSocket API Router

Endpoints:
  POST /telemetry           — ingest a single telemetry record
  GET  /telemetry/{engine_id}/recent — get recent telemetry
  GET  /alerts/{engine_id}  — get active alerts
  POST /alerts/{alert_id}/acknowledge
  WS   /ws                  — live WebSocket feed
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Header, WebSocket, WebSocketDisconnect

from api.pipeline import AerotwinPipeline, PipelineOutput
from api.ws_hub import get_hub
from configs.config_loader import load_settings
from data.schemas import TelemetryRecord
from storage.storage import get_sqlite

logger = logging.getLogger("aerotwin.api.telemetry")

router = APIRouter()

# Shared pipeline instance (one per server process per engine)
_pipelines: Dict[str, AerotwinPipeline] = {}


def get_pipeline(engine_id: str) -> AerotwinPipeline:
    if engine_id not in _pipelines:
        _pipelines[engine_id] = AerotwinPipeline(dt=1.0, persist=True)
    return _pipelines[engine_id]


def _verify_api_key(x_api_key: str = Header(None)) -> str:
    settings = load_settings()
    if x_api_key != settings.ingest_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ---------------------------------------------------------------------------
# Ingest endpoint
# ---------------------------------------------------------------------------

@router.post("/telemetry", summary="Ingest a telemetry record")
async def ingest_telemetry(
    record: TelemetryRecord,
    _: str = Depends(_verify_api_key),
) -> Dict[str, Any]:
    """
    Ingest one telemetry record, run through the full pipeline,
    broadcast to WebSocket clients, and return the pipeline output summary.
    """
    pipeline = get_pipeline(record.engine_id)
    output = pipeline.process(record)

    # Persist record
    db = get_sqlite()
    db.insert_telemetry(record)

    # Broadcast to WebSocket
    hub = get_hub()
    if hub.connection_count > 0:
        await hub.broadcast(output.to_ws_message())

    return {
        "status": "ok",
        "engine_id": record.engine_id,
        "decision": output.decision.decision,
        "alert_count": len(output.alerts),
        "health_score": output.twin_state.overall_health_score,
        "rul_median_h": output.rul.rul_median_h,
    }


@router.get("/telemetry/{engine_id}/recent", summary="Get recent telemetry")
async def get_recent_telemetry(engine_id: str, n: int = 100) -> List[Dict[str, Any]]:
    db = get_sqlite()
    return db.get_recent_telemetry(engine_id, n=n)


@router.get("/alerts/{engine_id}", summary="Get active alerts")
async def get_alerts(engine_id: str) -> List[Dict[str, Any]]:
    db = get_sqlite()
    return db.get_unacknowledged_alerts(engine_id)


@router.post("/alerts/{alert_id}/acknowledge", summary="Acknowledge an alert")
async def acknowledge_alert(alert_id: int) -> Dict[str, Any]:
    db = get_sqlite()
    db.acknowledge_alert(alert_id)
    return {"status": "acknowledged", "alert_id": alert_id}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Live WebSocket feed — broadcasts pipeline output for all engines."""
    hub = get_hub()
    await hub.connect(websocket)
    try:
        while True:
            # Keep connection alive — client can send ping
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        hub.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
        hub.disconnect(websocket)
