"""
AEROTWIN — WebSocket Hub

Manages WebSocket connections and broadcasts live twin state to all connected clients.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("aerotwin.api.ws")


class WebSocketHub:
    """
    Manages all active WebSocket connections.
    Broadcasts messages to all connected clients.
    """

    def __init__(self) -> None:
        self._connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        logger.info("WebSocket connected (total: %d)", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)
        logger.info("WebSocket disconnected (total: %d)", len(self._connections))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a JSON message to all connected clients."""
        if not self._connections:
            return
        payload = json.dumps(message, default=str)
        dead: List[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Singleton hub
_hub = WebSocketHub()


def get_hub() -> WebSocketHub:
    return _hub
