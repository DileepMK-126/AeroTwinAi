"""
AEROTWIN — System Router
General system-level API endpoints: status, config info, dataset summary.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from configs.config_loader import load_engine_config, load_settings
from storage.storage import get_parquet, get_sqlite

router = APIRouter()


@router.get("/status", summary="Full system status")
async def system_status() -> Dict[str, Any]:
    """Return full system status including storage and dataset info."""
    settings = load_settings()
    engine_cfg = load_engine_config()
    parquet = get_parquet()

    return {
        "status": "ok",
        "environment": settings.env,
        "engine": {
            "id": engine_cfg.meta.engine_id,
            "description": engine_cfg.meta.description,
            "max_power_kw": engine_cfg.power.max_power_kw,
        },
        "dataset_splits": parquet.split_summary(),
        "safety_notice": "Research prototype. Not flight-certified. Advisory only.",
    }


@router.get("/config/engine", summary="Get engine configuration")
async def get_engine_config() -> Dict[str, Any]:
    """Return the loaded engine configuration (read-only)."""
    engine_cfg = load_engine_config()
    return engine_cfg.model_dump()
