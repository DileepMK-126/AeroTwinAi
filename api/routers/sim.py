"""
AEROTWIN — Live Simulation & Fault Injection Router
Controls interactive flight simulation streaming via WebSocket and accepts live fault injections.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from api.pipeline import AerotwinPipeline
from api.routers.telemetry import get_pipeline
from api.ws_hub import get_hub
from core.faults.fault_injector import ActiveFault, FaultInjector
from core.simulator.flight_profile import FlightProfileGenerator
from core.simulator.simulator import Simulator

logger = logging.getLogger("aerotwin.api.sim")

router = APIRouter()


class FaultInjectRequest(BaseModel):
    fault_id: str
    severity: float = 1.0
    ramp_type: str = "abrupt"
    sensor_channel: Optional[str] = None
    sensor_bias_value: float = 50.0


class SimState:
    running: bool = False
    task: Optional[asyncio.Task] = None
    simulator: Optional[Simulator] = None
    injector: Optional[FaultInjector] = None
    speed_multiplier: float = 1.0
    current_step: int = 0
    engine_id: str = "rotax_914_like_001"
    last_fault_injected: Optional[str] = None
    fault_inject_time: Optional[float] = None
    time_to_detect: Optional[float] = None


_sim_state = SimState()


async def _simulation_loop() -> None:
    """Async background worker yielding telemetry records at ~1Hz."""
    hub = get_hub()
    engine_id = _sim_state.engine_id
    pipeline = get_pipeline(engine_id)

    # Initialize simulator with fault injector
    _sim_state.injector = FaultInjector.from_config()
    _sim_state.simulator = Simulator(fault_injector=_sim_state.injector, seed=42)

    from core.real_engine import get_real_manager
    real_mgr = get_real_manager()
    logger.info("Live flight simulation loop started for %s (dataset=%s)", engine_id, real_mgr.current_dataset)

    mission_generator = _sim_state.simulator.run_mission(
        mission_id="live_demo_001",
        engine_id=engine_id,
        profile_type="short",
        cruise_altitude_m=3500.0,
        loiter_duration_s=1800.0,
    )

    try:
        for record in mission_generator:
            if not _sim_state.running:
                break

            _sim_state.current_step += 1
            t_sim = float(_sim_state.current_step)

            # If real dataset mode is active, override record fields with real sensor readings
            real_sample = None
            if real_mgr.current_dataset != "rotax":
                real_sample = real_mgr.next_sample()
                record.source = real_sample["source"]
                record.rpm = real_sample["rpm"]
                record.throttle_pct = real_sample["throttle_pct"]
                record.map_kpa = real_sample["map_kpa"]
                record.boost_kpa = real_sample["boost_kpa"]
                record.fuel_flow_lph = real_sample["fuel_flow_lph"]
                record.oil_press_kpa = real_sample["oil_press_kpa"]
                record.oil_temp_c = real_sample["oil_temp_c"]
                record.coolant_temp_c = real_sample["coolant_temp_c"]
                record.vib_rms_g = real_sample["vib_rms_g"]
                record.cht_c = real_sample["cht_c"]
                record.egt_c = real_sample["egt_c"]
                record.altitude_m = real_sample["altitude_m"]
                record.airspeed_mps = real_sample["airspeed_mps"]
                record.flight_phase = real_sample["flight_phase"]

            # Process record through end-to-end digital twin pipeline
            output = pipeline.process(record)

            # Check if fault was detected for live TTD calculation
            if _sim_state.fault_inject_time is not None and _sim_state.time_to_detect is None:
                if any(a.severity in ("warning", "critical") for a in output.alerts):
                    _sim_state.time_to_detect = round(t_sim - _sim_state.fault_inject_time, 2)
                    logger.info("Fault %s detected in %.2fs!", _sim_state.last_fault_injected, _sim_state.time_to_detect)

            # Enrich ws message with live sim status and real model inference
            ws_msg = output.to_ws_message()
            ws_msg["dataset_source"] = real_mgr.current_dataset
            if real_sample and "model_inference" in real_sample:
                ws_msg["model_inference"] = real_sample["model_inference"]
                ws_msg["source"] = real_sample["source"]
                # Use real health score if available
                ws_msg["twin"]["health_score"] = real_sample["model_inference"]["health_score"]

            ws_msg["sim_status"] = {
                "running": True,
                "step": _sim_state.current_step,
                "sim_time_s": t_sim,
                "speed_multiplier": _sim_state.speed_multiplier,
                "active_fault": _sim_state.last_fault_injected,
                "time_to_detect_s": _sim_state.time_to_detect,
                "current_dataset": real_mgr.current_dataset,
            }

            # Broadcast to all connected WebSocket clients
            await hub.broadcast(ws_msg)

            # Sleep according to speed multiplier
            sleep_duration = max(0.05, 1.0 / _sim_state.speed_multiplier)
            await asyncio.sleep(sleep_duration)

    except asyncio.CancelledError:
        logger.info("Simulation loop cancelled")
    except Exception as e:
        logger.error("Simulation loop error: %s", e, exc_info=True)
    finally:
        _sim_state.running = False
        logger.info("Simulation loop ended")


@router.post("/sim/start", summary="Start or resume live simulation")
async def start_sim(speed: float = 1.0) -> Dict[str, Any]:
    _sim_state.speed_multiplier = max(0.2, min(10.0, speed))
    if not _sim_state.running:
        _sim_state.running = True
        _sim_state.task = asyncio.create_task(_simulation_loop())
    return {
        "status": "running",
        "speed": _sim_state.speed_multiplier,
        "step": _sim_state.current_step,
    }


@router.post("/sim/stop", summary="Stop/pause live simulation")
async def stop_sim() -> Dict[str, Any]:
    _sim_state.running = False
    if _sim_state.task and not _sim_state.task.done():
        _sim_state.task.cancel()
    return {"status": "stopped", "step": _sim_state.current_step}


@router.post("/sim/reset", summary="Reset simulation and twin state")
async def reset_sim() -> Dict[str, Any]:
    await stop_sim()
    _sim_state.current_step = 0
    _sim_state.last_fault_injected = None
    _sim_state.fault_inject_time = None
    _sim_state.time_to_detect = None

    # Reset twin pipeline
    pipeline = get_pipeline(_sim_state.engine_id)
    pipeline.reset()

    # Re-initialise simulator
    _sim_state.injector = FaultInjector.from_config()
    _sim_state.simulator = Simulator(fault_injector=_sim_state.injector, seed=42)

    return {"status": "reset", "step": 0}


@router.post("/sim/fault", summary="Inject an engine or sensor fault live")
async def inject_fault(req: FaultInjectRequest) -> Dict[str, Any]:
    if not _sim_state.injector:
        _sim_state.injector = FaultInjector.from_config()

    current_t = float(_sim_state.current_step)
    _sim_state.last_fault_injected = req.fault_id
    _sim_state.fault_inject_time = current_t
    _sim_state.time_to_detect = None

    fault = ActiveFault(
        fault_id=req.fault_id,
        onset_time_s=current_t,
        failure_time_s=current_t + (1.0 if req.ramp_type == "abrupt" else 120.0),
        ramp_type=req.ramp_type,
        sensor_channel=req.sensor_channel,
        sensor_bias_value=req.sensor_bias_value,
    )
    _sim_state.injector.add_fault(fault)

    logger.warning("Injected fault: %s at t=%.1fs (ramp=%s)", req.fault_id, current_t, req.ramp_type)
    return {
        "status": "injected",
        "fault_id": req.fault_id,
        "onset_time_s": current_t,
    }


@router.get("/sim/status", summary="Get live simulator status")
async def get_sim_status() -> Dict[str, Any]:
    return {
        "running": _sim_state.running,
        "step": _sim_state.current_step,
        "speed": _sim_state.speed_multiplier,
        "active_fault": _sim_state.last_fault_injected,
        "time_to_detect_s": _sim_state.time_to_detect,
    }


@router.get("/eval/report", summary="Get evaluation report markdown")
async def get_eval_report() -> Dict[str, str]:
    report_file = Path("eval") / "report.md"
    if report_file.exists():
        content = report_file.read_text(encoding="utf-8")
    else:
        content = "# Evaluation Report\n\nNo report generated yet. Run `make eval` or `python -m eval.run_eval`."
    return {"report_markdown": content}


@router.post("/sim/dataset", summary="Switch active real dataset stream")
async def set_dataset(name: str) -> Dict[str, Any]:
    from core.real_engine import get_real_manager
    mgr = get_real_manager()
    success = mgr.set_dataset(name)
    return {
        "status": "ok" if success else "error",
        "current_dataset": mgr.current_dataset,
        "message": f"Telemetry switched to real dataset: {mgr.current_dataset}",
    }


@router.get("/models/inventory", summary="Get inventory of real trained models and datasets")
async def get_models_inventory() -> List[Dict[str, Any]]:
    from core.real_engine import get_real_manager
    mgr = get_real_manager()
    return mgr.get_models_inventory()


@router.get("/models/benchmark", summary="Execute live real-time benchmark across all models")
async def run_models_benchmark() -> Dict[str, Any]:
    from core.real_engine import get_real_manager
    mgr = get_real_manager()
    return mgr.run_full_benchmark()


