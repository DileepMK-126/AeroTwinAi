"""
AEROTWIN — FastAPI Application Entry Point
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from configs.config_loader import load_settings
from core.logging_config import configure_logging
from storage.storage import get_sqlite

logger = logging.getLogger("aerotwin.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    settings = load_settings()
    configure_logging(level=settings.log_level)

    # Initialise SQLite connection
    db = get_sqlite()
    logger.info("AEROTWIN backend starting up (env=%s)", settings.env)

    yield

    # Shutdown
    db.close()
    logger.info("AEROTWIN backend shut down cleanly")


app = FastAPI(
    title="AEROTWIN API",
    description=(
        "AI-Enabled Real-Time Digital Twin for Aero Piston Engine Health Monitoring. "
        "⚠️ Research prototype only — not flight-certified."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check (required by Docker Compose healthcheck)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint. Returns status and version."""
    return {
        "status": "ok",
        "service": "aerotwin-backend",
        "version": "0.1.0",
        "safety_notice": "Research prototype. Not flight-certified. Advisory only.",
    }


@app.get("/login", response_class=FileResponse, tags=["Authentication"])
async def login_page() -> FileResponse:
    """Serve the AEROTWIN Local Login and Signin Portal."""
    login_path = Path("web") / "login.html"
    return FileResponse(login_path)


@app.get("/dashboard", response_class=FileResponse, tags=["Dashboard"])
async def dashboard() -> FileResponse:
    """Serve the AEROTWIN Ground Control Station Ops Dashboard."""
    dash_path = Path("web") / "index.html"
    return FileResponse(dash_path)


@app.get("/", tags=["System"])
async def root() -> RedirectResponse:
    """Root endpoint — redirects to live ops dashboard."""
    return RedirectResponse(url="/dashboard")


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

from api.routers import auth as auth_router  # noqa: E402
from api.routers import sim as sim_router  # noqa: E402
from api.routers import system as system_router  # noqa: E402
from api.routers import telemetry as telemetry_router  # noqa: E402

app.include_router(auth_router.router, prefix="/api/v1", tags=["Authentication"])
app.include_router(system_router.router, prefix="/api/v1", tags=["System"])
app.include_router(telemetry_router.router, prefix="/api/v1", tags=["Telemetry"])
app.include_router(sim_router.router, prefix="/api/v1", tags=["Simulation"])

# WebSocket is included directly in the telemetry router
app.include_router(telemetry_router.router, prefix="", tags=["WebSocket"], include_in_schema=False)
