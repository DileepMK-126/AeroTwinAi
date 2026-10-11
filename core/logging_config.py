"""
AEROTWIN — Logging Configuration
Structured logging with JSON option for production.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from typing import Any, Dict


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Configure application logging.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON-structured logs (for production)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if json_logs:
        # Use JSON formatter for structured logging in production
        try:
            import json_log_formatter  # type: ignore
            formatter: Any = json_log_formatter.JSONFormatter()
        except ImportError:
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for lib in ("uvicorn.access", "paho.mqtt", "httpx", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.getLogger("aerotwin").setLevel(numeric_level)
    logging.info("AEROTWIN logging configured: level=%s json=%s", level, json_logs)


def get_logger(name: str) -> logging.Logger:
    """Return a logger prefixed with 'aerotwin.'."""
    return logging.getLogger(f"aerotwin.{name}")
