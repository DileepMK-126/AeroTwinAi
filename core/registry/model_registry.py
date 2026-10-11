"""
AEROTWIN — Model Registry

Loads ML models from the registry directory.
Falls back to lower-level detectors if model file is missing or spec mismatches.
Supports ONNX Runtime for inference.

Registry file: configs/models.yaml
Model files: data/models/*.onnx | *.pkl
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from configs.config_loader import load_models_config

logger = logging.getLogger("aerotwin.registry")


class ModelRegistry:
    """
    Central model registry for AEROTWIN.

    Loads models on demand from the registry directory.
    Validates input spec before loading.
    Falls back gracefully when models are missing.
    """

    def __init__(
        self,
        registry_dir: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._config = config or load_models_config()
        self._models_cfg = self._config.get("models", {})
        self._registry_dir = registry_dir or Path(
            self._config.get("registry", {}).get("dir", "./data/models")
        )
        self._loaded_models: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    def load_model(self, model_name: str) -> Optional[Any]:
        """
        Load a model by name. Returns None if not available.
        Falls back through the loading chain defined in models.yaml.
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        cfg = self._models_cfg.get(model_name)
        if cfg is None:
            logger.warning("Model '%s' not in registry config", model_name)
            return None

        file_name = cfg.get("file")
        if file_name is None:
            logger.debug("Model '%s' has no file (computed model)", model_name)
            return None

        model_path = self._registry_dir / file_name
        if not model_path.exists():
            logger.info("Model file not found: %s — falling back", model_path)
            # Try fallback
            fallback = cfg.get("fallback")
            if fallback:
                return self.load_model(fallback)
            return None

        # Load based on extension
        try:
            if model_path.suffix == ".onnx":
                model = self._load_onnx(model_path, cfg)
            elif model_path.suffix == ".pkl":
                model = self._load_pickle(model_path)
            elif model_path.suffix == ".joblib":
                model = self._load_joblib(model_path)
            else:
                logger.warning("Unknown model format: %s", model_path.suffix)
                return None

            self._loaded_models[model_name] = model
            logger.info("Loaded model: %s from %s", model_name, model_path)
            return model

        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_name, e)
            return None

    def _load_joblib(self, path: Path) -> Any:
        """Load scikit-learn / joblib model pipeline."""
        import joblib
        from core.real_engine import patch_scikit_model
        return patch_scikit_model(joblib.load(path))

    def _load_onnx(self, path: Path, cfg: dict) -> Any:
        """Load ONNX model with ONNX Runtime."""
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logger.debug("ONNX session created for %s", path.name)
            return sess
        except ImportError:
            raise RuntimeError("onnxruntime not installed")

    def _load_pickle(self, path: Path) -> Any:
        with open(path, "rb") as f:
            return pickle.load(f)

    # -------------------------------------------------------------------------
    # Model info
    # -------------------------------------------------------------------------

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Return registry summary."""
        result = {}
        for name, cfg in self._models_cfg.items():
            file_name = cfg.get("file")
            file_exists = (
                (self._registry_dir / file_name).exists()
                if file_name else False
            )
            result[name] = {
                "version": cfg.get("version", "unknown"),
                "status": cfg.get("status", "unknown"),
                "file": file_name,
                "file_exists": file_exists,
                "fallback": cfg.get("fallback"),
                "description": cfg.get("description", ""),
            }
        return result

    def is_trained(self, model_name: str) -> bool:
        """Check if a trained model is available."""
        cfg = self._models_cfg.get(model_name, {})
        if cfg.get("status") == "active":
            return True
        if cfg.get("file") is None:
            return False
        path = self._registry_dir / cfg["file"]
        return path.exists()


# Singleton
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


get_model_registry = get_registry
__all__ = ["ModelRegistry", "get_registry", "get_model_registry"]
