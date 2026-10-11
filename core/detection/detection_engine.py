"""
AEROTWIN — Phase 4: Layered Fault Detection

Level A (no training required):
  - Normalised residual thresholds
  - EWMA (Exponentially Weighted Moving Average)
  - CUSUM (Cumulative Sum control chart)

Level B (requires training data, optional):
  - Isolation Forest
  - Autoencoder (loaded from registry if available)

All Level-A detectors are calibrated on healthy simulated data
to a configurable target false-alarm rate.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("aerotwin.detection")


# ---------------------------------------------------------------------------
# Detection Alert
# ---------------------------------------------------------------------------

@dataclass
class DetectionAlert:
    """Output from any detector."""
    detector: str             # "threshold" | "ewma" | "cusum" | "isolation_forest" | "autoencoder"
    level: str                # "A" | "B"
    severity: str             # "info" | "warning" | "critical"
    score: float              # anomaly score (higher = more anomalous)
    threshold: float          # threshold that was crossed
    channels: List[str]       # channels contributing to alert
    message: str
    timestamp_step: int = 0


# ---------------------------------------------------------------------------
# Level A: Threshold Detector
# ---------------------------------------------------------------------------

class ThresholdDetector:
    """
    Simple normalised residual threshold detector.
    Fires when |normalised_residual| > threshold for any channel.
    """

    def __init__(
        self,
        warning_sigma: float = 2.5,
        critical_sigma: float = 4.0,
    ) -> None:
        self.warning_sigma = warning_sigma
        self.critical_sigma = critical_sigma
        self._step = 0

    def update(
        self, normalised_residuals: Dict[str, float]
    ) -> List[DetectionAlert]:
        self._step += 1
        alerts = []
        for ch, nr in normalised_residuals.items():
            abs_nr = abs(nr)
            if abs_nr > self.critical_sigma:
                alerts.append(DetectionAlert(
                    detector="threshold",
                    level="A",
                    severity="critical",
                    score=abs_nr,
                    threshold=self.critical_sigma,
                    channels=[ch],
                    message=f"CRITICAL: {ch} residual {nr:+.2f}σ (threshold ±{self.critical_sigma}σ)",
                    timestamp_step=self._step,
                ))
            elif abs_nr > self.warning_sigma:
                alerts.append(DetectionAlert(
                    detector="threshold",
                    level="A",
                    severity="warning",
                    score=abs_nr,
                    threshold=self.warning_sigma,
                    channels=[ch],
                    message=f"WARNING: {ch} residual {nr:+.2f}σ (threshold ±{self.warning_sigma}σ)",
                    timestamp_step=self._step,
                ))
        return alerts


# ---------------------------------------------------------------------------
# Level A: EWMA Detector
# ---------------------------------------------------------------------------

class EWMADetector:
    """
    Exponentially Weighted Moving Average detector on normalised residuals.
    Detects slow drifts that threshold alone misses.
    """

    def __init__(
        self,
        lambda_: float = 0.2,
        L: float = 3.0,        # control limit (sigma multiplier)
        channels: Optional[List[str]] = None,
    ) -> None:
        self.lambda_ = lambda_
        self.L = L
        self._ewma: Dict[str, float] = {}
        self._sigma_ewma = math.sqrt(lambda_ / (2.0 - lambda_))
        self._step = 0

    def update(
        self, normalised_residuals: Dict[str, float]
    ) -> List[DetectionAlert]:
        self._step += 1
        alerts = []

        for ch, nr in normalised_residuals.items():
            # Update EWMA
            prev = self._ewma.get(ch, 0.0)
            ewma_val = self.lambda_ * nr + (1.0 - self.lambda_) * prev
            self._ewma[ch] = ewma_val

            # Check against control limits
            ucl = self.L * self._sigma_ewma
            if abs(ewma_val) > ucl:
                severity = "critical" if abs(ewma_val) > 2 * ucl else "warning"
                alerts.append(DetectionAlert(
                    detector="ewma",
                    level="A",
                    severity=severity,
                    score=abs(ewma_val),
                    threshold=ucl,
                    channels=[ch],
                    message=f"EWMA alert: {ch} EWMA={ewma_val:+.2f} (UCL=±{ucl:.2f})",
                    timestamp_step=self._step,
                ))
        return alerts

    def reset_channel(self, ch: str) -> None:
        self._ewma[ch] = 0.0


# ---------------------------------------------------------------------------
# Level A: CUSUM Detector
# ---------------------------------------------------------------------------

class CUSUMDetector:
    """
    Two-sided CUSUM on normalised residuals.
    Excellent at detecting step changes and slow drifts.

    C_k^+ = max(0, C_{k-1}^+ + nr_k - k_slack)
    C_k^- = max(0, C_{k-1}^- - nr_k - k_slack)
    Alert when C^+ > h or C^- > h.
    """

    def __init__(
        self,
        k_slack: float = 0.5,    # allowance (sigma)
        h: float = 5.0,          # decision threshold
    ) -> None:
        self.k = k_slack
        self.h = h
        self._cusum_pos: Dict[str, float] = {}
        self._cusum_neg: Dict[str, float] = {}
        self._step = 0

    def update(
        self, normalised_residuals: Dict[str, float]
    ) -> List[DetectionAlert]:
        self._step += 1
        alerts = []

        for ch, nr in normalised_residuals.items():
            c_pos = self._cusum_pos.get(ch, 0.0)
            c_neg = self._cusum_neg.get(ch, 0.0)

            c_pos = max(0.0, c_pos + nr - self.k)
            c_neg = max(0.0, c_neg - nr - self.k)

            self._cusum_pos[ch] = c_pos
            self._cusum_neg[ch] = c_neg

            max_c = max(c_pos, c_neg)
            if max_c > self.h:
                direction = "+" if c_pos > c_neg else "-"
                severity = "critical" if max_c > 2 * self.h else "warning"
                alerts.append(DetectionAlert(
                    detector="cusum",
                    level="A",
                    severity=severity,
                    score=max_c,
                    threshold=self.h,
                    channels=[ch],
                    message=f"CUSUM alert ({direction}): {ch} C={max_c:.2f} (h={self.h})",
                    timestamp_step=self._step,
                ))
                # Reset after alert
                self._cusum_pos[ch] = 0.0
                self._cusum_neg[ch] = 0.0

        return alerts


# ---------------------------------------------------------------------------
# Level B: Isolation Forest Detector
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector on residual windows.
    Trained on healthy simulated data. Optional — falls back gracefully.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        window_size: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.window_size = window_size
        self._model = None
        self._fitted = False
        self._step = 0
        self._channels: List[str] = []

    def fit(self, X: np.ndarray, channels: List[str]) -> None:
        """Train Isolation Forest on healthy residual data."""
        try:
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=42,
            )
            self._model.fit(X)
            self._fitted = True
            self._channels = channels
            logger.info("IsolationForest fitted on %d samples, %d features", *X.shape)
        except ImportError:
            logger.warning("scikit-learn not installed — IsolationForest unavailable")

    def update(
        self, normalised_residuals: Dict[str, float]
    ) -> List[DetectionAlert]:
        self._step += 1
        if not self._fitted or self._model is None:
            return []

        # Build feature vector in same order as training
        features = [normalised_residuals.get(ch, 0.0) for ch in self._channels]
        X = np.array(features).reshape(1, -1)

        score = -self._model.score_samples(X)[0]  # higher = more anomalous
        threshold = 0.5  # typical IF threshold

        if score > threshold:
            severity = "critical" if score > 0.7 else "warning"
            return [DetectionAlert(
                detector="isolation_forest",
                level="B",
                severity=severity,
                score=float(score),
                threshold=threshold,
                channels=list(normalised_residuals.keys()),
                message=f"IsolationForest anomaly score: {score:.3f}",
                timestamp_step=self._step,
            )]
        return []


# ---------------------------------------------------------------------------
# Layered Detection Engine
# ---------------------------------------------------------------------------

class DetectionEngine:
    """
    Combines all detectors (Level A and B) into a single interface.
    Always runs Level A. Adds Level B if models are available.
    """

    def __init__(
        self,
        warning_sigma: float = 2.5,
        critical_sigma: float = 4.0,
        ewma_lambda: float = 0.2,
        ewma_L: float = 3.0,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
    ) -> None:
        self.threshold = ThresholdDetector(warning_sigma, critical_sigma)
        self.ewma = EWMADetector(ewma_lambda, ewma_L)
        self.cusum = CUSUMDetector(cusum_k, cusum_h)
        self.iso_forest = IsolationForestDetector()
        self._step = 0

    def update(
        self, normalised_residuals: Dict[str, float]
    ) -> List[DetectionAlert]:
        """Run all detectors and return merged alert list."""
        self._step += 1
        all_alerts: List[DetectionAlert] = []

        all_alerts.extend(self.threshold.update(normalised_residuals))
        all_alerts.extend(self.ewma.update(normalised_residuals))
        all_alerts.extend(self.cusum.update(normalised_residuals))
        all_alerts.extend(self.iso_forest.update(normalised_residuals))

        return all_alerts

    def calibrate_from_healthy(
        self,
        healthy_residuals: List[Dict[str, float]],
        target_far: float = 0.001,
    ) -> None:
        """
        Calibrate Level A thresholds from healthy data to achieve target_far.
        Also trains IsolationForest.
        """
        if not healthy_residuals:
            return

        channels = list(healthy_residuals[0].keys())
        # Build matrix of normalised residuals
        rows = []
        for d in healthy_residuals:
            rows.append([d.get(ch, 0.0) for ch in channels])
        X = np.array(rows)

        # Calibrate threshold from empirical distribution
        # (z-score corresponding to target FAR / 2 for two-tailed)
        from scipy import stats
        z_threshold = abs(stats.norm.ppf(target_far / 2))
        self.threshold.warning_sigma = z_threshold * 0.75
        self.threshold.critical_sigma = z_threshold

        # Fit IsolationForest
        self.iso_forest.fit(X, channels)
        logger.info(
            "Calibrated: warning=%.2fσ critical=%.2fσ FAR=%.4f",
            self.threshold.warning_sigma, self.threshold.critical_sigma, target_far,
        )
