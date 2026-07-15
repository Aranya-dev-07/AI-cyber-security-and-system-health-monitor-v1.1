"""
ai_engine.predictive
======================
Explainable AI Predictive Alert Engine.

Extracted verbatim from the original single-file ``ai_engine.py`` as
part of a structural refactor — behavior is 100% identical, only the
file location changed. See ``ai_engine/__init__.py`` for the
package-level public API.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

import config

if TYPE_CHECKING:
    # Import-time only (never at runtime) to avoid a circular import with
    # ai_engine.core, which imports PredictiveAlertEngine from this module.
    # Safe because `from __future__ import annotations` makes all
    # annotations lazy strings.
    from ai_engine.core import AnomalyDetectionEngine

class PredictiveAlertEngine:
    """Explainable AI Predictive Alert Engine.

    Independent module, recalculated every monitoring cycle. Forecasts
    likely future threshold breaches by linearly extrapolating each
    metric's already-computed :class:`TrendAnalysisEngine` rate of
    change — it performs no trend computation or anomaly detection of
    its own, and only forecasts off metrics already classified as a
    sustained ``long_term_trend`` (never off spikes or stable readings).

    Attributes:
        _parent: Read-only back-reference to the owning
            ``AnomalyDetectionEngine``, used only to read the latest
            Root Cause Analysis result for root-cause-likelihood context.
        _history: Bounded history of past forecast bundles.
        _latest: The most recently computed list of predictive alerts
            (may be empty if no metric is currently trending toward its
            threshold).
    """

    _HORIZONS_MIN: List[int] = [5, 15, 30, 60]

    # metric_key -> (display label, callable returning current threshold)
    _METRIC_THRESHOLDS: Dict[str, Tuple[str, Any]] = {
        "cpu_percent":        ("CPU",     lambda: config.CPU_THRESHOLD),
        "ram_percent":        ("RAM",     lambda: config.RAM_THRESHOLD),
        "disk_percent":       ("Disk",    lambda: 90.0),
        "net_throughput_mb":  ("Network", lambda: config.NETWORK_THRESHOLD),
    }

    def __init__(self, parent_engine: "AnomalyDetectionEngine") -> None:
        self._parent: "AnomalyDetectionEngine" = parent_engine
        self._history: deque = deque(maxlen=200)
        self._latest: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forecast(
        self, trend_results: List[Dict[str, Any]], metric: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Forecast likely future threshold breaches from current trends.

        Args:
            trend_results: This cycle's output from
                :meth:`TrendAnalysisEngine.analyze`. Consumed only —
                never recomputed.
            metric: The raw system metric dict for this cycle (used only
                to timestamp/contextualize the forecast).

        Returns:
            A list of predictive-alert dicts (may be empty). Each has
            ``timestamp``, ``predicted_issue``, ``metric``,
            ``horizon_minutes``, ``probability``, ``confidence``,
            ``predicted_severity``, ``estimated_time_until``,
            ``root_cause_likelihood``, ``explanation``,
            ``recommended_action``.
        """
        timestamp = datetime.now().isoformat()
        trend_by_key = {t["metric_key"]: t for t in trend_results}
        rca_latest = self._parent._rca_engine.get_latest()

        alerts: List[Dict[str, Any]] = []

        for key, (label, threshold_fn) in self._METRIC_THRESHOLDS.items():
            trend = trend_by_key.get(key)
            if not trend or trend.get("classification") != "long_term_trend":
                continue  # only forecast off sustained trends, never spikes/stable/insufficient

            rate_per_min = trend.get("rate_of_change_per_min", 0.0)
            if rate_per_min <= 0:
                continue  # only forecast worsening trends

            current_value = trend.get("current_value", 0.0)
            threshold = threshold_fn()

            if current_value >= threshold:
                continue  # already breached — that's an active anomaly, not a prediction

            alert = self._first_breaching_horizon(
                key, label, current_value, threshold, rate_per_min,
                trend.get("confidence", 0.0), rca_latest, timestamp,
            )
            if alert:
                alerts.append(alert)

        with self._lock:
            self._latest = alerts
            self._history.append({"timestamp": timestamp, "alerts": alerts})

        return alerts

    def get_latest(self) -> List[Dict[str, Any]]:
        """Return the most recently forecast predictive alerts (may be empty)."""
        with self._lock:
            return list(self._latest)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past predictive-alert bundles, most recent first."""
        with self._lock:
            items = list(self._history)
        items.reverse()
        return items[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _first_breaching_horizon(
        self,
        key: str, label: str, current_value: float, threshold: float, rate_per_min: float,
        confidence: float, rca_latest: Optional[Dict[str, Any]], timestamp: str,
    ) -> Optional[Dict[str, Any]]:
        """Return a predictive-alert dict for the earliest horizon that
        projects a threshold breach, or ``None`` if no configured horizon
        (5/15/30/60 min) projects one.
        """
        for horizon in self._HORIZONS_MIN:
            projected = current_value + rate_per_min * horizon
            if projected < threshold:
                continue

            minutes_to_breach = (threshold - current_value) / rate_per_min
            probability = self._estimate_probability(confidence, projected, threshold)
            severity = self._severity_for(probability, horizon)
            root_cause_likelihood = self._root_cause_likelihood(key, rca_latest)

            explanation = (
                f"{label} usage is trending upward at {rate_per_min:.2f}%/min based on the current "
                f"growth trend. It is likely to exceed {threshold:.0f}% within the next {horizon} "
                f"minute(s) (projected ~{projected:.0f}%)."
            )

            return {
                "timestamp": timestamp,
                "predicted_issue": self._issue_name(key),
                "metric": label,
                "horizon_minutes": horizon,
                "probability": round(probability, 1),
                "confidence": round(confidence, 1),
                "predicted_severity": severity,
                "estimated_time_until": f"~{minutes_to_breach:.0f} minute(s)",
                "root_cause_likelihood": root_cause_likelihood,
                "explanation": explanation,
                "recommended_action": self._recommended_action(key),
            }

        return None

    @staticmethod
    def _estimate_probability(confidence: float, projected: float, threshold: float) -> float:
        margin = max(0.0, projected - threshold)
        margin_component = min(40.0, margin * 1.5)
        probability = 40.0 + margin_component + (confidence * 0.2)
        return float(np.clip(probability, 0.0, 99.0))

    @staticmethod
    def _severity_for(probability: float, horizon: int) -> str:
        if probability >= 80.0 and horizon <= 15:
            return "CRITICAL"
        if probability >= 65.0:
            return "HIGH"
        if probability >= 45.0:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _issue_name(key: str) -> str:
        names = {
            "cpu_percent": "Possible CPU Overload",
            "ram_percent": "Possible Memory Exhaustion",
            "disk_percent": "Disk Capacity Approaching Limit",
            "net_throughput_mb": "Possible Network Congestion",
        }
        return names.get(key, "Possible Resource Issue")

    @staticmethod
    def _recommended_action(key: str) -> str:
        actions = {
            "cpu_percent": "Identify and reduce CPU-intensive workloads before the projected threshold breach.",
            "ram_percent": "Investigate for a memory leak and consider proactively restarting the top memory-consuming process.",
            "disk_percent": "Free up disk space or expand storage capacity before it is exhausted.",
            "net_throughput_mb": "Investigate the rising network activity for possible congestion or data exfiltration.",
        }
        return actions.get(key, "Monitor the affected metric closely.")

    @staticmethod
    def _root_cause_likelihood(key: str, rca_latest: Optional[Dict[str, Any]]) -> str:
        if not rca_latest:
            return "Unknown — no recent Root Cause Analysis available to correlate."

        display_map = {
            "cpu_percent": "CPU Usage", "ram_percent": "RAM Usage",
            "disk_percent": "Disk Usage", "net_throughput_mb": "Network Throughput",
        }
        proc = rca_latest.get("responsible_process", {}).get("name", "N/A")
        if rca_latest.get("primary_metric") == display_map.get(key) and proc and proc != "N/A":
            return f"Likely '{proc}', consistent with the most recent Root Cause Analysis."
        return "No strong process-level correlation identified yet."