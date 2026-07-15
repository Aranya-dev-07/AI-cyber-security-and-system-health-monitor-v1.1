"""
ai_engine.trends
==================
Explainable AI Trend Analysis Engine.

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
    # ai_engine.core, which imports TrendAnalysisEngine from this module.
    # Safe because `from __future__ import annotations` makes all
    # annotations lazy strings.
    from ai_engine.core import AnomalyDetectionEngine

class TrendAnalysisEngine:
    """Explainable AI Trend Analysis Engine.

    Independent module, recalculated every monitoring cycle. Analyzes the
    shared ``config.metrics_data`` history (never re-collects metrics) to
    distinguish sustained long-term behavioural trends from temporary
    spikes, for CPU, RAM, Disk, and Network.

    This is a separate implementation from the pre-existing
    ``AnomalyDetectionEngine.analyze_trends()`` / ``_detect_trend()``
    (which remain untouched and continue to back existing
    recommendations/health penalties) — this engine adds duration in
    minutes, explicit spike-vs-trend classification, historical baseline
    comparison, and full-sentence explanations per your spec.

    Attributes:
        _parent: Read-only back-reference to the owning
            ``AnomalyDetectionEngine``, used only to read its learned
            baseline. No state is written back to it.
        _history: Bounded history of past trend-result bundles.
        _latest: The most recently computed list of trend results.
    """

    _SHORT_WINDOW = 12   # ~1 minute at the default 5s monitor interval
    _LONG_WINDOW = 120   # ~10 minutes at the default 5s monitor interval

    _METRIC_DEFS: List[Tuple[str, str]] = [
        ("cpu_percent", "CPU Usage"),
        ("ram_percent", "RAM Usage"),
        ("disk_percent", "Disk Usage"),
        ("net_throughput_mb", "Network Throughput"),
    ]

    def __init__(self, parent_engine: "AnomalyDetectionEngine") -> None:
        self._parent: "AnomalyDetectionEngine" = parent_engine
        self._history: deque = deque(maxlen=200)
        self._latest: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze current trend state for every tracked metric.

        Returns:
            A list of trend-result dicts, one per metric (``metric``,
            ``metric_key``, ``trend_name``, ``classification``,
            ``duration_minutes``, ``rate_of_change_per_min``,
            ``confidence``, ``severity``, ``historical_comparison``,
            ``explanation``, ``current_value``, ``timestamp``).
        """
        with config.data_lock:
            snapshot = list(config.metrics_data)

        timestamp = datetime.now().isoformat()
        results = [
            self._analyze_single(key, display, self._extract_values(snapshot, key), timestamp)
            for key, display in self._METRIC_DEFS
        ]

        with self._lock:
            self._latest = results
            self._history.append({"timestamp": timestamp, "trends": results})

        return results

    def get_latest(self) -> List[Dict[str, Any]]:
        """Return the most recently computed trend results."""
        with self._lock:
            return list(self._latest)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past trend-result bundles, most recent first."""
        with self._lock:
            items = list(self._history)
        items.reverse()
        return items[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_values(snapshot: List[Dict[str, Any]], key: str) -> List[float]:
        if key == "net_throughput_mb":
            return [
                float(m.get("net_sent_mb", 0.0)) + float(m.get("net_recv_mb", 0.0))
                for m in snapshot
            ]
        return [float(m.get(key, 0.0)) for m in snapshot]

    def _analyze_single(
        self, key: str, display: str, values: List[float], timestamp: str
    ) -> Dict[str, Any]:
        n = len(values)
        if n < 3:
            return {
                "metric": display, "metric_key": key,
                "trend_name": "Insufficient Data", "classification": "insufficient_data",
                "duration_minutes": 0.0, "rate_of_change_per_min": 0.0,
                "confidence": 0.0, "severity": "NORMAL",
                "historical_comparison": "Not enough data collected yet.",
                "explanation": f"Not enough {display.lower()} samples have been collected to analyze a trend.",
                "current_value": round(values[-1], 2) if values else 0.0,
                "timestamp": timestamp,
            }

        short = values[-self._SHORT_WINDOW:] if n >= self._SHORT_WINDOW else values
        long_ = values[-self._LONG_WINDOW:] if n >= self._LONG_WINDOW else values

        short_slope, short_conf = self._linreg_slope(short)
        long_slope, long_conf = self._linreg_slope(long_)

        interval_min = max(config.MONITOR_INTERVAL / 60.0, 1e-6)
        short_rate_per_min = short_slope / interval_min
        long_rate_per_min = long_slope / interval_min

        std_short = float(np.std(short))
        mean_short = float(np.mean(short)) or 1.0
        volatile = std_short > mean_short * 0.3 and mean_short > 5.0

        classification, rate_used, confidence = self._classify(
            short_rate_per_min, long_rate_per_min, short_conf, long_conf, volatile
        )

        duration_minutes = (len(long_) * config.MONITOR_INTERVAL) / 60.0
        current = values[-1]

        baseline = self._parent._baseline.get(key, 0.0)
        if baseline > 0:
            dev_pct = ((current - baseline) / baseline) * 100.0
            historical_comparison = (
                f"{display} is currently {abs(dev_pct):.0f}% "
                f"{'above' if dev_pct >= 0 else 'below'} its learned baseline."
            )
        else:
            historical_comparison = f"{display} baseline has not been learned yet."

        trend_name, severity, explanation = self._build_trend_narrative(
            key, display, classification, rate_used, duration_minutes, current, historical_comparison,
        )

        return {
            "metric": display, "metric_key": key,
            "trend_name": trend_name, "classification": classification,
            "duration_minutes": round(duration_minutes, 1),
            "rate_of_change_per_min": round(rate_used, 3),
            "confidence": round(confidence, 1), "severity": severity,
            "historical_comparison": historical_comparison,
            "explanation": explanation,
            "current_value": round(current, 2),
            "timestamp": timestamp,
        }

    @staticmethod
    def _linreg_slope(values: List[float]) -> Tuple[float, float]:
        """Return (slope-per-sample, confidence%) via simple linear regression."""
        n = len(values)
        if n < 2:
            return 0.0, 0.0
        arr = np.array(values, dtype=float)
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
        if np.std(arr) > 0:
            corr = float(np.corrcoef(x, arr)[0, 1])
            confidence = min(99.0, abs(corr) * 100.0)
        else:
            confidence = 20.0
        return slope, confidence

    @staticmethod
    def _classify(
        short_rate: float, long_rate: float, short_conf: float, long_conf: float, volatile: bool,
    ) -> Tuple[str, float, float]:
        """Return (classification, rate_used, confidence)."""
        if volatile:
            return "temporary_spike", short_rate, max(short_conf, 40.0)
        if abs(long_rate) < 0.05:
            return "stable", long_rate, long_conf
        if abs(short_rate) > abs(long_rate) * 2.5 and abs(short_rate) > 1.0:
            return "temporary_spike", short_rate, short_conf
        return "long_term_trend", long_rate, long_conf

    @staticmethod
    def _build_trend_narrative(
        key: str, display: str, classification: str, rate: float,
        duration_min: float, current: float, historical_comparison: str,
    ) -> Tuple[str, str, str]:
        """Return (trend_name, severity, explanation)."""
        direction = "increasing" if rate > 0 else "decreasing" if rate < 0 else "stable"

        if classification == "stable":
            return (
                f"{display} Stable", "NORMAL",
                f"{display} has remained stable at approximately {current:.1f}% with no significant trend.",
            )

        if classification == "temporary_spike":
            return (
                f"{display}: Repeated Spikes", "MEDIUM",
                (
                    f"{display} is showing volatile, spike-like behavior rather than a sustained "
                    f"trend. Current value is {current:.1f}%. {historical_comparison}"
                ),
            )

        # long_term_trend
        severity = "LOW"
        if abs(rate) > 2.0:
            severity = "HIGH"
        elif abs(rate) > 0.5:
            severity = "MEDIUM"

        special_name: Optional[str] = None
        if key == "ram_percent" and rate > 0.3:
            special_name = "Possible Memory Leak"
            if severity == "MEDIUM":
                severity = "HIGH"
        elif key == "disk_percent" and rate > 0.2 and current > 70.0:
            special_name = "Possible Resource Exhaustion"

        trend_name = special_name or f"{display} Steadily {'Increasing' if direction == 'increasing' else 'Decreasing'}"

        explanation = (
            f"{display} has been {direction} at approximately {abs(rate):.2f}%/min over the last "
            f"~{duration_min:.0f} minute(s), currently at {current:.1f}%. {historical_comparison}"
        )
        if special_name == "Possible Memory Leak":
            explanation += " This continuous, non-recovering growth pattern is consistent with a memory leak."
        elif special_name == "Possible Resource Exhaustion":
            explanation += " At this rate, available disk capacity may be significantly reduced if the trend continues."

        return trend_name, severity, explanation