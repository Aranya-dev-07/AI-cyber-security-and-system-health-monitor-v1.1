"""
ai_engine.health_score
========================
Explainable AI Health Score Engine (``HealthScoreWeights`` +
``HealthScoreEngine``).

Extracted verbatim from the original single-file ``ai_engine.py`` as
part of a structural refactor — behavior is 100% identical, only the
file location changed. See ``ai_engine/__init__.py`` for the
package-level public API.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

import config

if TYPE_CHECKING:
    # Import-time only (never at runtime) to avoid a circular import with
    # ai_engine.core, which imports HealthScoreEngine from this module.
    # Safe because `from __future__ import annotations` makes all
    # annotations lazy strings.
    from ai_engine.core import AnomalyDetectionEngine, AnomalyPrediction


@dataclass
class HealthScoreWeights:
    """Configurable weights for the AI Health Score Engine.

    Weights need not sum to exactly 1.0 — :meth:`HealthScoreEngine.compute`
    normalizes by the total automatically — but keeping them close to 1.0
    keeps the numbers intuitive when tuning.

    Future developers can change system health scoring behavior entirely
    by constructing ``HealthScoreEngine(parent, weights=HealthScoreWeights(...))``
    with different values — no changes to the scoring algorithm itself
    are required.

    Attributes:
        cpu: Weight for the CPU utilisation sub-score.
        ram: Weight for the RAM utilisation sub-score.
        disk: Weight for the disk utilisation sub-score.
        network: Weight for the network throughput sub-score.
        anomalies: Weight for the active-anomaly-severity sub-score.
        historical: Weight for the baseline-deviation sub-score.
    """
    cpu:        float = 0.25
    ram:        float = 0.25
    disk:       float = 0.15
    network:    float = 0.10
    anomalies:  float = 0.15
    historical: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        """Return the weights as a plain dict (API/DB friendly)."""
        return asdict(self)

    def total(self) -> float:
        """Return the sum of all weights, used for normalization."""
        return sum(self.as_dict().values())


class HealthScoreEngine:
    """Explainable AI Health Score Engine.

    Independent module recalculated every monitoring cycle. It combines
    the current metric snapshot, the active-anomaly list, and the latest
    Root Cause Analysis result into one weighted, human-explainable
    health score — without duplicating the anomaly-detection scoring,
    the Root Cause Analysis logic, or the pre-existing internal
    ``AnomalyDetectionEngine._health_score`` bookkeeping (which continues
    to independently drive existing recommendations/trend penalties,
    untouched).

    Attributes:
        _parent: Read-only back-reference to the owning
            ``AnomalyDetectionEngine``, used to read its rolling windows,
            learned baseline, active anomalies, and Root Cause Analysis
            engine. No state is written back to it.
        _weights: The configurable :class:`HealthScoreWeights` in use.
        _history: Bounded history of ``(timestamp, score)`` pairs, used to
            compute the "vs. last hour" historical comparison.
        _latest: The most recently computed result, or ``None``.
    """

    _STATUS_BANDS: List[Tuple[float, str]] = [
        (90.0, "Excellent"),
        (75.0, "Good"),
        (60.0, "Fair"),
        (40.0, "Poor"),
        (0.0,  "Critical"),
    ]

    def __init__(
        self,
        parent_engine: "AnomalyDetectionEngine",
        weights: Optional[HealthScoreWeights] = None,
    ) -> None:
        self._parent: "AnomalyDetectionEngine" = parent_engine
        self._weights: HealthScoreWeights = weights or HealthScoreWeights()
        # ~1 hour of history at the project's default 5s monitor interval.
        # Approximate if MONITOR_INTERVAL is changed; documented as such.
        self._history: deque = deque(maxlen=720)
        self._latest: Optional[Dict[str, Any]] = None
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def compute(
        self,
        metric: Dict[str, Any],
        processes: List[Dict[str, Any]],
        prediction: AnomalyPrediction,
    ) -> Dict[str, Any]:
        """Compute and explain the current AI Health Score.

        Intended to be called once per monitoring cycle, regardless of
        whether an anomaly was detected on this cycle.

        Args:
            metric: The raw system metric dict for this cycle.
            processes: The top-process snapshot for this cycle.
            prediction: The :class:`AnomalyPrediction` for this cycle
                (consumed only for its severity/confidence — never
                recomputed).

        Returns:
            A database/API-friendly dict matching the documented Health
            Score output schema (``timestamp``, ``health_score``,
            ``status``, ``confidence``, ``contributing_factors``,
            ``weights``, ``explanation``, ``historical_comparison``).
        """
        cpu = float(metric.get("cpu_percent", 0.0))
        ram = float(metric.get("ram_percent", 0.0))
        disk = float(metric.get("disk_percent", 0.0))
        net = float(metric.get("net_sent_mb", 0.0)) + float(metric.get("net_recv_mb", 0.0))

        cpu_sub = self._metric_subscore(cpu, config.CPU_THRESHOLD, config.CPU_THRESHOLD + 10.0)
        ram_sub = self._metric_subscore(ram, config.RAM_THRESHOLD, config.RAM_THRESHOLD + 10.0)
        disk_sub = self._metric_subscore(disk, 80.0, 90.0)
        net_sub = self._metric_subscore(net, config.NETWORK_THRESHOLD, config.NETWORK_THRESHOLD * 1.5)

        active_anomalies = self._parent.get_active_anomalies()
        anomaly_sub, anomaly_summary = self._anomaly_subscore(active_anomalies)

        deviations = self._parent._compute_deviations(metric)
        historical_sub, avg_abs_deviation = self._historical_subscore(deviations)

        w = self._weights
        total_weight = w.total() or 1.0
        weighted_score = (
            cpu_sub * w.cpu + ram_sub * w.ram + disk_sub * w.disk +
            net_sub * w.network + anomaly_sub * w.anomalies +
            historical_sub * w.historical
        ) / total_weight
        health_score = float(np.clip(weighted_score, 0.0, 100.0))

        status = self._score_to_status(health_score)

        rca_latest = self._parent._rca_engine.get_latest()

        contributing_factors = self._build_contributing_factors(
            cpu, ram, disk, net, anomaly_summary, rca_latest,
        )

        explanation = self._build_explanation(
            cpu, ram, disk, net, anomaly_summary, rca_latest, health_score, status,
        )

        confidence = self._compute_confidence()

        with self._lock:
            past_scores = [s for _, s in self._history]
        historical_comparison = self._historical_comparison_text(health_score, past_scores)

        result: Dict[str, Any] = {
            "timestamp": prediction.timestamp,
            "health_score": round(health_score, 1),
            "status": status,
            "confidence": round(confidence, 1),
            "contributing_factors": contributing_factors,
            "weights": w.as_dict(),
            "explanation": explanation,
            "historical_comparison": historical_comparison,
        }

        with self._lock:
            self._latest = result
            self._history.append((prediction.timestamp, health_score))

        return result

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently computed Health Score result, if any."""
        with self._lock:
            return dict(self._latest) if self._latest else None

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return past Health Score results, most recent first.

        Note: only the single most recent full result object is retained
        verbatim (``_latest``); ``_history`` stores compact
        ``(timestamp, score)`` pairs for trend/comparison purposes. This
        method reconstructs lightweight entries from that compact history.

        Args:
            limit: Maximum number of results to return.
        """
        with self._lock:
            items = list(self._history)
        items.reverse()
        return [
            {"timestamp": ts, "health_score": round(score, 1), "status": self._score_to_status(score)}
            for ts, score in items[:limit]
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @classmethod
    def _score_to_status(cls, score: float) -> str:
        for threshold, label in cls._STATUS_BANDS:
            if score >= threshold:
                return label
        return "Critical"

    @staticmethod
    def _metric_subscore(value: float, warn_threshold: float, danger_threshold: float) -> float:
        """Map a raw metric value to a 0-100 sub-score using its thresholds.

        Stays at 100 while comfortably below the warn threshold, glides
        down to ~85 as it approaches it, drops steeply between warn and
        danger, and bottoms out near 0 beyond danger.
        """
        if warn_threshold <= 0:
            return 100.0
        safe_zone = warn_threshold * 0.7

        if value <= safe_zone:
            return 100.0
        if value <= warn_threshold:
            frac = (value - safe_zone) / max(warn_threshold - safe_zone, 1e-6)
            return 100.0 - frac * 15.0
        if value <= danger_threshold:
            frac = (value - warn_threshold) / max(danger_threshold - warn_threshold, 1e-6)
            return 85.0 - frac * 45.0
        overshoot = value - danger_threshold
        return max(0.0, 40.0 - overshoot * 1.5)

    @staticmethod
    def _anomaly_subscore(active_anomalies: List[Dict[str, Any]]) -> Tuple[float, Dict[str, int]]:
        """Score based on severity-weighted count of currently active anomalies."""
        recent = active_anomalies[:10]
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for a in recent:
            sev = a.get("severity", "LOW")
            if sev in counts:
                counts[sev] += 1

        weighted = counts["CRITICAL"] * 4 + counts["HIGH"] * 2 + counts["MEDIUM"] * 1 + counts["LOW"] * 0.5
        subscore = max(0.0, 100.0 - weighted * 10.0)
        return subscore, counts

    @staticmethod
    def _historical_subscore(deviations: Dict[str, float]) -> Tuple[float, float]:
        """Score based on average absolute deviation from the learned baseline."""
        if not deviations:
            return 100.0, 0.0
        avg_abs_dev = float(np.mean([abs(v) for v in deviations.values()]))
        subscore = max(0.0, 100.0 - avg_abs_dev * 1.2)
        return subscore, avg_abs_dev

    @staticmethod
    def _compute_confidence() -> float:
        """Confidence reflects how much monitoring data backs this score."""
        sample_count = len(config.metrics_data)
        if sample_count < 5:
            return 45.0
        if sample_count < 20:
            return 60.0 + sample_count * 0.8
        return float(min(99.0, 80.0 + sample_count * 0.15))

    def _build_contributing_factors(
        self,
        cpu: float, ram: float, disk: float, net: float,
        anomaly_summary: Dict[str, int],
        rca_latest: Optional[Dict[str, Any]],
    ) -> List[str]:
        factors: List[str] = []

        if cpu > config.CPU_THRESHOLD:
            factors.append("High CPU Usage")
        if ram > config.RAM_THRESHOLD:
            if len(self._parent._rolling_ram) >= 3:
                vals = list(self._parent._rolling_ram)[-3:]
                if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
                    factors.append("Increasing RAM Usage")
                else:
                    factors.append("High RAM Usage")
            else:
                factors.append("High RAM Usage")
        if disk > 90:
            factors.append("Critical Disk Usage")
        if net > config.NETWORK_THRESHOLD:
            factors.append("Abnormal Network Activity")

        total_anomalies = sum(anomaly_summary.values())
        if total_anomalies == 1:
            factors.append("One Active Anomaly")
        elif total_anomalies > 1:
            factors.append(f"{total_anomalies} Active Anomalies")

        if rca_latest and rca_latest.get("root_cause"):
            factors.append(rca_latest["root_cause"])

        if not factors:
            factors.append("All Metrics Nominal")

        return factors

    def _build_explanation(
        self,
        cpu: float, ram: float, disk: float, net: float,
        anomaly_summary: Dict[str, int],
        rca_latest: Optional[Dict[str, Any]],
        health_score: float,
        status: str,
    ) -> str:
        parts: List[str] = [f"Health Score is {health_score:.0f} ({status})."]

        if cpu > config.CPU_THRESHOLD:
            duration_above = sum(1 for v in self._parent._rolling_cpu if v > config.CPU_THRESHOLD)
            dur_min = (duration_above * config.MONITOR_INTERVAL) / 60.0
            parts.append(
                f"CPU utilization remained above {config.CPU_THRESHOLD:.0f}% for "
                f"approximately {dur_min:.0f} minute(s)."
            )

        if ram > config.RAM_THRESHOLD:
            if len(self._parent._rolling_ram) >= 3:
                vals = list(self._parent._rolling_ram)[-3:]
                if all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)):
                    increase = vals[-1] - vals[0]
                    parts.append(f"RAM usage increased continuously by {increase:.0f}%.")
                else:
                    parts.append(f"RAM usage is elevated at {ram:.1f}%.")
            else:
                parts.append(f"RAM usage is elevated at {ram:.1f}%.")

        high = anomaly_summary.get("HIGH", 0)
        crit = anomaly_summary.get("CRITICAL", 0)
        if crit > 0:
            parts.append(f"{crit} Critical severity anomaly(ies) are currently active.")
        if high > 0:
            parts.append(f"{high} High severity anomaly(ies) are currently active.")

        if rca_latest:
            proc = rca_latest.get("responsible_process", {}).get("name", "N/A")
            if proc and proc != "N/A":
                parts.append(
                    f"The AI Root Cause Analysis identified '{proc}' as the primary "
                    f"contributor ({rca_latest.get('root_cause', 'unknown cause')})."
                )

        if health_score < 60:
            parts.append(
                "Historical comparison indicates current resource utilization is "
                "significantly above the learned baseline, and overall system "
                "stability has decreased."
            )

        return " ".join(parts)

    def _historical_comparison_text(self, current_score: float, past_scores: List[float]) -> str:
        if not past_scores:
            return "Not enough history yet to compare against the last hour."

        avg_past = float(np.mean(past_scores))
        if avg_past <= 0:
            return "Not enough history yet to compare against the last hour."

        delta_pct = ((current_score - avg_past) / avg_past) * 100.0
        direction = "higher" if delta_pct >= 0 else "lower"
        return (
            f"Current health is {abs(delta_pct):.0f}% {direction} than the average "
            f"health score over the recent monitoring history."
        )