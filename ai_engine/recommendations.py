"""
ai_engine.recommendations
===========================
Explainable AI Recommendation Engine.

Extracted verbatim from the original single-file ``ai_engine.py`` as
part of a structural refactor — behavior is 100% identical, only the
file location changed. See ``ai_engine/__init__.py`` for the
package-level public API.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import config

if TYPE_CHECKING:
    # Import-time only (never at runtime) to avoid a circular import with
    # ai_engine.core, which imports RecommendationEngine from this module.
    # Safe because `from __future__ import annotations` makes all
    # annotations lazy strings.
    from ai_engine.core import AnomalyDetectionEngine, AnomalyPrediction

class RecommendationEngine:
    """Explainable AI Recommendation Engine.

    Independent module, recalculated every monitoring cycle. Synthesizes
    across this cycle's anomaly prediction, the latest Root Cause
    Analysis, the latest Trend Analysis, and the latest Health Score into
    prioritized, metric-referenced recommendations — never generic advice.

    This is a separate implementation from the pre-existing
    ``AnomalyDetectionEngine.generate_recommendations()`` (which remains
    untouched and continues to back any existing consumers) — this
    engine adds explicit ``reason``, ``confidence``, and
    ``estimated_urgency`` fields per your spec, and reasons over the
    already-computed downstream engine outputs rather than re-deriving
    everything from raw metrics.

    Attributes:
        _parent: Read-only back-reference to the owning
            ``AnomalyDetectionEngine``, used only to read the latest
            Root Cause Analysis and Health Score results.
        _history: Bounded history of past recommendation bundles.
        _latest: The most recently generated list of recommendations.
    """

    def __init__(self, parent_engine: "AnomalyDetectionEngine") -> None:
        self._parent: "AnomalyDetectionEngine" = parent_engine
        self._history: deque = deque(maxlen=200)
        self._latest: List[Dict[str, Any]] = []
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prediction: AnomalyPrediction,
        metric: Dict[str, Any],
        processes: List[Dict[str, Any]],
        trend_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate prioritized, explainable recommendations for this cycle.

        Args:
            prediction: This cycle's :class:`AnomalyPrediction`.
            metric: The raw system metric dict for this cycle.
            processes: The top-process snapshot for this cycle.
            trend_results: This cycle's output from
                :meth:`TrendAnalysisEngine.analyze`.

        Returns:
            A list of recommendation dicts, sorted by priority, each with
            ``timestamp``, ``priority``, ``recommendation``, ``reason``,
            ``expected_impact``, ``confidence``, ``estimated_urgency``,
            ``category``. Always non-empty (falls back to an explicit
            "all normal" entry).
        """
        timestamp = datetime.now().isoformat()
        rca_latest = self._parent._rca_engine.get_latest()
        health_latest = self._parent._health_score_engine.get_latest()
        trend_by_key = {t["metric_key"]: t for t in trend_results}

        cpu = float(metric.get("cpu_percent", 0.0))
        ram = float(metric.get("ram_percent", 0.0))
        disk = float(metric.get("disk_percent", 0.0))
        net = float(metric.get("net_sent_mb", 0.0)) + float(metric.get("net_recv_mb", 0.0))

        recs: List[Dict[str, Any]] = []

        if prediction.is_anomaly and rca_latest:
            recs.append(self._from_rca(rca_latest))

        cpu_trend = trend_by_key.get("cpu_percent")
        if cpu > config.CPU_THRESHOLD or (cpu_trend and cpu_trend.get("classification") == "long_term_trend" and cpu_trend.get("rate_of_change_per_min", 0) > 0.5):
            recs.append(self._cpu_recommendation(cpu, cpu_trend, processes))

        ram_trend = trend_by_key.get("ram_percent")
        if ram > config.RAM_THRESHOLD or (ram_trend and ram_trend.get("trend_name") == "Possible Memory Leak"):
            recs.append(self._ram_recommendation(ram, ram_trend, processes))

        disk_trend = trend_by_key.get("disk_percent")
        if disk > 85.0 or (disk_trend and disk_trend.get("trend_name") == "Possible Resource Exhaustion"):
            recs.append(self._disk_recommendation(disk, disk_trend))

        net_trend = trend_by_key.get("net_throughput_mb")
        if net > config.NETWORK_THRESHOLD or (net_trend and net_trend.get("classification") == "long_term_trend" and net_trend.get("rate_of_change_per_min", 0) > 0):
            recs.append(self._network_recommendation(net, net_trend))

        if health_latest and health_latest.get("health_score", 100.0) < 50.0:
            recs.append(self._health_recommendation(health_latest))

        if not recs:
            recs.append({
                "priority": "LOW",
                "recommendation": "All systems operating within normal parameters. No action required.",
                "reason": "No active anomalies, adverse trends, or degraded health score were detected this cycle.",
                "expected_impact": "Continued stable operation.",
                "confidence": 90.0,
                "estimated_urgency": "None",
                "category": "status",
            })

        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recs.sort(key=lambda r: priority_order.get(r.get("priority", "LOW"), 4))

        for r in recs:
            r["timestamp"] = timestamp

        with self._lock:
            self._latest = recs
            self._history.append({"timestamp": timestamp, "recommendations": recs})

        return recs

    def get_latest(self) -> List[Dict[str, Any]]:
        """Return the most recently generated recommendations."""
        with self._lock:
            return list(self._latest)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past recommendation bundles, most recent first."""
        with self._lock:
            items = list(self._history)
        items.reverse()
        return items[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _from_rca(rca: Dict[str, Any]) -> Dict[str, Any]:
        severity = rca.get("severity", "MEDIUM")
        return {
            "priority": severity,
            "recommendation": rca.get("recommendation", "Investigate the detected anomaly."),
            "reason": (
                f"Root Cause Analysis identified '{rca.get('root_cause', 'an anomaly')}' via "
                f"{rca.get('primary_metric', 'an affected metric')}."
            ),
            "expected_impact": "Directly addresses the detected anomaly's root cause.",
            "confidence": rca.get("confidence", 70.0),
            "estimated_urgency": "Immediate" if severity in ("CRITICAL", "HIGH") else "Within 15 minutes",
            "category": "anomaly",
        }

    @staticmethod
    def _cpu_recommendation(
        cpu: float, trend: Optional[Dict[str, Any]], processes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        top_proc = processes[0].get("name", "unknown") if processes else "unknown"
        reason = f"CPU usage is at {cpu:.1f}%"
        reason += f", {trend['explanation']}" if trend else "."
        return {
            "priority": "CRITICAL" if cpu > 95 else "HIGH" if cpu > config.CPU_THRESHOLD else "MEDIUM",
            "recommendation": f"Reduce CPU-intensive workloads; consider restarting '{top_proc}' if usage does not recover.",
            "reason": reason,
            "expected_impact": "Could reduce CPU usage by 10-30%.",
            "confidence": trend["confidence"] if trend else 70.0,
            "estimated_urgency": "Immediate" if cpu > 95 else "Within 15 minutes" if cpu > config.CPU_THRESHOLD else "Monitor closely",
            "category": "cpu",
        }

    @staticmethod
    def _ram_recommendation(
        ram: float, trend: Optional[Dict[str, Any]], processes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        mem_proc = "unknown"
        if processes:
            sorted_p = sorted(processes, key=lambda p: p.get("memory_percent", 0.0), reverse=True)
            if sorted_p:
                mem_proc = sorted_p[0].get("name", "unknown")

        is_leak = bool(trend and trend.get("trend_name") == "Possible Memory Leak")
        reason = f"RAM usage is at {ram:.1f}%"
        reason += f", {trend['explanation']}" if trend else "."

        return {
            "priority": "CRITICAL" if ram > 95 else "HIGH",
            "recommendation": (
                f"Restart '{mem_proc}' and inspect its memory allocation for a leak."
                if is_leak else
                f"Close unused applications and monitor '{mem_proc}' memory consumption."
            ),
            "reason": reason,
            "expected_impact": "May free 15-40% RAM and prevent OOM conditions.",
            "confidence": trend["confidence"] if trend else 70.0,
            "estimated_urgency": "Immediate" if ram > 95 else "Within 15 minutes",
            "category": "ram",
        }

    @staticmethod
    def _disk_recommendation(disk: float, trend: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        reason = f"Disk usage is at {disk:.1f}%"
        reason += f", {trend['explanation']}" if trend else "."
        return {
            "priority": "CRITICAL" if disk > 95 else "HIGH",
            "recommendation": "Clear temporary files/logs and consider expanding storage capacity.",
            "reason": reason,
            "expected_impact": "Prevents disk-full system failure.",
            "confidence": trend["confidence"] if trend else 70.0,
            "estimated_urgency": "Immediate" if disk > 95 else "Within 30 minutes",
            "category": "disk",
        }

    @staticmethod
    def _network_recommendation(net: float, trend: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        reason = f"Network throughput is {net:.3f} MB/cycle"
        reason += f", {trend['explanation']}" if trend else "."
        return {
            "priority": "HIGH",
            "recommendation": "Investigate abnormal network activity for possible data exfiltration or misconfiguration.",
            "reason": reason,
            "expected_impact": "May identify data exfiltration or bandwidth abuse.",
            "confidence": trend["confidence"] if trend else 65.0,
            "estimated_urgency": "Within 15 minutes",
            "category": "network",
        }

    @staticmethod
    def _health_recommendation(health: Dict[str, Any]) -> Dict[str, Any]:
        score = health.get("health_score", 100.0)
        return {
            "priority": "HIGH" if score < 40 else "MEDIUM",
            "recommendation": "Perform a full system review; multiple subsystems are contributing to a degraded health score.",
            "reason": health.get("explanation", f"Health score is {score:.0f}/100."),
            "expected_impact": "Restores overall system health to acceptable levels.",
            "confidence": health.get("confidence", 70.0),
            "estimated_urgency": "Within 15 minutes",
            "category": "health",
        }