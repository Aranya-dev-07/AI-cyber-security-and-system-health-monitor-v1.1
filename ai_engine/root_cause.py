"""
ai_engine.root_cause
=====================
Explainable AI Root Cause Analysis Engine.

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
    # ai_engine.core, which imports RootCauseAnalysisEngine from this
    # module. Safe because `from __future__ import annotations` makes all
    # annotations lazy strings.
    from ai_engine.core import AnomalyDetectionEngine, AnomalyPrediction

class RootCauseAnalysisEngine:
    """Explainable AI Root Cause Analysis Engine.

    Independent module that runs strictly *after* ``AnomalyDetectionEngine``
    has already flagged a sample as anomalous. It performs NO anomaly
    detection of its own — it only consumes an already-computed
    :class:`AnomalyPrediction` plus the raw metric/process snapshot that
    produced it, and explains *why* the anomaly happened.

    This keeps the original Isolation Forest detection logic completely
    untouched: this class is a pure downstream consumer, wired in from
    ``AnomalyDetectionEngine._register_anomaly`` (an additive "dashboard
    intelligence" hook, not part of the core scoring path).

    Attributes:
        _parent: Read-only back-reference to the owning
            ``AnomalyDetectionEngine``, used only to read its rolling
            windows and learned baseline. No state is written back to it.
        _history: Bounded history of past root-cause results, most
            recent last.
        _latest: The most recently computed root-cause result, or
            ``None`` if no anomaly has been analyzed yet this session.
    """

    # Root-cause classification labels
    _LABEL_CPU_SATURATION      = "CPU Saturation"
    _LABEL_MEMORY_LEAK         = "Possible Memory Leak"
    _LABEL_DISK_BOTTLENECK     = "Disk Bottleneck"
    _LABEL_NETWORK_TRAFFIC     = "Heavy Network Traffic"
    _LABEL_RUNAWAY_PROCESS     = "Runaway Process"
    _LABEL_RESOURCE_SPIKE      = "Abnormal Resource Spike"
    _LABEL_TOO_MANY_PROCESSES  = "Too Many Background Processes"
    _LABEL_MULTI_HIGH_PROCESS  = "Multiple High Resource Processes"

    def __init__(self, parent_engine: "AnomalyDetectionEngine") -> None:
        self._parent: "AnomalyDetectionEngine" = parent_engine
        self._history: deque = deque(maxlen=100)
        self._latest: Optional[Dict[str, Any]] = None
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        prediction: AnomalyPrediction,
        metric: Dict[str, Any],
        processes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Explain a single already-detected anomaly.

        Args:
            prediction: The :class:`AnomalyPrediction` returned by
                ``AnomalyDetectionEngine.predict_anomaly`` for this cycle.
                Only consumed — never recomputed.
            metric: The raw system metric dict for this cycle.
            processes: The top-process snapshot for this cycle.

        Returns:
            A database/API-friendly dict matching the documented Root
            Cause Analysis output schema (``timestamp``, ``root_cause``,
            ``primary_metric``, ``responsible_process``, ``confidence``,
            ``severity``, ``historical_comparison``, ``recommendation``,
            ``explanation``).
        """
        deviations = self._parent._compute_deviations(metric)
        primary_key, primary_dev = self._identify_primary_metric(deviations)

        responsible_process = self._identify_responsible_process(primary_key, processes)

        duration_samples, continuous_increase = self._trend_context(primary_key)
        duration_min = (duration_samples * config.MONITOR_INTERVAL) / 60.0

        root_cause = self._classify_root_cause(
            primary_key, primary_dev, deviations, processes, continuous_increase
        )

        explanation = self._build_explanation(
            primary_key, primary_dev, metric, responsible_process,
            duration_min, continuous_increase, root_cause,
        )

        historical_comparison = self._historical_comparison_text(primary_key, primary_dev)

        recommendation = self._build_recommendation(root_cause, responsible_process, primary_key)

        confidence = self._compute_confidence(prediction, primary_dev)

        result: Dict[str, Any] = {
            "timestamp": prediction.timestamp,
            "root_cause": root_cause,
            "primary_metric": self._display_metric_name(primary_key),
            "responsible_process": responsible_process,
            "confidence": round(confidence, 1),
            "severity": prediction.severity,
            "historical_comparison": historical_comparison,
            "recommendation": recommendation,
            "explanation": explanation,
        }

        with self._lock:
            self._latest = result
            self._history.append(result)

        return result

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recently computed root-cause result, if any."""
        with self._lock:
            return dict(self._latest) if self._latest else None

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past root-cause results, most recent first."""
        with self._lock:
            items = list(self._history)
        items.reverse()
        return items[:limit]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _display_metric_name(key: str) -> str:
        names = {
            "cpu_percent": "CPU Usage",
            "ram_percent": "RAM Usage",
            "disk_percent": "Disk Usage",
            "net_throughput_mb": "Network Throughput",
            "process_activity": "Process Activity",
        }
        return names.get(key, key.replace("_", " ").title())

    def _identify_primary_metric(self, deviations: Dict[str, float]) -> Tuple[str, float]:
        """Pick the metric with the largest absolute deviation from baseline."""
        if not deviations:
            return "cpu_percent", 0.0
        sorted_devs = sorted(deviations.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return sorted_devs[0]

    def _identify_responsible_process(
        self, primary_key: str, processes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Pick the top process for the resource type behind the anomaly.

        Note: the project's process collector (``config.py``) tracks
        ``memory_percent`` rather than an absolute MB figure, so the
        ``memory`` field below is a percentage, not megabytes.
        """
        if not processes:
            return {
                "name": "N/A", "pid": -1, "cpu": 0.0, "memory": 0.0, "status": "unknown",
            }

        if "ram" in primary_key or "memory" in primary_key:
            sorted_p = sorted(processes, key=lambda p: p.get("memory_percent", 0.0), reverse=True)
        else:
            sorted_p = sorted(processes, key=lambda p: p.get("cpu_percent", 0.0), reverse=True)

        top = sorted_p[0]
        return {
            "name":   top.get("name", "unknown"),
            "pid":    top.get("pid", -1),
            "cpu":    round(float(top.get("cpu_percent", 0.0)), 2),
            "memory": round(float(top.get("memory_percent", 0.0)), 2),
            "status": top.get("status", "unknown"),
        }

    def _trend_context(self, primary_key: str) -> Tuple[int, bool]:
        """Return (samples above baseline, is continuously increasing)."""
        window_map = {
            "cpu_percent": self._parent._rolling_cpu,
            "ram_percent": self._parent._rolling_ram,
            "net_throughput_mb": self._parent._rolling_net,
        }
        window = window_map.get(primary_key)
        if not window or len(window) < 2:
            return 0, False

        values = list(window)
        baseline = self._parent._baseline.get(primary_key, 0.0)
        duration = sum(1 for v in values if baseline and v > baseline)

        tail = values[-5:] if len(values) >= 5 else values
        continuous_increase = len(tail) >= 3 and all(
            tail[i] <= tail[i + 1] for i in range(len(tail) - 1)
        )
        return duration, continuous_increase

    def _classify_root_cause(
        self,
        primary_key: str,
        primary_dev: float,
        deviations: Dict[str, float],
        processes: List[Dict[str, Any]],
        continuous_increase: bool,
    ) -> str:
        """Map metric + pattern context to a labelled root cause."""
        high_dev_count = sum(1 for v in deviations.values() if abs(v) >= 30)

        if high_dev_count >= 2:
            return self._LABEL_MULTI_HIGH_PROCESS

        high_cpu_procs = sum(1 for p in processes if p.get("cpu_percent", 0.0) > 50)
        if len(processes) >= 5 and high_cpu_procs >= 3:
            return self._LABEL_TOO_MANY_PROCESSES

        if "ram" in primary_key:
            return self._LABEL_MEMORY_LEAK if continuous_increase else self._LABEL_RESOURCE_SPIKE
        if "cpu" in primary_key:
            if processes and processes[0].get("cpu_percent", 0.0) > 90:
                return self._LABEL_RUNAWAY_PROCESS
            return self._LABEL_CPU_SATURATION
        if "disk" in primary_key:
            return self._LABEL_DISK_BOTTLENECK
        if "net" in primary_key:
            return self._LABEL_NETWORK_TRAFFIC

        return self._LABEL_RESOURCE_SPIKE

    def _build_explanation(
        self,
        primary_key: str,
        primary_dev: float,
        metric: Dict[str, Any],
        responsible_process: Dict[str, Any],
        duration_min: float,
        continuous_increase: bool,
        root_cause: str,
    ) -> str:
        proc_name = responsible_process.get("name", "N/A")
        current_val = float(metric.get(primary_key, 0.0)) if primary_key in metric else None
        metric_disp = self._display_metric_name(primary_key)

        parts: List[str] = []

        if current_val is not None:
            parts.append(f"{metric_disp} is currently at {current_val:.1f}%.")

        if continuous_increase and duration_min > 0:
            parts.append(
                f"{metric_disp} has increased continuously over the last "
                f"~{duration_min:.0f} minute(s)."
            )
        elif duration_min > 0:
            parts.append(
                f"{metric_disp} has remained above its learned baseline for "
                f"~{duration_min:.0f} minute(s)."
            )

        if proc_name != "N/A":
            if root_cause == self._LABEL_MEMORY_LEAK:
                parts.append(
                    f"The process '{proc_name}' continuously consumed memory "
                    f"without releasing it, consistent with a leak pattern."
                )
            elif root_cause == self._LABEL_RUNAWAY_PROCESS:
                parts.append(
                    f"The process '{proc_name}' is consuming an unusually large "
                    f"share of CPU, consistent with a runaway or stuck process."
                )
            elif root_cause == self._LABEL_MULTI_HIGH_PROCESS:
                parts.append(
                    f"Multiple metrics deviated from baseline simultaneously, with "
                    f"'{proc_name}' among the top resource consumers — suggesting "
                    f"coordinated resource exhaustion rather than a single cause."
                )
            else:
                parts.append(f"'{proc_name}' is the top consumer of the affected resource.")

        parts.append(f"Deviation from learned baseline: {abs(primary_dev):.0f}%.")

        return " ".join(parts)

    def _historical_comparison_text(self, primary_key: str, primary_dev: float) -> str:
        metric_disp = self._display_metric_name(primary_key)
        direction = "above" if primary_dev >= 0 else "below"
        return f"{metric_disp} is currently {abs(primary_dev):.0f}% {direction} its learned baseline."

    def _build_recommendation(
        self,
        root_cause: str,
        responsible_process: Dict[str, Any],
        primary_key: str,
    ) -> str:
        proc_name = responsible_process.get("name", "the responsible process")

        recommendations = {
            self._LABEL_MEMORY_LEAK: (
                f"Restart '{proc_name}' and inspect its memory allocation for a leak."
            ),
            self._LABEL_CPU_SATURATION: (
                f"Reduce CPU-intensive workloads and review '{proc_name}' for "
                f"unnecessary compute usage."
            ),
            self._LABEL_RUNAWAY_PROCESS: (
                f"Terminate or restart '{proc_name}'; investigate for an infinite "
                f"loop or stuck operation."
            ),
            self._LABEL_DISK_BOTTLENECK: (
                "Check disk-intensive applications, clear temporary files or logs, "
                "and consider expanding storage."
            ),
            self._LABEL_NETWORK_TRAFFIC: (
                f"Investigate unusual network activity from '{proc_name}'; check for "
                f"unauthorized outbound connections or possible data exfiltration."
            ),
            self._LABEL_TOO_MANY_PROCESSES: (
                "Review background processes and close unnecessary ones to free up "
                "system resources."
            ),
            self._LABEL_MULTI_HIGH_PROCESS: (
                f"Investigate '{proc_name}' and other high-resource processes for "
                f"coordinated abnormal activity, including possible malware."
            ),
            self._LABEL_RESOURCE_SPIKE: (
                f"Monitor '{proc_name}' for recurring spikes; if they persist, "
                f"inspect the process for potential malware or misconfiguration."
            ),
        }

        return recommendations.get(
            root_cause,
            f"Investigate '{proc_name}' and monitor {self._display_metric_name(primary_key)} closely.",
        )

    @staticmethod
    def _compute_confidence(prediction: AnomalyPrediction, primary_dev: float) -> float:
        """Blend the anomaly model's confidence with deviation strength."""
        base = prediction.confidence
        dev_component = min(40.0, abs(primary_dev) * 0.8)
        blended = (base * 0.6) + (dev_component + 50.0) * 0.4
        return float(np.clip(blended, 0.0, 99.9))