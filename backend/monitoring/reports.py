"""
backend/monitoring/reports.py
==============================
Post-session report generation for the Lavender Trinetra platform.

Responsible ONLY for summarizing a completed monitoring session into a
single structured report and appending it to ``system_report.csv``.
Triggered by ``main.py`` once a monitoring run has been stopped — this
module never starts/stops monitoring itself and refuses to run while a
session is still active (see :func:`generate_report`).

Dependencies (one-directional):
    reports.py --> config.py    (REPORT_STORAGE_PATH)
    reports.py --> collector.py (session start/stop times, active flag)
    reports.py --> alerts.py    (alert counts/severity breakdown)
    reports.py --> metrics.py   (raw metrics/process history for the session)

ASSUMED metrics.py INTERFACE
------------------------------
This module expects ``metrics.py`` to expose:
    * ``get_session_metrics() -> List[Dict[str, Any]]``
        Every system-metrics snapshot collected this session (each with
        ``cpu_percent``, ``ram_percent``, ``disk_percent``,
        ``net_sent_mb``, ``net_recv_mb``).
    * ``get_session_processes() -> List[List[Dict[str, Any]]]``
        Every per-cycle top-processes list collected this session.
If ``metrics.py`` ends up shaped differently, only the two read calls in
:func:`_load_session_data` need to change — everything downstream works
off plain dicts/lists.

AI INTEGRATION HOOK
--------------------
``register_report_hook()`` lets another module (most likely a future
``ai_engine`` report generator) subscribe to receive the structured report
dict right after it's built — e.g. to produce a narrative summary, feed a
trend model, or flag the run as anomalous. A hook that raises is caught
and logged; it never breaks report generation.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    from backend.config import REPORT_STORAGE_PATH
except ImportError:
    REPORT_STORAGE_PATH = "."

try:
    from backend.config import TOP_PROCESS_COUNT
except ImportError:
    TOP_PROCESS_COUNT = 5

from backend.monitoring import collector as collector_module
from backend.monitoring import alerts as alert_engine
from backend.monitoring import metrics as metrics_store


REPORT_CSV_FILENAME = "system_report.csv"
REPORT_CSV_PATH = os.path.join(REPORT_STORAGE_PATH, REPORT_CSV_FILENAME)

REPORT_CSV_FIELDS: List[str] = [
    "generated_at",
    "start_time",
    "end_time",
    "duration_seconds",
    "avg_cpu",
    "peak_cpu",
    "avg_ram",
    "peak_ram",
    "avg_disk",
    "avg_network_mb",
    "total_alerts",
    "warning_alerts",
    "critical_alerts",
    "top_processes",  # JSON-encoded list, since CSV rows are flat
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class SessionReport:
    """Structured summary of one completed monitoring session.

    Attributes:
        generated_at: ISO-8601 timestamp the report itself was built.
        start_time: ISO-8601 timestamp the monitoring session started.
        end_time: ISO-8601 timestamp the monitoring session stopped.
        duration_seconds: Total session duration, in seconds.
        avg_cpu: Average CPU utilisation across the session, in percent.
        peak_cpu: Peak CPU utilisation observed, in percent.
        avg_ram: Average RAM utilisation across the session, in percent.
        peak_ram: Peak RAM utilisation observed, in percent.
        avg_disk: Average disk utilisation across the session, in percent.
        avg_network_mb: Average network throughput per cycle (sent +
            received), in MB.
        total_alerts: Total number of alerts raised during the session.
        warning_alerts: Count of WARNING-severity alerts.
        critical_alerts: Count of CRITICAL-severity alerts.
        top_processes: Top resource-consuming processes across the
            session, ranked by peak CPU usage.
    """

    generated_at: str
    start_time: Optional[str]
    end_time: Optional[str]
    duration_seconds: float
    avg_cpu: float
    peak_cpu: float
    avg_ram: float
    peak_ram: float
    avg_disk: float
    avg_network_mb: float
    total_alerts: int
    warning_alerts: int
    critical_alerts: int
    top_processes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_state_lock = threading.Lock()
_last_report: Optional[Dict[str, Any]] = None
_report_hooks: List[Callable[[Dict[str, Any]], None]] = []


def register_report_hook(hook: Callable[[Dict[str, Any]], None]) -> None:
    """Register a callback invoked with the report dict after generation.

    Intended for future AI-engine integration (narrative report
    generation, trend modeling, anomaly flagging). A hook that raises is
    caught and logged — it never breaks report generation for other hooks
    or the caller.

    Args:
        hook: A callable accepting a single report dict (see
            :meth:`SessionReport.to_dict`).
    """
    with _state_lock:
        _report_hooks.append(hook)
    logger.info("Registered report hook: {}", getattr(hook, "__name__", repr(hook)))


def get_last_report() -> Optional[Dict[str, Any]]:
    """Return the most recently generated report, or ``None`` if none yet."""
    with _state_lock:
        return _last_report


# ---------------------------------------------------------------------------
# Data loading / aggregation helpers
# ---------------------------------------------------------------------------
def _load_session_data() -> Dict[str, Any]:
    """Pull everything needed to build a report from the other modules.

    Isolated into its own function so a failure reading any one source
    (metrics history, alert history, session bookkeeping) is caught and
    logged individually rather than aborting the whole report.

    Returns:
        A dict with keys: ``session``, ``metrics``, ``processes``,
        ``alerts``. Missing/failed sources default to empty
        lists/dicts so downstream aggregation degrades gracefully.
    """
    data: Dict[str, Any] = {
        "session": {},
        "metrics": [],
        "processes": [],
        "alerts": [],
    }

    try:
        data["session"] = collector_module.get_session_snapshot()
    except Exception:
        logger.exception("Failed to read session bookkeeping from collector.py.")

    try:
        data["metrics"] = metrics_store.get_session_metrics()
    except Exception:
        logger.exception("Failed to read session metrics history from metrics.py.")

    try:
        data["processes"] = metrics_store.get_session_processes()
    except Exception:
        logger.exception("Failed to read session process history from metrics.py.")

    try:
        # Request the full session's worth of alerts; get_recent_alerts()
        # is bounded by alerts.py's internal history cap (500), so very
        # long/noisy sessions may lose granular severity breakdown beyond
        # that cap even though get_alert_count() stays accurate.
        data["alerts"] = alert_engine.get_recent_alerts(limit=10_000)
    except Exception:
        logger.exception("Failed to read alert history from alerts.py.")

    return data


def _safe_avg(values: List[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(sum(clean) / len(clean), 2) if clean else 0.0


def _safe_peak(values: List[float]) -> float:
    clean = [v for v in values if isinstance(v, (int, float))]
    return round(max(clean), 2) if clean else 0.0


def _aggregate_top_processes(
    process_cycles: List[List[Dict[str, Any]]], limit: int = TOP_PROCESS_COUNT
) -> List[Dict[str, Any]]:
    """Collapse every cycle's top-processes list into one session-wide ranking.

    Groups all observed process snapshots by PID, keeps each process's
    peak CPU/memory usage across the session, and returns the top
    ``limit`` processes ranked by peak CPU.

    Args:
        process_cycles: A list of per-cycle process snapshot lists, as
            produced by ``processes.get_top_processes()`` each cycle.
        limit: Maximum number of processes to include in the result.

    Returns:
        A list of dicts: ``pid``, ``name``, ``peak_cpu_percent``,
        ``peak_memory_percent``. Empty list if no process data available.
    """
    try:
        by_pid: Dict[int, Dict[str, Any]] = {}

        for cycle in process_cycles or []:
            for proc in cycle or []:
                pid = proc.get("pid")
                if pid is None:
                    continue
                cpu = float(proc.get("cpu_percent") or 0.0)
                mem = float(proc.get("memory_percent") or 0.0)

                existing = by_pid.get(pid)
                if existing is None:
                    by_pid[pid] = {
                        "pid": pid,
                        "name": proc.get("name", "unknown"),
                        "peak_cpu_percent": cpu,
                        "peak_memory_percent": mem,
                    }
                else:
                    existing["peak_cpu_percent"] = max(existing["peak_cpu_percent"], cpu)
                    existing["peak_memory_percent"] = max(existing["peak_memory_percent"], mem)

        ranked = sorted(by_pid.values(), key=lambda p: p["peak_cpu_percent"], reverse=True)
        return ranked[:limit]

    except Exception:
        logger.exception("Failed to aggregate top processes for report.")
        return []


def _compute_duration_seconds(start_time: Optional[str], end_time: Optional[str]) -> float:
    """Compute session duration in seconds from ISO-8601 timestamp strings."""
    if not start_time or not end_time:
        return 0.0
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        return round((end_dt - start_dt).total_seconds(), 2)
    except Exception:
        logger.warning("Failed to compute session duration from timestamps.", exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# CSV persistence
# ---------------------------------------------------------------------------
def _save_report_to_csv(report: Dict[str, Any]) -> None:
    """Append one report row to ``system_report.csv``, writing a header if new.

    Never raises: IO errors are caught and logged so a failed write does
    not prevent the report dict from being returned to the caller.
    """
    try:
        os.makedirs(os.path.dirname(REPORT_CSV_PATH) or ".", exist_ok=True)
        file_exists = os.path.isfile(REPORT_CSV_PATH)

        row = dict(report)
        row["top_processes"] = json.dumps(row.get("top_processes", []))

        with open(REPORT_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in REPORT_CSV_FIELDS})

        logger.info("Report saved to {}.", REPORT_CSV_PATH)

    except Exception:
        logger.exception("Failed to save report to {}.", REPORT_CSV_PATH)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_report() -> Dict[str, Any]:
    """Generate and persist a summary report for the just-completed session.

    Refuses to run while monitoring is still active — call this only
    after ``collector.stop_monitoring()`` has completed. This is enforced
    defensively (logged + empty dict returned) rather than raising, to
    keep this module's error-handling style consistent with the rest of
    the monitoring package.

    Returns:
        The generated report as a structured dict (see
        :meth:`SessionReport.to_dict`), or an empty dict if generation
        was refused or failed entirely.
    """
    global _last_report

    try:
        if collector_module.is_monitoring_active():
            logger.error(
                "generate_report() called while monitoring is still active. "
                "Call collector.stop_monitoring() first."
            )
            return {}
    except Exception:
        logger.exception("Failed to check monitoring status before generating report.")
        return {}

    try:
        data = _load_session_data()
        session = data["session"]
        metrics_history = data["metrics"]
        process_history = data["processes"]
        alert_history = data["alerts"]

        cpu_values = [m.get("cpu_percent") for m in metrics_history]
        ram_values = [m.get("ram_percent") for m in metrics_history]
        disk_values = [m.get("disk_percent") for m in metrics_history]
        network_values = [
            (m.get("net_sent_mb") or 0) + (m.get("net_recv_mb") or 0) for m in metrics_history
        ]

        warning_count = sum(1 for a in alert_history if a.get("severity") == alert_engine.SEVERITY_WARNING)
        critical_count = sum(1 for a in alert_history if a.get("severity") == alert_engine.SEVERITY_CRITICAL)

        start_time = session.get("started_at")
        end_time = session.get("stopped_at")

        report = SessionReport(
            generated_at=datetime.now().isoformat(),
            start_time=start_time,
            end_time=end_time,
            duration_seconds=_compute_duration_seconds(start_time, end_time),
            avg_cpu=_safe_avg(cpu_values),
            peak_cpu=_safe_peak(cpu_values),
            avg_ram=_safe_avg(ram_values),
            peak_ram=_safe_peak(ram_values),
            avg_disk=_safe_avg(disk_values),
            avg_network_mb=_safe_avg(network_values),
            total_alerts=alert_engine.get_alert_count(),
            warning_alerts=warning_count,
            critical_alerts=critical_count,
            top_processes=_aggregate_top_processes(process_history),
        )
        report_dict = report.to_dict()

        _save_report_to_csv(report_dict)

        with _state_lock:
            _last_report = report_dict

        for hook in list(_report_hooks):
            try:
                hook(report_dict)
            except Exception:
                logger.exception("Report hook {} raised an exception.", getattr(hook, "__name__", hook))

        logger.info(
            "Report generated: duration={}s avg_cpu={}% avg_ram={}% total_alerts={}",
            report_dict["duration_seconds"], report_dict["avg_cpu"],
            report_dict["avg_ram"], report_dict["total_alerts"],
        )
        return report_dict

    except Exception:
        logger.exception("Failed to generate session report.")
        return {}