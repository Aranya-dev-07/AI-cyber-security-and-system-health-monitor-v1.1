"""
backend/monitoring/alerts.py
=============================
Alert engine for the Lavender Trinetra platform.

Responsible ONLY for evaluating already-collected metrics against
configured thresholds, classifying severity, building structured alert
objects, printing them to the terminal, and keeping an in-memory count/
history for the current session. Triggered by ``collector.py`` once per
cycle; contains no scheduling or orchestration logic of its own.

Dependencies (one-directional):
    alerts.py --> config.py  (thresholds)

Explicitly OUT of scope for this module (by requirement):
    * Persisting alerts to the database — that's database.py's job, not
      this module's. Callers (e.g. api.py) are responsible for storing
      the dicts this module returns, if desired.

AI INTEGRATION HOOK
--------------------
``register_alert_hook()`` lets any other module (most likely the future
``ai_engine``) subscribe to be called synchronously with every alert as
it's raised — e.g. to feed anomaly correlation, root-cause analysis, or
predictive models without this module needing to know anything about AI.
A hook that raises is caught and logged; it never breaks alert evaluation.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from rich.console import Console

try:
    from backend.config import (
        CPU_THRESHOLD,
        RAM_THRESHOLD,
        DISK_THRESHOLD,
        NETWORK_THRESHOLD,
    )
except ImportError:
    # Fallback defaults so this module is importable/testable before
    # config.py is fully wired up. Real values should come from config.py.
    CPU_THRESHOLD = 85.0
    RAM_THRESHOLD = 85.0
    DISK_THRESHOLD = 90.0
    NETWORK_THRESHOLD = 100.0

try:
    # Per-process thresholds — not part of the original .env/config.py
    # spec, so these fall back to sensible defaults if not yet defined.
    # Recommend adding PROCESS_CPU_THRESHOLD / PROCESS_MEMORY_THRESHOLD
    # to config.py and .env for consistency with the system-level ones.
    from backend.config import PROCESS_CPU_THRESHOLD
except ImportError:
    PROCESS_CPU_THRESHOLD = 50.0

try:
    from backend.config import PROCESS_MEMORY_THRESHOLD
except ImportError:
    PROCESS_MEMORY_THRESHOLD = 30.0

try:
    # How far above the warning threshold a value must be to escalate to
    # CRITICAL, expressed as a multiplier (e.g. 1.15 = 15% over the
    # warning threshold). Recommend adding this to config.py too.
    from backend.config import CRITICAL_SEVERITY_MULTIPLIER
except ImportError:
    CRITICAL_SEVERITY_MULTIPLIER = 1.15


console = Console()

# Severity labels (string constants rather than an Enum so alert dicts
# serialize to JSON/CSV/DB without any custom encoding).
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

_SEVERITY_STYLES = {
    SEVERITY_WARNING: "bold yellow",
    SEVERITY_CRITICAL: "bold red",
}

_MAX_HISTORY = 500  # bounded in-memory history; not a persistence layer


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """A single structured alert event.

    Attributes:
        timestamp: ISO-8601 timestamp the alert was raised.
        severity: ``"WARNING"`` or ``"CRITICAL"``.
        metric: Name of the metric that triggered the alert (e.g. "CPU",
            "PROCESS_CPU").
        current_value: The measured value that triggered the alert.
        threshold: The threshold that was exceeded.
        process: Dict with ``pid``/``name`` if this alert is attributable
            to a specific process, otherwise ``None`` for system-wide
            alerts.
        message: Human-readable explanation of the alert.
    """

    timestamp: str
    severity: str
    metric: str
    current_value: float
    threshold: float
    process: Optional[Dict[str, Any]]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_alert_count: int = 0
_alert_history: List[Dict[str, Any]] = []
_alert_hooks: List[Callable[[Dict[str, Any]], None]] = []


def register_alert_hook(hook: Callable[[Dict[str, Any]], None]) -> None:
    """Register a callback to be invoked with every alert as it's raised.

    Intended for future AI-engine integration (anomaly correlation,
    root-cause analysis, predictive modeling) without alerts.py needing
    any knowledge of those modules. A hook that raises an exception is
    caught and logged — it never breaks alert evaluation for other hooks
    or the caller.

    Args:
        hook: A callable accepting a single alert dict (see
            :meth:`Alert.to_dict`).
    """
    with _state_lock:
        _alert_hooks.append(hook)
    logger.info("Registered alert hook: {}", getattr(hook, "__name__", repr(hook)))


def reset_alert_session() -> None:
    """Reset the alert counter and history for a new monitoring session.

    Intended to be called by ``collector.py`` (or ``main.py``) at the
    start of each ``start_monitoring()`` run, so alert counts reflect the
    current session rather than accumulating across runs.
    """
    with _state_lock:
        global _alert_count
        _alert_count = 0
        _alert_history.clear()
    logger.info("Alert session reset.")


def get_alert_count() -> int:
    """Return the total number of alerts raised in the current session."""
    with _state_lock:
        return _alert_count


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Return the most recent alerts from the current session's history.

    Args:
        limit: Maximum number of alerts to return, most recent first.

    Returns:
        A list of alert dicts, newest first, length <= ``limit``.
    """
    with _state_lock:
        return list(reversed(_alert_history[-limit:]))


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------
def _classify_severity(value: float, threshold: float) -> Optional[str]:
    """Classify a value against a threshold as WARNING, CRITICAL, or None.

    Args:
        value: The current measured value.
        threshold: The configured warning threshold for this metric.

    Returns:
        ``"CRITICAL"`` if the value exceeds
        ``threshold * CRITICAL_SEVERITY_MULTIPLIER``, ``"WARNING"`` if it
        merely exceeds ``threshold``, otherwise ``None`` (no alert).
    """
    if value is None:
        return None
    if value > threshold * CRITICAL_SEVERITY_MULTIPLIER:
        return SEVERITY_CRITICAL
    if value > threshold:
        return SEVERITY_WARNING
    return None


_HUMAN_EXPLANATIONS = {
    "CPU": "System-wide CPU usage is at {value:.1f}%, exceeding the {threshold:.1f}% threshold.",
    "RAM": "System-wide RAM usage is at {value:.1f}%, exceeding the {threshold:.1f}% threshold.",
    "DISK": "Disk usage is at {value:.1f}%, exceeding the {threshold:.1f}% threshold.",
    "NETWORK": "Network throughput is at {value:.1f}MB this interval, exceeding the {threshold:.1f}MB threshold.",
    "PROCESS_CPU": "Process '{process_name}' (PID {pid}) is using {value:.1f}% CPU, exceeding the {threshold:.1f}% threshold.",
    "PROCESS_MEMORY": "Process '{process_name}' (PID {pid}) is using {value:.1f}% memory, exceeding the {threshold:.1f}% threshold.",
}


def _build_message(metric: str, value: float, threshold: float, process: Optional[Dict[str, Any]]) -> str:
    """Build a human-readable explanation string for an alert.

    Falls back to a generic template if ``metric`` isn't one of the known
    keys in ``_HUMAN_EXPLANATIONS`` (keeps this module resilient to new
    metric types being added later without needing a code change here).
    """
    template = _HUMAN_EXPLANATIONS.get(metric)
    if template is None:
        return f"{metric} is at {value:.2f}, exceeding threshold {threshold:.2f}."

    pid = process.get("pid") if process else None
    process_name = process.get("name") if process else None
    return template.format(value=value, threshold=threshold, pid=pid, process_name=process_name)


# ---------------------------------------------------------------------------
# Alert construction, display, and dispatch
# ---------------------------------------------------------------------------
def _raise_alert(
    metric: str,
    value: float,
    threshold: float,
    severity: str,
    process: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build, count, display, and dispatch a single alert.

    Args:
        metric: Metric name (e.g. "CPU", "PROCESS_CPU").
        value: The measured value that triggered the alert.
        threshold: The threshold that was exceeded.
        severity: ``SEVERITY_WARNING`` or ``SEVERITY_CRITICAL``.
        process: Optional dict with ``pid``/``name`` for process-level
            alerts.

    Returns:
        The alert as a plain dict (see :meth:`Alert.to_dict`).
    """
    global _alert_count

    alert = Alert(
        timestamp=datetime.now().isoformat(),
        severity=severity,
        metric=metric,
        current_value=round(float(value), 2),
        threshold=round(float(threshold), 2),
        process=process,
        message=_build_message(metric, value, threshold, process),
    )
    alert_dict = alert.to_dict()

    with _state_lock:
        _alert_count += 1
        _alert_history.append(alert_dict)
        if len(_alert_history) > _MAX_HISTORY:
            _alert_history.pop(0)

    _display_alert(alert_dict)
    logger.warning(
        "ALERT [{}] {}: value={} threshold={}",
        severity, metric, alert_dict["current_value"], alert_dict["threshold"],
    )

    for hook in list(_alert_hooks):
        try:
            hook(alert_dict)
        except Exception:
            logger.exception("Alert hook {} raised an exception.", getattr(hook, "__name__", hook))

    return alert_dict


def _display_alert(alert_dict: Dict[str, Any]) -> None:
    """Print a single alert to the terminal with severity-based styling."""
    style = _SEVERITY_STYLES.get(alert_dict["severity"], "white")
    console.print(
        f"[{style}][{alert_dict['severity']}] {alert_dict['metric']}[/{style}] "
        f"— {alert_dict['message']} "
        f"(value={alert_dict['current_value']}, threshold={alert_dict['threshold']})"
    )


# ---------------------------------------------------------------------------
# Public API — callable from collector.py
# ---------------------------------------------------------------------------
def evaluate_alerts(
    metrics: Dict[str, Any],
    processes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate one cycle's metrics/processes against configured thresholds.

    This is the entry point ``collector.py`` calls once per monitoring
    cycle. Never raises: a failure evaluating one metric or process does
    not prevent the others from being checked.

    Args:
        metrics: The system-wide metrics snapshot from
            ``collector.collect_system_metrics()`` — expected to contain
            ``cpu_percent``, ``ram_percent``, ``disk_percent``, and
            optionally ``net_sent_mb`` / ``net_recv_mb``.
        processes: Optional list of process snapshot dicts from
            ``processes.get_top_processes()`` — expected to contain
            ``pid``, ``name``, ``cpu_percent``, ``memory_percent`` per
            entry. Pass ``None`` (or omit) to skip process-level checks.

    Returns:
        A list of alert dicts raised this cycle (empty if nothing
        exceeded its threshold).
    """
    raised: List[Dict[str, Any]] = []

    system_checks = (
        ("CPU", metrics.get("cpu_percent"), CPU_THRESHOLD),
        ("RAM", metrics.get("ram_percent"), RAM_THRESHOLD),
        ("DISK", metrics.get("disk_percent"), DISK_THRESHOLD),
        (
            "NETWORK",
            (metrics.get("net_sent_mb") or 0) + (metrics.get("net_recv_mb") or 0)
            if metrics.get("net_sent_mb") is not None or metrics.get("net_recv_mb") is not None
            else None,
            NETWORK_THRESHOLD,
        ),
    )

    for metric_name, value, threshold in system_checks:
        try:
            severity = _classify_severity(value, threshold)
            if severity:
                raised.append(_raise_alert(metric_name, value, threshold, severity))
        except Exception:
            logger.exception("Failed to evaluate alert for metric {}.", metric_name)

    if processes:
        for proc in processes:
            proc_ref = {"pid": proc.get("pid"), "name": proc.get("name")}
            try:
                cpu_val = proc.get("cpu_percent")
                severity = _classify_severity(cpu_val, PROCESS_CPU_THRESHOLD)
                if severity:
                    raised.append(
                        _raise_alert("PROCESS_CPU", cpu_val, PROCESS_CPU_THRESHOLD, severity, proc_ref)
                    )
            except Exception:
                logger.exception("Failed to evaluate PROCESS_CPU alert for PID {}.", proc.get("pid"))

            try:
                mem_val = proc.get("memory_percent")
                severity = _classify_severity(mem_val, PROCESS_MEMORY_THRESHOLD)
                if severity:
                    raised.append(
                        _raise_alert("PROCESS_MEMORY", mem_val, PROCESS_MEMORY_THRESHOLD, severity, proc_ref)
                    )
            except Exception:
                logger.exception("Failed to evaluate PROCESS_MEMORY alert for PID {}.", proc.get("pid"))

    return raised