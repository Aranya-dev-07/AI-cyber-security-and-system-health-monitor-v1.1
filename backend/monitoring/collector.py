"""
backend/monitoring/collector.py
================================
Central monitoring engine for the Lavender Trinetra platform.

Responsible ONLY for continuously collecting raw system metrics via
``psutil`` at a configurable interval, handing each snapshot off to the
metrics/processes/alerts modules, and maintaining an in-memory record of
the current monitoring session (start time, cycle count, last snapshot).

This module deliberately contains NO orchestration logic (no CLI, no
FastAPI startup hooks, no signal handling) — ``main.py`` owns the
application lifecycle and simply calls ``start_monitoring()`` /
``stop_monitoring()``. This keeps the collector reusable and testable in
isolation.

Dependencies (by design, one-directional):
    collector.py --> config.py   (thresholds, interval, paths)
    collector.py --> metrics.py  (persistence of system-level snapshots)
    collector.py --> processes.py (per-process collection, triggered per cycle)
    collector.py --> alerts.py   (threshold evaluation, triggered per cycle)

``config.py`` is expected to expose:
    MONITORING_INTERVAL, CPU_THRESHOLD, RAM_THRESHOLD, DISK_THRESHOLD,
    NETWORK_THRESHOLD  (see backend/.env / config.py)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger

from backend.config import (
    MONITORING_INTERVAL,
    CPU_THRESHOLD,
    RAM_THRESHOLD,
    DISK_THRESHOLD,
    NETWORK_THRESHOLD,
)
from backend.monitoring import metrics as metrics_store
from backend.monitoring import processes as process_monitor
from backend.monitoring import alerts as alert_engine


# ---------------------------------------------------------------------------
# In-memory session state for the current monitoring run
# ---------------------------------------------------------------------------
@dataclass
class MonitoringSession:
    """Bookkeeping for a single start->stop monitoring run.

    Attributes:
        active: Whether a monitoring run is currently in progress.
        started_at: ISO-8601 timestamp the run started, or ``None``.
        stopped_at: ISO-8601 timestamp the run stopped, or ``None``.
        cycle_count: Number of successful collection cycles this run.
        last_snapshot: The most recent metrics dict collected, or ``None``.
    """

    active: bool = False
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    cycle_count: int = 0
    last_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_session = MonitoringSession()
_session_lock = threading.Lock()

_stop_event = threading.Event()
_worker_thread: Optional[threading.Thread] = None

# Bookkeeping for computing network throughput as a delta between cycles
# (psutil.net_io_counters() returns cumulative bytes since boot).
_prev_net_sent: Optional[int] = None
_prev_net_recv: Optional[int] = None

# Bookkeeping for computing disk I/O throughput as a delta between cycles.
_prev_disk_read: Optional[int] = None
_prev_disk_write: Optional[int] = None


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------
def collect_system_metrics() -> Dict[str, Any]:
    """Collect a single structured snapshot of system-wide health metrics.

    Gathers CPU / RAM / disk usage, disk read/write throughput, network
    throughput, uptime, boot time, and CPU temperature (where the platform
    exposes it). All values are returned in a plain dict — this function
    never prints and never raises; collection failures are logged and
    degrade gracefully (missing fields are omitted or set to ``None``).

    Returns:
        A dictionary of collected metrics. Empty dict only in the
        catastrophic case where even the timestamp/CPU read fails.
    """
    global _prev_net_sent, _prev_net_recv, _prev_disk_read, _prev_disk_write

    snapshot: Dict[str, Any] = {}

    try:
        snapshot["timestamp"] = datetime.now().isoformat()
        snapshot["cpu_percent"] = psutil.cpu_percent(interval=1)
        snapshot["ram_percent"] = psutil.virtual_memory().percent
        snapshot["disk_percent"] = psutil.disk_usage("/").percent
    except Exception:
        logger.exception("Failed to collect core CPU/RAM/disk metrics.")
        return snapshot

    # --- Disk read/write throughput (delta since last cycle) -------------
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io is None:
            snapshot["disk_read_mb"] = 0.0
            snapshot["disk_write_mb"] = 0.0
        elif _prev_disk_read is None or _prev_disk_write is None:
            snapshot["disk_read_mb"] = 0.0
            snapshot["disk_write_mb"] = 0.0
        else:
            snapshot["disk_read_mb"] = round(
                max(0, disk_io.read_bytes - _prev_disk_read) / (1024 ** 2), 4
            )
            snapshot["disk_write_mb"] = round(
                max(0, disk_io.write_bytes - _prev_disk_write) / (1024 ** 2), 4
            )
        if disk_io is not None:
            _prev_disk_read = disk_io.read_bytes
            _prev_disk_write = disk_io.write_bytes
    except Exception:
        logger.warning("Disk I/O metrics unavailable on this platform.", exc_info=True)
        snapshot["disk_read_mb"] = None
        snapshot["disk_write_mb"] = None

    # --- Network throughput (delta since last cycle) ----------------------
    try:
        net_io = psutil.net_io_counters()
        if _prev_net_sent is None or _prev_net_recv is None:
            snapshot["net_sent_mb"] = 0.0
            snapshot["net_recv_mb"] = 0.0
        else:
            snapshot["net_sent_mb"] = round(
                max(0, net_io.bytes_sent - _prev_net_sent) / (1024 ** 2), 4
            )
            snapshot["net_recv_mb"] = round(
                max(0, net_io.bytes_recv - _prev_net_recv) / (1024 ** 2), 4
            )
        _prev_net_sent = net_io.bytes_sent
        _prev_net_recv = net_io.bytes_recv
    except Exception:
        logger.warning("Network I/O metrics unavailable.", exc_info=True)
        snapshot["net_sent_mb"] = None
        snapshot["net_recv_mb"] = None

    # --- Boot time / uptime -------------------------------------------
    try:
        boot_ts = psutil.boot_time()
        snapshot["boot_time"] = datetime.fromtimestamp(boot_ts).isoformat()
        snapshot["uptime_seconds"] = round(time.time() - boot_ts, 2)
    except Exception:
        logger.warning("Failed to read boot time / uptime.", exc_info=True)
        snapshot["boot_time"] = None
        snapshot["uptime_seconds"] = None

    # --- CPU temperature (not supported on all platforms) --------------
    snapshot["cpu_temp_celsius"] = _read_cpu_temperature()

    return snapshot


def _read_cpu_temperature() -> Optional[float]:
    """Best-effort CPU temperature reading.

    ``psutil.sensors_temperatures()`` is only implemented on Linux, and
    even there depends on hardware sensor availability. Returns ``None``
    silently (not an error) whenever temperature data isn't obtainable,
    since this is expected on most Windows/macOS/VM/container environments.

    Returns:
        The average of the first available sensor's readings in Celsius,
        or ``None`` if unsupported/unavailable.
    """
    try:
        sensors_fn = getattr(psutil, "sensors_temperatures", None)
        if sensors_fn is None:
            return None

        temps = sensors_fn()
        if not temps:
            return None

        for entries in temps.values():
            if entries:
                readings = [e.current for e in entries if e.current is not None]
                if readings:
                    return round(sum(readings) / len(readings), 2)
        return None
    except Exception:
        logger.debug("CPU temperature not available on this platform.", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Per-cycle pipeline: collect -> persist -> process scan -> alert evaluation
# ---------------------------------------------------------------------------
def _run_cycle() -> None:
    """Execute a single monitoring cycle end-to-end.

    Collects one system metrics snapshot, persists it via ``metrics.py``,
    triggers a process scan via ``processes.py``, and evaluates alert
    thresholds via ``alerts.py``. Every stage is individually guarded so
    that a failure in one (e.g. a DB write error) does not prevent the
    others from running or crash the monitoring thread.
    """
    snapshot = collect_system_metrics()
    if not snapshot:
        logger.error("Skipping cycle: system metrics collection returned nothing.")
        return

    with _session_lock:
        _session.cycle_count += 1
        _session.last_snapshot = snapshot

    try:
        metrics_store.save_metrics(snapshot)
    except Exception:
        logger.exception("Failed to persist metrics snapshot via metrics.py.")

    try:
        process_monitor.collect_process_metrics()
    except Exception:
        logger.exception("Failed to trigger process monitoring via processes.py.")

    try:
        alert_engine.evaluate_alerts(
            {
                "CPU": (snapshot.get("cpu_percent"), CPU_THRESHOLD),
                "RAM": (snapshot.get("ram_percent"), RAM_THRESHOLD),
                "DISK": (snapshot.get("disk_percent"), DISK_THRESHOLD),
                "NETWORK": (
                    (snapshot.get("net_sent_mb") or 0) + (snapshot.get("net_recv_mb") or 0),
                    NETWORK_THRESHOLD,
                ),
            }
        )
    except Exception:
        logger.exception("Failed to evaluate alerts via alerts.py.")

    logger.info(
        "Cycle #{} complete: CPU={}% RAM={}% DISK={}%",
        _session.cycle_count,
        snapshot.get("cpu_percent"),
        snapshot.get("ram_percent"),
        snapshot.get("disk_percent"),
    )


def _monitoring_loop(interval: float) -> None:
    """Background worker loop: run cycles until ``stop_monitoring()`` is called.

    Uses ``Event.wait(timeout)`` rather than ``time.sleep`` so that
    ``stop_monitoring()`` interrupts the wait immediately instead of
    blocking for up to a full interval.

    Args:
        interval: Seconds to wait between the end of one cycle and the
            start of the next.
    """
    logger.info("Monitoring loop started (interval={}s).", interval)
    while not _stop_event.is_set():
        try:
            _run_cycle()
        except Exception:
            # _run_cycle already guards its own stages, but this is a final
            # safety net so a truly unexpected error never kills the thread.
            logger.exception("Unhandled error in monitoring cycle; continuing.")
        _stop_event.wait(timeout=interval)
    logger.info("Monitoring loop exited cleanly.")


# ---------------------------------------------------------------------------
# Public control surface
# ---------------------------------------------------------------------------
def start_monitoring(interval: Optional[float] = None) -> bool:
    """Start the background monitoring thread.

    Idempotent: calling this while monitoring is already active is a no-op
    (logged as a warning) rather than spawning a second thread.

    Args:
        interval: Seconds between collection cycles. Defaults to
            ``MONITORING_INTERVAL`` from ``config.py`` if not provided.

    Returns:
        ``True`` if monitoring was started, ``False`` if it was already
        running.
    """
    global _worker_thread

    with _session_lock:
        if _session.active:
            logger.warning("start_monitoring() called but monitoring is already active.")
            return False

        _stop_event.clear()
        _session.active = True
        _session.started_at = datetime.now().isoformat()
        _session.stopped_at = None
        _session.cycle_count = 0
        _session.last_snapshot = None

    resolved_interval = float(interval) if interval is not None else float(MONITORING_INTERVAL)

    _worker_thread = threading.Thread(
        target=_monitoring_loop,
        args=(resolved_interval,),
        name="lavender-trinetra-collector",
        daemon=True,
    )
    _worker_thread.start()
    logger.info("Monitoring started at {} (interval={}s).", _session.started_at, resolved_interval)
    return True


def stop_monitoring(timeout: float = 10.0) -> bool:
    """Stop the background monitoring thread and finalize the session.

    Signals the loop to exit, joins the worker thread (bounded by
    ``timeout``), and marks the session inactive.

    Args:
        timeout: Maximum seconds to wait for the worker thread to exit
            cleanly before giving up on the join.

    Returns:
        ``True`` if monitoring was stopped, ``False`` if it wasn't running.
    """
    with _session_lock:
        if not _session.active:
            logger.warning("stop_monitoring() called but monitoring is not active.")
            return False

    _stop_event.set()
    if _worker_thread is not None:
        _worker_thread.join(timeout=timeout)
        if _worker_thread.is_alive():
            logger.warning("Monitoring thread did not exit within {}s timeout.", timeout)

    with _session_lock:
        _session.active = False
        _session.stopped_at = datetime.now().isoformat()

    logger.info("Monitoring stopped at {}.", _session.stopped_at)
    return True


def is_monitoring_active() -> bool:
    """Return whether a monitoring run is currently active."""
    with _session_lock:
        return _session.active


def get_session_snapshot() -> Dict[str, Any]:
    """Return a copy of the current session's bookkeeping state.

    Safe to call from any thread (e.g. an API route) while monitoring is
    running; returns a snapshot dict, not a live reference.
    """
    with _session_lock:
        return _session.to_dict()