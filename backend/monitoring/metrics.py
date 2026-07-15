"""
backend/monitoring/metrics.py
==============================
Monitoring data storage layer for the Lavender Trinetra platform.

Responsible ONLY for persisting and retrieving monitoring data — system
metrics, process metrics, and monitoring statistics. Contains NO
monitoring logic of its own (no psutil calls, no threshold checks, no
scheduling): it is a pure storage layer that ``collector.py``,
``processes.py``, and ``reports.py`` write to and read from.

Two responsibilities, kept deliberately separate:
    1. Immediate CSV persistence (system_metrics.csv / system_processes.csv)
       — durable, append-only, survives process restarts.
    2. In-memory session history (cleared per monitoring run via
       ``reset_session()``) — fast reads for reports.py and any live
       dashboard/API endpoint, without re-parsing CSV files.

Dependencies (one-directional):
    metrics.py --> config.py  (CSV_STORAGE_PATH / individual file paths)

No other project module is imported here, so metrics.py can be safely
imported by collector.py, processes.py, and reports.py without circular
imports.
"""

from __future__ import annotations

import csv
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from backend.config import CSV_STORAGE_PATH
except ImportError:
    CSV_STORAGE_PATH = "."

try:
    from backend.config import CSV_METRICS_PATH
except ImportError:
    CSV_METRICS_PATH = os.path.join(CSV_STORAGE_PATH, "system_metrics.csv")

try:
    from backend.config import CSV_PROCESSES_PATH
except ImportError:
    CSV_PROCESSES_PATH = os.path.join(CSV_STORAGE_PATH, "system_processes.csv")


# ---------------------------------------------------------------------------
# CSV column order (locked — changing this changes the on-disk schema;
# append new columns at the end rather than reordering existing ones)
# ---------------------------------------------------------------------------
METRICS_CSV_FIELDS: List[str] = [
    "timestamp",
    "cpu_percent",
    "ram_percent",
    "disk_percent",
    "disk_read_mb",
    "disk_write_mb",
    "net_sent_mb",
    "net_recv_mb",
    "boot_time",
    "uptime_seconds",
    "cpu_temp_celsius",
]

PROCESSES_CSV_FIELDS: List[str] = [
    "timestamp",
    "pid",
    "name",
    "cpu_percent",
    "memory_percent",
    "disk_read_mb",
    "disk_write_mb",
    "net_io",
    "status",
    "username",
    "num_threads",
    "start_time",
]


# ---------------------------------------------------------------------------
# Thread-safety primitives
# ---------------------------------------------------------------------------
# Separate locks for CSV I/O vs. in-memory state: a slow disk write should
# never block a concurrent read of the in-memory session history (e.g. an
# API route checking get_latest_metrics() while a write is in flight).
_csv_lock = threading.Lock()
_session_lock = threading.Lock()

# In-memory session history — cleared per monitoring run via reset_session().
_session_metrics: List[Dict[str, Any]] = []
_session_processes: List[List[Dict[str, Any]]] = []
_session_statistics: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _sanitize_for_csv(row: Dict[str, Any], fieldnames: List[str]) -> Dict[str, Any]:
    """Coerce a dict into flat, CSV-writable values for the given fieldnames.

    ``None`` becomes an empty string; any non-primitive value (list/dict)
    is JSON-encoded so it survives a round trip through a single CSV cell
    without corrupting the row structure.

    Args:
        row: The raw data to sanitize.
        fieldnames: The CSV column order to sanitize against; keys not in
            this list are dropped, missing keys become "".

    Returns:
        A dict containing exactly ``fieldnames`` as keys, CSV-safe values.
    """
    clean: Dict[str, Any] = {}
    for field_name in fieldnames:
        value = row.get(field_name)
        if value is None:
            clean[field_name] = ""
        elif isinstance(value, (list, dict)):
            clean[field_name] = json.dumps(value)
        else:
            clean[field_name] = value
    return clean


def _init_csv(path: str, fieldnames: List[str]) -> None:
    """Ensure a CSV file exists with a header row, creating parent dirs.

    Safe to call repeatedly — a no-op if the file already exists. Never
    raises: IO errors are caught and logged.

    Args:
        path: Target CSV file path.
        fieldnames: Header row / column order to write if the file is new.
    """
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        if not os.path.isfile(path):
            with _csv_lock:
                if not os.path.isfile(path):  # re-check inside the lock
                    with open(path, mode="w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                    logger.info("Initialized CSV file: {}", path)
    except Exception:
        logger.exception("Failed to initialize CSV file: {}", path)


def _write_row(path: str, fieldnames: List[str], row: Dict[str, Any]) -> None:
    """Append a single sanitized row to a CSV file, thread-safely.

    Assumes the file has already been initialized (header written) via
    :func:`_init_csv`; initializes it defensively if missing. Never
    raises: IO errors are caught and logged so a failed write never
    crashes the calling monitoring cycle.

    Args:
        path: Target CSV file path.
        fieldnames: Column order — must match the file's existing header.
        row: Data to write; sanitized against ``fieldnames`` before write.
    """
    try:
        _init_csv(path, fieldnames)
        clean_row = _sanitize_for_csv(row, fieldnames)
        with _csv_lock:
            with open(path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(clean_row)
    except Exception:
        logger.exception("Failed to write row to {}.", path)


def initialize_csv_files() -> None:
    """Ensure both monitoring CSV files exist with correct headers.

    Idempotent — safe to call once at application startup (e.g. from
    ``main.py``) even if the files already exist from a previous run.
    """
    _init_csv(CSV_METRICS_PATH, METRICS_CSV_FIELDS)
    _init_csv(CSV_PROCESSES_PATH, PROCESSES_CSV_FIELDS)


# ---------------------------------------------------------------------------
# Timestamp consistency helper
# ---------------------------------------------------------------------------
def _ensure_timestamp(row: Dict[str, Any]) -> str:
    """Return ``row``'s timestamp if present, otherwise stamp and return one.

    Mutates ``row`` in place when a timestamp needs to be generated, so
    every downstream consumer (CSV row, in-memory history) sees the exact
    same value for this record.
    """
    if not row.get("timestamp"):
        row["timestamp"] = datetime.now().isoformat()
    return row["timestamp"]


# ---------------------------------------------------------------------------
# Public API — save functions
# ---------------------------------------------------------------------------
def save_metrics(snapshot: Dict[str, Any]) -> None:
    """Persist a single system metrics snapshot.

    Writes immediately to ``system_metrics.csv`` and appends to the
    in-memory session history for the current monitoring run.

    Args:
        snapshot: A system metrics dict, typically the output of
            ``collector.collect_system_metrics()``. Must be JSON/CSV
            friendly (plain types); missing fields are written as "".
    """
    if not snapshot:
        logger.warning("save_metrics() called with empty snapshot; skipping.")
        return

    _ensure_timestamp(snapshot)

    with _session_lock:
        _session_metrics.append(dict(snapshot))

    _write_row(CSV_METRICS_PATH, METRICS_CSV_FIELDS, snapshot)
    logger.debug("Saved system metrics snapshot at {}.", snapshot.get("timestamp"))


def save_processes(processes: List[Dict[str, Any]]) -> None:
    """Persist one cycle's worth of top process snapshots.

    Writes each process as its own row to ``system_processes.csv`` and
    appends the whole cycle (as one list) to the in-memory session
    history. All processes in the same call share a single timestamp if
    any of them are missing one, so a cycle's rows stay correlated.

    Args:
        processes: A list of process snapshot dicts, typically the output
            of ``processes.get_top_processes()``.
    """
    if not processes:
        logger.warning("save_processes() called with empty process list; skipping.")
        return

    shared_timestamp = datetime.now().isoformat()
    for proc in processes:
        if not proc.get("timestamp"):
            proc["timestamp"] = shared_timestamp

    with _session_lock:
        _session_processes.append([dict(p) for p in processes])

    for proc in processes:
        _write_row(CSV_PROCESSES_PATH, PROCESSES_CSV_FIELDS, proc)

    logger.debug("Saved {} process snapshots at {}.", len(processes), shared_timestamp)


def save_monitoring_statistics(stats: Dict[str, Any]) -> None:
    """Record a monitoring-statistics entry for the current session.

    Intended for lightweight, computed/aggregate figures (e.g. a running
    "session so far" summary) that other modules want to snapshot
    periodically — distinct from raw per-cycle metrics/process rows.
    Kept in-memory only (no dedicated CSV): full-session summaries are
    ``reports.py``'s responsibility and already get their own
    ``system_report.csv``. This function exists so intermediate/partial
    statistics have somewhere reusable to live in the meantime (e.g. for
    a live dashboard).

    Args:
        stats: An arbitrary dict of computed statistics; a timestamp is
            added automatically if not present.
    """
    if not stats:
        logger.warning("save_monitoring_statistics() called with empty stats; skipping.")
        return

    _ensure_timestamp(stats)

    with _session_lock:
        _session_statistics.append(dict(stats))

    logger.debug("Saved monitoring statistics entry at {}.", stats.get("timestamp"))


# ---------------------------------------------------------------------------
# Public API — read functions
# ---------------------------------------------------------------------------
def get_latest_metrics() -> Optional[Dict[str, Any]]:
    """Return the most recently saved system metrics snapshot (in-memory).

    Returns:
        A copy of the latest snapshot dict, or ``None`` if nothing has
        been saved yet this session.
    """
    with _session_lock:
        return dict(_session_metrics[-1]) if _session_metrics else None


def get_latest_processes() -> List[Dict[str, Any]]:
    """Return the most recently saved process cycle (in-memory).

    Returns:
        A copy of the latest cycle's process list, or an empty list if
        nothing has been saved yet this session.
    """
    with _session_lock:
        return [dict(p) for p in _session_processes[-1]] if _session_processes else []


def get_session_metrics() -> List[Dict[str, Any]]:
    """Return every system metrics snapshot saved this session (in-memory).

    Used by ``reports.py`` to compute session-wide averages/peaks.

    Returns:
        A shallow copy of the full in-memory metrics history.
    """
    with _session_lock:
        return [dict(m) for m in _session_metrics]


def get_session_processes() -> List[List[Dict[str, Any]]]:
    """Return every process cycle saved this session (in-memory).

    Used by ``reports.py`` to compute session-wide top-process rankings.

    Returns:
        A shallow copy of the full in-memory process cycle history —
        a list of cycles, each a list of process dicts.
    """
    with _session_lock:
        return [[dict(p) for p in cycle] for cycle in _session_processes]


def get_session_statistics() -> List[Dict[str, Any]]:
    """Return every monitoring-statistics entry saved this session.

    Returns:
        A shallow copy of the full in-memory statistics history.
    """
    with _session_lock:
        return [dict(s) for s in _session_statistics]


def read_latest_metrics_from_csv(count: int = 10) -> List[Dict[str, Any]]:
    """Read the last ``count`` rows directly from ``system_metrics.csv``.

    Unlike :func:`get_session_metrics`, this reads from disk rather than
    in-memory state — useful for inspecting history from a *previous*
    (already-ended) session, e.g. on application startup before any new
    monitoring run has occurred.

    Args:
        count: Maximum number of most-recent rows to return.

    Returns:
        A list of row dicts (oldest to newest), length <= ``count``.
        Empty list if the file doesn't exist yet or reading fails.
    """
    try:
        if not os.path.isfile(CSV_METRICS_PATH):
            return []
        with _csv_lock, open(CSV_METRICS_PATH, mode="r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-count:] if count > 0 else rows
    except Exception:
        logger.exception("Failed to read latest metrics from {}.", CSV_METRICS_PATH)
        return []


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------
def reset_session() -> None:
    """Clear all in-memory session history for a new monitoring run.

    Does NOT touch the on-disk CSV files (those remain a durable,
    cross-session log) — only the in-memory history used by
    ``get_session_*()`` / ``reports.py`` is cleared. Intended to be called
    by ``collector.start_monitoring()`` at the start of each run.
    """
    with _session_lock:
        _session_metrics.clear()
        _session_processes.clear()
        _session_statistics.clear()
    logger.info("Monitoring data session history reset.")