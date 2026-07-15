"""
backend/monitoring/processes.py
================================
Per-process monitoring for the Lavender Trinetra platform.

Responsible ONLY for identifying and describing the top resource-consuming
processes on each monitoring cycle. Triggered by ``collector.py`` once per
cycle; contains no scheduling, threading, or orchestration logic of its
own — it is a pure "collect on demand" module.

Dependencies (one-directional):
    processes.py --> config.py   (TOP_PROCESS_COUNT)
    processes.py --> metrics.py  (persistence of process snapshots)

``config.py`` is expected to expose ``TOP_PROCESS_COUNT`` (defaults to 5
here if not present, so this module still works standalone).

NON-BLOCKING CPU% NOTE
-----------------------
``psutil.Process.cpu_percent()`` returns a meaningful value only when
called at least twice on the *same* ``Process`` object, with real time
elapsed between calls — calling it with ``interval=<seconds>`` blocks the
thread for that long, which would stall the whole monitoring loop.

To avoid blocking, this module keeps a persistent cache of ``Process``
objects across cycles (keyed by PID). A newly seen PID gets a "priming"
call to ``cpu_percent(None)`` (returns 0.0, establishes the internal
baseline) and its real value is reported starting the *next* cycle, once
enough wall-clock time has passed since that baseline call.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import psutil
from loguru import logger

try:
    from backend.config import TOP_PROCESS_COUNT
except ImportError:
    # Allows this module to be imported/tested standalone before
    # config.py defines TOP_PROCESS_COUNT.
    TOP_PROCESS_COUNT = 5

from backend.monitoring import metrics as metrics_store


# ---------------------------------------------------------------------------
# Persistent process cache (survives across cycles; required for
# non-blocking cpu_percent() readings — see module docstring)
# ---------------------------------------------------------------------------
_process_cache: Dict[int, psutil.Process] = {}


@dataclass
class ProcessSnapshot:
    """A single process's resource usage at one point in time.

    Attributes:
        timestamp: ISO-8601 timestamp the snapshot was taken.
        pid: Process ID.
        name: Process executable/name.
        cpu_percent: Process CPU utilisation, in percent.
        memory_percent: Process memory utilisation, in percent.
        disk_read_mb: Cumulative bytes read by the process, in MB
            (``None`` if unavailable — I/O counters are often
            permission-restricted per-OS).
        disk_write_mb: Cumulative bytes written by the process, in MB
            (``None`` if unavailable).
        net_io: Placeholder for per-process network I/O. ``psutil`` does
            not expose per-process network usage on most platforms, so
            this is ``None`` unless a platform-specific extension is
            wired in later.
        status: Process status string (e.g. "running", "sleeping").
        username: Owning user of the process, where accessible.
        num_threads: Number of threads owned by the process.
        start_time: ISO-8601 timestamp the process was created.
    """

    timestamp: str
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    disk_read_mb: Optional[float]
    disk_write_mb: Optional[float]
    net_io: Optional[Any]
    status: str
    username: Optional[str]
    num_threads: Optional[int]
    start_time: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cache maintenance
# ---------------------------------------------------------------------------
def _sync_process_cache() -> None:
    """Add newly seen PIDs to the cache and evict PIDs that have exited.

    Newly added processes get a "priming" ``cpu_percent(None)`` call so
    their *next* reading is meaningful (see module docstring). This never
    raises: individual process errors are caught and logged.
    """
    try:
        live_pids = set(psutil.pids())
    except Exception:
        logger.exception("Failed to enumerate live PIDs.")
        return

    # Evict processes that no longer exist.
    for pid in list(_process_cache.keys()):
        if pid not in live_pids:
            _process_cache.pop(pid, None)

    # Add newly seen processes.
    for pid in live_pids:
        if pid in _process_cache:
            continue
        try:
            proc = psutil.Process(pid)
            proc.cpu_percent(None)  # priming call; establishes baseline
            _process_cache[pid] = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process exited between pids() and Process() construction,
            # or we're not permitted to inspect it. Skip silently.
            continue
        except Exception:
            logger.debug("Unexpected error caching PID {}.", pid, exc_info=True)
            continue


def _describe_process(proc: psutil.Process, timestamp: str) -> Optional[ProcessSnapshot]:
    """Build a :class:`ProcessSnapshot` for one cached ``Process`` object.

    Args:
        proc: A cached ``psutil.Process`` instance.
        timestamp: ISO-8601 timestamp to stamp this snapshot with.

    Returns:
        A populated :class:`ProcessSnapshot`, or ``None`` if the process
        vanished or became inaccessible while reading its attributes.
    """
    try:
        with proc.oneshot():
            name = proc.name()
            cpu_percent = proc.cpu_percent(None)
            memory_percent = proc.memory_percent()
            status = proc.status()
            num_threads = proc.num_threads()

            try:
                username = proc.username()
            except (psutil.AccessDenied, Exception):
                username = None

            try:
                create_ts = proc.create_time()
                start_time = datetime.fromtimestamp(create_ts).isoformat()
            except Exception:
                start_time = None

        disk_read_mb: Optional[float] = None
        disk_write_mb: Optional[float] = None
        try:
            io_counters = proc.io_counters()
            disk_read_mb = round(io_counters.read_bytes / (1024 ** 2), 4)
            disk_write_mb = round(io_counters.write_bytes / (1024 ** 2), 4)
        except (psutil.AccessDenied, AttributeError, NotImplementedError):
            # io_counters() is unavailable on some platforms (e.g. macOS
            # for non-root) and unsupported for some process types.
            pass

        return ProcessSnapshot(
            timestamp=timestamp,
            pid=proc.pid,
            name=name or "unknown",
            cpu_percent=float(cpu_percent or 0.0),
            memory_percent=round(float(memory_percent or 0.0), 4),
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            net_io=None,  # per-process network I/O not exposed by psutil
            status=status or "unknown",
            username=username,
            num_threads=num_threads,
            start_time=start_time,
        )

    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        # Process exited mid-read; drop it from the cache and skip.
        _process_cache.pop(proc.pid, None)
        return None
    except psutil.AccessDenied:
        logger.debug("Access denied reading process PID {}.", proc.pid)
        return None
    except Exception:
        logger.debug("Unexpected error describing PID {}.", proc.pid, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Public API — callable from collector.py (or directly, e.g. from routes.py)
# ---------------------------------------------------------------------------
def get_top_processes(limit: int = TOP_PROCESS_COUNT) -> List[Dict[str, Any]]:
    """Return the top ``limit`` processes by CPU usage, as plain dicts.

    Does not persist anything — pure read. Safe to call independently of
    ``collect_process_metrics()`` (e.g. from an API route for a live
    on-demand view) since it shares the same non-blocking process cache.

    Args:
        limit: Maximum number of processes to return, sorted by CPU
            usage descending. Defaults to ``TOP_PROCESS_COUNT``.

    Returns:
        A list of process snapshot dicts, length <= ``limit``. Returns an
        empty list if collection fails entirely.
    """
    try:
        _sync_process_cache()
        timestamp = datetime.now().isoformat()

        snapshots: List[ProcessSnapshot] = []
        for proc in list(_process_cache.values()):
            snapshot = _describe_process(proc, timestamp)
            if snapshot is not None:
                snapshots.append(snapshot)

        snapshots.sort(key=lambda s: s.cpu_percent, reverse=True)
        top = snapshots[:limit]

        logger.info("Identified top {} processes by CPU usage.", len(top))
        return [s.to_dict() for s in top]

    except Exception:
        logger.exception("Failed to collect top processes.")
        return []


def collect_process_metrics() -> List[Dict[str, Any]]:
    """Collect and persist the top resource-consuming processes.

    This is the entry point ``collector.py`` calls once per monitoring
    cycle: gathers the top processes via :func:`get_top_processes` and
    saves the result through ``metrics.py``.

    Returns:
        The list of top process snapshot dicts that were persisted
        (possibly empty on failure — never raises).
    """
    top_processes = get_top_processes(TOP_PROCESS_COUNT)

    if not top_processes:
        logger.warning("No process data collected this cycle; skipping persistence.")
        return top_processes

    try:
        metrics_store.save_processes(top_processes)
    except Exception:
        logger.exception("Failed to persist process snapshot via metrics.py.")

    return top_processes