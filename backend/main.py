from __future__ import annotations

import asyncio
import signal
import sys
import threading
from datetime import datetime
from typing import Optional

import uvicorn

from backend.config import settings
from backend.core import (
    get_logger,
    test_run_manager,
    application_status,
    StatusValue,
    startup_initialize,
    register_cleanup,
    safe_shutdown,
    safe_execute,
)
from backend.api.api import app as fastapi_app

from backend.monitoring import collector as monitoring_collector
from backend.monitoring import processes as monitoring_processes
from backend.monitoring import metrics as monitoring_metrics
from backend.monitoring import alerts as monitoring_alerts
from backend.monitoring import reports as monitoring_reports

from backend.ai import ai_engine as ai_engine_module

from backend.database import database as db_module
from backend.database import crud as db_crud

logger = get_logger("lavender_trinetra.main")

BANNER = """=========================================
\u222b Lavender Trinetra
Observe. Learn. Protect.
=========================================
Type "start" to begin monitoring.
Type "stop" to stop monitoring."""

METRICS_HEADERS = [
    "timestamp", "cpu_percent", "memory_percent",
    "disk_percent", "network_sent_mb", "network_received_mb",
]
PROCESSES_HEADERS = ["timestamp", "pid", "name", "cpu_percent", "memory_percent"]
REPORT_HEADERS = ["timestamp", "run_id", "summary"]

COLLECTION_INTERVAL_SECONDS = settings.COLLECTION_INTERVAL_SECONDS
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT


class CybersecurityEngineUnavailable(Exception):
    pass


def _load_cybersecurity_engine():
    """
    Loads the cybersecurity coordination entrypoint. Imported lazily and
    isolated behind a try/except so the orchestrator can still run
    monitoring, AI and the API even if the cybersecurity module is not
    yet present or fails to import.
    """
    try:
        from backend.cybersecurity import threat_detector

        return threat_detector
    except Exception as exc:
        logger.warning("Cybersecurity module unavailable: %s", exc)
        raise CybersecurityEngineUnavailable(str(exc))


class Orchestrator:
    """
    Sole backend orchestrator. Coordinates monitoring, AI, cybersecurity,
    database and API modules without implementing their internal logic.
    All shared utilities (logging, status, CSV, cleanup) are delegated to
    core.py; all configuration is sourced from config.py.
    """

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        self._api_server: Optional[uvicorn.Server] = None

        self._run_id: Optional[int] = None
        self._alert_tracker = monitoring_alerts.get_session_tracker()
        self._ai_engine: Optional[ai_engine_module.AIEngine] = None
        self._cyber_module = None

    # ------------------------------------------------------------------
    # API server lifecycle
    # ------------------------------------------------------------------
    def start_api_server(self) -> None:
        config = uvicorn.Config(
            fastapi_app,
            host=API_HOST,
            port=API_PORT,
            log_level=settings.LOG_LEVEL.lower(),
            loop="asyncio",
        )
        self._api_server = uvicorn.Server(config)

        def _run_server() -> None:
            with safe_execute("api-server"):
                self._api_server.run()

        self._api_thread = threading.Thread(target=_run_server, name="api-server", daemon=True)
        self._api_thread.start()
        application_status.set_api_status(StatusValue.OPERATIONAL)
        logger.info("API service started at http://%s:%s", API_HOST, API_PORT)
        register_cleanup(self.stop_api_server)

    def stop_api_server(self) -> None:
        if self._api_server is not None:
            self._api_server.should_exit = True
        if self._api_thread is not None:
            self._api_thread.join(timeout=5)
        application_status.set_api_status(StatusValue.STOPPED)
        logger.info("API service stopped.")

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------
    def init_database(self) -> None:
        with safe_execute("database-init", reraise=False):
            db_module.init_db()
            run = db_crud.create_test_run()
            self._run_id = run.get("id") if isinstance(run, dict) else None
            test_run_manager.start_run(run_id=self._run_id)
            application_status.set_database_status(StatusValue.OPERATIONAL)
            logger.info("Database initialized. Run ID: %s", self._run_id)
            register_cleanup(self.finalize_database)
            return
        application_status.set_database_status(StatusValue.UNAVAILABLE)
        self._run_id = None

    def finalize_database(self) -> None:
        with safe_execute("database-finalize"):
            if self._run_id is not None:
                context = test_run_manager.end_run()
                alert_count = context.alert_count if context else 0
                db_crud.end_test_run(self._run_id, alert_count=alert_count)
            logger.info("Database writes finalized.")

    # ------------------------------------------------------------------
    # AI / Cybersecurity initialization
    # ------------------------------------------------------------------
    def init_ai_engine(self) -> None:
        if not settings.AI_ENABLED:
            application_status.set_ai_status(StatusValue.STOPPED)
            logger.info("AI engine disabled via configuration.")
            return
        with safe_execute("ai-engine-init"):
            self._ai_engine = ai_engine_module.get_engine()
            application_status.set_ai_status(StatusValue.OPERATIONAL)
            logger.info("AI engine started.")
            return
        application_status.set_ai_status(StatusValue.UNAVAILABLE)
        self._ai_engine = None

    def init_cybersecurity_engine(self) -> None:
        try:
            self._cyber_module = _load_cybersecurity_engine()
            if hasattr(self._cyber_module, "start"):
                self._cyber_module.start()
            application_status.set_cybersecurity_status(StatusValue.OPERATIONAL)
            logger.info("Cybersecurity engine started.")
            register_cleanup(self.shutdown_cybersecurity_engine)
        except CybersecurityEngineUnavailable:
            self._cyber_module = None
            application_status.set_cybersecurity_status(StatusValue.UNAVAILABLE)
            logger.warning("Cybersecurity engine not started (module unavailable).")
        except Exception as exc:
            self._cyber_module = None
            application_status.set_cybersecurity_status(StatusValue.UNAVAILABLE)
            logger.error("Failed to start cybersecurity engine: %s", exc)

    def shutdown_cybersecurity_engine(self) -> None:
        if self._cyber_module is not None and hasattr(self._cyber_module, "stop"):
            with safe_execute("cybersecurity-shutdown"):
                self._cyber_module.stop()
        application_status.set_cybersecurity_status(StatusValue.STOPPED)

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------
    def _monitoring_cycle(self) -> None:
        timestamp = datetime.utcnow().isoformat()

        system_metrics = monitoring_collector.collect_system_metrics()
        top_processes = monitoring_processes.collect_top_processes()

        metrics_row = {
            "timestamp": timestamp,
            "cpu_percent": getattr(system_metrics, "cpu_percent", None),
            "memory_percent": getattr(system_metrics, "memory_percent", None),
            "disk_percent": getattr(system_metrics, "disk_percent", None),
            "network_sent_mb": getattr(system_metrics, "network_sent_mb", None),
            "network_received_mb": getattr(system_metrics, "network_received_mb", None),
        }
        # Delegates CSV persistence to monitoring/metrics.py, which uses
        # core.py's CSV helpers internally - no duplicated CSV logic here.
        monitoring_metrics.save_system_metrics(metrics_row)

        process_rows = [
            {
                "timestamp": timestamp,
                "pid": p.pid if hasattr(p, "pid") else p.get("pid"),
                "name": p.name if hasattr(p, "name") else p.get("name"),
                "cpu_percent": p.cpu_percent if hasattr(p, "cpu_percent") else p.get("cpu_percent"),
                "memory_percent": p.memory_percent if hasattr(p, "memory_percent") else p.get("memory_percent"),
            }
            for p in top_processes
        ]
        monitoring_metrics.save_process_metrics(process_rows, timestamp=timestamp)

        alerts = monitoring_alerts.generate_alerts(metrics_row, tracker=self._alert_tracker)
        if alerts:
            test_run_manager.record_alert(len(alerts) if hasattr(alerts, "__len__") else 1)

        with safe_execute("database-write-cycle"):
            with db_module.session_scope() as session:
                db_crud.insert_system_metrics(metrics_row, run_id=self._run_id, db=session)
                db_crud.insert_process_metrics(process_rows, run_id=self._run_id, db=session)

        if self._ai_engine is not None:
            with safe_execute("ai-engine-cycle"):
                ai_engine_module.run_ai_cycle(self._ai_engine, metrics_row, process_rows)

        if self._cyber_module is not None and hasattr(self._cyber_module, "run_cycle"):
            with safe_execute("cybersecurity-cycle"):
                self._cyber_module.run_cycle(metrics_row, process_rows)

    def _monitoring_loop(self) -> None:
        logger.info("Monitoring loop started.")
        application_status.set_monitoring_status(StatusValue.OPERATIONAL)
        while not self._stop_event.is_set():
            with safe_execute("monitoring-cycle"):
                self._monitoring_cycle()
            self._stop_event.wait(COLLECTION_INTERVAL_SECONDS)
        application_status.set_monitoring_status(StatusValue.STOPPED)
        logger.info("Monitoring loop terminated.")

    # ------------------------------------------------------------------
    # Public start / stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        logger.info("Starting Lavender Trinetra services...")

        startup_initialize(METRICS_HEADERS, PROCESSES_HEADERS, REPORT_HEADERS)
        self.init_database()
        self.init_ai_engine()
        self.init_cybersecurity_engine()
        self.start_api_server()

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="monitoring-loop", daemon=True
        )
        self._monitoring_thread.start()
        register_cleanup(self._stop_monitoring_thread)

        logger.info("All services started. Monitoring is now active.")

    def _stop_monitoring_thread(self) -> None:
        self._stop_event.set()
        if self._monitoring_thread is not None:
            self._monitoring_thread.join(timeout=10)

    def stop(self) -> None:
        logger.info("Stopping Lavender Trinetra services...")

        self._stop_monitoring_thread()

        with safe_execute("csv-report-finalize"):
            report = monitoring_reports.generate_report_on_stop(run_id=self._run_id)
            monitoring_metrics.save_system_report(report)

        if self._ai_engine is not None:
            with safe_execute("ai-report-finalize"):
                result = ai_engine_module.get_latest_result_dict(
                    getattr(self._ai_engine, "last_result", None)
                )
                if self._run_id is not None:
                    with db_module.session_scope() as session:
                        db_crud.insert_ai_result(result, run_id=self._run_id, db=session)

        # safe_shutdown() runs all registered cleanup callbacks (API,
        # cybersecurity, database) in reverse order and marks every
        # component's status as stopped.
        safe_shutdown()

        logger.info("All services stopped cleanly.")


async def command_loop(orchestrator: Orchestrator) -> None:
    monitoring_active = False
    loop = asyncio.get_event_loop()

    while True:
        command = (await loop.run_in_executor(None, input, "> ")).strip().lower()

        if command == "start":
            if monitoring_active:
                print("Monitoring is already active.")
                continue
            orchestrator.start()
            monitoring_active = True

        elif command == "stop":
            if not monitoring_active:
                print("Monitoring is not currently active.")
                continue
            orchestrator.stop()
            monitoring_active = False
            print("User has stopped data collection.")
            print("Exiting!!")
            print("Thank You for using The System Health Monitor \U0001F600")
            break

        else:
            print('Unrecognized command. Type "start" or "stop".')


def _install_signal_handlers(orchestrator: Orchestrator) -> None:
    def _handle_signal(signum, frame) -> None:  # noqa: ANN001
        logger.info("Received signal %s, shutting down.", signum)
        orchestrator.stop()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass


def main() -> None:
    print(BANNER)
    orchestrator = Orchestrator()
    _install_signal_handlers(orchestrator)

    try:
        asyncio.run(command_loop(orchestrator))
    except KeyboardInterrupt:
        orchestrator.stop()
        print("User has stopped data collection.")
        print("Exiting!!")
        print("Thank You for using The System Health Monitor \U0001F600")


if __name__ == "__main__":
    main()