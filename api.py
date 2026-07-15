"""
api.py
======
FastAPI REST interface for the System Health Monitor and Cybersecurity
Monitoring Platform.

Exposes monitoring data collected by ``config.py`` and persisted by
``database.py`` over HTTP, with interactive Swagger (``/docs``) and Redoc
(``/redoc``) documentation enabled by default.

ARCHITECTURE NOTE
------------------
``api.py`` imports from both ``config.py`` and ``database.py``:

    config.py  <-- database.py  <-- api.py  <-- main.py

DATA SOURCE NOTE (per project decision)
-----------------------------------------
Endpoints are intentionally split between two data sources:

    * ``/metrics`` and ``/processes`` read the shared **in-memory** state
      (``config.metrics_data`` / ``config.process_data``) for the lowest
      possible latency on "what is happening right now" queries.
    * ``/runs`` reads from **SQLite** via ``database.get_run_history()``,
      since run history is inherently a durable, historical record.
    * ``/summary`` has no dedicated database table (only a CSV report is
      ever produced), so it is computed live via
      ``config.generate_run_summary()`` against the current in-memory
      ``metrics_data`` / ``run_start_time`` / ``run_end_time`` /
      ``alert_count``.

This file never duplicates SQL or schema logic - all database access goes
through the functions already defined in ``database.py``.

NO MONITORING CONTROL ENDPOINTS
----------------------------------
This API is read-only by design (per the locked endpoint list:
``/health``, ``/metrics``, ``/processes``, ``/runs``, ``/summary``).
Starting and stopping the monitoring loop is controlled exclusively by
``main.py`` calling ``start_monitoring()`` / ``stop_monitoring()``
directly - there are no ``POST /monitoring/start`` or
``POST /monitoring/stop`` routes here. If remote start/stop control is
ever needed, two additional endpoints can be added following the same
pattern used below.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import database

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="System Health Monitor & Cybersecurity Monitoring Platform",
    description=(
        "REST API exposing live and historical CPU, RAM, disk, network, "
        "and process monitoring data, plus run history and run summaries."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Permissive CORS so any frontend (dev or otherwise) can consume this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic response models — Original
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Response model for the liveness check endpoint."""
    status: str = Field(..., description="Service health status.", examples=["healthy"])


class MetricResponse(BaseModel):
    """Response model for a single system metric snapshot."""
    timestamp: str = Field(..., description="ISO-8601 timestamp of the snapshot.")
    cpu_percent: float = Field(..., description="CPU utilisation, in percent.")
    ram_percent: float = Field(..., description="RAM utilisation, in percent.")
    disk_percent: float = Field(..., description="Disk utilisation, in percent.")
    net_sent_mb: float = Field(..., description="MB sent since the previous cycle.")
    net_recv_mb: float = Field(..., description="MB received since the previous cycle.")


class ProcessResponse(BaseModel):
    """Response model for a single process snapshot entry."""
    timestamp: str = Field(..., description="ISO-8601 timestamp of the snapshot.")
    pid: int = Field(..., description="Process ID.")
    name: str = Field(..., description="Process name.")
    cpu_percent: float = Field(..., description="Process CPU utilisation, in percent.")
    memory_percent: float = Field(..., description="Process memory utilisation, in percent.")
    status: str = Field(..., description="Process status (e.g. 'running', 'sleeping').")


class RunHistoryResponse(BaseModel):
    """Response model for a single historical monitoring run."""
    id: int = Field(..., description="Unique run identifier.")
    start_time: str = Field(..., description="ISO-8601 timestamp the run started.")
    end_time: Optional[str] = Field(None, description="ISO-8601 timestamp the run ended.")
    duration_seconds: Optional[float] = Field(None, description="Run duration, in seconds.")
    alert_count: int = Field(..., description="Total alerts raised during the run.")


class RunSummaryResponse(BaseModel):
    """Response model for the current/most recent run summary."""
    run_id: int = Field(..., description="Identifier of the run this summary covers.")
    start_time: str = Field(..., description="ISO-8601 timestamp the run started.")
    end_time: str = Field(..., description="ISO-8601 timestamp the run ended (or 'now').")
    duration_sec: float = Field(..., description="Run duration, in seconds.")
    avg_cpu: float = Field(..., description="Average CPU utilisation across the run.")
    avg_ram: float = Field(..., description="Average RAM utilisation across the run.")
    avg_disk: float = Field(..., description="Average disk utilisation across the run.")
    total_alerts: int = Field(..., description="Total alerts raised during the run.")


class ErrorResponse(BaseModel):
    """Standard error response shape for documented error cases."""
    detail: str = Field(..., description="Human-readable error message.")


# ---------------------------------------------------------------------------
# Startup: ensure the database schema exists before serving requests
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup() -> None:
    """Initialize the database schema when the API starts."""
    try:
        database.initialize_database()
        logger.info("API startup complete: database initialized.")
    except Exception:
        logger.exception("API startup failed during database initialization.")


# ---------------------------------------------------------------------------
# Endpoints — Original
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness check",
)
async def get_health() -> HealthResponse:
    """Return a simple liveness indicator for the API service."""
    return HealthResponse(status="healthy")


@app.get(
    "/metrics",
    response_model=MetricResponse,
    tags=["Metrics"],
    summary="Get the latest system metrics",
    responses={404: {"model": ErrorResponse, "description": "No metrics collected yet."}},
)
async def get_metrics() -> MetricResponse:
    """Return the most recently collected system metric snapshot."""
    try:
        with config.data_lock:
            if not config.metrics_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No metrics data available yet. Start monitoring first.",
                )
            latest = config.metrics_data[-1]
        return MetricResponse(**latest)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve latest metrics.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving metrics.",
        )


@app.get(
    "/processes",
    response_model=List[ProcessResponse],
    tags=["Processes"],
    summary="Get the latest top-process snapshot",
    responses={404: {"model": ErrorResponse, "description": "No process data collected yet."}},
)
async def get_processes() -> List[ProcessResponse]:
    """Return the most recently collected top-process snapshot."""
    try:
        with config.data_lock:
            if not config.process_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No process data available yet. Start monitoring first.",
                )
            latest_snapshot = config.process_data[-1]
        processes = latest_snapshot.get("processes", [])
        return [ProcessResponse(**proc) for proc in processes]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve latest process data.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving process data.",
        )


@app.get(
    "/metrics/history",
    response_model=List[MetricResponse],
    tags=["Metrics"],
    summary="Get full in-memory metrics history (for charts)",
    responses={404: {"model": ErrorResponse, "description": "No metrics collected yet."}},
)
async def get_metrics_history() -> List[MetricResponse]:
    """Return all metric snapshots collected since monitoring started."""
    try:
        with config.data_lock:
            if not config.metrics_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No metrics data available yet. Start monitoring first.",
                )
            snapshot = list(config.metrics_data)
        return [MetricResponse(**m) for m in snapshot]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve metrics history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving metrics history.",
        )


@app.get(
    "/runs",
    response_model=List[RunHistoryResponse],
    tags=["Runs"],
    summary="Get test run history",
)
async def get_runs(limit: int = 50) -> List[RunHistoryResponse]:
    """Return historical monitoring run records from the database."""
    try:
        runs = database.get_run_history(limit=limit)
        return [RunHistoryResponse(**run) for run in runs]
    except Exception:
        logger.exception("Failed to retrieve run history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving run history.",
        )


@app.get(
    "/summary",
    response_model=RunSummaryResponse,
    tags=["Summary"],
    summary="Get the current/most recent run summary",
    responses={404: {"model": ErrorResponse, "description": "No run data available yet."}},
)
async def get_summary() -> RunSummaryResponse:
    """Return aggregate statistics for the current (or most recent) run."""
    try:
        summary = config.generate_run_summary()
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No run summary available yet. Start monitoring first.",
            )
        return RunSummaryResponse(**summary)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to generate run summary.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the run summary.",
        )


# ---------------------------------------------------------------------------
# AI anomaly detection endpoints — Original + New Dashboard endpoints
# ---------------------------------------------------------------------------
import ai_engine as _ai_engine  # noqa: E402


class AnomalyResponse(BaseModel):
    """Response model for a single stored anomaly prediction."""
    id:            int   = Field(..., description="Unique anomaly record ID.")
    run_id:        int   = Field(..., description="Run this anomaly belongs to.")
    timestamp:     str   = Field(..., description="ISO-8601 timestamp.")
    anomaly_score: float = Field(..., description="Raw Isolation Forest score.")
    confidence:    float = Field(..., description="Confidence percentage (0-100).")
    severity:      str   = Field(..., description="NORMAL / LOW / MEDIUM / HIGH / CRITICAL.")
    reason:        str   = Field(..., description="Natural-language explanation.")
    is_anomaly:    int   = Field(..., description="1 if anomaly, 0 otherwise.")
    model_tier:    str   = Field(..., description="Model tier that produced this prediction.")


class AIStatisticsResponse(BaseModel):
    """Response model for aggregate AI anomaly statistics."""
    total_anomalies: int   = Field(..., description="Total anomalies detected.")
    critical_count:  int   = Field(..., description="CRITICAL severity count.")
    high_count:      int   = Field(..., description="HIGH severity count.")
    medium_count:    int   = Field(..., description="MEDIUM severity count.")
    low_count:       int   = Field(..., description="LOW severity count.")
    avg_confidence:  float = Field(..., description="Average detection confidence.")
    avg_score:       float = Field(..., description="Average anomaly score.")
    model_trained:   bool  = Field(..., description="Whether the AI model is currently trained.")
    total_predictions: int = Field(..., description="Total predictions made this session.")


@app.get(
    "/ai/latest",
    response_model=List[AnomalyResponse],
    tags=["AI Anomaly Detection"],
    summary="Get the most recent AI anomaly detections",
    responses={404: {"model": ErrorResponse, "description": "No anomalies recorded yet."}},
)
async def get_ai_latest(limit: int = 5) -> List[AnomalyResponse]:
    """Return the most recently detected anomalies from the database."""
    try:
        rows = database.get_latest_ai_prediction(limit=limit)
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No AI anomaly predictions recorded yet.",
            )
        return [AnomalyResponse(**row) for row in rows]
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve latest AI predictions.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving AI predictions.",
        )


@app.get(
    "/ai/history",
    response_model=List[AnomalyResponse],
    tags=["AI Anomaly Detection"],
    summary="Get full AI anomaly detection history",
)
async def get_ai_history(limit: int = 100) -> List[AnomalyResponse]:
    """Return historical anomaly detections from the database."""
    try:
        rows = database.get_ai_prediction_history(limit=limit)
        return [AnomalyResponse(**row) for row in rows]
    except Exception:
        logger.exception("Failed to retrieve AI prediction history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving AI history.",
        )


@app.get(
    "/ai/statistics",
    response_model=AIStatisticsResponse,
    tags=["AI Anomaly Detection"],
    summary="Get aggregate AI anomaly detection statistics",
)
async def get_ai_statistics() -> AIStatisticsResponse:
    """Return aggregate statistics across all stored anomaly predictions."""
    try:
        stats = database.get_ai_statistics()
        return AIStatisticsResponse(
            total_anomalies=  stats.get("total_anomalies", 0),
            critical_count=   stats.get("critical_count",  0),
            high_count=       stats.get("high_count",       0),
            medium_count=     stats.get("medium_count",     0),
            low_count=        stats.get("low_count",        0),
            avg_confidence=   stats.get("avg_confidence",  0.0),
            avg_score=        stats.get("avg_score",        0.0),
            model_trained=    _ai_engine.engine.is_trained,
            total_predictions=_ai_engine.engine.total_predictions,
        )
    except Exception:
        logger.exception("Failed to retrieve AI statistics.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving AI statistics.",
        )


# ---------------------------------------------------------------------------
# NEW: AI Dashboard Intelligence Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/ai/dashboard",
    tags=["AI Dashboard"],
    summary="Get complete AI dashboard data bundle",
)
async def get_ai_dashboard() -> Dict[str, Any]:
    """Return all AI dashboard data in a single response.

    This is the primary endpoint consumed by the dashboard's AI tab.
    Returns health score, root causes, trends, anomalies,
    recommendations, timeline, insights, and baseline comparison
    in one HTTP round-trip.
    """
    try:
        return _ai_engine.engine.get_dashboard_bundle()
    except Exception:
        logger.exception("Failed to generate AI dashboard bundle.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating AI dashboard data.",
        )


@app.get(
    "/ai/health",
    tags=["AI Dashboard"],
    summary="Get AI health score and status",
)
async def get_ai_health() -> Dict[str, Any]:
    """Return the current AI-computed system health score."""
    try:
        return _ai_engine.engine.compute_health_score()
    except Exception:
        logger.exception("Failed to retrieve AI health score.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving AI health.",
        )


@app.get(
    "/ai/root-cause",
    tags=["AI Dashboard"],
    summary="Get AI root cause analysis",
)
async def get_ai_root_cause() -> List[Dict[str, Any]]:
    """Return root cause analysis for active anomalies."""
    try:
        return _ai_engine.engine.analyze_root_cause()
    except Exception:
        logger.exception("Failed to retrieve AI root cause analysis.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while analyzing root cause.",
        )


@app.get(
    "/ai/trends",
    tags=["AI Dashboard"],
    summary="Get AI trend analysis",
)
async def get_ai_trends() -> List[Dict[str, Any]]:
    """Return detected trends across all monitored metrics."""
    try:
        return _ai_engine.engine.analyze_trends()
    except Exception:
        logger.exception("Failed to retrieve AI trends.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while analyzing trends.",
        )


@app.get(
    "/ai/anomalies",
    tags=["AI Dashboard"],
    summary="Get active AI anomalies",
)
async def get_ai_anomalies() -> List[Dict[str, Any]]:
    """Return all currently active anomalies."""
    try:
        return _ai_engine.engine.get_active_anomalies()
    except Exception:
        logger.exception("Failed to retrieve active anomalies.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving anomalies.",
        )


@app.get(
    "/ai/recommendations",
    tags=["AI Dashboard"],
    summary="Get AI-generated recommendations",
)
async def get_ai_recommendations() -> List[Dict[str, Any]]:
    """Return prioritized actionable recommendations."""
    try:
        return _ai_engine.engine.generate_recommendations()
    except Exception:
        logger.exception("Failed to generate AI recommendations.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating recommendations.",
        )


@app.get(
    "/ai/timeline",
    tags=["AI Dashboard"],
    summary="Get system health timeline",
)
async def get_ai_timeline(limit: int = 50) -> List[Dict[str, Any]]:
    """Return chronological system health events."""
    try:
        return _ai_engine.engine.get_timeline(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve AI timeline.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving timeline.",
        )


@app.get(
    "/ai/insights",
    tags=["AI Dashboard"],
    summary="Get AI-generated system insights",
)
async def get_ai_insights() -> Dict[str, Any]:
    """Return natural-language AI summary of current system state."""
    try:
        return _ai_engine.engine.generate_insights()
    except Exception:
        logger.exception("Failed to generate AI insights.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating insights.",
        )


# ---------------------------------------------------------------------------
# NEW: Explainable AI Root Cause Analysis Engine — endpoints
# ---------------------------------------------------------------------------
# NOTE ON NAMING: the pre-existing "/ai/root-cause" endpoint above already
# serves the original AnomalyDetectionEngine.analyze_root_cause() method
# and is left completely untouched. The new, independent
# RootCauseAnalysisEngine (which produces a richer, single-object,
# explanation-first result keyed to the most recently *detected* anomaly)
# is exposed under "/ai/rca" instead, to avoid any collision with the
# existing route or response shape.
@app.get(
    "/ai/rca",
    tags=["AI Root Cause Analysis"],
    summary="Get the latest Explainable AI Root Cause Analysis result",
    responses={404: {"model": ErrorResponse, "description": "No anomaly has been analyzed yet."}},
)
async def get_ai_rca_latest() -> Dict[str, Any]:
    """Return the most recent Explainable AI Root Cause Analysis result.

    Produced by the independent ``RootCauseAnalysisEngine`` in
    ai_engine.py, which runs only after an anomaly has already been
    detected. Returns 404 until the first anomaly of the session occurs.
    """
    try:
        result = _ai_engine.engine.get_latest_root_cause_analysis()
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No anomaly has been analyzed yet.",
            )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve latest Root Cause Analysis result.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving root cause analysis.",
        )


@app.get(
    "/ai/rca/history",
    tags=["AI Root Cause Analysis"],
    summary="Get historical Explainable AI Root Cause Analysis results",
)
async def get_ai_rca_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return past Root Cause Analysis results, most recent first."""
    try:
        return _ai_engine.engine.get_root_cause_analysis_history(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve Root Cause Analysis history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving root cause analysis history.",
        )


# ---------------------------------------------------------------------------
# NEW: Explainable AI Health Score Engine — endpoints
# ---------------------------------------------------------------------------
# NOTE ON NAMING: the pre-existing "/ai/health" endpoint above already
# serves the original AnomalyDetectionEngine.compute_health_score()
# bookkeeping and is left completely untouched. The new, independent
# HealthScoreEngine (weighted, explainable, recalculated every cycle) is
# exposed under "/ai/health-score" instead.
@app.get(
    "/ai/health-score",
    tags=["AI Health Score"],
    summary="Get the latest Explainable AI Health Score",
    responses={404: {"model": ErrorResponse, "description": "No monitoring cycle has run yet."}},
)
async def get_ai_health_score() -> Dict[str, Any]:
    """Return the current Explainable AI Health Score.

    Produced by the independent ``HealthScoreEngine`` in ai_engine.py,
    recalculated every monitoring cycle. Returns 404 until the first
    monitoring cycle of the session completes.
    """
    try:
        result = _ai_engine.engine.get_latest_health_score()
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No monitoring cycle has run yet. Start monitoring first.",
            )
        return result
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to retrieve latest AI Health Score.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the health score.",
        )


@app.get(
    "/ai/health-score/history",
    tags=["AI Health Score"],
    summary="Get historical Explainable AI Health Score values",
)
async def get_ai_health_score_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Return past Health Score values, most recent first (for a sparkline)."""
    try:
        return _ai_engine.engine.get_health_score_history(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve AI Health Score history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving health score history.",
        )


# ---------------------------------------------------------------------------
# NEW: Explainable AI Trend Analysis Engine — endpoints
# ---------------------------------------------------------------------------
# NOTE ON NAMING: the pre-existing "/ai/trends" endpoint above already
# serves the original AnomalyDetectionEngine.analyze_trends() method and
# is left completely untouched. The new, independent TrendAnalysisEngine
# (duration-in-minutes, spike-vs-long-term-trend classification,
# historical comparison) is exposed under "/ai/trend-analysis" instead.
@app.get(
    "/ai/trend-analysis",
    tags=["AI Trend Analysis"],
    summary="Get the latest Explainable AI Trend Analysis (one entry per metric)",
)
async def get_ai_trend_analysis() -> List[Dict[str, Any]]:
    """Return the most recent Trend Analysis result set.

    Produced by the independent ``TrendAnalysisEngine`` in ai_engine.py,
    recalculated every monitoring cycle. Returns an empty list until
    enough samples have been collected.
    """
    try:
        return _ai_engine.engine.get_latest_trend_analysis()
    except Exception:
        logger.exception("Failed to retrieve latest Trend Analysis.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving trend analysis.",
        )


@app.get(
    "/ai/trend-analysis/history",
    tags=["AI Trend Analysis"],
    summary="Get historical Explainable AI Trend Analysis bundles",
)
async def get_ai_trend_analysis_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return past Trend Analysis result bundles, most recent first."""
    try:
        return _ai_engine.engine.get_trend_analysis_history(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve Trend Analysis history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving trend analysis history.",
        )


# ---------------------------------------------------------------------------
# NEW: Explainable AI Predictive Alert Engine — endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/ai/predictive-alerts",
    tags=["AI Predictive Alerts"],
    summary="Get the latest AI Predictive Alerts (forecast, not yet occurred)",
)
async def get_ai_predictive_alerts() -> List[Dict[str, Any]]:
    """Return the most recently forecast predictive alerts.

    Produced by the independent ``PredictiveAlertEngine`` in ai_engine.py.
    These are forward-looking forecasts (5/15/30/60 minute horizons) and
    are intentionally distinct from ``/ai/anomalies`` (already-occurred
    events). May return an empty list if no metric is currently trending
    toward its threshold.
    """
    try:
        return _ai_engine.engine.get_latest_predictive_alerts()
    except Exception:
        logger.exception("Failed to retrieve latest Predictive Alerts.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving predictive alerts.",
        )


@app.get(
    "/ai/predictive-alerts/history",
    tags=["AI Predictive Alerts"],
    summary="Get historical AI Predictive Alert bundles",
)
async def get_ai_predictive_alerts_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return past predictive-alert bundles, most recent first."""
    try:
        return _ai_engine.engine.get_predictive_alerts_history(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve Predictive Alert history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving predictive alert history.",
        )


# ---------------------------------------------------------------------------
# NEW: Explainable AI Recommendation Engine — endpoints
# ---------------------------------------------------------------------------
# NOTE ON NAMING: the pre-existing "/ai/recommendations" endpoint above
# already serves the original AnomalyDetectionEngine.generate_recommendations()
# method and is left completely untouched. The new, independent
# RecommendationEngine (adds "reason", "confidence", "estimated_urgency"
# and synthesizes across RCA + Trend Analysis + Health Score) is exposed
# under "/ai/smart-recommendations" instead.
@app.get(
    "/ai/smart-recommendations",
    tags=["AI Recommendations"],
    summary="Get the latest Explainable AI Recommendations",
)
async def get_ai_smart_recommendations() -> List[Dict[str, Any]]:
    """Return the most recently generated recommendations.

    Produced by the independent ``RecommendationEngine`` in ai_engine.py,
    synthesizing across the current anomaly prediction, Root Cause
    Analysis, Trend Analysis, and Health Score. Always non-empty (falls
    back to an explicit "all normal" entry).
    """
    try:
        return _ai_engine.engine.get_latest_smart_recommendations()
    except Exception:
        logger.exception("Failed to retrieve latest AI recommendations.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving recommendations.",
        )


@app.get(
    "/ai/smart-recommendations/history",
    tags=["AI Recommendations"],
    summary="Get historical Explainable AI Recommendation bundles",
)
async def get_ai_smart_recommendations_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return past recommendation bundles, most recent first."""
    try:
        return _ai_engine.engine.get_smart_recommendations_history(limit=limit)
    except Exception:
        logger.exception("Failed to retrieve AI recommendation history.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving recommendation history.",
        )