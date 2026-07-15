"""
ai_engine.core
==============
Core Isolation Forest anomaly detection engine — the original
``ai_engine.py`` logic, unchanged, moved into the ``ai_engine`` package
as part of a structural refactor (behavior is 100% identical; only the
file location changed).

This module owns:
    * ``ModelConfig`` / ``AnomalyPrediction`` / ``TimelineEvent`` — the
      core data models shared across the AI layer.
    * ``FEATURE_NAMES`` — the locked feature vector used for
      training/inference.
    * ``AnomalyDetectionEngine`` — the full ML lifecycle: initialization,
      feature engineering, training, inference, alerting, persistence,
      plus the "dashboard intelligence" hooks (health bookkeeping,
      timeline, active-anomaly tracking) that the five sibling engines
      (Root Cause Analysis, Health Score, Trend Analysis, Predictive
      Alerts, Recommendations) plug into.

See ``ai_engine/__init__.py`` for the package-level public API, which is
identical to the original single-file ``ai_engine.py`` module.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config

from ai_engine.root_cause import RootCauseAnalysisEngine
from ai_engine.health_score import HealthScoreEngine
from ai_engine.trends import TrendAnalysisEngine
from ai_engine.predictive import PredictiveAlertEngine
from ai_engine.recommendations import RecommendationEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# NOTE: this file now lives one directory deeper than the original
# ai_engine.py (which sat directly at the project root). To keep saved
# models at the exact same on-disk location as before (project_root/models/,
# not project_root/ai_engine/models/), we go up two directories instead of
# one — preserving full backward compatibility with any model files
# already saved by the pre-refactor single-file version.
_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(_HERE, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "isolation_forest.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Tunable parameters for the Isolation Forest engine.

    Attributes:
        n_estimators: Number of isolation trees. Higher = more robust,
            slower to train. 100 is a strong default.
        contamination: Expected proportion of anomalies in training data.
            ``"auto"`` lets the algorithm decide; set to e.g. ``0.05``
            if you expect ~5% anomalous samples in your history.
        max_samples: Samples drawn per tree. ``"auto"`` uses
            min(256, n_samples).
        random_state: Seed for reproducibility.
        rolling_window: Number of past readings kept for rolling
            feature computation (rolling mean, moving average).
        min_train_samples: Minimum samples required before the model
            will attempt its first training run.
        retrain_interval: Retrain after this many new samples have
            been collected since the last training run.
        zscore_spike_threshold: Z-score above which a single-feature
            spike is flagged in the reason string.
    """
    n_estimators:          int   = 100
    contamination:         Any   = "auto"
    max_samples:           Any   = "auto"
    random_state:          int   = 42
    rolling_window:        int   = 20
    min_train_samples:     int   = 10
    retrain_interval:      int   = 100
    zscore_spike_threshold: float = 2.5


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnomalyPrediction:
    """The result of one call to ``predict_anomaly``.

    Attributes:
        timestamp: ISO-8601 timestamp of the analyzed metric snapshot.
        is_anomaly: ``True`` if the sample is classified as anomalous.
        anomaly_score: Raw Isolation Forest score. More negative =
            more anomalous. Range approximately [-0.5, 0.5].
        confidence: Human-readable confidence percentage (0–100).
            Derived from the anomaly score.
        severity: One of ``NORMAL``, ``LOW``, ``MEDIUM``, ``HIGH``,
            ``CRITICAL``.
        reason: Intelligent natural-language explanation of why the
            anomaly was detected.
        features_used: Names of the engineered features fed to the
            model for this prediction.
        model_tier: Which model tier produced this prediction
            (``"isolation_forest"`` for Tier 1).
    """
    timestamp:     str
    is_anomaly:    bool
    anomaly_score: float
    confidence:    float
    severity:      str
    reason:        str
    features_used: List[str] = field(default_factory=list)
    model_tier:    str       = "isolation_forest"

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict (database / API friendly)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Feature names (locked — used in training AND inference)
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "cpu_percent",
    "ram_percent",
    "disk_percent",
    "net_sent_mb",
    "net_recv_mb",
    "net_throughput_mb",
    "cpu_ram_ratio",
    "avg_process_cpu",
    "avg_process_memory",
    "total_active_processes",
    "rolling_cpu_avg",
    "rolling_ram_avg",
    "rolling_net_avg",
]


# ---------------------------------------------------------------------------
# Timeline event dataclass
# ---------------------------------------------------------------------------
@dataclass
class TimelineEvent:
    """One entry in the chronological system health timeline."""
    timestamp: str
    event_type: str          # "metric", "anomaly", "alert", "health", "trend", "root_cause"
    title: str
    description: str
    severity: str = "NORMAL"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnomalyDetectionEngine:
    """Isolation Forest-based anomaly detection engine.

    Owns the full ML lifecycle: initialization, feature engineering,
    training, inference, alerting, and model persistence.

    This class is designed to be instantiated once and shared across
    the monitoring thread and FastAPI request threads. All mutable
    state is protected by ``_training_lock`` where necessary.

    Args:
        model_config: Tunable parameters. Uses :class:`ModelConfig`
            defaults if not provided.

    Example::

        engine = AnomalyDetectionEngine()
        engine.initialize_model()
        engine.load_model()          # loads saved model if available

        # after enough data has been collected:
        engine.train_model(metrics_list)

        # every monitoring cycle:
        prediction = engine.predict_anomaly(metric_dict, process_list)
        if prediction.is_anomaly:
            engine.generate_ai_alert(prediction)
    """

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        self.cfg             = model_config or ModelConfig()
        self._model: Optional[IsolationForest]  = None
        self._scaler: Optional[StandardScaler]  = None
        self._is_trained: bool                  = False
        self._training_lock: threading.Lock     = threading.Lock()
        self._samples_since_retrain: int        = 0
        self._total_predictions: int            = 0

        # Rolling windows for feature computation (per-metric deques)
        self._rolling_cpu:  deque = deque(maxlen=self.cfg.rolling_window)
        self._rolling_ram:  deque = deque(maxlen=self.cfg.rolling_window)
        self._rolling_net:  deque = deque(maxlen=self.cfg.rolling_window)

        # ── NEW: state for dashboard intelligence layer ──
        self._timeline: deque = deque(maxlen=500)
        self._health_score: float = 100.0
        self._health_status: str = "Healthy"
        self._health_reasons: List[str] = []
        self._health_confidence: float = 100.0
        self._last_health_update: str = datetime.now().isoformat()
        self._active_anomalies: List[Dict[str, Any]] = []
        self._baseline: Dict[str, float] = {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "disk_percent": 0.0,
            "net_throughput_mb": 0.0,
        }
        self._baseline_computed: bool = False

        # ── NEW: Explainable AI Root Cause Analysis Engine ──
        # Independent module, consumes predictions only after they are
        # made — never re-runs or alters anomaly detection itself.
        self._rca_engine: "RootCauseAnalysisEngine" = RootCauseAnalysisEngine(self)

        # ── NEW: Explainable AI Health Score Engine ──
        # Independent module, recalculated every monitoring cycle.
        # Does not read or modify self._health_score / _health_status
        # (the pre-existing internal bookkeeping used by recommendations
        # and trend penalties) — this is a separate, explicitly weighted,
        # fully explainable score computed in parallel.
        self._health_score_engine: "HealthScoreEngine" = HealthScoreEngine(self)

        # ── NEW: Explainable AI Trend Analysis Engine ──
        # Independent module, recalculated every monitoring cycle. Does
        # not read or modify the pre-existing analyze_trends()/_detect_trend
        # logic — a separate, richer implementation with duration-in-minutes,
        # spike-vs-long-term-trend classification, and historical comparison.
        self._trend_engine: "TrendAnalysisEngine" = TrendAnalysisEngine(self)

        # ── NEW: Explainable AI Predictive Alert Engine ──
        # Independent module, recalculated every monitoring cycle. Only
        # consumes this cycle's TrendAnalysisEngine output — never
        # recomputes trends or reads raw metrics for its own regression.
        self._predictive_engine: "PredictiveAlertEngine" = PredictiveAlertEngine(self)

        # ── NEW: Explainable AI Recommendation Engine ──
        # Independent module, recalculated every monitoring cycle. Does
        # not read or modify the pre-existing generate_recommendations()
        # method — a separate implementation that synthesizes across the
        # current prediction, Root Cause Analysis, Trend Analysis, and
        # Health Score outputs.
        self._recommendation_engine: "RecommendationEngine" = RecommendationEngine(self)

    # ------------------------------------------------------------------
    # Public API — original methods
    # ------------------------------------------------------------------
    def initialize_model(self) -> None:
        """Initialise a fresh Isolation Forest model and StandardScaler.

        Safe to call multiple times — subsequent calls reset the model
        to an untrained state (use :meth:`load_model` immediately after
        to restore a persisted model instead of retraining from scratch).

        Raises:
            Never raises: errors are logged.
        """
        try:
            with self._training_lock:
                self._model = IsolationForest(
                    n_estimators=self.cfg.n_estimators,
                    contamination=self.cfg.contamination,
                    max_samples=self.cfg.max_samples,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )
                self._scaler    = StandardScaler()
                self._is_trained = False

            logger.info(
                "Isolation Forest initialized: n_estimators=%d contamination=%s",
                self.cfg.n_estimators, self.cfg.contamination,
            )
        except Exception:
            logger.exception("Failed to initialize the anomaly detection model.")

    def prepare_training_data(
        self,
        metrics_list: List[Dict[str, Any]],
        process_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[np.ndarray]:
        """Convert raw monitoring dicts into a scaled feature matrix.

        Performs:
            * Missing value imputation (median fill).
            * Feature engineering (ratios, rolling averages, throughput).
            * ``StandardScaler`` normalization.

        Args:
            metrics_list: List of metric dicts as produced by
                :func:`config.collect_system_metrics`.
            process_snapshots: Optional list of process snapshot dicts
                (each element is a ``{"processes": [...]}`` dict from
                ``config.process_data``). Aligned by index with
                ``metrics_list`` where available.

        Returns:
            A 2-D ``numpy`` array of shape
            ``(n_samples, len(FEATURE_NAMES))``, scaled and ready for
            training or inference. Returns ``None`` if preparation fails
            or if there are fewer than 2 valid samples.
        """
        if not metrics_list:
            logger.warning("prepare_training_data: empty metrics_list.")
            return None

        try:
            rows: List[List[float]] = []

            # Temporary rolling windows just for batch preparation
            tmp_cpu: deque = deque(maxlen=self.cfg.rolling_window)
            tmp_ram: deque = deque(maxlen=self.cfg.rolling_window)
            tmp_net: deque = deque(maxlen=self.cfg.rolling_window)

            for i, m in enumerate(metrics_list):
                procs = []
                if process_snapshots and i < len(process_snapshots):
                    procs = process_snapshots[i].get("processes", [])

                row = self._engineer_features(m, procs, tmp_cpu, tmp_ram, tmp_net)
                rows.append(row)

            matrix = np.array(rows, dtype=float)

            # Impute NaN / Inf with column medians
            matrix = self._impute(matrix)

            if matrix.shape[0] < 2:
                logger.warning("prepare_training_data: fewer than 2 valid samples.")
                return None

            return matrix

        except Exception:
            logger.exception("Failed to prepare training data.")
            return None

    def train_model(
        self,
        metrics_list: List[Dict[str, Any]],
        process_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """Fit the Isolation Forest on historical monitoring data.

        Will not train if fewer than ``cfg.min_train_samples`` samples
        are available. After successful training, persists the model and
        scaler to disk via :meth:`save_model`.

        Args:
            metrics_list: Historical metric dicts from
                ``config.metrics_data``.
            process_snapshots: Matching process snapshot dicts from
                ``config.process_data``.

        Returns:
            ``True`` if training succeeded, ``False`` otherwise.
        """
        if len(metrics_list) < self.cfg.min_train_samples:
            logger.info(
                "train_model: only %d samples available, need %d. Skipping.",
                len(metrics_list), self.cfg.min_train_samples,
            )
            return False

        try:
            matrix = self.prepare_training_data(metrics_list, process_snapshots)
            if matrix is None:
                return False

            with self._training_lock:
                if self._model is None or self._scaler is None:
                    self.initialize_model()

                scaled = self._scaler.fit_transform(matrix)      # type: ignore[union-attr]
                self._model.fit(scaled)                           # type: ignore[union-attr]
                self._is_trained       = True
                self._samples_since_retrain = 0

            # Compute baseline from training data
            self._compute_baseline(metrics_list)

            self.save_model()

            self._add_timeline_event(
                "health", "Model Trained",
                f"Isolation Forest trained on {matrix.shape[0]} samples "
                f"with {matrix.shape[1]} features.",
                severity="NORMAL",
            )

            logger.info(
                "Isolation Forest trained on %d samples (%d features).",
                matrix.shape[0], matrix.shape[1],
            )
            return True

        except Exception:
            logger.exception("Failed to train the anomaly detection model.")
            return False

    def predict_anomaly(
        self,
        metric: Dict[str, Any],
        processes: Optional[List[Dict[str, Any]]] = None,
    ) -> AnomalyPrediction:
        """Classify one monitoring sample and return a full prediction.

        Updates the internal rolling windows with the new sample before
        computing features, so the rolling averages reflect the most
        recent history.

        Automatically triggers a background retrain when
        ``_samples_since_retrain`` reaches ``cfg.retrain_interval``
        and enough data is available in ``config.metrics_data``.

        Args:
            metric: A single metric dict as produced by
                :func:`config.collect_system_metrics`.
            processes: Top-process list for this cycle from
                :func:`config.collect_process_metrics`.

        Returns:
            An :class:`AnomalyPrediction` with all fields populated.
            If the model is not yet trained, returns a ``NORMAL``
            prediction with a note in the reason field.
        """
        timestamp = metric.get("timestamp", datetime.now().isoformat())

        # Update live rolling windows with this new reading
        self._rolling_cpu.append(float(metric.get("cpu_percent", 0.0)))
        self._rolling_ram.append(float(metric.get("ram_percent", 0.0)))
        net = float(metric.get("net_sent_mb", 0.0)) + float(metric.get("net_recv_mb", 0.0))
        self._rolling_net.append(net)

        self._samples_since_retrain += 1
        self._total_predictions     += 1

        # Schedule background retrain if interval reached
        if (
            self._samples_since_retrain >= self.cfg.retrain_interval
            and len(config.metrics_data) >= self.cfg.min_train_samples
        ):
            self._schedule_retrain()

        if not self._is_trained or self._model is None or self._scaler is None:
            prediction = AnomalyPrediction(
                timestamp=timestamp,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity="NORMAL",
                reason=(
                    f"Model not yet trained. Need {self.cfg.min_train_samples} samples; "
                    f"collected {len(config.metrics_data)} so far."
                ),
                features_used=FEATURE_NAMES,
            )
            # Still update health score even when untrained
            self._update_health_score(metric, prediction, processes)

            # ── NEW: Explainable AI Health Score — recalculated every cycle ──
            try:
                self._health_score_engine.compute(metric, processes or [], prediction)
            except Exception:
                logger.exception("Health Score computation failed (untrained-model branch).")

            # ── NEW: Trend Analysis / Predictive Alerts / Recommendations ──
            self._run_extended_ai_pipeline(metric, processes or [], prediction)

            return prediction

        try:
            features = self._engineer_features(
                metric, processes or [],
                self._rolling_cpu, self._rolling_ram, self._rolling_net,
            )
            feature_array = np.array([features], dtype=float)
            feature_array = self._impute(feature_array)

            with self._training_lock:
                scaled  = self._scaler.transform(feature_array)
                raw_score = float(self._model.score_samples(scaled)[0])

            is_anomaly, confidence = self._score_to_decision(raw_score)
            severity = self._compute_severity(raw_score, feature_array[0], metric)
            reason   = self._build_reason(feature_array[0], metric, is_anomaly)

            prediction = AnomalyPrediction(
                timestamp=timestamp,
                is_anomaly=is_anomaly,
                anomaly_score=round(raw_score, 4),
                confidence=round(confidence, 2),
                severity=severity,
                reason=reason,
                features_used=FEATURE_NAMES,
            )

            # Update all dashboard intelligence
            self._update_health_score(metric, prediction, processes)

            # ── NEW: Explainable AI Health Score — recalculated every cycle ──
            try:
                self._health_score_engine.compute(metric, processes or [], prediction)
            except Exception:
                logger.exception("Health Score computation failed.")

            if is_anomaly:
                self._register_anomaly(prediction, metric, processes)

            # ── NEW: Trend Analysis / Predictive Alerts / Recommendations ──
            # Placed after _register_anomaly so this cycle's Root Cause
            # Analysis (if any) is already available to the Recommendation
            # and Predictive Alert engines.
            self._run_extended_ai_pipeline(metric, processes or [], prediction)

            # Add metric event to timeline
            self._add_timeline_event(
                "metric", "Metric Collected",
                f"CPU={metric.get('cpu_percent', 0):.1f}% "
                f"RAM={metric.get('ram_percent', 0):.1f}% "
                f"Disk={metric.get('disk_percent', 0):.1f}%",
                severity="NORMAL",
                metadata={"cpu": metric.get("cpu_percent"), "ram": metric.get("ram_percent")},
            )

            return prediction

        except Exception:
            logger.exception("predict_anomaly failed; returning safe default.")
            return AnomalyPrediction(
                timestamp=timestamp,
                is_anomaly=False,
                anomaly_score=0.0,
                confidence=0.0,
                severity="NORMAL",
                reason="Prediction error — check logs.",
            )

    def generate_ai_alert(self, prediction: AnomalyPrediction) -> None:
        """Print a structured AI alert to the terminal and log it.

        Only prints if ``prediction.is_anomaly`` is ``True``.

        Args:
            prediction: The result of a :meth:`predict_anomaly` call.
        """
        if not prediction.is_anomaly:
            return

        severity_colors = {
            "LOW":      "⚠ ",
            "MEDIUM":   "🔶",
            "HIGH":     "🔴",
            "CRITICAL": "🚨",
        }
        icon = severity_colors.get(prediction.severity, "⚠ ")

        alert = (
            f"\n{'=' * 60}\n"
            f"  {icon}  [AI ALERT]  Potential anomaly detected.\n"
            f"{'=' * 60}\n"
            f"  Confidence : {prediction.confidence:.1f}%\n"
            f"  Severity   : {prediction.severity}\n"
            f"  Score      : {prediction.anomaly_score:.4f}\n"
            f"  Reason     :\n"
            f"    {prediction.reason}\n"
            f"  Timestamp  : {prediction.timestamp}\n"
            f"  Model Tier : {prediction.model_tier}\n"
            f"{'=' * 60}\n"
        )
        print(alert)
        logger.warning(
            "AI ALERT | severity=%s confidence=%.1f%% score=%.4f | %s",
            prediction.severity, prediction.confidence,
            prediction.anomaly_score, prediction.reason,
        )

    def save_model(self) -> bool:
        """Persist the trained model and scaler to disk using joblib.

        Saves to:
            * ``models/isolation_forest.joblib``
            * ``models/scaler.joblib``

        Returns:
            ``True`` on success, ``False`` on failure.
        """
        if not self._is_trained or self._model is None or self._scaler is None:
            logger.warning("save_model: model not trained yet; nothing to save.")
            return False

        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            joblib.dump(self._model,  MODEL_PATH)
            joblib.dump(self._scaler, SCALER_PATH)
            logger.info("Model saved to '%s'.", MODEL_PATH)
            return True
        except Exception:
            logger.exception("Failed to save model.")
            return False

    def load_model(self) -> bool:
        """Load a previously saved model and scaler from disk.

        If both ``isolation_forest.joblib`` and ``scaler.joblib`` are
        found, loads them and marks the engine as trained — skipping
        the need for an immediate retraining cycle.

        Returns:
            ``True`` if both files were loaded successfully,
            ``False`` if either is missing or loading fails.
        """
        if not os.path.isfile(MODEL_PATH) or not os.path.isfile(SCALER_PATH):
            logger.info(
                "No saved model found at '%s'. Will train from scratch once "
                "%d samples are collected.",
                MODEL_PATH, self.cfg.min_train_samples,
            )
            return False

        try:
            with self._training_lock:
                self._model   = joblib.load(MODEL_PATH)
                self._scaler  = joblib.load(SCALER_PATH)
                self._is_trained = True

            logger.info("Loaded saved model from '%s'.", MODEL_PATH)
            return True
        except Exception:
            logger.exception("Failed to load saved model; will retrain.")
            self._is_trained = False
            return False

    # ------------------------------------------------------------------
    # NEW: Dashboard intelligence methods
    # ------------------------------------------------------------------
    def compute_health_score(self) -> Dict[str, Any]:
        """Return the current AI health assessment for the dashboard.

        Returns:
            A dict with ``score``, ``status``, ``confidence``,
            ``last_updated``, and ``reasons``.
        """
        return {
            "score": round(self._health_score, 1),
            "status": self._health_status,
            "confidence": round(self._health_confidence, 1),
            "last_updated": self._last_health_update,
            "reasons": list(self._health_reasons),
        }

    def analyze_root_cause(self) -> List[Dict[str, Any]]:
        """Identify the root cause of active anomalies.

        Examines the most recent anomalies and the rolling metric
        windows to attribute the primary responsible metric and
        process.

        Returns:
            A list of root-cause analysis dicts, one per active
            anomaly (most recent first, max 5).
        """
        results: List[Dict[str, Any]] = []

        with config.data_lock:
            recent_metrics = list(config.metrics_data[-20:]) if config.metrics_data else []
            recent_procs = list(config.process_data[-5:]) if config.process_data else []

        if not recent_metrics:
            return results

        latest = recent_metrics[-1] if recent_metrics else {}

        # Determine which metric is most abnormal
        deviations = self._compute_deviations(latest)
        sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)

        primary_metric = sorted_devs[0][0] if sorted_devs else "cpu_percent"
        primary_dev = sorted_devs[0][1] if sorted_devs else 0.0

        # Find responsible process
        responsible_process = "N/A"
        if recent_procs:
            last_procs = recent_procs[-1].get("processes", [])
            if last_procs:
                if "cpu" in primary_metric.lower():
                    sorted_p = sorted(last_procs, key=lambda p: p.get("cpu_percent", 0), reverse=True)
                elif "ram" in primary_metric.lower() or "memory" in primary_metric.lower():
                    sorted_p = sorted(last_procs, key=lambda p: p.get("memory_percent", 0), reverse=True)
                else:
                    sorted_p = sorted(last_procs, key=lambda p: p.get("cpu_percent", 0), reverse=True)
                if sorted_p:
                    responsible_process = sorted_p[0].get("name", "unknown")

        # Determine root cause description
        root_cause, recommendation = self._determine_root_cause(
            primary_metric, primary_dev, latest, responsible_process
        )

        # Severity from deviation magnitude
        dev_abs = abs(primary_dev)
        if dev_abs >= 50:
            severity = "CRITICAL"
        elif dev_abs >= 30:
            severity = "HIGH"
        elif dev_abs >= 15:
            severity = "MEDIUM"
        elif dev_abs >= 5:
            severity = "LOW"
        else:
            severity = "NORMAL"

        confidence = min(99.0, 50.0 + dev_abs * 1.2)

        # Historical comparison
        baseline_val = self._baseline.get(primary_metric, 0.0)
        current_val = float(latest.get(primary_metric, 0.0))

        metric_display = primary_metric.replace("_", " ").replace("percent", "").strip().upper()
        if not metric_display:
            metric_display = primary_metric.upper()

        results.append({
            "primary_metric": metric_display,
            "responsible_process": responsible_process,
            "root_cause": root_cause,
            "severity": severity,
            "confidence": round(confidence, 1),
            "recommendation": recommendation,
            "current_value": round(current_val, 2),
            "baseline_value": round(baseline_val, 2),
            "deviation_percent": round(primary_dev, 1),
            "timestamp": latest.get("timestamp", datetime.now().isoformat()),
        })

        return results

    def analyze_trends(self) -> List[Dict[str, Any]]:
        """Detect trends in monitored metrics using rolling analysis.

        Examines the last N readings for each key metric and identifies
        sustained increasing, decreasing, stable, or spike patterns.

        Returns:
            A list of trend dicts, one per analyzed metric.
        """
        trends: List[Dict[str, Any]] = []

        with config.data_lock:
            recent = list(config.metrics_data[-30:]) if config.metrics_data else []

        if len(recent) < 3:
            return trends

        metric_keys = [
            ("cpu_percent", "CPU Usage"),
            ("ram_percent", "RAM Usage"),
            ("disk_percent", "Disk Usage"),
        ]

        # Compute net throughput trend separately
        net_values = [
            float(m.get("net_sent_mb", 0)) + float(m.get("net_recv_mb", 0))
            for m in recent
        ]

        for key, display_name in metric_keys:
            values = [float(m.get(key, 0)) for m in recent]
            trend_info = self._detect_trend(values, display_name)
            trend_info["metric_key"] = key
            trends.append(trend_info)

        # Network trend
        net_trend = self._detect_trend(net_values, "Network Activity")
        net_trend["metric_key"] = "net_throughput_mb"
        trends.append(net_trend)

        return trends

    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        """Return all currently active (recent) anomalies.

        Returns:
            A list of anomaly dicts, most recent first, max 20.
        """
        return list(reversed(self._active_anomalies[-20:]))

    # ------------------------------------------------------------------
    # NEW: Explainable AI Root Cause Analysis — public accessors
    # ------------------------------------------------------------------
    def get_latest_root_cause_analysis(self) -> Optional[Dict[str, Any]]:
        """Return the most recent Root Cause Analysis result, if any.

        This is produced by the independent :class:`RootCauseAnalysisEngine`
        and reflects the most recently *detected* anomaly only — it does
        not perform any new detection itself.

        Returns:
            The result dict (see :meth:`RootCauseAnalysisEngine.analyze`
            for the schema), or ``None`` if no anomaly has occurred yet.
        """
        return self._rca_engine.get_latest()

    def get_root_cause_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past Root Cause Analysis results, most recent first.

        Args:
            limit: Maximum number of results to return.
        """
        return self._rca_engine.get_history(limit=limit)

    # ------------------------------------------------------------------
    # NEW: Explainable AI Health Score — public accessors
    # ------------------------------------------------------------------
    def get_latest_health_score(self) -> Optional[Dict[str, Any]]:
        """Return the most recent Health Score result, if any.

        Produced by the independent :class:`HealthScoreEngine`, recomputed
        every monitoring cycle. Does not read the pre-existing internal
        ``compute_health_score()`` bookkeeping — this is a separate,
        explicitly weighted, fully explainable score.

        Returns:
            The result dict (see :meth:`HealthScoreEngine.compute` for the
            schema), or ``None`` if no monitoring cycle has run yet.
        """
        return self._health_score_engine.get_latest()

    def get_health_score_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return past Health Score results, most recent first.

        Args:
            limit: Maximum number of results to return.
        """
        return self._health_score_engine.get_history(limit=limit)

    # ------------------------------------------------------------------
    # NEW: Explainable AI Trend Analysis — public accessors
    # ------------------------------------------------------------------
    def get_latest_trend_analysis(self) -> List[Dict[str, Any]]:
        """Return the most recent Trend Analysis result set (one entry per metric).

        Produced by the independent :class:`TrendAnalysisEngine`, recomputed
        every monitoring cycle. Does not read the pre-existing
        ``analyze_trends()``/``_detect_trend`` logic.
        """
        return self._trend_engine.get_latest()

    def get_trend_analysis_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past Trend Analysis result bundles, most recent first."""
        return self._trend_engine.get_history(limit=limit)

    # ------------------------------------------------------------------
    # NEW: Explainable AI Predictive Alerts — public accessors
    # ------------------------------------------------------------------
    def get_latest_predictive_alerts(self) -> List[Dict[str, Any]]:
        """Return the most recently forecast predictive alerts (may be empty).

        Produced by the independent :class:`PredictiveAlertEngine`, which
        only extrapolates off this cycle's Trend Analysis output — it
        performs no anomaly detection or trend computation of its own.
        """
        return self._predictive_engine.get_latest()

    def get_predictive_alerts_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past predictive alert bundles, most recent first."""
        return self._predictive_engine.get_history(limit=limit)

    # ------------------------------------------------------------------
    # NEW: Explainable AI Recommendations — public accessors
    # ------------------------------------------------------------------
    def get_latest_smart_recommendations(self) -> List[Dict[str, Any]]:
        """Return the most recent Recommendation Engine output.

        Produced by the independent :class:`RecommendationEngine`, which
        synthesizes across this cycle's prediction, Root Cause Analysis,
        Trend Analysis, and Health Score — without duplicating the
        pre-existing ``generate_recommendations()`` method.
        """
        return self._recommendation_engine.get_latest()

    def get_smart_recommendations_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return past recommendation bundles, most recent first."""
        return self._recommendation_engine.get_history(limit=limit)

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on current state.

        Examines health score, active anomalies, and trends to produce
        prioritized recommendations.

        Returns:
            A list of recommendation dicts with ``priority``,
            ``recommendation``, ``estimated_impact``, and ``category``.
        """
        recs: List[Dict[str, Any]] = []

        with config.data_lock:
            latest = config.metrics_data[-1] if config.metrics_data else {}
            recent_procs = list(config.process_data[-3:]) if config.process_data else []

        if not latest:
            recs.append({
                "priority": "LOW",
                "recommendation": "Start monitoring to enable AI-powered recommendations.",
                "estimated_impact": "Enables full system health visibility.",
                "category": "setup",
            })
            return recs

        cpu = float(latest.get("cpu_percent", 0))
        ram = float(latest.get("ram_percent", 0))
        disk = float(latest.get("disk_percent", 0))
        net = float(latest.get("net_sent_mb", 0)) + float(latest.get("net_recv_mb", 0))

        # CPU recommendations
        if cpu > config.CPU_THRESHOLD:
            top_proc = "unknown"
            if recent_procs:
                procs = recent_procs[-1].get("processes", [])
                if procs:
                    top_proc = procs[0].get("name", "unknown")
            recs.append({
                "priority": "CRITICAL" if cpu > 95 else "HIGH",
                "recommendation": (
                    f"CPU usage is critically high at {cpu:.1f}%. "
                    f"Consider terminating or restarting '{top_proc}' which is the "
                    f"top CPU consumer."
                ),
                "estimated_impact": f"Could reduce CPU usage by 10-30%.",
                "category": "cpu",
            })
        elif cpu > 70:
            recs.append({
                "priority": "MEDIUM",
                "recommendation": (
                    f"CPU usage is elevated at {cpu:.1f}%. "
                    f"Monitor for sustained increase. Consider deferring "
                    f"non-critical background tasks."
                ),
                "estimated_impact": "Prevents potential CPU exhaustion.",
                "category": "cpu",
            })

        # RAM recommendations
        if ram > config.RAM_THRESHOLD:
            mem_proc = "unknown"
            if recent_procs:
                procs = recent_procs[-1].get("processes", [])
                sorted_procs = sorted(procs, key=lambda p: p.get("memory_percent", 0), reverse=True)
                if sorted_procs:
                    mem_proc = sorted_procs[0].get("name", "unknown")
            recs.append({
                "priority": "CRITICAL" if ram > 95 else "HIGH",
                "recommendation": (
                    f"RAM usage at {ram:.1f}% exceeds threshold. "
                    f"'{mem_proc}' is consuming the most memory. "
                    f"Investigate for possible memory leak and consider restarting."
                ),
                "estimated_impact": "May free 15-40% RAM and prevent OOM conditions.",
                "category": "ram",
            })
        elif ram > 70:
            recs.append({
                "priority": "MEDIUM",
                "recommendation": (
                    f"RAM usage at {ram:.1f}% is trending high. "
                    f"Close unused applications to prevent threshold breach."
                ),
                "estimated_impact": "Maintains system stability.",
                "category": "ram",
            })

        # Disk recommendations
        if disk > 90:
            recs.append({
                "priority": "CRITICAL" if disk > 95 else "HIGH",
                "recommendation": (
                    f"Disk usage at {disk:.1f}% is critically high. "
                    f"Immediately clear temporary files, logs, and unused data. "
                    f"Consider expanding storage."
                ),
                "estimated_impact": "Prevents disk-full system failure.",
                "category": "disk",
            })
        elif disk > 80:
            recs.append({
                "priority": "MEDIUM",
                "recommendation": (
                    f"Disk usage at {disk:.1f}%. Plan storage cleanup "
                    f"or expansion in the near term."
                ),
                "estimated_impact": "Avoids future disk pressure.",
                "category": "disk",
            })

        # Network recommendations
        if net > config.NETWORK_THRESHOLD:
            recs.append({
                "priority": "HIGH",
                "recommendation": (
                    f"Network throughput ({net:.3f} MB/cycle) exceeds threshold. "
                    f"Investigate for unauthorized data transfer or "
                    f"misconfigured services."
                ),
                "estimated_impact": "May identify data exfiltration or bandwidth abuse.",
                "category": "network",
            })

        # Anomaly-based recommendations
        active = self.get_active_anomalies()
        critical_count = sum(1 for a in active if a.get("severity") == "CRITICAL")
        high_count = sum(1 for a in active if a.get("severity") == "HIGH")

        if critical_count > 0:
            recs.append({
                "priority": "CRITICAL",
                "recommendation": (
                    f"{critical_count} CRITICAL anomaly(ies) detected. "
                    f"Immediate investigation required. Review the anomaly details "
                    f"and root cause analysis for specific actions."
                ),
                "estimated_impact": "Prevents potential security incident or system failure.",
                "category": "anomaly",
            })

        if high_count > 0:
            recs.append({
                "priority": "HIGH",
                "recommendation": (
                    f"{high_count} HIGH severity anomaly(ies) active. "
                    f"Schedule investigation within the next 15 minutes."
                ),
                "estimated_impact": "Reduces risk of escalation to CRITICAL.",
                "category": "anomaly",
            })

        # Health score recommendations
        if self._health_score < 50:
            recs.append({
                "priority": "HIGH",
                "recommendation": (
                    f"Overall health score is {self._health_score:.0f}/100. "
                    f"Multiple subsystems are under stress. Consider a full "
                    f"system review and load reduction."
                ),
                "estimated_impact": "Restores system health to acceptable levels.",
                "category": "health",
            })

        # If everything looks good
        if not recs:
            recs.append({
                "priority": "LOW",
                "recommendation": "All systems operating within normal parameters. No action required.",
                "estimated_impact": "Continued stable operation.",
                "category": "status",
            })

        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 4))

        return recs

    def get_timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the chronological event timeline.

        Args:
            limit: Maximum events to return (most recent first).

        Returns:
            A list of timeline event dicts.
        """
        events = list(self._timeline)
        events.reverse()
        return events[:limit]

    def generate_insights(self) -> Dict[str, Any]:
        """Generate a natural-language AI summary of the current state.

        Returns:
            A dict with ``summary``, ``key_findings``, ``risk_level``,
            and ``timestamp``.
        """
        with config.data_lock:
            recent = list(config.metrics_data[-20:]) if config.metrics_data else []
            proc_data = list(config.process_data[-5:]) if config.process_data else []

        if not recent:
            return {
                "summary": "Monitoring has not started yet. Start monitoring to receive AI-powered insights.",
                "key_findings": [],
                "risk_level": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
            }

        latest = recent[-1]
        cpu = float(latest.get("cpu_percent", 0))
        ram = float(latest.get("ram_percent", 0))
        disk = float(latest.get("disk_percent", 0))
        net = float(latest.get("net_sent_mb", 0)) + float(latest.get("net_recv_mb", 0))

        findings: List[str] = []
        paragraphs: List[str] = []

        # Load assessment
        if cpu > 85:
            load_desc = "heavy"
        elif cpu > 60:
            load_desc = "moderate"
        elif cpu > 30:
            load_desc = "light"
        else:
            load_desc = "minimal"

        paragraphs.append(
            f"The system is currently operating under {load_desc} load."
        )

        # CPU analysis
        if cpu > config.CPU_THRESHOLD:
            duration_above = sum(
                1 for m in recent if float(m.get("cpu_percent", 0)) > config.CPU_THRESHOLD
            )
            duration_min = (duration_above * config.MONITOR_INTERVAL) / 60
            paragraphs.append(
                f"CPU utilization is at {cpu:.1f}% and has exceeded the "
                f"{config.CPU_THRESHOLD}% threshold for approximately "
                f"{duration_min:.0f} minutes."
            )
            findings.append(f"CPU above threshold for ~{duration_min:.0f} min")
        elif cpu > 70:
            paragraphs.append(
                f"CPU utilization at {cpu:.1f}% is elevated but below the alert threshold."
            )

        # RAM analysis
        if ram > config.RAM_THRESHOLD:
            # Check for continuous increase (potential leak)
            if len(recent) >= 5:
                ram_values = [float(m.get("ram_percent", 0)) for m in recent[-5:]]
                if all(ram_values[i] <= ram_values[i + 1] for i in range(len(ram_values) - 1)):
                    increase = ram_values[-1] - ram_values[0]
                    top_mem = "unknown"
                    if proc_data:
                        procs = proc_data[-1].get("processes", [])
                        sp = sorted(procs, key=lambda p: p.get("memory_percent", 0), reverse=True)
                        if sp:
                            top_mem = sp[0].get("name", "unknown")
                    paragraphs.append(
                        f"RAM usage at {ram:.1f}% indicates a possible memory leak "
                        f"originating from {top_mem}. Usage increased by {increase:.1f}% "
                        f"over the last {len(ram_values)} readings."
                    )
                    findings.append(f"Possible memory leak via {top_mem}")
                else:
                    paragraphs.append(f"RAM usage is elevated at {ram:.1f}%.")
        elif ram > 70:
            paragraphs.append(f"RAM usage at {ram:.1f}% is within acceptable range but trending upward.")

        # Disk analysis
        if disk > 90:
            paragraphs.append(
                f"Disk usage is critically high at {disk:.1f}%. Immediate attention required."
            )
            findings.append("Disk usage critical")

        # Network analysis
        if net > config.NETWORK_THRESHOLD:
            paragraphs.append(
                f"Network activity of {net:.3f} MB/cycle is above normal thresholds. "
                f"This may indicate data exfiltration or unusually heavy traffic."
            )
            findings.append("Abnormal network activity detected")

        # Health score
        paragraphs.append(
            f"Overall Health Score is {self._health_score:.0f}/100. "
            f"Status: {self._health_status}."
        )

        # Active anomalies
        active = self.get_active_anomalies()
        if active:
            crit = sum(1 for a in active if a.get("severity") == "CRITICAL")
            high = sum(1 for a in active if a.get("severity") == "HIGH")
            if crit > 0:
                paragraphs.append(f"{crit} CRITICAL anomaly(ies) require immediate attention.")
                findings.append(f"{crit} CRITICAL anomalies active")
            if high > 0:
                paragraphs.append(f"{high} HIGH severity anomaly(ies) detected.")
                findings.append(f"{high} HIGH anomalies active")

        # Recommendation
        if self._health_score < 50:
            paragraphs.append("Immediate action is recommended.")
        elif self._health_score < 70:
            paragraphs.append("Monitoring and preemptive action are advised.")
        else:
            paragraphs.append("No immediate action required.")

        # Risk level
        if self._health_score >= 80:
            risk = "LOW"
        elif self._health_score >= 60:
            risk = "MODERATE"
        elif self._health_score >= 40:
            risk = "HIGH"
        else:
            risk = "CRITICAL"

        return {
            "summary": " ".join(paragraphs),
            "key_findings": findings,
            "risk_level": risk,
            "timestamp": datetime.now().isoformat(),
        }

    def compare_to_baseline(self) -> Dict[str, Any]:
        """Compare current metric values against their learned baseline.

        Returns:
            A dict with per-metric current/baseline/deviation values.
        """
        with config.data_lock:
            latest = config.metrics_data[-1] if config.metrics_data else {}

        if not latest:
            return {"metrics": [], "baseline_computed": self._baseline_computed}

        net_now = float(latest.get("net_sent_mb", 0)) + float(latest.get("net_recv_mb", 0))

        comparisons = []
        metric_pairs = [
            ("CPU", "cpu_percent", float(latest.get("cpu_percent", 0))),
            ("RAM", "ram_percent", float(latest.get("ram_percent", 0))),
            ("Disk", "disk_percent", float(latest.get("disk_percent", 0))),
            ("Network", "net_throughput_mb", net_now),
        ]

        for display, key, current in metric_pairs:
            baseline = self._baseline.get(key, 0.0)
            if baseline > 0:
                deviation = ((current - baseline) / baseline) * 100
            else:
                deviation = 0.0

            status = "normal"
            if deviation > 30:
                status = "critical"
            elif deviation > 15:
                status = "warning"
            elif deviation < -15:
                status = "low"

            comparisons.append({
                "metric": display,
                "key": key,
                "current": round(current, 2),
                "baseline": round(baseline, 2),
                "deviation_percent": round(deviation, 1),
                "status": status,
            })

        return {
            "metrics": comparisons,
            "baseline_computed": self._baseline_computed,
            "timestamp": latest.get("timestamp", datetime.now().isoformat()),
        }

    def get_dashboard_bundle(self) -> Dict[str, Any]:
        """Return all AI dashboard data in a single response.

        This is the preferred endpoint for the dashboard to call —
        one HTTP request returns everything the AI tab needs, reducing
        the number of polling round-trips.

        Returns:
            A dict with keys: ``health``, ``root_causes``, ``trends``,
            ``active_anomalies``, ``recommendations``, ``timeline``,
            ``insights``, ``baseline_comparison``.
        """
        return {
            "health": self.compute_health_score(),
            "root_causes": self.analyze_root_cause(),
            "trends": self.analyze_trends(),
            "active_anomalies": self.get_active_anomalies(),
            "recommendations": self.generate_recommendations(),
            "timeline": self.get_timeline(limit=30),
            "insights": self.generate_insights(),
            "baseline_comparison": self.compare_to_baseline(),
        }

    # ------------------------------------------------------------------
    # Properties (read-only access for api.py / main.py)
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        """``True`` if the model has been fitted at least once."""
        return self._is_trained

    @property
    def total_predictions(self) -> int:
        """Total number of ``predict_anomaly`` calls made so far."""
        return self._total_predictions

    @property
    def samples_until_retrain(self) -> int:
        """Samples remaining before the next scheduled retrain."""
        return max(0, self.cfg.retrain_interval - self._samples_since_retrain)

    # ------------------------------------------------------------------
    # Private helpers — original
    # ------------------------------------------------------------------
    def _engineer_features(
        self,
        metric: Dict[str, Any],
        processes: List[Dict[str, Any]],
        rolling_cpu: deque,
        rolling_ram: deque,
        rolling_net: deque,
    ) -> List[float]:
        """Compute the full feature vector for one monitoring sample."""
        cpu    = float(metric.get("cpu_percent",  0.0))
        ram    = float(metric.get("ram_percent",  0.0))
        disk   = float(metric.get("disk_percent", 0.0))
        sent   = float(metric.get("net_sent_mb",  0.0))
        recv   = float(metric.get("net_recv_mb",  0.0))
        net_tp = sent + recv

        cpu_ram_ratio = cpu / max(ram, 1.0)

        proc_cpus  = [float(p.get("cpu_percent",    0.0)) for p in processes]
        proc_mems  = [float(p.get("memory_percent", 0.0)) for p in processes]
        avg_proc_cpu = float(np.mean(proc_cpus))  if proc_cpus else 0.0
        avg_proc_mem = float(np.mean(proc_mems))  if proc_mems else 0.0
        total_procs  = float(len(processes))

        rolling_cpu.append(cpu)
        rolling_ram.append(ram)
        rolling_net.append(net_tp)

        roll_cpu_avg = float(np.mean(rolling_cpu)) if rolling_cpu else cpu
        roll_ram_avg = float(np.mean(rolling_ram)) if rolling_ram else ram
        roll_net_avg = float(np.mean(rolling_net)) if rolling_net else net_tp

        return [
            cpu, ram, disk, sent, recv, net_tp, cpu_ram_ratio,
            avg_proc_cpu, avg_proc_mem, total_procs,
            roll_cpu_avg, roll_ram_avg, roll_net_avg,
        ]

    @staticmethod
    def _impute(matrix: np.ndarray) -> np.ndarray:
        """Replace NaN and Inf values with column medians."""
        matrix = np.where(np.isinf(matrix), np.nan, matrix)
        col_medians = np.nanmedian(matrix, axis=0)
        nan_mask    = np.isnan(matrix)
        if nan_mask.any():
            matrix[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        return matrix

    @staticmethod
    def _score_to_decision(raw_score: float) -> Tuple[bool, float]:
        """Convert an Isolation Forest score to (is_anomaly, confidence)."""
        is_anomaly = raw_score < 0.0
        confidence = float(np.clip((0.5 - raw_score) * 100.0, 0.0, 100.0))
        return is_anomaly, confidence

    def _compute_severity(
        self,
        raw_score: float,
        features: np.ndarray,
        metric: Dict[str, Any],
    ) -> str:
        """Map anomaly score + feature context to a severity label."""
        if raw_score >= 0.0:
            return "NORMAL"

        abnormal_count = 0
        if len(self._rolling_cpu) >= 3:
            roll_mean_cpu = float(np.mean(self._rolling_cpu))
            if float(metric.get("cpu_percent", 0.0)) > roll_mean_cpu * 1.5:
                abnormal_count += 1
        if len(self._rolling_ram) >= 3:
            roll_mean_ram = float(np.mean(self._rolling_ram))
            if float(metric.get("ram_percent", 0.0)) > roll_mean_ram * 1.3:
                abnormal_count += 1
        if len(self._rolling_net) >= 3:
            roll_mean_net = float(np.mean(self._rolling_net))
            net_now = float(metric.get("net_sent_mb", 0.0)) + float(metric.get("net_recv_mb", 0.0))
            if net_now > roll_mean_net * 2.0:
                abnormal_count += 1

        score_abs = abs(raw_score)
        if score_abs >= 0.35 or abnormal_count >= 3:
            return "CRITICAL"
        if score_abs >= 0.25 or abnormal_count == 2:
            return "HIGH"
        if score_abs >= 0.15 or abnormal_count == 1:
            return "MEDIUM"
        return "LOW"

    def _build_reason(
        self,
        features: np.ndarray,
        metric: Dict[str, Any],
        is_anomaly: bool,
    ) -> str:
        """Construct a human-readable explanation of the prediction."""
        if not is_anomaly:
            return "All metrics within learned normal range."

        reasons: List[str] = []

        cpu = float(metric.get("cpu_percent", 0.0))
        ram = float(metric.get("ram_percent", 0.0))
        net = float(metric.get("net_sent_mb", 0.0)) + float(metric.get("net_recv_mb", 0.0))

        if len(self._rolling_cpu) >= 3:
            baseline_cpu = float(np.mean(self._rolling_cpu))
            if baseline_cpu > 0 and cpu > baseline_cpu * 1.5:
                pct = ((cpu - baseline_cpu) / baseline_cpu) * 100
                reasons.append(
                    f"CPU usage ({cpu:.1f}%) is {pct:.0f}% above its "
                    f"learned baseline ({baseline_cpu:.1f}%)."
                )

        if len(self._rolling_ram) >= 3:
            baseline_ram = float(np.mean(self._rolling_ram))
            if baseline_ram > 0 and ram > baseline_ram * 1.3:
                pct = ((ram - baseline_ram) / baseline_ram) * 100
                reasons.append(
                    f"RAM usage ({ram:.1f}%) is {pct:.0f}% above its "
                    f"learned baseline ({baseline_ram:.1f}%)."
                )

        if len(self._rolling_net) >= 3:
            baseline_net = float(np.mean(self._rolling_net))
            if baseline_net > 0 and net > baseline_net * 2.0:
                reasons.append(
                    f"Network throughput ({net:.3f} MB) is "
                    f"{(net / baseline_net):.1f}x the learned baseline "
                    f"({baseline_net:.3f} MB) — possible data exfiltration or spike."
                )

        if cpu > config.CPU_THRESHOLD:
            reasons.append(
                f"CPU usage ({cpu:.1f}%) exceeds the configured threshold "
                f"({config.CPU_THRESHOLD}%)."
            )
        if ram > config.RAM_THRESHOLD:
            reasons.append(
                f"RAM usage ({ram:.1f}%) exceeds the configured threshold "
                f"({config.RAM_THRESHOLD}%)."
            )

        if len(reasons) > 1:
            reasons.append(
                "Multiple abnormal metrics detected simultaneously — "
                "possible coordinated resource exhaustion or attack."
            )

        if not reasons:
            reasons.append(
                "Multivariate feature combination is inconsistent with "
                "learned system behavior (no single metric dominates)."
            )

        return " | ".join(reasons)

    def _schedule_retrain(self) -> None:
        """Trigger a background retraining cycle without blocking inference."""
        def _retrain() -> None:
            logger.info("Background retrain triggered after %d new samples.", self.cfg.retrain_interval)
            with config.data_lock:
                metrics_snap  = list(config.metrics_data)
                process_snap  = list(config.process_data)
            self.train_model(metrics_snap, process_snap)

        t = threading.Thread(target=_retrain, daemon=True, name="AIRetrainThread")
        t.start()

    # ------------------------------------------------------------------
    # NEW: shared pipeline helper for the three newest explainable engines
    # ------------------------------------------------------------------
    def _run_extended_ai_pipeline(
        self,
        metric: Dict[str, Any],
        processes: List[Dict[str, Any]],
        prediction: AnomalyPrediction,
    ) -> None:
        """Run Trend Analysis, Predictive Alerts, and Recommendations, in order.

        Called once per monitoring cycle from both branches of
        ``predict_anomaly`` (trained and not-yet-trained), after health
        score and (if applicable) Root Cause Analysis have already run
        this cycle. Each stage is independently guarded so a failure in
        one never blocks the others or the caller.

        Pipeline order: Trend Analysis -> Predictive Alerts (consumes
        trend output) -> Recommendations (consumes prediction, RCA,
        trend output, and health score).
        """
        trend_results: List[Dict[str, Any]] = []
        try:
            trend_results = self._trend_engine.analyze()
        except Exception:
            logger.exception("Trend Analysis failed for cycle at %s.", prediction.timestamp)

        try:
            self._predictive_engine.forecast(trend_results, metric)
        except Exception:
            logger.exception("Predictive Alert forecasting failed for cycle at %s.", prediction.timestamp)

        try:
            self._recommendation_engine.generate(prediction, metric, processes, trend_results)
        except Exception:
            logger.exception("Recommendation generation failed for cycle at %s.", prediction.timestamp)

    # ------------------------------------------------------------------
    # Private helpers — NEW dashboard intelligence
    # ------------------------------------------------------------------
    def _update_health_score(
        self,
        metric: Dict[str, Any],
        prediction: AnomalyPrediction,
        processes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Recompute the composite health score after each prediction.

        The health score is a weighted combination of:
            - Individual metric proximity to thresholds (60% weight)
            - Recent anomaly density (25% weight)
            - Trend stability (15% weight)
        """
        reasons: List[str] = []
        score = 100.0

        cpu = float(metric.get("cpu_percent", 0))
        ram = float(metric.get("ram_percent", 0))
        disk = float(metric.get("disk_percent", 0))
        net = float(metric.get("net_sent_mb", 0)) + float(metric.get("net_recv_mb", 0))

        # ── Metric-based penalties (60% weight) ──
        # CPU penalty
        if cpu > config.CPU_THRESHOLD:
            penalty = min(20.0, (cpu - config.CPU_THRESHOLD) * 0.8)
            score -= penalty
            duration_above = sum(
                1 for v in self._rolling_cpu if v > config.CPU_THRESHOLD
            )
            if duration_above > 0:
                dur_min = (duration_above * config.MONITOR_INTERVAL) / 60
                reasons.append(f"CPU remained above {config.CPU_THRESHOLD}% for ~{dur_min:.0f} minutes.")
        elif cpu > 70:
            score -= (cpu - 70) * 0.3

        # RAM penalty
        if ram > config.RAM_THRESHOLD:
            penalty = min(20.0, (ram - config.RAM_THRESHOLD) * 0.8)
            score -= penalty
            # Check for continuous increase
            if len(self._rolling_ram) >= 5:
                ram_list = list(self._rolling_ram)[-5:]
                if all(ram_list[i] <= ram_list[i + 1] for i in range(len(ram_list) - 1)):
                    increase = ram_list[-1] - ram_list[0]
                    reasons.append(f"RAM usage increased continuously by {increase:.0f}%.")
                else:
                    reasons.append(f"RAM usage at {ram:.1f}% exceeds threshold.")
        elif ram > 70:
            score -= (ram - 70) * 0.3

        # Disk penalty
        if disk > 90:
            score -= min(15.0, (disk - 90) * 1.5)
            reasons.append(f"Disk usage critically high at {disk:.1f}%.")
        elif disk > 80:
            score -= (disk - 80) * 0.3

        # Network penalty
        if net > config.NETWORK_THRESHOLD:
            score -= min(10.0, (net - config.NETWORK_THRESHOLD) * 0.1)
            reasons.append(f"Network throughput {net:.3f} MB exceeds threshold.")

        # ── Anomaly-based penalties (25% weight) ──
        active = self._active_anomalies[-10:]
        recent_critical = sum(1 for a in active if a.get("severity") == "CRITICAL")
        recent_high = sum(1 for a in active if a.get("severity") == "HIGH")
        recent_medium = sum(1 for a in active if a.get("severity") == "MEDIUM")

        if recent_critical > 0:
            score -= recent_critical * 8
            reasons.append(f"{recent_critical} critical-severity anomaly(ies) detected.")
        if recent_high > 0:
            score -= recent_high * 5
            reasons.append(f"{recent_high} high-severity anomaly(ies) detected.")
        if recent_medium > 0:
            score -= recent_medium * 2

        # ── Trend stability penalty (15% weight) ──
        trends = self.analyze_trends()
        for t in trends:
            if t.get("direction") == "increasing" and t.get("severity") in ("HIGH", "CRITICAL"):
                score -= 5
                if t.get("trend") not in [r for r in reasons]:
                    reasons.append(f"{t.get('metric', 'Metric')} trending upward rapidly.")

        # Clamp
        score = max(0.0, min(100.0, score))

        # Determine status
        if score >= 90:
            status = "Healthy"
        elif score >= 70:
            status = "Good"
        elif score >= 50:
            status = "Degraded"
        elif score >= 30:
            status = "Poor"
        else:
            status = "Critical"

        # Confidence in health assessment
        sample_count = len(config.metrics_data)
        if sample_count < 5:
            confidence = 40.0
        elif sample_count < 20:
            confidence = 60.0 + sample_count
        else:
            confidence = min(98.0, 80.0 + sample_count * 0.2)

        if not reasons:
            reasons.append("All metrics within normal operating parameters.")

        self._health_score = score
        self._health_status = status
        self._health_confidence = confidence
        self._health_reasons = reasons
        self._last_health_update = datetime.now().isoformat()

        # Timeline event for health changes
        if score < 50:
            self._add_timeline_event(
                "health", f"Health Score Drop: {score:.0f}",
                " | ".join(reasons[:3]),
                severity="HIGH" if score >= 30 else "CRITICAL",
                metadata={"score": score},
            )

    def _register_anomaly(
        self,
        prediction: AnomalyPrediction,
        metric: Dict[str, Any],
        processes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Register a detected anomaly into the active anomalies list and timeline."""
        # Find affected metric
        deviations = self._compute_deviations(metric)
        sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
        affected_metric = sorted_devs[0][0] if sorted_devs else "unknown"

        # Find responsible process
        responsible = "N/A"
        if processes:
            if "cpu" in affected_metric:
                sorted_p = sorted(processes, key=lambda p: p.get("cpu_percent", 0), reverse=True)
            else:
                sorted_p = sorted(processes, key=lambda p: p.get("memory_percent", 0), reverse=True)
            if sorted_p:
                responsible = sorted_p[0].get("name", "unknown")

        anomaly_entry = {
            "timestamp": prediction.timestamp,
            "severity": prediction.severity,
            "confidence": prediction.confidence,
            "anomaly_score": prediction.anomaly_score,
            "affected_metric": affected_metric.replace("_", " ").replace("percent", "").strip().upper(),
            "responsible_process": responsible,
            "reason": prediction.reason,
        }

        self._active_anomalies.append(anomaly_entry)
        # Keep only last 50
        if len(self._active_anomalies) > 50:
            self._active_anomalies = self._active_anomalies[-50:]

        self._add_timeline_event(
            "anomaly",
            f"Anomaly Detected ({prediction.severity})",
            prediction.reason[:120],
            severity=prediction.severity,
            metadata={"score": prediction.anomaly_score, "confidence": prediction.confidence},
        )

        # ── NEW: Explainable AI Root Cause Analysis ──
        # Runs strictly after the anomaly above has already been detected
        # and registered. Performs no detection of its own — it only
        # explains this prediction.
        try:
            rca_result = self._rca_engine.analyze(prediction, metric, processes or [])
            self._add_timeline_event(
                "root_cause",
                f"Root Cause: {rca_result['root_cause']}",
                rca_result["explanation"][:160],
                severity=rca_result["severity"],
                metadata={
                    "primary_metric": rca_result["primary_metric"],
                    "responsible_process": rca_result["responsible_process"].get("name"),
                    "confidence": rca_result["confidence"],
                },
            )
        except Exception:
            logger.exception("Root Cause Analysis failed for anomaly at %s.", prediction.timestamp)

    def _compute_deviations(self, metric: Dict[str, Any]) -> Dict[str, float]:
        """Compute how far each metric deviates from its baseline (%)."""
        deviations: Dict[str, float] = {}
        net_now = float(metric.get("net_sent_mb", 0)) + float(metric.get("net_recv_mb", 0))

        pairs = [
            ("cpu_percent", float(metric.get("cpu_percent", 0))),
            ("ram_percent", float(metric.get("ram_percent", 0))),
            ("disk_percent", float(metric.get("disk_percent", 0))),
            ("net_throughput_mb", net_now),
        ]

        for key, current in pairs:
            baseline = self._baseline.get(key, 0.0)
            if baseline > 0:
                deviations[key] = ((current - baseline) / baseline) * 100
            else:
                # Use rolling mean as fallback
                if key == "cpu_percent" and self._rolling_cpu:
                    bl = float(np.mean(self._rolling_cpu))
                    deviations[key] = ((current - bl) / max(bl, 1)) * 100
                elif key == "ram_percent" and self._rolling_ram:
                    bl = float(np.mean(self._rolling_ram))
                    deviations[key] = ((current - bl) / max(bl, 1)) * 100
                elif key == "net_throughput_mb" and self._rolling_net:
                    bl = float(np.mean(self._rolling_net))
                    deviations[key] = ((current - bl) / max(bl, 0.001)) * 100
                else:
                    deviations[key] = 0.0

        return deviations

    def _compute_baseline(self, metrics_list: List[Dict[str, Any]]) -> None:
        """Compute baseline averages from training data."""
        if not metrics_list:
            return

        cpus = [float(m.get("cpu_percent", 0)) for m in metrics_list]
        rams = [float(m.get("ram_percent", 0)) for m in metrics_list]
        disks = [float(m.get("disk_percent", 0)) for m in metrics_list]
        nets = [
            float(m.get("net_sent_mb", 0)) + float(m.get("net_recv_mb", 0))
            for m in metrics_list
        ]

        self._baseline = {
            "cpu_percent": float(np.mean(cpus)),
            "ram_percent": float(np.mean(rams)),
            "disk_percent": float(np.mean(disks)),
            "net_throughput_mb": float(np.mean(nets)),
        }
        self._baseline_computed = True
        logger.info("Baseline computed: %s", self._baseline)

    def _detect_trend(self, values: List[float], display_name: str) -> Dict[str, Any]:
        """Analyze a list of metric values and detect trends."""
        if len(values) < 3:
            return {
                "metric": display_name,
                "trend": "Insufficient data",
                "direction": "stable",
                "duration_samples": len(values),
                "confidence": 0.0,
                "severity": "NORMAL",
                "current_value": values[-1] if values else 0,
                "change_rate": 0.0,
            }

        arr = np.array(values, dtype=float)
        n = len(arr)

        # Simple linear regression for trend direction
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])

        # Calculate change rate per sample
        change_rate = slope

        # Determine direction
        if abs(slope) < 0.1:
            direction = "stable"
            trend_desc = f"{display_name} is stable"
        elif slope > 0:
            direction = "increasing"
            trend_desc = f"{display_name} increasing steadily"
        else:
            direction = "decreasing"
            trend_desc = f"{display_name} decreasing"

        # Check for spikes (high variance)
        std = float(np.std(arr))
        mean = float(np.mean(arr))
        if std > mean * 0.3 and mean > 5:
            trend_desc = f"{display_name}: repeated spikes detected"
            direction = "volatile"

        # Trend confidence based on R²
        if n >= 3:
            correlation = float(np.corrcoef(x, arr)[0, 1]) if np.std(arr) > 0 else 0
            confidence = min(99.0, abs(correlation) * 100)
        else:
            confidence = 30.0

        # Severity
        if direction == "increasing" and abs(slope) > 2.0:
            severity = "HIGH"
        elif direction == "increasing" and abs(slope) > 0.5:
            severity = "MEDIUM"
        elif direction == "volatile":
            severity = "MEDIUM"
        else:
            severity = "LOW" if direction != "stable" else "NORMAL"

        return {
            "metric": display_name,
            "trend": trend_desc,
            "direction": direction,
            "duration_samples": n,
            "confidence": round(confidence, 1),
            "severity": severity,
            "current_value": round(float(arr[-1]), 2),
            "change_rate": round(change_rate, 3),
        }

    def _determine_root_cause(
        self,
        primary_metric: str,
        deviation: float,
        metric: Dict[str, Any],
        process_name: str,
    ) -> Tuple[str, str]:
        """Return (root_cause_description, recommendation) for a metric."""
        cpu = float(metric.get("cpu_percent", 0))
        ram = float(metric.get("ram_percent", 0))

        if "cpu" in primary_metric:
            if cpu > 90:
                cause = f"Extreme CPU load ({cpu:.1f}%). Likely caused by '{process_name}' consuming excessive compute resources."
                rec = f"Terminate or restart '{process_name}'. Investigate for infinite loops or runaway computation."
            else:
                cause = f"Elevated CPU usage ({cpu:.1f}%), {abs(deviation):.0f}% above baseline."
                rec = f"Monitor '{process_name}' and consider reducing concurrent workloads."
        elif "ram" in primary_metric:
            # Check for steady increase (leak indicator)
            if len(self._rolling_ram) >= 5:
                ram_list = list(self._rolling_ram)[-5:]
                if all(ram_list[i] <= ram_list[i + 1] for i in range(len(ram_list) - 1)):
                    cause = f"Possible memory leak. RAM at {ram:.1f}% with continuous upward trend via '{process_name}'."
                    rec = f"Restart '{process_name}' and inspect memory allocation. Check for unclosed handles."
                else:
                    cause = f"High RAM usage ({ram:.1f}%), {abs(deviation):.0f}% above baseline."
                    rec = f"Close unused applications. Monitor '{process_name}' memory consumption."
            else:
                cause = f"RAM usage ({ram:.1f}%) deviating from baseline."
                rec = f"Monitor '{process_name}' and consider freeing memory."
        elif "disk" in primary_metric:
            disk = float(metric.get("disk_percent", 0))
            cause = f"Disk usage at {disk:.1f}% is growing. Possible log accumulation or data growth."
            rec = "Clear temporary files, rotate logs, or expand storage capacity."
        elif "net" in primary_metric:
            net = float(metric.get("net_sent_mb", 0)) + float(metric.get("net_recv_mb", 0))
            cause = f"Abnormal network activity ({net:.3f} MB/cycle). Possible data exfiltration or service misconfiguration."
            rec = f"Investigate '{process_name}' network connections. Check for unauthorized outbound traffic."
        else:
            cause = f"Multivariate anomaly — no single metric dominates. Combined behavior deviates from learned baseline."
            rec = "Review all system metrics and active processes for coordinated anomalies."

        return cause, rec

    def _add_timeline_event(
        self,
        event_type: str,
        title: str,
        description: str,
        severity: str = "NORMAL",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an event to the internal timeline."""
        # Throttle metric events (only keep every 3rd to avoid flooding)
        if event_type == "metric":
            metric_events = sum(1 for e in self._timeline if e.get("event_type") == "metric")
            if metric_events > 0 and metric_events % 3 != 0:
                return

        event = TimelineEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            title=title,
            description=description,
            severity=severity,
            metadata=metadata or {},
        )
        self._timeline.append(event.to_dict())