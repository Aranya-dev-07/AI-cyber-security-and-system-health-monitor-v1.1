"""
ai_engine
=========
Package entry point for the AI Engine — a structural refactor of what
used to be a single ~3,500-line file ``ai_engine.py``. Behavior is 100%
unchanged; only the internal file organization changed, for
maintainability as the engine grew to six independent AI modules.

BACKWARD COMPATIBILITY (important)
------------------------------------
Every name that was previously importable from ``ai_engine.py`` is still
importable exactly the same way, with the exact same behavior:

    import ai_engine
    ai_engine.engine.predict_anomaly(metric, processes)
    ai_engine.get_latest_health_score()
    ai_engine.AnomalyDetectionEngine, ai_engine.AnomalyPrediction, ...

No other project file needs to change because of this refactor:
``main.py`` (``import ai_engine``), ``api.py``
(``import ai_engine as _ai_engine``), and ``database.py`` all continue
to work unmodified.

SUBMODULE MAP
--------------
    ai_engine/core.py             AnomalyDetectionEngine, AnomalyPrediction,
                                   ModelConfig, TimelineEvent, FEATURE_NAMES
    ai_engine/root_cause.py       RootCauseAnalysisEngine
    ai_engine/health_score.py     HealthScoreWeights, HealthScoreEngine
    ai_engine/trends.py           TrendAnalysisEngine
    ai_engine/predictive.py       PredictiveAlertEngine
    ai_engine/recommendations.py  RecommendationEngine

Each submodule only depends on ``config`` and (for type hints only, via
``TYPE_CHECKING``) on ``ai_engine.core`` — never on each other — so new
AI modules can be added the same way without touching existing files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ai_engine.core import (
    AnomalyDetectionEngine,
    AnomalyPrediction,
    ModelConfig,
    TimelineEvent,
    FEATURE_NAMES,
    MODELS_DIR,
    MODEL_PATH,
    SCALER_PATH,
)
from ai_engine.root_cause import RootCauseAnalysisEngine
from ai_engine.health_score import HealthScoreWeights, HealthScoreEngine
from ai_engine.trends import TrendAnalysisEngine
from ai_engine.predictive import PredictiveAlertEngine
from ai_engine.recommendations import RecommendationEngine

__all__ = [
    "AnomalyDetectionEngine", "AnomalyPrediction", "ModelConfig", "TimelineEvent",
    "FEATURE_NAMES", "MODELS_DIR", "MODEL_PATH", "SCALER_PATH",
    "RootCauseAnalysisEngine", "HealthScoreWeights", "HealthScoreEngine",
    "TrendAnalysisEngine", "PredictiveAlertEngine", "RecommendationEngine",
    "engine",
    "initialize_model", "prepare_training_data", "train_model", "predict_anomaly",
    "generate_ai_alert", "save_model", "load_model",
    "get_latest_root_cause_analysis", "get_root_cause_analysis_history",
    "get_latest_health_score", "get_health_score_history",
    "get_latest_trend_analysis", "get_trend_analysis_history",
    "get_latest_predictive_alerts", "get_predictive_alerts_history",
    "get_latest_smart_recommendations", "get_smart_recommendations_history",
]

# ---------------------------------------------------------------------------
# Module-level singleton (shared by main.py, api.py) — unchanged behavior
# ---------------------------------------------------------------------------
engine: AnomalyDetectionEngine = AnomalyDetectionEngine()


# ---------------------------------------------------------------------------
# Convenience functions (match the public API spec from the prompt)
# ---------------------------------------------------------------------------
def initialize_model() -> None:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.initialize_model`."""
    engine.initialize_model()


def prepare_training_data(
    metrics_list: List[Dict[str, Any]],
    process_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> Optional[np.ndarray]:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.prepare_training_data`."""
    return engine.prepare_training_data(metrics_list, process_snapshots)


def train_model(
    metrics_list: List[Dict[str, Any]],
    process_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.train_model`."""
    return engine.train_model(metrics_list, process_snapshots)


def predict_anomaly(
    metric: Dict[str, Any],
    processes: Optional[List[Dict[str, Any]]] = None,
) -> AnomalyPrediction:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.predict_anomaly`."""
    return engine.predict_anomaly(metric, processes)


def generate_ai_alert(prediction: AnomalyPrediction) -> None:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.generate_ai_alert`."""
    engine.generate_ai_alert(prediction)


def save_model() -> bool:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.save_model`."""
    return engine.save_model()


def load_model() -> bool:
    """Module-level convenience wrapper for :meth:`AnomalyDetectionEngine.load_model`."""
    return engine.load_model()


def get_latest_root_cause_analysis() -> Optional[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_latest_root_cause_analysis`.

    Intended for a future ``GET /ai/root-cause`` endpoint in api.py —
    see the field list documented on :meth:`RootCauseAnalysisEngine.analyze`.
    """
    return engine.get_latest_root_cause_analysis()


def get_root_cause_analysis_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_root_cause_analysis_history`.

    Intended for a future ``GET /ai/root-cause/history`` endpoint in api.py.
    """
    return engine.get_root_cause_analysis_history(limit=limit)


def get_latest_health_score() -> Optional[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_latest_health_score`.

    Intended for a future ``GET /ai/health-score`` endpoint in api.py.
    """
    return engine.get_latest_health_score()


def get_health_score_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_health_score_history`.

    Intended for a future ``GET /ai/health-score/history`` endpoint in api.py.
    """
    return engine.get_health_score_history(limit=limit)


def get_latest_trend_analysis() -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_latest_trend_analysis`.

    Intended for a future ``GET /ai/trend-analysis`` endpoint in api.py.
    """
    return engine.get_latest_trend_analysis()


def get_trend_analysis_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_trend_analysis_history`.

    Intended for a future ``GET /ai/trend-analysis/history`` endpoint in api.py.
    """
    return engine.get_trend_analysis_history(limit=limit)


def get_latest_predictive_alerts() -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_latest_predictive_alerts`.

    Intended for a future ``GET /ai/predictive-alerts`` endpoint in api.py.
    """
    return engine.get_latest_predictive_alerts()


def get_predictive_alerts_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_predictive_alerts_history`.

    Intended for a future ``GET /ai/predictive-alerts/history`` endpoint in api.py.
    """
    return engine.get_predictive_alerts_history(limit=limit)


def get_latest_smart_recommendations() -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_latest_smart_recommendations`.

    Intended for a future ``GET /ai/smart-recommendations`` endpoint in api.py.
    """
    return engine.get_latest_smart_recommendations()


def get_smart_recommendations_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Module-level convenience wrapper for
    :meth:`AnomalyDetectionEngine.get_smart_recommendations_history`.

    Intended for a future ``GET /ai/smart-recommendations/history`` endpoint in api.py.
    """
    return engine.get_smart_recommendations_history(limit=limit)