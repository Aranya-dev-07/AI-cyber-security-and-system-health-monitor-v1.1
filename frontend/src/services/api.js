import axios from "axios";

/**
 * api.js
 * ======
 * Centralized API communication layer for Lavender Trinetra. Every
 * network call the frontend makes to the FastAPI backend goes through
 * this module — no component talks to axios or fetch directly.
 *
 * Responsibilities:
 *   - Own the single Axios instance (base URL, timeout, headers).
 *   - Normalize every response to its payload (`response.data`) so
 *     callers can write `const data = await getX()` directly.
 *   - Normalize every failure to a plain Error with a readable
 *     `.message`, a `.status`, and the original `.cause`, so callers
 *     can do `catch (err) { toast.error(err.message) }` uniformly.
 *   - Expose one small, well-named function per backend capability.
 *
 * This module intentionally contains NO business logic — no scoring,
 * no threshold evaluation, no data transformation beyond generic
 * envelope unwrapping. It is a thin, typed-by-convention transport
 * layer between the React app and backend/api/routes.py.
 */

// ---------------------------------------------------------------------------
// Base configuration
// ---------------------------------------------------------------------------

/**
 * Backend origin. In development, Vite's dev server proxies `/api` to
 * `VITE_BACKEND_URL` (see vite.config), so leaving this blank keeps
 * requests relative and proxy-friendly. In production, set
 * VITE_BACKEND_URL to the deployed backend's origin (e.g.
 * "https://api.lavender-trinetra.example.com").
 */
const BACKEND_ORIGIN = (import.meta.env?.VITE_BACKEND_URL || "").replace(/\/+$/, "");

/** All backend routes are mounted under this prefix (backend/api/routes.py). */
const API_PREFIX = "/api";

const API_BASE_URL = `${BACKEND_ORIGIN}${API_PREFIX}`;

const REQUEST_TIMEOUT_MS = 15000;

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
});

// ---------------------------------------------------------------------------
// Request interceptor
// ---------------------------------------------------------------------------
// Attaches request timing (for logging) and a bearer token when one is
// present, without requiring auth to exist yet. Future-ready for when
// Lavender Trinetra adds authenticated sessions.
apiClient.interceptors.request.use(
  (config) => {
    config.metadata = { startedAt: Date.now() };

    const token = typeof window !== "undefined" ? window.localStorage?.getItem("lt_auth_token") : null;
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }

    if (import.meta.env?.DEV) {
      // eslint-disable-next-line no-console
      console.debug(`[api] → ${config.method?.toUpperCase()} ${config.url}`);
    }

    return config;
  },
  (error) => Promise.reject(normalizeError(error))
);

// ---------------------------------------------------------------------------
// Response interceptor
// ---------------------------------------------------------------------------
// Unwraps `response.data` so every exported function returns the
// payload directly, and normalizes all failures to a single shape.
apiClient.interceptors.response.use(
  (response) => {
    if (import.meta.env?.DEV) {
      const elapsed = Date.now() - (response.config.metadata?.startedAt ?? Date.now());
      // eslint-disable-next-line no-console
      console.debug(
        `[api] ← ${response.config.method?.toUpperCase()} ${response.config.url} (${response.status}, ${elapsed}ms)`
      );
    }
    return response.data;
  },
  (error) => Promise.reject(normalizeError(error))
);

/**
 * normalizeError — converts any Axios/network failure into a plain
 * Error with a human-readable `.message`, the HTTP `.status` (if any),
 * and the raw error preserved as `.cause`. Centralizes error handling
 * so every caller can rely on `err.message` regardless of whether the
 * failure was a network error, a timeout, or a backend 4xx/5xx.
 */
function normalizeError(error) {
  if (axios.isCancel?.(error)) {
    const cancelError = new Error("Request was cancelled.");
    cancelError.status = null;
    cancelError.cause = error;
    return cancelError;
  }

  const status = error?.response?.status ?? null;
  const backendDetail = error?.response?.data?.detail;

  let message;
  if (backendDetail) {
    message = typeof backendDetail === "string" ? backendDetail : JSON.stringify(backendDetail);
  } else if (error?.code === "ECONNABORTED") {
    message = "The request timed out. Please try again.";
  } else if (error?.message === "Network Error") {
    message = "Unable to reach the Lavender Trinetra backend. Please check your connection.";
  } else if (status) {
    message = `Request failed with status ${status}.`;
  } else {
    message = error?.message || "An unexpected error occurred.";
  }

  if (import.meta.env?.DEV) {
    // eslint-disable-next-line no-console
    console.error(`[api] ✗ ${error?.config?.method?.toUpperCase() || ""} ${error?.config?.url || ""}`, error);
  }

  const normalized = new Error(message);
  normalized.status = status;
  normalized.cause = error;
  return normalized;
}

// ---------------------------------------------------------------------------
// Generic HTTP verb helpers
// ---------------------------------------------------------------------------
// Thin wrappers kept for reuse/extension. All domain functions below
// are built on top of these rather than calling apiClient directly.

const get = (url, config) => apiClient.get(url, config);
const post = (url, body, config) => apiClient.post(url, body, config);
const put = (url, body, config) => apiClient.put(url, body, config);
const del = (url, config) => apiClient.delete(url, config);

// ---------------------------------------------------------------------------
// System Status
// ---------------------------------------------------------------------------

/** GET /api/status — overall API/AI/monitoring/database status. */
export function getSystemStatus() {
  return get("/status");
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

/**
 * GET /api/dashboard/statistics — aggregate PostgreSQL statistics
 * (total runs, metric/process sample counts, alert totals, latest
 * run/health snapshot). Backing endpoint pending backend
 * implementation on top of database/crud.py::get_dashboard_statistics.
 */
export function getDashboardStatistics() {
  return get("/dashboard/statistics");
}

// ---------------------------------------------------------------------------
// Monitoring
// ---------------------------------------------------------------------------

/** GET /api/monitoring/metrics — latest live system metrics snapshot. */
export function getLatestMetrics(params = {}) {
  return get("/monitoring/metrics", { params });
}

/** GET /api/monitoring/processes — top running processes (supports `limit`). */
export function getLatestProcesses(params = {}) {
  return get("/monitoring/processes", { params });
}

/**
 * POST /api/monitoring/control/start — begin a monitoring session.
 * Backing endpoint pending backend implementation.
 */
export function startMonitoring() {
  return post("/monitoring/control/start");
}

/**
 * POST /api/monitoring/control/stop — end the current monitoring
 * session and generate its report. Backing endpoint pending backend
 * implementation.
 */
export function stopMonitoring() {
  return post("/monitoring/control/stop");
}

/**
 * POST /api/monitoring/control/reset — clear in-memory alert counters
 * for the current session. Backing endpoint pending backend
 * implementation.
 */
export function resetMonitoringSession() {
  return post("/monitoring/control/reset");
}

/**
 * POST /api/monitoring/control/refresh — force an immediate metrics
 * collection cycle. Backing endpoint pending backend implementation.
 */
export function refreshMetrics() {
  return post("/monitoring/control/refresh");
}

// ---------------------------------------------------------------------------
// AI Workspace
// ---------------------------------------------------------------------------

/** GET /api/ai/health-score — current AI-computed system health score. */
export function getHealthScore() {
  return get("/ai/health-score");
}

/** GET /api/ai/root-cause — latest root cause analysis. */
export function getRootCause() {
  return get("/ai/root-cause");
}

/** GET /api/ai/recommendations — current AI recommendations. */
export function getRecommendations() {
  return get("/ai/recommendations");
}

/** GET /api/ai/trends — trend analysis series and summary (supports `window`). */
export function getTrendAnalysis(params = {}) {
  return get("/ai/trends", { params });
}

/** GET /api/ai/predictive-alerts — current predictive alerts. */
export function getPredictiveAlerts() {
  return get("/ai/predictive-alerts");
}

/** GET /api/ai/anomalies — current anomaly detection results. */
export function getAnomalies() {
  return get("/ai/anomalies");
}

/**
 * GET /api/ai/results/latest — the most recent combined AI analysis
 * cycle (health, anomalies, trends, predictions, recommendations).
 * Backing endpoint pending backend implementation on top of
 * ai/ai_engine.py.
 */
export function getLatestAIResult() {
  return get("/ai/results/latest");
}

/**
 * GET /api/ai/results — historical AI analysis cycles (supports
 * `limit`). Backing endpoint pending backend implementation.
 */
export function getAIResults(params = {}) {
  return get("/ai/results", { params });
}

// ---------------------------------------------------------------------------
// Cybersecurity
// ---------------------------------------------------------------------------

/** GET /api/cybersecurity/score — overall security score. */
export function getSecurityScore() {
  return get("/cybersecurity/score");
}

/**
 * GET /api/cybersecurity/threats — active detected threats. Backing
 * endpoint pending backend implementation on top of
 * cybersecurity/threat_detector.py.
 */
export function getThreats() {
  return get("/cybersecurity/threats");
}

/**
 * GET /api/cybersecurity/firewall — firewall status. Backing endpoint
 * pending backend implementation on top of
 * cybersecurity/firewall_monitor.py.
 */
export function getFirewallStatus() {
  return get("/cybersecurity/firewall");
}

/**
 * GET /api/cybersecurity/ports — open port scan results. Backing
 * endpoint pending backend implementation on top of
 * cybersecurity/port_scanner.py.
 */
export function getPortScanResults() {
  return get("/cybersecurity/ports");
}

/**
 * GET /api/cybersecurity/intrusion — intrusion detection events.
 * Backing endpoint pending backend implementation on top of
 * cybersecurity/intrusion_detector.py.
 */
export function getIntrusionEvents() {
  return get("/cybersecurity/intrusion");
}

/**
 * GET /api/cybersecurity/vulnerabilities — vulnerability scan
 * results. Backing endpoint pending backend implementation on top of
 * cybersecurity/vulnerability_scan.py.
 */
export function getVulnerabilities() {
  return get("/cybersecurity/vulnerabilities");
}

// ---------------------------------------------------------------------------
// Reports / Test Runs
// ---------------------------------------------------------------------------

/** GET /api/reports — all stored session reports / test runs (supports `limit`). */
export function getReports(params = {}) {
  return get("/reports", { params });
}

/** GET /api/reports/{id} — a single report's full detail. */
export function getReportById(reportId) {
  return get(`/reports/${reportId}`);
}

/**
 * DELETE /api/reports/{id} — remove a stored report. Backing endpoint
 * pending backend implementation.
 */
export function deleteReport(reportId) {
  return del(`/reports/${reportId}`);
}

/**
 * GET /api/reports/export — export report data as a downloadable
 * blob. `category` is one of "monitoring" | "ai" | "cybersecurity" |
 * "test_runs"; `format` is "csv" | "json" | "pdf". Backing endpoint
 * pending backend implementation.
 */
export function exportReport(category, format) {
  return get("/reports/export", {
    params: { category, format },
    responseType: "blob",
  });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

/**
 * GET /api/settings/alert-policy — current alert threshold/severity/
 * frequency configuration. Backing endpoint pending backend
 * implementation on top of config.py's CPU_THRESHOLD, RAM_THRESHOLD,
 * NETWORK_THRESHOLD, etc.
 */
export function getAlertPolicy() {
  return get("/settings/alert-policy");
}

/** PUT /api/settings/alert-policy — persist alert policy changes. */
export function updateAlertPolicy(policy) {
  return put("/settings/alert-policy", policy);
}

/**
 * GET /api/settings/preferences — current user preferences
 * (monitoring interval, refresh behavior, landing page, notifications,
 * time format). Backing endpoint pending backend implementation.
 */
export function getPreferences() {
  return get("/settings/preferences");
}

/** PUT /api/settings/preferences — persist preference changes. */
export function updatePreferences(preferences) {
  return put("/settings/preferences", preferences);
}

/**
 * GET /api/settings/appearance — current workspace appearance
 * configuration (theme mode, sidebar width, card density, animation
 * toggle, font size, accent color). Backing endpoint pending backend
 * implementation.
 */
export function getAppearanceSettings() {
  return get("/settings/appearance");
}

/** PUT /api/settings/appearance — persist appearance changes. */
export function updateAppearanceSettings(settings) {
  return put("/settings/appearance", settings);
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

export { apiClient, API_BASE_URL };

export default {
  getSystemStatus,
  getDashboardStatistics,
  getLatestMetrics,
  getLatestProcesses,
  startMonitoring,
  stopMonitoring,
  resetMonitoringSession,
  refreshMetrics,
  getHealthScore,
  getRootCause,
  getRecommendations,
  getTrendAnalysis,
  getPredictiveAlerts,
  getAnomalies,
  getLatestAIResult,
  getAIResults,
  getSecurityScore,
  getThreats,
  getFirewallStatus,
  getPortScanResults,
  getIntrusionEvents,
  getVulnerabilities,
  getReports,
  getReportById,
  deleteReport,
  exportReport,
  getAlertPolicy,
  updateAlertPolicy,
  getPreferences,
  updatePreferences,
  getAppearanceSettings,
  updateAppearanceSettings,
};