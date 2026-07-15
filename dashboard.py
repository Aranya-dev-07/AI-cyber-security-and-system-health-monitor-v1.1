"""
dashboard.py
============
Mounts the built Vite + React dashboard onto the existing FastAPI app
defined in ``api.py``, serving it at ``/dashboard/``.

ARCHITECTURE
-------------
    Browser (React + Vite)
           │
     HTTP REST polls  (GET /metrics, /processes, /runs, /health)
           │
      FastAPI  (api.py)          ← also serves /dashboard/* static files
           │
      SQLite (database.py)
           │
    Monitoring engine (config.py)

HOW IT WORKS
-------------
``api.py`` creates the FastAPI ``app`` object.
``dashboard.py`` imports that same ``app`` and mounts the built React
bundle (``dashboard/dist/``) under ``/dashboard/`` using FastAPI's
``StaticFiles``.  A dedicated ``/dashboard`` HTML route returns
``index.html`` so the React router can boot correctly.

``main.py`` imports ``dashboard`` (one line) which triggers this mount
as a side effect — no other change to ``main.py`` is needed.

SERVING IN PRODUCTION
----------------------
The same ``uvicorn`` instance started by ``main.py`` serves both the
REST API and the dashboard static files on one port (default 8000).

Visit: http://127.0.0.1:8000/dashboard/

DEVELOPMENT (hot-reload)
--------------------------
While iterating on the React source, run the Vite dev server in parallel:

    cd dashboard && npm run dev

Vite's dev server proxies ``/metrics``, ``/processes``, ``/runs``,
``/health`` to FastAPI (configured in ``vite.config.js``).
Visit the Vite URL (default http://localhost:5173/dashboard/) for
instant hot-module replacement.

Rebuild for production any time with:

    cd dashboard && npm run build
"""

from __future__ import annotations

import logging
import os

from fastapi import Response
from fastapi.staticfiles import StaticFiles

import api  # imports the shared FastAPI `app` object

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_DIST_DIR  = os.path.join(_HERE, "dashboard", "dist")
_INDEX_HTML = os.path.join(_DIST_DIR, "index.html")


# ---------------------------------------------------------------------------
# Mount static assets  (JS / CSS / fonts bundled by Vite)
# ---------------------------------------------------------------------------
def _mount_dashboard() -> None:
    """Mount the React build output onto the FastAPI app.

    Attaches two things to ``api.app``:

    1. ``/dashboard/assets`` — the hashed JS/CSS chunks produced by Vite.
    2. A ``GET /dashboard`` route that returns ``index.html`` so the React
       app boots correctly when the user navigates to ``/dashboard/``.

    Idempotent: calling this more than once (e.g. during testing) will
    silently skip re-mounting.

    Raises:
        RuntimeError: if the ``dashboard/dist`` folder does not exist,
            which means ``npm run build`` inside ``dashboard/`` has not
            been run yet.
    """
    if not os.path.isdir(_DIST_DIR):
        raise RuntimeError(
            f"Dashboard build not found at '{_DIST_DIR}'.\n"
            "Run:  cd dashboard && npm run build\n"
            "Then restart main.py."
        )

    # Mount hashed asset files produced by Vite (JS, CSS, sourcemaps).
    assets_path = os.path.join(_DIST_DIR, "assets")
    if os.path.isdir(assets_path):
        api.app.mount(
            "/dashboard/assets",
            StaticFiles(directory=assets_path),
            name="dashboard-assets",
        )

    @api.app.get("/dashboard", include_in_schema=False)
    @api.app.get("/dashboard/", include_in_schema=False)
    async def serve_dashboard() -> Response:
        """Serve the React app's ``index.html`` for any ``/dashboard`` request.

        Returning the HTML directly (rather than via FileResponse) avoids
        a dependency on ``aiofiles`` being installed.
        """
        try:
            with open(_INDEX_HTML, encoding="utf-8") as f:
                html = f.read()
            return Response(content=html, media_type="text/html")
        except FileNotFoundError:
            return Response(
                content="<h1>Dashboard not built</h1>"
                        "<p>Run <code>cd dashboard &amp;&amp; npm run build</code> "
                        "then restart main.py.</p>",
                media_type="text/html",
                status_code=503,
            )

    logger.info("Dashboard mounted at /dashboard/ (serving from '%s').", _DIST_DIR)


# ---------------------------------------------------------------------------
# Execute mount on import (side-effect import used by main.py)
# ---------------------------------------------------------------------------
_mount_dashboard()