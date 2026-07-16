"""
database.py

SQLite Database Connection Manager — Lavender Trinetra Platform
=====================================================================

Owns the SQLAlchemy engine, session factory, and declarative Base for
the platform's SQLite persistence layer. Responsible for creating all
tables on application startup and providing reusable, safely-scoped
database sessions to the rest of the backend.

Integrates with:
    - database/models.py   (declares ORM models against Base)
    - database/crud.py     (uses get_db() / SessionLocal for queries)
    - api/api.py            (wires get_db() as a FastAPI dependency)
    - main.py               (calls init_db() on application startup)

Author: Lavender Trinetra Backend Engineering
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("lavender_trinetra.database")
logger.addHandler(logging.NullHandler())


# =====================================================================
# CONFIGURATION
# =====================================================================

_DB_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
)
_DEFAULT_DB_PATH = os.path.join(_DB_DIR, "lavender_trinetra.db")

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", f"sqlite:///{_DEFAULT_DB_PATH}"
)

# SQLite requires this flag when the connection is shared across threads
# (e.g. FastAPI's threaded request handling, or background monitoring
# threads writing while the API reads).
_CONNECT_ARGS = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}


# =====================================================================
# ENGINE
# =====================================================================

def _build_engine(database_url: str = DATABASE_URL) -> Engine:
    """
    Construct the SQLAlchemy engine for the given database URL.

    Args:
        database_url: SQLAlchemy-compatible connection string.

    Returns:
        A configured Engine instance.

    Raises:
        SQLAlchemyError: if engine creation fails.
    """
    try:
        os.makedirs(_DB_DIR, exist_ok=True)

        engine = create_engine(
            database_url,
            connect_args=_CONNECT_ARGS,
            pool_pre_ping=True,
            future=True,
        )

        logger.info("Database engine created for %s", database_url)
        return engine

    except SQLAlchemyError as exc:
        logger.exception("Failed to create database engine: %s", exc)
        raise


engine: Engine = _build_engine()


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    """Enable SQLite foreign key enforcement on every new connection."""
    if DATABASE_URL.startswith("sqlite"):
        try:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to enable SQLite foreign_keys pragma: %s", exc)


# =====================================================================
# SESSION FACTORY
# =====================================================================

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)


# =====================================================================
# DECLARATIVE BASE
# =====================================================================

Base = declarative_base()


# =====================================================================
# TABLE INITIALIZATION
# =====================================================================

def init_db() -> None:
    """
    Create all tables registered against Base.metadata that do not yet
    exist. Intended to be called once during application startup
    (main.py), after models.py has been imported so its model classes
    are registered with Base.

    Raises:
        SQLAlchemyError: re-raised after logging if table creation fails.
    """
    try:
        # Import models here (not at module top) to avoid a circular
        # import between database.py and models.py, while still
        # guaranteeing all models are registered before create_all().
        from . import models  # noqa: F401
    except ImportError:
        try:
            import models  # type: ignore # noqa: F401
        except ImportError:
            logger.warning("models.py not importable during init_db(); "
                            "tables will only be created for already-imported models.")

    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully at %s", DATABASE_URL)
    except SQLAlchemyError as exc:
        logger.exception("Failed to initialize database tables: %s", exc)
        raise


def drop_all_tables() -> None:
    """
    Drop all tables registered against Base.metadata. Intended for use
    in test/reset scenarios only — never called automatically.

    Raises:
        SQLAlchemyError: re-raised after logging if the drop fails.
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.warning("All database tables dropped for %s", DATABASE_URL)
    except SQLAlchemyError as exc:
        logger.exception("Failed to drop database tables: %s", exc)
        raise


# =====================================================================
# SESSION ACCESSORS
# =====================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI-style dependency that yields a database session and
    guarantees it is closed afterward, even on error.

    Usage (api/api.py):
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...

    Yields:
        An active SQLAlchemy Session.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as exc:
        logger.exception("Database session error, rolling back: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Context-manager form of a database session for use outside FastAPI's
    dependency injection (e.g. crud.py helper functions, background
    monitoring threads, scripts). Commits on success, rolls back and
    re-raises on error, and always closes the session.

    Usage:
        with session_scope() as db:
            db.add(record)
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as exc:
        logger.exception("Database transaction failed, rolling back: %s", exc)
        db.rollback()
        raise
    finally:
        db.close()


def get_session() -> Session:
    """
    Return a new, unmanaged SQLAlchemy Session. Caller is responsible
    for closing it (and committing/rolling back as needed). Prefer
    get_db() or session_scope() where possible; this is provided for
    integrations that need direct session control.
    """
    return SessionLocal()


# =====================================================================
# HEALTH CHECK
# =====================================================================

def check_connection() -> bool:
    """
    Verify the database is reachable by executing a trivial query.
    Useful for startup diagnostics and health-check API endpoints.

    Returns:
        True if the connection succeeds, False otherwise.
    """
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return True
    except SQLAlchemyError as exc:
        logger.error("Database connection check failed: %s", exc)
        return False


__all__ = [
    "DATABASE_URL",
    "engine",
    "SessionLocal",
    "Base",
    "init_db",
    "drop_all_tables",
    "get_db",
    "session_scope",
    "get_session",
    "check_connection",
]