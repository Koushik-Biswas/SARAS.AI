"""Database engine and session handling. SQLite on the 4 GB laptop, PostgreSQL where
RAM allows — same schema, switched only by SARAS_DATABASE_URL."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
SessionLocal = sessionmaker(autoflush=False, expire_on_commit=False)

# Nullable columns added to existing tables after their first release. create_all()
# never alters an existing table, so ensure_additive_columns() adds these to databases
# created earlier (until Alembic migrations exist, spec O-2). Only nullable columns
# without defaults belong here: adding them is safe on SQLite and PostgreSQL.
ADDITIVE_COLUMNS: dict[str, tuple[str, ...]] = {
    "answers": ("verdict", "feedback", "grading_trace"),
    "submissions": ("progress",),
}


def init_engine(url: str) -> Engine:
    global _engine
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    _engine = create_engine(url, **kwargs)

    if url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()

    SessionLocal.configure(bind=_engine)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine not initialised")
    return _engine


def ensure_additive_columns(engine: Engine | None = None) -> list[str]:
    """Add the ADDITIVE_COLUMNS missing from existing tables; returns "table.column" for
    each one added. Idempotent (a second run adds nothing) and safe when two processes
    start at once (a column another process just added is accepted)."""
    from . import models  # noqa: F401  (register tables)

    engine = engine or get_engine()
    quote = engine.dialect.identifier_preparer.quote
    added: list[str] = []
    for table_name, columns in ADDITIVE_COLUMNS.items():
        insp = inspect(engine)
        if not insp.has_table(table_name):
            continue  # create_all() makes it with every column
        existing = {c["name"] for c in insp.get_columns(table_name)}
        table = Base.metadata.tables[table_name]
        for name in columns:
            if name in existing:
                continue
            column = table.c[name]
            if not column.nullable or column.server_default is not None:
                raise RuntimeError(f"{table_name}.{name}: only nullable columns without defaults can be added here")
            ddl = (f"ALTER TABLE {quote(table_name)} ADD COLUMN {quote(name)} "
                   f"{column.type.compile(dialect=engine.dialect)}")
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
            except SQLAlchemyError:
                if name not in {c["name"] for c in inspect(engine).get_columns(table_name)}:
                    raise
                continue  # added concurrently by another process
            log.info("Added column %s.%s", table_name, name)
            added.append(f"{table_name}.{name}")
    return added


def create_schema() -> None:
    from . import models  # noqa: F401  (register tables)

    engine = get_engine()
    Base.metadata.create_all(engine)
    ensure_additive_columns(engine)
    if engine.dialect.name == "sqlite":
        # The audit log is append-only (PRD 15.1). PostgreSQL enforces this with role
        # privileges; SQLite has no roles, so triggers do the same job.
        with engine.begin() as conn:
            conn.execute(text(
                "CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log "
                "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;"
            ))
            conn.execute(text(
                "CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log "
                "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;"
            ))


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
