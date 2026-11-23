"""Additive columns for the grading ensemble (answers.verdict, feedback, grading_trace)
on databases created before they existed: db.ensure_additive_columns() adds them
idempotently, keeps existing rows, and emits DDL valid for SQLite and PostgreSQL."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, Table, create_engine, inspect, text
from sqlalchemy.dialects import postgresql, sqlite

from app import db as dbmod
from app.db import ADDITIVE_COLUMNS, Base, create_schema, ensure_additive_columns

NEW = set(ADDITIVE_COLUMNS["answers"])


def _old_schema_db(path: Path):  # noqa: ANN202
    """A SQLite file with every table, but `answers` as it was before the ensemble."""
    from app import models  # noqa: F401

    engine = create_engine(f"sqlite:///{path.as_posix()}")
    Base.metadata.create_all(engine, tables=[t for t in Base.metadata.sorted_tables if t.name != "answers"])
    old = MetaData()
    answers = Base.metadata.tables["answers"]
    Table("answers", old, *[Column(c.name, c.type, primary_key=c.primary_key, nullable=c.nullable)
                            for c in answers.columns if c.name not in NEW])
    old.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO answers (id, submission_id, question_id, ocr_text, text_corrected, "
                          "ocr_confidence, ai_marks, grading_confidence, segmentation_confidence, confidence, "
                          "evidence, flags, grader, status, updated_at) VALUES ('a1', 's1', 'q1', 'old text', 0, "
                          "0.9, 2.5, 0.6, 1.0, 0.5, '[]', '[]', 'lexical', 'needs_review', '2026-01-01 00:00:00')"))
    return engine


def _columns(engine) -> set[str]:  # noqa: ANN001
    return {c["name"] for c in inspect(engine).get_columns("answers")}


def test_columns_are_added_once_and_old_rows_survive(tmp_path):
    engine = _old_schema_db(tmp_path / "old.db")
    assert not NEW & _columns(engine)
    assert sorted(ensure_additive_columns(engine)) == sorted(f"answers.{c}" for c in NEW)
    assert NEW <= _columns(engine)
    assert ensure_additive_columns(engine) == []  # idempotent
    with engine.connect() as conn:
        row = conn.execute(text("SELECT ai_marks, grader, verdict, feedback, grading_trace FROM answers")).one()
    assert tuple(row) == (2.5, "lexical", None, None, None)
    engine.dispose()


def test_create_schema_upgrades_an_existing_database(tmp_path, monkeypatch):
    engine = _old_schema_db(tmp_path / "old.db")
    monkeypatch.setattr(dbmod, "_engine", engine)
    create_schema()
    create_schema()  # a second start changes nothing
    assert NEW <= _columns(engine)
    from sqlalchemy.orm import Session

    from app.models import Answer

    with Session(engine) as s:  # the ORM reads the old row and writes the new fields
        a = s.get(Answer, "a1")
        assert a.verdict is None and a.grading_trace is None
        a.verdict, a.grading_trace = "partially_correct", {"graders_used": ["lexical"]}
        s.commit()
    with Session(engine) as s:
        assert s.get(Answer, "a1").grading_trace == {"graders_used": ["lexical"]}
    engine.dispose()


@pytest.mark.parametrize(("dialect", "expected"), [
    (sqlite.dialect(), {"verdict": "TEXT", "feedback": "TEXT", "grading_trace": "JSON"}),
    (postgresql.dialect(), {"verdict": "TEXT", "feedback": "TEXT", "grading_trace": "JSON"}),
])
def test_additive_columns_are_nullable_and_portable(dialect, expected):
    from app import models  # noqa: F401

    table = Base.metadata.tables["answers"]
    for name in NEW:
        col = table.c[name]
        assert col.nullable and col.server_default is None and col.default is None
        assert col.type.compile(dialect=dialect) == expected[name]
