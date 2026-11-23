"""scripts/seed_demo.py: seeds the database named by the environment through the
real API, processes every sheet, and is idempotent."""

from __future__ import annotations

import pytest

from scripts import make_samples, seed_demo


def test_seed_demo_is_idempotent(tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models import Exam, Submission, User

    monkeypatch.setenv("SARAS_ENV", "test")
    monkeypatch.setenv("SARAS_DATABASE_URL", f"sqlite:///{(tmp_path / 'seed.db').as_posix()}")
    monkeypatch.setenv("SARAS_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("SARAS_FRONTEND_DIST", str(tmp_path / "no-frontend"))
    monkeypatch.setenv("SARAS_GRADER", "lexical")
    samples_dir = tmp_path / "samples"  # generated on demand

    first = seed_demo.seed(samples_dir)
    n_sheets = len(make_samples.TYPED_SHEETS) + 1
    assert first["uploaded"] == n_sheets and first["rejected"] == [] and first["processed"] == n_sheets
    assert (samples_dir / "rubric.json").is_file()

    second = seed_demo.seed(samples_dir)
    assert second["skipped"] is True and second["exam_id"] == first["exam_id"] and second["processed"] == 0

    from app.config import Settings
    from app.db import init_engine

    engine = init_engine(Settings().database_url)
    try:
        with SessionLocal() as db:
            assert db.query(User).filter_by(email=seed_demo.DEMO_EMAIL).count() == 1
            assert db.query(Exam).count() == 1
            subs = db.query(Submission).all()
            assert len(subs) == n_sheets and {s.status for s in subs} == {"processed"}
    finally:
        engine.dispose()


def test_seed_demo_refuses_production(tmp_path, monkeypatch):
    monkeypatch.setenv("SARAS_ENV", "prod")
    monkeypatch.setenv("SARAS_JWT_SECRET", "x" * 40)
    monkeypatch.setenv("SARAS_DATABASE_URL", f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    monkeypatch.setenv("SARAS_STORAGE_DIR", str(tmp_path / "storage"))
    with pytest.raises(seed_demo.SeedError, match="development only"):
        seed_demo.seed(tmp_path / "samples")
