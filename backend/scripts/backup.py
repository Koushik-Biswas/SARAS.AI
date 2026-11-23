"""Back up SARAS AI data: the database and the uploaded answer sheets / page images.

    python -m scripts.backup                       # -> backend/backups/saras-<UTC time>.zip
    python -m scripts.backup --out D:/backups --keep 14

SQLite (the default profile): a consistent ONLINE copy made with SQLite's backup API, so
the app can keep running. The zip holds `saras.db` and `storage/` (every submission's
original PDF and rendered pages). PostgreSQL (the Docker extended profile): the zip
holds `storage/` only; dump the database separately, e.g.

    docker compose exec db pg_dump -U saras -Fc saras > saras-$(date +%F).dump

Restore (SQLite): stop the app, put saras.db at SARAS_DATABASE_URL's path and unzip
storage/ into SARAS_STORAGE_DIR, start the app.

Backups contain student work: keep them private (backend/backups/ is git-ignored).
`--keep N` deletes all but the newest N backups in the output folder.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


def sqlite_path(database_url: str) -> Path | None:
    if database_url.startswith("sqlite:///"):
        return Path(database_url.removeprefix("sqlite:///"))
    return None


def backup(out_dir: Path, keep: int | None = None) -> Path:
    from app.config import get_settings

    s = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = out_dir / f"saras-{stamp}.zip"
    db_file = sqlite_path(s.database_url)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        if db_file is not None and db_file.exists():
            copy = Path(tmp) / "saras.db"
            src = sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)
            dst = sqlite3.connect(copy)
            try:
                src.backup(dst)  # consistent snapshot, works while the app writes (WAL)
            finally:
                dst.close()
                src.close()
            zf.write(copy, "saras.db")
        storage = Path(s.storage_dir)
        if storage.exists():
            for f in sorted(storage.rglob("*")):
                if f.is_file():
                    zf.write(f, Path("storage") / f.relative_to(storage))
    if keep:
        old = sorted(out_dir.glob("saras-*.zip"))[:-keep]
        for f in old:
            f.unlink()
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Back up the SARAS AI database and uploaded sheets.")
    ap.add_argument("--out", type=Path, default=BACKEND / "backups")
    ap.add_argument("--keep", type=int, default=None, help="keep only the newest N backups")
    args = ap.parse_args(argv)
    path = backup(args.out, args.keep)
    print(f"Backup written: {path} ({path.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
