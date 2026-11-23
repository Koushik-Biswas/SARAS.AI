"""Download the local NLI model used by the ONNX grader (no PyTorch needed).

    cd backend && .venv/Scripts/python -m scripts.download_models
    cd backend && .venv/Scripts/python -m scripts.download_models --preset minilm   (alternative, for comparison)

Default model: cross-encoder/nli-deberta-v3-xsmall (Apache-2.0; trained on SNLI +
MultiNLI; SNLI-test 91.64 %, MNLI-mismatched 87.77 % as published by the authors),
using the dynamic-int8 ONNX export the authors ship (onnx/model_quint8_avx2.onnx,
87 MB). The u8/u8 variant is used because it runs on any x86-64 (AVX2) and ARM CPU
and avoids the u8s8 saturation issue of AVX2 CPUs without VNNI.

Writes backend/models/nli/<preset>/ (``default`` for the default preset):

    model.onnx       the int8 cross-encoder
    tokenizer.json   HF fast tokenizer (read with the ``tokenizers`` package)
    config.json      model config (id2label gives the label order)
    manifest.json    repo, pinned commit, license, per-file size + SHA-256

Behaviour:
* Pinned: files come from a fixed commit (``/resolve/<sha>/...``), not from ``main``.
* Licence-checked: the model card licence at the pinned commit must equal the preset's
  (Apache-2.0 for both presets) and is recorded in manifest.json.
* Verified: every file is checked against the size and hash Hugging Face reports
  for that commit (LFS SHA-256, or the git blob SHA-1 for small files) and, for the
  default preset, against SHA-256 values pinned below.
* Idempotent: a re-run re-hashes the files on disk and only downloads what is
  missing or corrupt. If everything matches the manifest no network is used.
* Resumable: partial downloads are kept as ``*.part`` and continued with HTTP Range.

Only ``httpx`` is used (no huggingface_hub). The model is never fetched at runtime;
the grader only reads these files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parent.parent
MODELS_DIR = BACKEND / "models" / "nli"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CHUNK = 1 << 20  # 1 MiB
RETRIES = 4


@dataclass(frozen=True)
class Preset:
    repo: str
    revision: str  # full commit SHA: downloads are reproducible
    license: str
    files: dict[str, str]  # local name -> path in the repo
    sha256: dict[str, str] = field(default_factory=dict)  # local name -> pinned SHA-256 (optional)
    notes: str = ""


PRESETS: dict[str, Preset] = {
    "default": Preset(
        repo="cross-encoder/nli-deberta-v3-xsmall",
        revision="a150876415327c80daeff35ca6f68f5ed8cf5c24",
        license="apache-2.0",
        files={"model.onnx": "onnx/model_quint8_avx2.onnx", "tokenizer.json": "tokenizer.json",
               "config.json": "config.json"},
        # Verified on download against the Hub (LFS SHA-256 / git blob ids) on 2026-11-23.
        sha256={"model.onnx": "21b14751a95520953bfcc607ceeb617de7cbeaeb6d60f4c8966716c743985337",
                "tokenizer.json": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
                "config.json": "8d9f07bf7ba54a6fc3b1962483056f94c39dcf188db4cf61843e1c88f94b2342"},
        notes="DeBERTa-v3-xsmall NLI cross-encoder, dynamic int8 (u8/u8) ONNX export by the model authors.",
    ),
    "minilm": Preset(
        repo="cross-encoder/nli-MiniLM2-L6-H768",
        revision="b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
        license="apache-2.0",
        files={"model.onnx": "onnx/model_quint8_avx2.onnx", "tokenizer.json": "tokenizer.json",
               "config.json": "config.json"},
        sha256={"model.onnx": "44391a5241a62e0083c1a8899a71e69a092b95aea5ba89e14062925468eceac7"},
        notes="MiniLM2-L6-H768 NLI cross-encoder (6 layers), dynamic int8 (u8/u8) ONNX export. Comparison only.",
    ),
}


# ----------------------------------------------------------------------------- hashing
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file (what the Hub reports as ``oid`` for non-LFS files)."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())  # noqa: S324 - git object id, not security
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_size(n: int) -> str:
    return f"{n / 1e6:8.1f} MB" if n >= 100_000 else f"{n / 1e3:8.1f} kB"


# ----------------------------------------------------------------------------- hub metadata
def remote_metadata(client: httpx.Client, preset: Preset) -> dict[str, dict]:
    """Size and hashes of each wanted file at the pinned commit, from the Hub tree API."""
    wanted = set(preset.files.values())
    meta: dict[str, dict] = {}
    for folder in sorted({str(Path(p).parent).replace("\\", "/") for p in wanted}):
        suffix = "" if folder in ("", ".") else f"/{folder}"
        r = client.get(f"{ENDPOINT}/api/models/{preset.repo}/tree/{preset.revision}{suffix}")
        r.raise_for_status()
        for entry in r.json():
            if entry.get("type") == "file" and entry.get("path") in wanted:
                lfs = entry.get("lfs") or {}
                meta[entry["path"]] = {"size": int(entry["size"]), "git_oid": entry.get("oid"),
                                       "lfs_sha256": lfs.get("oid")}
    missing = wanted - meta.keys()
    if missing:
        raise SystemExit(f"error: {preset.repo}@{preset.revision[:12]} has no file(s) {sorted(missing)}")
    return meta


def remote_license(client: httpx.Client, preset: Preset) -> str | None:
    """Licence declared in the model card at the pinned commit (None if the card has none)."""
    r = client.get(f"{ENDPOINT}/api/models/{preset.repo}/revision/{preset.revision}")
    r.raise_for_status()
    info = r.json()
    lic = (info.get("cardData") or {}).get("license")
    if lic is None:
        lic = next((t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")), None)
    return str(lic).lower() if lic else None


def verify_against(path: Path, name: str, preset: Preset, remote: dict | None) -> tuple[bool, str, str]:
    """Return (ok, sha256, reason)."""
    digest = sha256_file(path)
    pinned = preset.sha256.get(name)
    if pinned and digest != pinned:
        return False, digest, "SHA-256 differs from the pinned value"
    if remote is not None:
        if path.stat().st_size != remote["size"]:
            return False, digest, f"size {path.stat().st_size} != expected {remote['size']}"
        if remote.get("lfs_sha256"):
            if digest != remote["lfs_sha256"]:
                return False, digest, "SHA-256 differs from the Hub LFS hash"
        elif remote.get("git_oid") and git_blob_sha1(path) != remote["git_oid"]:
            return False, digest, "git blob id differs from the Hub"
    return True, digest, "ok"


# ----------------------------------------------------------------------------- download
def download(client: httpx.Client, url: str, dest: Path, expected_size: int) -> None:
    """Resumable download into ``dest`` (via ``dest.part`` and HTTP Range)."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:
            part.unlink()
            have = 0
        if have == expected_size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with client.stream("GET", url, headers=headers) as r:
                if r.status_code == 416:  # range past the end: start again
                    part.unlink(missing_ok=True)
                    continue
                if r.status_code == 200:
                    have, mode = 0, "wb"
                elif r.status_code == 206:
                    mode = "ab"
                else:
                    r.raise_for_status()
                    raise httpx.HTTPError(f"unexpected status {r.status_code}")
                last = time.monotonic()
                with open(part, mode) as f:
                    for chunk in r.iter_bytes(CHUNK):
                        f.write(chunk)
                        have += len(chunk)
                        if time.monotonic() - last > 1.0:
                            last = time.monotonic()
                            print(f"\r    {dest.name}: {have / 1e6:6.1f} / {expected_size / 1e6:.1f} MB",
                                  end="", flush=True)
                print(f"\r    {dest.name}: {have / 1e6:6.1f} / {expected_size / 1e6:.1f} MB", flush=True)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429:
                raise
            if attempt == RETRIES:
                raise
            wait = 2 ** attempt
            print(f"\n    retry {attempt}/{RETRIES - 1} in {wait}s ({type(exc).__name__})", flush=True)
            time.sleep(wait)
    if not part.exists() or part.stat().st_size != expected_size:
        raise SystemExit(f"error: incomplete download of {dest.name}")
    os.replace(part, dest)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _manifest_matches(dest: Path, preset: Preset) -> bool:
    """Offline check: every file present and hashing to what the manifest recorded."""
    mpath = dest / "manifest.json"
    if not mpath.exists():
        return False
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if manifest.get("repo") != preset.repo or manifest.get("revision") != preset.revision:
        return False
    files = manifest.get("files", {})
    if set(files) != set(preset.files):
        return False
    for name, info in files.items():
        path = dest / name
        if not path.exists() or path.stat().st_size != info.get("size"):
            print(f"  {name}: missing or wrong size")
            return False
        digest = sha256_file(path)
        if digest != info.get("sha256") or (preset.sha256.get(name) and digest != preset.sha256[name]):
            print(f"  {name}: checksum mismatch")
            return False
        print(f"  {name:15s} {_fmt_size(info['size'])}  sha256 {digest[:16]}...  verified")
    return True


def fetch(preset_name: str, dest: Path | None = None) -> Path:
    preset = PRESETS[preset_name]
    dest = dest or MODELS_DIR / preset_name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{preset.repo} @ {preset.revision[:12]} ({preset.license}) -> {dest}")

    if _manifest_matches(dest, preset):
        total = sum((dest / n).stat().st_size for n in preset.files)
        print(f"All files present and verified ({total / 1e6:.1f} MB). Nothing to download.")
        return dest

    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0),
                      headers={"User-Agent": "saras-ai-model-downloader/1.0"}) as client:
        lic = remote_license(client, preset)
        if lic != preset.license:
            raise SystemExit(f"error: {preset.repo}@{preset.revision[:12]} declares licence {lic!r}, expected "
                             f"{preset.license!r}; check it before using the model")
        print(f"  licence {lic} (model card at the pinned commit)")
        meta = remote_metadata(client, preset)
        files: dict[str, dict] = {}
        for name, repo_path in preset.files.items():
            target = dest / name
            remote = meta[repo_path]
            if target.exists():
                ok, digest, reason = verify_against(target, name, preset, remote)
                if not ok:
                    print(f"  {name}: {reason}; downloading again")
                    target.unlink()
            if not target.exists():
                url = f"{ENDPOINT}/{preset.repo}/resolve/{preset.revision}/{repo_path}"
                print(f"  downloading {repo_path} ({remote['size'] / 1e6:.1f} MB)")
                download(client, url, target, remote["size"])
                ok, digest, reason = verify_against(target, name, preset, remote)
                if not ok:
                    target.unlink()
                    raise SystemExit(f"error: {name}: {reason}. The file was deleted; run again.")
            files[name] = {"source": repo_path, "size": target.stat().st_size, "sha256": digest,
                           "git_oid": remote.get("git_oid"), "lfs_sha256": remote.get("lfs_sha256")}
            print(f"  {name:15s} {_fmt_size(files[name]['size'])}  sha256 {digest[:16]}...  verified")

    config = json.loads((dest / "config.json").read_text(encoding="utf-8"))
    _write_json(dest / "manifest.json", {
        "preset": preset_name,
        "repo": preset.repo,
        "revision": preset.revision,
        "license": preset.license,
        "license_source": f"{ENDPOINT}/{preset.repo} model card at the pinned revision",
        "notes": preset.notes,
        "id2label": config.get("id2label"),
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    })
    total = sum(f["size"] for f in files.values())
    print(f"Done: {len(files)} files, {total / 1e6:.1f} MB in {dest}")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--dest", type=Path, help="target folder (default: backend/models/nli/<preset>)")
    args = ap.parse_args(argv)
    try:
        fetch(args.preset, args.dest)
    except httpx.HTTPError as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
