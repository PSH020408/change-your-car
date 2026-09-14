"""Path resolution shared by every ingest stage.

Locations are read from `configs/scope.yaml` rather than walked from
`__file__`. The previous relative-walk approach silently put the cache under
`backend/` because `Path("configs/scope.yaml").parent.parent.parent`
saturates at `.` (docs/recon/DECISIONS.md D9).
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import yaml


def load_scope(scope_path: Path) -> dict:
    return yaml.safe_load(Path(scope_path).read_text())


def _base(scope_path: Path) -> Path:
    """The backend/ directory, whatever cwd the module was invoked from."""
    return Path(scope_path).resolve().parent.parent


def _resolve(scope_path: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (_base(scope_path) / p).resolve()


def cache_dir(scope_path: Path, scope: dict) -> Path:
    return _resolve(scope_path, (scope.get("paths") or {}).get("cache_dir", "data/cache"))


def lake_dir(scope_path: Path, scope: dict) -> Path:
    return _resolve(scope_path, (scope.get("paths") or {}).get("lake_dir", "../data"))


def bronze_dir(scope_path: Path, scope: dict) -> Path:
    return lake_dir(scope_path, scope) / "bronze"


def slug(text: str) -> str:
    """Filesystem-safe event key. Stable across runs — it becomes a path.

    Accents are folded to ASCII first. The era calendar contains "Sao Paulo
    Grand Prix" and several "Autodromo ..." circuit names, and a path key
    carrying combining characters is not stable across filesystems (macOS
    stores NFD, Linux NFC) — the same session would land in two directories
    depending on which machine ingested it.
    """
    norm = unicodedata.normalize("NFKD", str(text).strip().lower())
    out = []
    for ch in norm:
        if unicodedata.combining(ch):
            continue
        if ch.isalnum() and ch.isascii():
            out.append(ch)
        elif ch.isalnum():
            out.append("_")          # non-latin script: keep a separator
        elif ch in " -_/":
            out.append("_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def session_dir(scope_path: Path, scope: dict, season: int, event: str, ses: str) -> Path:
    return bronze_dir(scope_path, scope) / str(season) / slug(event) / ses
