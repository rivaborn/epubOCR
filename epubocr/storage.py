"""The per-book project folder and content-addressed cache (epubOCR.md §10).

OCR is expensive, so every stage reads/writes a cache keyed on
``hash(content) + params + model + prompt_version`` — editing a prompt re-runs
only the affected stage, never OCR.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "book"


@dataclass(frozen=True)
class BookProject:
    """Filesystem layout for one input EPUB."""

    root: Path

    @classmethod
    def for_epub(cls, epub_path: Path, projects_root: Path) -> "BookProject":
        return cls(root=projects_root / _slug(Path(epub_path).stem))

    # -- subdirectories (mirror epubOCR.md §10) --------------------------
    @property
    def source(self) -> Path:      return self.root / "source"
    @property
    def extracted(self) -> Path:   return self.root / "extracted"
    @property
    def pages(self) -> Path:       return self.extracted / "pages"
    @property
    def ocr(self) -> Path:         return self.root / "ocr"
    @property
    def cleaned(self) -> Path:     return self.root / "cleaned"
    @property
    def qa(self) -> Path:          return self.root / "qa"
    @property
    def cache(self) -> Path:       return self.root / "cache"
    @property
    def output(self) -> Path:      return self.root / "output"

    @property
    def manifest_path(self) -> Path:
        return self.extracted / "manifest.json"

    def ensure(self) -> "BookProject":
        for d in (self.source, self.pages, self.ocr, self.cleaned, self.qa, self.cache, self.output):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- json helpers ----------------------------------------------------
    def write_json(self, path: Path, data) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_json(self, path: Path):
        return json.loads(path.read_text(encoding="utf-8"))

    # -- cache -----------------------------------------------------------
    def cache_get(self, stage: str, key: str):
        p = self.cache / stage / f"{key}.json"
        return self.read_json(p) if p.exists() else None

    def cache_put(self, stage: str, key: str, data) -> Path:
        return self.write_json(self.cache / stage / f"{key}.json", data)


def content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def cache_key(*, content: bytes | str, params: dict | None = None,
              model: str = "", version: str = "1") -> str:
    """Stable key over content + params + model + stage/prompt version."""
    h = hashlib.sha256()
    h.update(content if isinstance(content, bytes) else content.encode("utf-8"))
    h.update(json.dumps(params or {}, sort_keys=True).encode("utf-8"))
    h.update(model.encode("utf-8"))
    h.update(version.encode("utf-8"))
    return h.hexdigest()[:16]
