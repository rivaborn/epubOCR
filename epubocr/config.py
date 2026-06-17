"""Configuration loading for epubocr.

Reads ``config.toml`` at the project root (override with ``$EPUBOCR_CONFIG``).
Resolves the homelab 3090 endpoints, per-endpoint model aliases, OCR default,
fidelity thresholds, and preprocessing toggles described in epubOCR.md.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / "config.toml"


@dataclass(frozen=True)
class Endpoint:
    """An OpenAI-compatible inference endpoint (Ollama or vLLM on the 3090 box)."""

    name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class FidelityThresholds:
    """Limits past which an LLM-cleaned page is held for human QA (epubOCR.md §7)."""

    max_cer_vs_ocr: float
    max_insertion_ratio: float
    max_net_length_delta: float


@dataclass(frozen=True)
class Config:
    raw: dict
    endpoints: dict[str, Endpoint]
    roles: dict[str, str]
    models: dict[str, dict]
    ocr_default_engine: str
    fidelity: FidelityThresholds
    preprocess: dict
    projects_root: Path

    # -- endpoint / model resolution -------------------------------------
    def endpoint(self, name: str) -> Endpoint:
        try:
            return self.endpoints[name]
        except KeyError:
            raise KeyError(f"no endpoint '{name}' in config; have {list(self.endpoints)}")

    def role_endpoint(self, role: str) -> Endpoint:
        """The endpoint serving a role ('vlm_ocr' | 'text_cleanup')."""
        return self.endpoint(self.roles.get(role, "ollama"))

    def model(self, key: str, *, endpoint: str | None = None) -> str:
        """Resolve a model alias (e.g. 'vlm_ocr', 'text_xhtml') on its endpoint.

        ``endpoint`` defaults to the endpoint bound to that role, falling back to
        whichever endpoint defines the key.
        """
        ep = endpoint or self.roles.get(key, "ollama")
        table = self.models.get(ep, {})
        if key in table:
            return table[key]
        # fall back: first endpoint that defines the key
        for name, t in self.models.items():
            if key in t:
                return t[key]
        raise KeyError(f"no model alias '{key}' for endpoint '{ep}'")


def _project_root_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_PROJECT_ROOT / path)


_LOCAL_CONFIG = _PROJECT_ROOT / "config.local.toml"


def _default_config_path() -> Path:
    """Prefer a gitignored config.local.toml (real endpoints) over the committed template."""
    if os.environ.get("EPUBOCR_CONFIG"):
        return Path(os.environ["EPUBOCR_CONFIG"])
    return _LOCAL_CONFIG if _LOCAL_CONFIG.exists() else _DEFAULT_CONFIG


@lru_cache(maxsize=8)
def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else _default_config_path()
    with open(cfg_path, "rb") as fh:
        raw = tomllib.load(fh)

    endpoints = {
        name: Endpoint(name=name, base_url=d["base_url"], api_key=d.get("api_key", "EMPTY"))
        for name, d in raw.get("endpoints", {}).items()
    }
    fid = raw.get("fidelity", {})
    return Config(
        raw=raw,
        endpoints=endpoints,
        roles=raw.get("roles", {}),
        models=raw.get("models", {}),
        ocr_default_engine=raw.get("ocr", {}).get("default_engine", "tesseract"),
        fidelity=FidelityThresholds(
            max_cer_vs_ocr=fid.get("max_cer_vs_ocr", 0.15),
            max_insertion_ratio=fid.get("max_insertion_ratio", 0.05),
            max_net_length_delta=fid.get("max_net_length_delta", 0.20),
        ),
        preprocess=raw.get("preprocess", {}),
        projects_root=_project_root_path(raw.get("paths", {}).get("projects_root", "book_projects")),
    )
