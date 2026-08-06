"""OCR engines behind a common interface so the eval harness compares them fairly.

Engines are lazily imported by :func:`get_engine` so that installing one backend
(e.g. Tesseract) doesn't require the others (Surya/torch, Paddle).

Any served engine can be **pooled across free fleet units** by setting ``[ocr] pool``
(see :mod:`.pool`): ``get_engine`` then returns a :class:`~.pool.PoolEngine` whose
``identity()`` — and therefore the page cache — is identical to the single-unit engine,
so pooling is a pure throughput change.
"""
from __future__ import annotations

from ..config import Config
from .base import OCREngine, OcrResult, OcrWord

__all__ = ["OCREngine", "OcrResult", "OcrWord", "get_engine"]

# Engines that talk to an OpenAI-compatible server and can therefore be pooled across
# units. Each entry builds ONE sub-engine bound to a given (url, model).
_POOLABLE = {"surya2", "chandra", "paddleocr", "vlm"}


def _pool_factory(name: str, config: Config):
    """Return ``make_engine(url, model)`` for a poolable engine name."""
    ocr_cfg = config.raw.get("ocr", {}) or {}

    if name == "surya2":
        from .surya2 import Surya2Engine, _FULL_PAGE_TOKEN_CAP

        def make(url: str, model: str | None):
            return Surya2Engine(
                backend=ocr_cfg.get("surya2_backend", "vllm"),
                layout=bool(ocr_cfg.get("layout", False)),
                max_tokens=int(ocr_cfg.get("surya2_max_tokens", _FULL_PAGE_TOKEN_CAP)),
                inference_url=url, model=model or ocr_cfg.get("surya2_model"),
                parallel=int(ocr_cfg["surya2_parallel"]) if ocr_cfg.get("surya2_parallel") else None)
        return make

    from ..config import Endpoint
    from ..llm.client import LLMClient

    if name == "chandra":
        from .chandra import ChandraEngine

        def make(url: str, model: str | None):
            return ChandraEngine(
                LLMClient(Endpoint(name, url, "EMPTY")), model or ocr_cfg.get("chandra_model"),
                layout=bool(ocr_cfg.get("layout", False)),
                max_tokens=int(ocr_cfg.get("chandra_max_tokens", 6144)),
                logprobs=bool(ocr_cfg.get("logprob_conf", False)))
        return make

    from .vlm_openai import VlmOcrEngine

    def make(url: str, model: str | None):
        return VlmOcrEngine(LLMClient(Endpoint(name, url, "EMPTY")),
                            model or ocr_cfg.get(f"{name}_model"),
                            logprobs=bool(ocr_cfg.get("logprob_conf", False)))
    return make


def get_engine(name: str, config: Config) -> OCREngine:
    name = name.lower()
    ocr_cfg = config.raw.get("ocr", {}) or {}

    if ocr_cfg.get("pool") or ocr_cfg.get("pool_urls"):
        if name in _POOLABLE:
            from .pool import build_pool
            return build_pool(config, name, make_engine=_pool_factory(name, config))
        raise ValueError(
            f"[ocr] pool is set but engine '{name}' runs locally and cannot be pooled "
            f"(poolable: {', '.join(sorted(_POOLABLE))})")

    if name == "tesseract":
        from .tesseract import TesseractEngine
        return TesseractEngine()
    if name in ("vlm", "vlm_ocr", "qwen-vl"):
        from .vlm_openai import VlmOcrEngine
        return VlmOcrEngine.from_config(config)
    if name in ("surya", "marker"):
        from .surya_marker import SuryaEngine
        return SuryaEngine(layout=bool(ocr_cfg.get("layout", False)))
    if name == "surya2":
        from .surya2 import Surya2Engine, _FULL_PAGE_TOKEN_CAP
        return Surya2Engine(backend=ocr_cfg.get("surya2_backend", "llamacpp"),
                            layout=bool(ocr_cfg.get("layout", False)),
                            max_tokens=int(ocr_cfg.get("surya2_max_tokens", _FULL_PAGE_TOKEN_CAP)),
                            inference_url=ocr_cfg.get("surya2_inference_url") or None,
                            model=ocr_cfg.get("surya2_model") or None,
                            parallel=int(ocr_cfg["surya2_parallel"]) if ocr_cfg.get("surya2_parallel") else None)
    if name == "chandra":
        from .chandra import ChandraEngine
        return ChandraEngine.from_config(config)
    if name == "paddleocr":                    # served PaddleOCR-VL (not the local extra)
        from .vlm_openai import VlmOcrEngine
        return VlmOcrEngine.from_config(config, model_key="paddleocr")
    if name == "paddle":
        from .paddle import PaddleEngine
        return PaddleEngine()
    raise ValueError(
        f"unknown OCR engine '{name}' (tesseract|surya|surya2|chandra|paddleocr|paddle|vlm)")
