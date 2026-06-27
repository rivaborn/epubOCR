"""OCR engines behind a common interface so the eval harness compares them fairly.

Engines are lazily imported by :func:`get_engine` so that installing one backend
(e.g. Tesseract) doesn't require the others (Surya/torch, Paddle).
"""
from __future__ import annotations

from ..config import Config
from .base import OCREngine, OcrResult, OcrWord

__all__ = ["OCREngine", "OcrResult", "OcrWord", "get_engine"]


def get_engine(name: str, config: Config) -> OCREngine:
    name = name.lower()
    if name == "tesseract":
        from .tesseract import TesseractEngine
        return TesseractEngine()
    if name in ("vlm", "vlm_ocr", "qwen-vl"):
        from .vlm_openai import VlmOcrEngine
        return VlmOcrEngine.from_config(config)
    if name in ("surya", "marker"):
        from .surya_marker import SuryaEngine
        return SuryaEngine(layout=bool(config.raw.get("ocr", {}).get("layout", False)))
    if name == "surya2":
        from .surya2 import Surya2Engine, _FULL_PAGE_TOKEN_CAP
        ocr_cfg = config.raw.get("ocr", {})
        return Surya2Engine(backend=ocr_cfg.get("surya2_backend", "llamacpp"),
                            layout=bool(ocr_cfg.get("layout", False)),
                            max_tokens=int(ocr_cfg.get("surya2_max_tokens", _FULL_PAGE_TOKEN_CAP)),
                            inference_url=ocr_cfg.get("surya2_inference_url") or None,
                            model=ocr_cfg.get("surya2_model") or None,
                            parallel=int(ocr_cfg["surya2_parallel"]) if ocr_cfg.get("surya2_parallel") else None)
    if name == "paddle":
        from .paddle import PaddleEngine
        return PaddleEngine()
    raise ValueError(f"unknown OCR engine '{name}' (tesseract|surya|surya2|paddle|vlm)")
