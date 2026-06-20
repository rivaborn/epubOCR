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
    if name == "paddle":
        from .paddle import PaddleEngine
        return PaddleEngine()
    raise ValueError(f"unknown OCR engine '{name}' (tesseract|surya|paddle|vlm)")
