"""PaddleOCR engine — lightweight conventional first-pass alternative.

STUB. Install with ``uv sync --extra paddle`` (mind the paddlepaddle-gpu CUDA wheel
match on Windows), then implement run() with PaddleOCR / PP-StructureV3.
"""
from __future__ import annotations

from pathlib import Path

from .base import OCREngine, OcrResult


class PaddleEngine(OCREngine):
    name = "paddle"

    def run(self, image_path: Path) -> OcrResult:
        raise NotImplementedError(
            "PaddleOCR not wired yet. Install with `uv sync --extra paddle` and implement run()."
        )
