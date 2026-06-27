"""Common OCR result types and the engine interface."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OcrWord:
    text: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    conf: float | None               # 0..1, None if the engine has no confidence


@dataclass
class OcrResult:
    text: str
    words: list[OcrWord] = field(default_factory=list)
    mean_conf: float | None = None
    engine: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "engine": self.engine,
            "mean_conf": self.mean_conf,
            "text": self.text,
            "words": [
                {"text": w.text, "bbox": list(w.bbox), "conf": w.conf} for w in self.words
            ],
            "meta": self.meta,
        }


class OCREngine(abc.ABC):
    """Transcribe a single page image to text (+ boxes/confidence when available)."""

    name: str = "base"
    # Traditional engines benefit from grayscale/deskew; VLMs and Surya do their own
    # normalization and generally read the raw page better.
    wants_preprocess: bool = True

    def identity(self) -> str:
        """Stable string for cache keys — overridden by engines with a model."""
        return self.name

    @abc.abstractmethod
    def run(self, image_path: Path) -> OcrResult:  # pragma: no cover - interface
        ...

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        """OCR several page images at once, returning results in the same order.

        The default is sequential — correct for any engine. Served-VLM / Surya engines
        override this to hand the whole list to one batched call (concurrent HTTP
        requests against the vLLM/llama.cpp server, or one GPU batch), which is where
        the throughput win for whole-book OCR comes from.
        """
        return [self.run(p) for p in image_paths]
