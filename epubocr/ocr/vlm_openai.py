"""VLM-OCR engine via an OpenAI-compatible endpoint (Qwen2.5-VL on the 3090).

The challenger in the fidelity-first policy (epubOCR.md §4). The prompt is tight:
transcribe exactly, invent nothing. Output still passes the fidelity verifier
before it is trusted.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Config
from ..llm.client import LLMClient
from .base import OCREngine, OcrResult, run_batch_threaded

TRANSCRIBE_PROMPT = (
    "Transcribe ALL text in this scanned book page exactly as printed.\n"
    "Rules:\n"
    "- Output only the transcription, no commentary.\n"
    "- Do NOT add, complete, paraphrase, translate, or correct words.\n"
    "- Preserve reading order and line breaks; keep hyphenation as printed.\n"
    "- If a region is illegible or the page is blank, simply omit it. Never repeat a\n"
    "  placeholder word, and never output the same line more than once."
)


class VlmOcrEngine(OCREngine):
    name = "vlm"
    wants_preprocess = False  # the VLM reads the raw page better than a thresholded one

    def __init__(self, client: LLMClient, model: str, *, logprobs: bool = False,
                 max_tokens: int = 2048, parallel: int = 8):
        self.client = client
        self.model = model
        self._logprobs = logprobs
        self._max_tokens = max_tokens
        self._parallel = parallel

    def identity(self) -> str:
        return f"vlm:{self.model}"

    @classmethod
    def from_config(cls, config: Config, *, hard: bool = False,
                    model_key: str | None = None) -> "VlmOcrEngine":
        client = LLMClient.for_role(config, "vlm_ocr")
        key = model_key or ("vlm_ocr_hard" if hard else "vlm_ocr")
        ocr = config.raw.get("ocr", {}) or {}
        return cls(client, config.model(key, endpoint=client.endpoint.name),
                   logprobs=bool(ocr.get("logprob_conf", False)),
                   parallel=int(ocr.get("vlm_parallel", 8)))

    def run(self, image_path: Path) -> OcrResult:
        # No frequency/presence penalty: penalties perturb legitimate transcription (they
        # shift proper nouns). For fidelity we instead cap max_tokens to bound loop damage
        # and let the degeneracy detector (pipeline) route runaway pages to facsimile.
        text, conf = self.client.vision_scored(
            self.model, Path(image_path), TRANSCRIBE_PROMPT,
            temperature=0.0, max_tokens=self._max_tokens, logprobs=self._logprobs)
        return OcrResult(text=text, words=[], mean_conf=conf, engine=self.name,
                         meta={"model": self.model, "endpoint": self.client.endpoint.name})

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        return run_batch_threaded(self.run, image_paths, self._parallel)
