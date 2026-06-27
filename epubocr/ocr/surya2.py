"""Surya 2 (surya-ocr >= 0.20) — the served-VLM Surya engine; the **default** engine.

Surya 0.20 ("Surya 2") replaced the local detection/recognition/layout models with a
single VLM served through a backend — **vllm** (NVIDIA) or **llama.cpp** (CPU/Apple) —
via ``SuryaInferenceManager``. So it is a *generative* OCR engine. Validation on real
scans (a 1965 hardback) showed it earns the default slot on fidelity grounds: it matches
or beats the 0.17 engine on clean prose, drops trailing noise on marginal pages, and —
critically for a fidelity-first tool — returns **empty** (not confabulated text) on
unreadable pages (covers, near-blank), which routes them straight to facsimile.

Two costs make 0.17 worth keeping as the fast ``--engine surya`` option:
  * it is ~6-10x slower per page (autoregressive generation vs a parallel recognition
    transformer); and
  * it needs a running backend (a ``llama-server`` binary or a vLLM endpoint) that the
    ``surya2`` extra does not ship.

Contrary to an earlier assumption, Surya 2 **does** report a per-block confidence (~0.99
on clean prose, ``None`` on pages it couldn't read), so the conf floor in ``build_book``
applies normally — the degeneracy guard + consensus remain as backstops.

A generative model can churn toward its token ceiling on a hard image (a cover OCR'd for
188s and produced nothing); :data:`_FULL_PAGE_TOKEN_CAP` lowers Surya's full-page token
limit so a pathological page aborts in bounded time. A real page never approaches it.

Install with ``uv sync --extra surya2`` (mutually exclusive with the ``surya`` extra —
both are the same ``surya-ocr`` package at incompatible versions) and provide a backend.
"""
from __future__ import annotations

import importlib.metadata as _md

from pathlib import Path

from .base import OCREngine, OcrResult, OcrWord

# Cap full-page generation so a hard image (e.g. a cover the VLM can't read) can't churn
# toward Surya's 12288-token default for minutes. Real pages stay well under this — page 7
# of the test book used ~450 tokens; the HIGH_ACCURACY bbox-HTML format inflates that but a
# dense page is still ~2-4k. Tunable via [ocr] surya2_max_tokens.
_FULL_PAGE_TOKEN_CAP = 6144


def _surya_minor() -> tuple[int, int]:
    parts = _md.version("surya-ocr").split(".")
    return (int(parts[0]), int(parts[1]))


def _poly_to_bbox(polygon) -> tuple[int, int, int, int]:
    if not polygon:
        return (0, 0, 0, 0)
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _block_json(b) -> dict | None:
    """Map a Surya 2 ``BlockOCRResult`` to our ``structure.Block`` JSON (heading/list/table/…)."""
    from ..structure import _DROP, _FIGURE, _TABLE, _block_for, _plain

    label = getattr(b, "label", "Text") or "Text"
    if label in _DROP or getattr(b, "skipped", False):
        return None
    html = getattr(b, "html", "") or ""
    if label in _TABLE and html:                 # keep the model's table HTML verbatim
        return {"type": "table", "text": _plain(html), "html": html}
    if label in _FIGURE:                         # a mislabeled text region -> prose
        label = "Text"
    block = _block_for(label, _plain(html))
    return block.to_json() if block else None


class Surya2Engine(OCREngine):
    name = "surya2"
    wants_preprocess = False  # the VLM reads the raw page

    def __init__(self, backend: str | None = "llamacpp", layout: bool = False,
                 max_tokens: int | None = _FULL_PAGE_TOKEN_CAP,
                 inference_url: str | None = None, model: str | None = None,
                 parallel: int | None = None):
        self._rec = None
        self._manager = None
        self._backend = backend          # "llamacpp" | "vllm" | None (let Surya autodetect)
        self._layout = layout
        self._max_tokens = max_tokens    # full-page generation ceiling (runaway guard)
        # vllm backend: attach to an already-running server (e.g. the homelab 3090 relay)
        # instead of spawning Docker. `model` must equal that server's served name.
        self._inference_url = inference_url
        self._model = model
        # Concurrent requests Surya fires per batched call (SURYA_INFERENCE_PARALLEL,
        # default 8). Higher exploits more of the server's batching headroom.
        self._parallel = parallel

    def identity(self) -> str:
        try:
            major, minor = _surya_minor()
            ver = f"{major}.{minor}"
        except Exception:
            ver = "0.20"
        return f"surya2:{ver}:{self._backend or 'auto'}{':layout' if self._layout else ''}"

    def _ensure_loaded(self) -> None:
        if self._rec is not None:
            return
        if _surya_minor() < (0, 20):
            raise RuntimeError(
                f"the 'surya2' engine needs surya-ocr>=0.20 (Surya 2); installed "
                f"{_md.version('surya-ocr')}. Install the 'surya2' extra "
                f"(`uv sync --extra surya2`) — it is mutually exclusive with 'surya'."
            )
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
            from surya.settings import settings as _surya_settings
            # _full_page_ocr reads this at call time off the settings singleton; lowering it
            # bounds a runaway on an unreadable page without truncating any real page.
            if self._max_tokens:
                _surya_settings.SURYA_MAX_TOKENS_FULL_PAGE = int(self._max_tokens)
            # vllm backend: point at an already-running server (no Docker spawn). Surya's
            # attach path requires the server's served model id to equal this checkpoint.
            if self._inference_url:
                _surya_settings.SURYA_INFERENCE_URL = self._inference_url
            if self._model:
                _surya_settings.SURYA_MODEL_CHECKPOINT = self._model
            if self._parallel:
                _surya_settings.SURYA_INFERENCE_PARALLEL = int(self._parallel)
            # The backend (and model load) only engages on the first OCR call.
            self._manager = SuryaInferenceManager(method=self._backend)
            self._rec = RecognitionPredictor(self._manager)
        except ImportError as exc:
            raise RuntimeError(
                f"Surya 2 needs a serving backend ('{self._backend}'): {exc}. Install it "
                f"(llama.cpp: `pip install llama-cpp-python`; or run vllm) and make the "
                f"Surya 2 model available. The 3090's vLLM is one option once it's free."
            ) from exc

    def run(self, image_path: Path) -> OcrResult:
        return self.run_batch([image_path])[0]

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        """OCR a chunk of pages in one call. Surya's RecognitionPredictor fires the
        requests concurrently (``SURYA_INFERENCE_PARALLEL`` workers) and the server
        continuous-batches them — the throughput win over one-page-at-a-time."""
        from PIL import Image

        self._ensure_loaded()
        images = [Image.open(p).convert("RGB") for p in image_paths]
        pages = self._rec(images, full_page=True)            # List[PageOCRResult], in order
        return [self._page_to_result(pg) for pg in pages]

    def _page_to_result(self, page) -> OcrResult:
        from ..structure import _plain

        blocks = sorted(getattr(page, "blocks", []), key=lambda b: getattr(b, "reading_order", 0))
        texts: list[str] = []
        words: list[OcrWord] = []
        confs: list[float] = []
        for b in blocks:
            if getattr(b, "skipped", False) or getattr(b, "error", False):
                continue
            txt = _plain(getattr(b, "html", "") or "")
            if not txt:
                continue
            texts.append(txt)
            conf = getattr(b, "confidence", None)            # Surya 2 may not expose one -> None
            words.append(OcrWord(text=txt, bbox=_poly_to_bbox(getattr(b, "polygon", None)), conf=conf))
            if conf is not None:
                confs.append(float(conf))

        meta = {"backend": self._backend, "blocks_n": len(getattr(page, "blocks", []))}
        if self._layout:
            meta["blocks"] = [bj for bj in (_block_json(b) for b in blocks) if bj]

        return OcrResult(
            # Blank line between blocks so the deterministic paragraph rejoin (which keys on
            # blank-line boundaries) keeps one <p> per layout block instead of collapsing the
            # whole page into a single paragraph. Within-block lines stay '\n'-joined upstream.
            text="\n\n".join(texts),
            words=words,
            mean_conf=(sum(confs) / len(confs) if confs else None),
            engine=self.name,
            meta=meta,
        )
