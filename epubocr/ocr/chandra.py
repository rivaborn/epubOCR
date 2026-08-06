"""Chandra 2 (``datalab-to/chandra-ocr-2``) — a served-VLM OCR engine, 5B params.

Datalab's larger sibling to Surya 2. It does **not** answer a plain "transcribe this"
prompt with plain text: whatever you ask, it emits its own layout markup —

    <div data-bbox="18 417 986 726" data-label="Text"><p>THE cold drizzle …</p></div>

which is why running it through the generic ``vlm`` engine scores ~0.20 CER of pure
markup against a text gold set. Parsed (this module) it lands at ~0.004 CER, i.e.
statistically level with Surya 2 on clean prose (measured 2026-08-06, see fleet.md).

Two behavioural differences worth knowing:

* **It de-hyphenates line breaks itself** ("tip-\\ntilted" -> "tiptilted" joined as one
  word), so its output is *already* past the de-hyphenation `layout.py` does. That costs
  it a little against an as-printed gold set and gains it a little downstream.
* **It reports no confidence.** Like the ``vlm`` engine it relies on the degeneracy guard
  and the fidelity verifier rather than the conf floor — unless ``logprobs=True``, which
  derives a pseudo-confidence from the mean token logprob (see :mod:`.vlm_openai`).

``data-label`` values match Surya's layout vocabulary, so the labels feed the same
``structure.Block`` mapping and ``layout=True`` gets real headings/lists/tables.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..config import Config
from ..llm.client import LLMClient
from .base import OCREngine, OcrResult, OcrWord, run_batch_threaded

# Chandra ignores prompt wording for format, but a tight instruction still suppresses
# commentary and discourages it from "helpfully" completing illegible words.
TRANSCRIBE_PROMPT = (
    "Transcribe this scanned book page exactly as printed.\n"
    "Do NOT add, complete, paraphrase, translate, or correct any word.\n"
    "If a region is illegible, omit it rather than guessing."
)

_DIV = re.compile(
    r'<div\b[^>]*?data-bbox="(?P<bbox>[^"]*)"[^>]*?data-label="(?P<label>[^"]*)"[^>]*>'
    r'(?P<inner>.*?)</div>',
    re.S | re.I,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def _label(raw: str) -> str:
    """Chandra writes ``Section-Header`` / ``Page-Footer`` where Surya's vocabulary (which
    ``structure.py`` keys on) writes ``SectionHeader`` / ``PageFooter``. Normalize, or
    every heading silently degrades to a paragraph and running heads are never dropped."""
    return (raw or "Text").replace("-", "").replace("_", "").replace(" ", "")


def _bbox(raw: str) -> tuple[int, int, int, int]:
    try:
        x0, y0, x1, y1 = (int(float(v)) for v in raw.split()[:4])
        return (x0, y0, x1, y1)
    except Exception:  # noqa: BLE001 - a malformed bbox must not lose the text
        return (0, 0, 0, 0)


def _text(html: str) -> str:
    # <br> is a real line break inside a block; every other tag is markup.
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    html = re.sub(r"</(p|h[1-6]|li|tr)>", "\n", html, flags=re.I)
    return _WS.sub(" ", _TAG.sub("", html)).strip()


def parse_blocks(raw: str) -> list[tuple[str, str, str, str]]:
    """Chandra's markup -> ``[(label, plain_text, inner_html, bbox_raw)]`` in reading order.

    Falls back to one synthetic ``Text`` block when the model answered with plain text
    (it does on near-blank pages), so a caller never has to special-case that.
    """
    out = [(_label(m.group("label")), _text(m.group("inner")), m.group("inner"),
            m.group("bbox")) for m in _DIV.finditer(raw)]
    out = [row for row in out if row[1]]
    if not out:
        plain = _text(raw)
        return [("Text", plain, raw, "")] if plain else []
    return out


def _blocks_json(parsed: list[tuple[str, str, str, str]]) -> list[dict]:
    from ..structure import _DROP, _FIGURE, _TABLE, _block_for

    js: list[dict] = []
    for label, text, html, _bb in parsed:
        if label in _DROP:                       # running heads / folios
            continue
        if label in _TABLE and html:
            js.append({"type": "table", "text": text, "html": html})
            continue
        block = _block_for("Text" if label in _FIGURE else label, text)
        if block:
            js.append(block.to_json())
    return js


class ChandraEngine(OCREngine):
    name = "chandra"
    wants_preprocess = False  # like every VLM here, it reads the raw page better

    def __init__(self, client: LLMClient, model: str, *, layout: bool = False,
                 max_tokens: int = 6144, logprobs: bool = False, parallel: int = 8):
        self.client = client
        self.model = model
        self._layout = layout
        self._max_tokens = max_tokens
        self._logprobs = logprobs
        self._parallel = parallel

    def identity(self) -> str:
        return f"chandra:{self.model}{':layout' if self._layout else ''}"

    @classmethod
    def from_config(cls, config: Config) -> "ChandraEngine":
        ocr = config.raw.get("ocr", {}) or {}
        client = LLMClient.for_role(config, "vlm_ocr")
        model = ocr.get("chandra_model") or config.model("chandra", endpoint=client.endpoint.name)
        return cls(client, model, layout=bool(ocr.get("layout", False)),
                   max_tokens=int(ocr.get("chandra_max_tokens", 6144)),
                   logprobs=bool(ocr.get("logprob_conf", False)),
                   parallel=int(ocr.get("chandra_parallel", 8)))

    def run(self, image_path: Path) -> OcrResult:
        raw, conf = self.client.vision_scored(
            self.model, Path(image_path), TRANSCRIBE_PROMPT,
            temperature=0.0, max_tokens=self._max_tokens, logprobs=self._logprobs)
        parsed = parse_blocks(raw)

        # Blank line between blocks: the deterministic paragraph rejoin keys on blank-line
        # boundaries, so this keeps one <p> per layout block (same contract as surya2).
        from ..structure import _DROP

        texts = [t for lbl, t, _h, _b in parsed if lbl not in _DROP]
        words = [OcrWord(text=t, bbox=_bbox(bb), conf=None) for _l, t, _h, bb in parsed]
        meta: dict = {"model": self.model, "endpoint": self.client.endpoint.name,
                      "blocks_n": len(parsed)}
        if self._layout:
            meta["blocks"] = _blocks_json(parsed)

        return OcrResult(text="\n\n".join(texts), words=words, mean_conf=conf,
                         engine=self.name, meta=meta)

    def run_batch(self, image_paths: list[Path]) -> list[OcrResult]:
        return run_batch_threaded(self.run, image_paths, self._parallel)
