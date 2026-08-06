"""Thin OpenAI-compatible client for the homelab 3090 endpoints (Ollama / vLLM).

Lazy-imports the ``openai`` SDK so importing the package stays cheap. Handles both
text chat and image (VLM) messages. Note: the SDK is pointed at the IP literal in
config, never ``localhost`` (see epubOCR.md topology notes).
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from ..config import Config, Endpoint


class LLMClient:
    def __init__(self, endpoint: Endpoint):
        self.endpoint = endpoint
        self._client = None

    @classmethod
    def for_role(cls, config: Config, role: str) -> "LLMClient":
        return cls(config.role_endpoint(role))

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # lazy
            self._client = OpenAI(base_url=self.endpoint.base_url, api_key=self.endpoint.api_key)
        return self._client

    def list_models(self) -> list[str]:
        return [m.id for m in self.client.models.list().data]

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int | None = None, frequency_penalty: float | None = None,
             presence_penalty: float | None = None) -> str:
        kw: dict = {"model": model, "messages": messages,
                    "temperature": temperature, "max_tokens": max_tokens}
        if frequency_penalty is not None:
            kw["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kw["presence_penalty"] = presence_penalty
        resp = self.client.chat.completions.create(**kw)
        return resp.choices[0].message.content or ""

    def vision(self, model: str, image_path: Path, prompt: str, *,
               temperature: float = 0.0, max_tokens: int | None = 2048,
               frequency_penalty: float | None = None,
               presence_penalty: float | None = None) -> str:
        """Single-image VLM call (transcription/structure). Low temperature for fidelity;
        penalties available to suppress degenerate repetition loops."""
        return self.vision_scored(model, image_path, prompt, temperature=temperature,
                                  max_tokens=max_tokens, frequency_penalty=frequency_penalty,
                                  presence_penalty=presence_penalty, logprobs=False)[0]

    def vision_scored(self, model: str, image_path: Path, prompt: str, *,
                      temperature: float = 0.0, max_tokens: int | None = 2048,
                      frequency_penalty: float | None = None,
                      presence_penalty: float | None = None,
                      logprobs: bool = False) -> tuple[str, float | None]:
        """``vision()`` plus an optional logprob-derived pseudo-confidence.

        A generative OCR model reports no confidence of its own, which is why the
        ``vlm`` engine has always been ``conf=None`` and leans on the degeneracy guard.
        vLLM does expose per-token logprobs, and ``exp(mean logprob)`` over the answer is
        a usable *relative* certainty: near 1.0 when the model is reading cleanly,
        dropping sharply where it is guessing. It is NOT calibrated like Surya's
        per-block confidence — do not reuse Surya's 0.80 floor on it without measuring
        the distribution first (fleet.md bake-off #9).
        """
        b64, mime = _data_url(Path(image_path))
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        kw: dict = {"model": model, "messages": messages, "temperature": temperature,
                    "max_tokens": max_tokens}
        if frequency_penalty is not None:
            kw["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            kw["presence_penalty"] = presence_penalty
        if logprobs:
            kw["logprobs"] = True
        resp = self.client.chat.completions.create(**kw)
        choice = resp.choices[0]
        text = choice.message.content or ""
        return text, (_mean_logprob_conf(choice) if logprobs else None)


def _data_url(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return base64.b64encode(path.read_bytes()).decode("ascii"), mime


def _mean_logprob_conf(choice) -> float | None:
    """exp(mean token logprob) over the completion, or None if the server sent none.

    Backends differ in where they hang logprobs and some ignore the flag entirely, so
    every access is defensive: a missing field means "no signal", never an exception.
    """
    import math

    lp = getattr(getattr(choice, "logprobs", None), "content", None)
    vals = [t.logprob for t in (lp or []) if getattr(t, "logprob", None) is not None]
    if not vals:
        return None
    return math.exp(sum(vals) / len(vals))
