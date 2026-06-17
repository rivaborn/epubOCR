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
        b64, mime = _data_url(Path(image_path))
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        return self.chat(model, messages, temperature=temperature, max_tokens=max_tokens,
                         frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)


def _data_url(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return base64.b64encode(path.read_bytes()).decode("ascii"), mime
