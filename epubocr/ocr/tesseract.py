"""Tesseract OCR engine — the zero-setup CPU baseline / fallback (epubOCR.md §4).

Requires the Tesseract binary on PATH (or set ``pytesseract.pytesseract.tesseract_cmd``)
and the ``ocr-local`` extra (``uv sync --extra ocr-local``).
"""
from __future__ import annotations

from pathlib import Path

from .base import OCREngine, OcrResult, OcrWord


class TesseractEngine(OCREngine):
    name = "tesseract"

    def run(self, image_path: Path) -> OcrResult:
        import pytesseract                       # lazy (extra)
        from PIL import Image

        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words: list[OcrWord] = []
        confs: list[float] = []
        for i, txt in enumerate(data["text"]):
            if not txt.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:
                continue
            words.append(OcrWord(
                text=txt,
                bbox=(data["left"][i], data["top"][i],
                      data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]),
                conf=conf / 100.0,
            ))
            confs.append(conf / 100.0)

        text = pytesseract.image_to_string(img)
        mean_conf = sum(confs) / len(confs) if confs else None
        return OcrResult(text=text, words=words, mean_conf=mean_conf, engine=self.name)
