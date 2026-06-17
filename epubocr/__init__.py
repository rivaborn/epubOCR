"""epubocr — fidelity-first OCR pipeline for improving image-only EPUBs.

See epubOCR.md for the full design. Stage modules:
  ingest -> preprocess -> ocr -> (eval gate) -> layout -> llm cleanup
         -> assemble -> validate

Every stage is cache-first and reversible; the LLM restructures OCR output but
never re-transcribes the image (enforced by llm.fidelity_verifier).
"""

__version__ = "0.1.0"
