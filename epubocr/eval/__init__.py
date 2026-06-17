"""Evaluation harness — the piece that makes 'optimize' measurable (epubOCR.md §5).

Score every OCR engine / prompt variant against a small hand-keyed gold set so
engine selection is empirical. Key metrics live in :mod:`epubocr.eval.metrics`.
"""
