"""Adaptive image preprocessing (epubOCR.md §3).

Steps are conditional, not a fixed chain — over-processing already-clean page scans
hurts OCR. The minimal default (grayscale + autocontrast) is pure Pillow; deskew and
friends use OpenCV (``uv sync --extra ocr-local``) and are enabled per config only
when they improve measured OCR confidence on the gold set.
"""
from __future__ import annotations

from pathlib import Path


def preprocess_image(src: Path, dst: Path, cfg: dict | None = None) -> Path:
    """Write a processed copy of ``src`` to ``dst`` and return ``dst``.

    Always keeps the original (caller passes a separate dst) so OCR is regenerable.
    """
    from PIL import Image, ImageOps  # lazy

    cfg = cfg or {}
    img = ImageOps.exif_transpose(Image.open(src))   # normalize orientation
    img = ImageOps.grayscale(img)
    img = ImageOps.autocontrast(img)

    if cfg.get("deskew"):
        img = _deskew(img)
    # denoise / crop_borders / upscale_lowres / split_spreads: wire when eval justifies.

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dst


def _deskew(pil_img):
    """Estimate and correct small skew with OpenCV; no-op if OpenCV isn't installed."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return pil_img

    arr = np.array(pil_img)
    inv = cv2.bitwise_not(arr)
    coords = np.column_stack(np.where(inv > 0))
    if coords.size == 0:
        return pil_img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3:
        return pil_img
    h, w = arr.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    from PIL import Image
    return Image.fromarray(rotated)
