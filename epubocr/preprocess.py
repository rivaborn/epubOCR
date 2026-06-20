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

    if cfg.get("clahe") or cfg.get("binarize"):
        img = _cv_enhance(img, cfg)                  # local contrast + thresholding (faded scans)
    else:
        img = ImageOps.autocontrast(img, cutoff=cfg.get("autocontrast_cutoff", 0))
        factor = float(cfg.get("contrast", 1.0))     # >1 lifts faded scans (linear)
        if factor != 1.0:
            from PIL import ImageEnhance
            img = ImageEnhance.Contrast(img).enhance(factor)

    if cfg.get("deskew"):
        img = _deskew(img)
    # denoise / crop_borders / upscale_lowres / split_spreads: wire when eval justifies.

    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)
    return dst


def _cv_enhance(pil_img, cfg: dict):
    """CLAHE (local contrast) and/or binarization for faded/low-contrast scans (OpenCV).

    ``clahe`` = local histogram equalization; ``binarize`` = 'adaptive' (Sauvola-style
    local threshold, best for uneven fading) or 'otsu' (global). Falls back to the input
    if OpenCV isn't installed.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return pil_img
    from PIL import Image

    arr = np.array(pil_img)
    if cfg.get("clahe"):
        clahe = cv2.createCLAHE(clipLimit=float(cfg.get("clahe_clip", 2.0)), tileGridSize=(8, 8))
        arr = clahe.apply(arr)
    mode = cfg.get("binarize")
    if mode == "adaptive":
        block = int(cfg.get("adaptive_block", 31)) | 1          # must be odd
        arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, block, int(cfg.get("adaptive_c", 15)))
    elif mode == "otsu":
        _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(arr)


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
