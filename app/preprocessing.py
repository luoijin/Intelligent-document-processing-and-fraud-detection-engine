"""
app.preprocessing
------------------
Vision preprocessing utilities for the IDP pipeline (Phase 1).

Responsibilities (per docs/01-ARCHITECTURE.md §2.1):
    - Deskew: correct rotation so text lines are horizontal.
    - Denoise: reduce scan/photo noise.
    - Normalize: convert to a consistent grayscale, contrast-adjusted
      format that downstream OCR performs best on.

All functions operate on numpy arrays (OpenCV's native format) and are
side-effect-free: given the same input array, output is deterministic.
"""
from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file exists but is not a decodable image.
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image at path: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Apply light denoising suited to scanned/photographed documents."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate and correct small rotational skew using the minAreaRect
    of foreground (text) pixels.

    For blank or near-blank images (no discernible foreground), the
    input is returned unchanged rather than raising — a defensive
    choice since skew correction on an empty page is undefined.
    """
    # Invert + threshold so text pixels are foreground (white) for
    # cv2.minAreaRect, which expects the object of interest as
    # non-zero pixels.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 20:
        # Not enough foreground pixels to estimate a reliable angle.
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angles in (-90, 0]; normalize to a
    # small rotation correction rather than a full re-orientation.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible skew — avoids introducing
    # interpolation blur on already-straight images.
    if abs(angle) < 0.5:
        return gray

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to improve contrast
    on unevenly lit photos."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline: load -> grayscale -> denoise ->
    deskew -> contrast normalize.

    Returns a single-channel numpy array ready for OCR.
    """
    image = load_image(image_path)
    gray = to_grayscale(image)
    gray = denoise(gray)
    gray = deskew(gray)
    gray = normalize_contrast(gray)
    return gray
