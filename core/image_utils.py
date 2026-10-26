"""
Image utilities for Viral-Lens.
Shared between training, data preparation, evaluation, and live serving.
"""
import io
import os
from pathlib import Path
from typing import Tuple, Optional, Union
from PIL import Image


DEFAULT_PLACEHOLDER_COLOR = (128, 128, 128)  # Neutral gray (RGB)
DEFAULT_MAX_DIM = 512
DEFAULT_MIN_WIDTH = 200
MODEL_IMAGE_SIZE = (224, 224)


def create_placeholder_image(
    size: Tuple[int, int] = (DEFAULT_MAX_DIM, DEFAULT_MAX_DIM),
    color: Tuple[int, int, int] = DEFAULT_PLACEHOLDER_COLOR
) -> Image.Image:
    """
    Creates a neutral uniform gray RGB placeholder image.
    Used for failed/missing test thumbnails to prevent inference failures.
    """
    return Image.new("RGB", size, color=color)


def validate_image_bytes(
    raw_bytes: bytes,
    min_width: int = DEFAULT_MIN_WIDTH
) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates image bytes using Pillow:
    1. Attempts to open and verify the integrity of the image data.
    2. Re-opens and converts to RGB.
    3. Checks whether image width meets the minimum threshold (>= min_width).
       Filters out YouTube 120x90 default placeholder images.
       
    Returns:
        (is_valid, reason, pil_image_or_None)
    """
    if not raw_bytes:
        return False, "empty_bytes", None

    try:
        # Step 1: Pillow verify detects corrupt or truncated payloads
        bio_verify = io.BytesIO(raw_bytes)
        verify_img = Image.open(bio_verify)
        verify_img.verify()
    except Exception as e:
        return False, f"corrupt_or_invalid_format_{type(e).__name__}", None

    try:
        # Step 2: Re-open image after verify() as documented by Pillow
        bio_read = io.BytesIO(raw_bytes)
        img = Image.open(bio_read).convert("RGB")
    except Exception as e:
        return False, f"decode_failed_{type(e).__name__}", None

    # Step 3: Check resolution threshold
    if img.width < min_width:
        return False, f"placeholder_or_too_small_w{img.width}", None

    return True, "", img


def validate_image_file(
    file_path: Union[str, Path],
    min_width: int = DEFAULT_MIN_WIDTH
) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validates an existing image file on disk.
    """
    path = Path(file_path)
    if not path.is_file() or path.stat().st_size == 0:
        return False, "file_missing_or_empty", None

    try:
        with open(path, "rb") as f:
            content = f.read()
        return validate_image_bytes(content, min_width=min_width)
    except Exception as e:
        return False, f"file_read_error_{type(e).__name__}", None


def resize_thumbnail(
    img: Image.Image,
    max_dim: int = DEFAULT_MAX_DIM
) -> Image.Image:
    """
    Resizes image maintaining aspect ratio so the longest side is at most max_dim.
    """
    img_copy = img.copy()
    img_copy.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img_copy


def save_thumbnail(
    img: Image.Image,
    out_path: Union[str, Path],
    max_dim: int = DEFAULT_MAX_DIM,
    quality: int = 90
) -> Tuple[int, int]:
    """
    Resizes image to max_dim and saves as JPEG.
    Returns: (width, height)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    resized = resize_thumbnail(img, max_dim=max_dim)
    resized.save(out_path, format="JPEG", quality=quality)
    return resized.size


def load_image_for_model(
    img_source: Union[str, Path, bytes, Image.Image],
    target_size: Optional[Tuple[int, int]] = MODEL_IMAGE_SIZE,
    fallback_to_placeholder: bool = True
) -> Image.Image:
    """Loads an RGB image from path, bytes, or PIL Image, with optional gray placeholder fallback."""
    img = None
    if isinstance(img_source, Image.Image):
        img = img_source.convert("RGB")
    elif isinstance(img_source, (str, Path)):
        path = Path(img_source)
        if path.is_file():
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = None
    elif isinstance(img_source, bytes):
        try:
            img = Image.open(io.BytesIO(img_source)).convert("RGB")
        except Exception:
            img = None

    if img is None:
        if fallback_to_placeholder:
            img = create_placeholder_image(size=target_size or MODEL_IMAGE_SIZE)
        else:
            raise ValueError(f"Could not load valid image from source: {type(img_source)}")

    if target_size is not None and img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    return img
