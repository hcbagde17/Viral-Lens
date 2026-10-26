"""
Feature extraction and preprocessing pipeline for Viral-Lens.
Shared verbatim by training, offline evaluation, and live serving to eliminate train-serving skew.
"""
import math
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from core.image_utils import (
    load_image_for_model,
    MODEL_IMAGE_SIZE
)

TABULAR_FEATURE_NAMES: List[str] = [
    "log_subs",
    "log_duration",
    "is_short",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos"
]

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Evaluation / Serving image transform
EVAL_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(MODEL_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Training image transform (light augmentation: crops, slight color jitter; NO horizontal flips)
TRAIN_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomCrop(MODEL_IMAGE_SIZE),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Neutral fallback cyclical encoding (start of cycle / neutral orientation)
NEUTRAL_TIME_ENCODING: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


def _parse_cyclical_time(
    published_at: Optional[Union[str, pd.Timestamp]]
) -> Tuple[float, float, float, float]:
    """
    Parses published timestamp into cyclical sine/cosine pairs:
    (hour_sin, hour_cos, weekday_sin, weekday_cos).
    """
    if published_at is None or pd.isna(published_at) or not str(published_at).strip():
        return NEUTRAL_TIME_ENCODING

    try:
        dt = pd.to_datetime(published_at, utc=True)
        if pd.isna(dt):
            return NEUTRAL_TIME_ENCODING

        h_rad = 2.0 * math.pi * dt.hour / 24.0
        w_rad = 2.0 * math.pi * dt.weekday() / 7.0
        return (
            math.sin(h_rad),
            math.cos(h_rad),
            math.sin(w_rad),
            math.cos(w_rad)
        )
    except Exception:
        return NEUTRAL_TIME_ENCODING


def compute_tabular_raw(
    duration: Union[int, float],
    subscriber_count: Union[int, float],
    published_at: Optional[Union[str, pd.Timestamp]] = None
) -> np.ndarray:
    """
    Computes unscaled raw tabular features:
    [log_subs, log_duration, is_short, hour_sin, hour_cos, weekday_sin, weekday_cos].
    """
    d = max(0.0, float(duration))
    s = max(0.0, float(subscriber_count))
    h_sin, h_cos, w_sin, w_cos = _parse_cyclical_time(published_at)

    return np.array([
        math.log1p(s),
        math.log1p(d),
        1.0 if d <= 60.0 else 0.0,
        h_sin,
        h_cos,
        w_sin,
        w_cos
    ], dtype=np.float32)


def compute_tabular_raw_batch(
    df: pd.DataFrame
) -> np.ndarray:
    """
    Vectorized computation of raw tabular features for an entire DataFrame.
    """
    durations = np.maximum(0.0, df["duration"].values.astype(np.float64))
    subs = np.maximum(0.0, df["subscriber_count"].values.astype(np.float64))

    log_subs = np.log1p(subs)
    log_duration = np.log1p(durations)
    is_short = (durations <= 60.0).astype(np.float32)

    if "published_at" in df.columns:
        dt = pd.to_datetime(df["published_at"], utc=True)
        hours = dt.dt.hour.fillna(0).values.astype(np.float64)
        weekdays = dt.dt.weekday.fillna(0).values.astype(np.float64)
        hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
        weekday_sin = np.sin(2.0 * np.pi * weekdays / 7.0)
        weekday_cos = np.cos(2.0 * np.pi * weekdays / 7.0)
    else:
        n = len(df)
        hour_sin = np.zeros(n, dtype=np.float32)
        hour_cos = np.ones(n, dtype=np.float32)
        weekday_sin = np.zeros(n, dtype=np.float32)
        weekday_cos = np.ones(n, dtype=np.float32)

    return np.column_stack([
        log_subs,
        log_duration,
        is_short,
        hour_sin,
        hour_cos,
        weekday_sin,
        weekday_cos
    ]).astype(np.float32)


def fit_tabular_scaler(
    train_df: pd.DataFrame
) -> Dict[str, Union[List[str], List[float]]]:
    """
    Computes mean and std statistics for tabular features on the training split only.
    """
    raw_feats = compute_tabular_raw_batch(train_df)
    mean = np.mean(raw_feats, axis=0)
    raw_std = np.std(raw_feats, axis=0)
    std = np.where(raw_std > 1e-6, raw_std, 1.0)

    return {
        "feature_names": TABULAR_FEATURE_NAMES,
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist()
    }


def standardize_tabular(
    features: np.ndarray,
    scaler_stats: Optional[Dict[str, Union[List[str], List[float]]]] = None
) -> np.ndarray:
    """
    Standardizes features using pre-computed mean and std. Supports 1D and 2D arrays.
    """
    if scaler_stats is None:
        return features.astype(np.float32)

    mean = np.array(scaler_stats["mean"], dtype=np.float32)
    std = np.array(scaler_stats["std"], dtype=np.float32)
    return ((features - mean) / std).astype(np.float32)


def clean_title(title: str) -> str:
    """
    Strips leading/trailing whitespace while preserving emojis and multilingual characters.
    """
    if not title or not isinstance(title, str):
        return ""
    return title.strip()


def preprocess_single_sample(
    image: Union[str, Image.Image, bytes],
    title: str,
    duration: Union[int, float],
    subscriber_count: Union[int, float],
    published_at: Optional[Union[str, pd.Timestamp]] = None,
    tokenizer=None,
    scaler_stats: Optional[Dict] = None,
    is_train: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Transforms multimodal inputs into model-ready PyTorch tensors.
    """
    # 1. Image
    pil_img = load_image_for_model(image, target_size=MODEL_IMAGE_SIZE, fallback_to_placeholder=True)
    img_transform = TRAIN_IMAGE_TRANSFORM if is_train else EVAL_IMAGE_TRANSFORM
    output = {"image": img_transform(pil_img)}

    # 2. Tabular
    raw_tab = compute_tabular_raw(duration, subscriber_count, published_at)
    scaled_tab = standardize_tabular(raw_tab, scaler_stats=scaler_stats)
    output["tabular"] = torch.from_numpy(scaled_tab).float()

    # 3. Text
    if tokenizer is not None:
        tokens = tokenizer(
            clean_title(title),
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        output["input_ids"] = tokens["input_ids"].squeeze(0)
        output["attention_mask"] = tokens["attention_mask"].squeeze(0)

    return output


def compute_target(view_count: Union[int, float]) -> float:
    """
    Computes log views target: ln(view_count + 1).
    """
    return float(math.log1p(max(0.0, float(view_count))))


def compute_virality_score(
    predicted_log_views: float,
    subscriber_count: Union[int, float]
) -> float:
    """
    Computes virality score: predicted_log_views - ln(subs + 1).
    """
    return float(predicted_log_views - compute_target(subscriber_count))
