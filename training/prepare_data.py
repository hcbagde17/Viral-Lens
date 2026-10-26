"""
Data Preparation and Leak-Free Group Splitting Pipeline for Viral-Lens.
Filters verified thumbnails, applies GroupShuffleSplit on channels, fits tabular scaler,
and saves clean train/val/test datasets.
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.features import (
    TABULAR_FEATURE_NAMES,
    fit_tabular_scaler,
    IMAGENET_MEAN,
    IMAGENET_STD
)
from core.image_utils import MODEL_IMAGE_SIZE


def filter_valid_samples(train_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Drops training rows where thumbnail acquisition failed."""
    train_manifest = manifest_df[manifest_df["split"] == "train"]
    valid_ids = set(train_manifest[train_manifest["status"] == "ok"]["video_id"])
    clean_df = train_df[train_df["video_id"].isin(valid_ids)].copy().reset_index(drop=True)
    clean_df["thumbnail_path"] = clean_df["video_id"].apply(lambda vid: f"data/thumbnails/{vid}.jpg")
    clean_df["target_log_views"] = np.log1p(clean_df["view_count"])
    return clean_df


def split_by_channel(
    df: pd.DataFrame, val_size: float = 0.20, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits dataset into train/val using subscriber_count to prevent channel leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["subscriber_count"]))
    train_split = df.iloc[train_idx].copy().reset_index(drop=True)
    val_split = df.iloc[val_idx].copy().reset_index(drop=True)

    overlap = len(set(train_split["subscriber_count"]).intersection(set(val_split["subscriber_count"])))
    assert overlap == 0, f"Channel leakage detected: {overlap} groups appear in both splits!"
    return train_split, val_split


def save_config(artifacts_dir: Path, scaler_stats: dict) -> Path:
    """Saves model configuration and tabular scaling parameters."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_version": "1.0.0",
        "tabular_feature_names": scaler_stats["feature_names"],
        "scaler_mean": scaler_stats["mean"],
        "scaler_std": scaler_stats["std"],
        "image_size": list(MODEL_IMAGE_SIZE),
        "image_mean": IMAGENET_MEAN,
        "image_std": IMAGENET_STD,
        "text_model_name": "distilbert-base-multilingual-cased",
        "max_token_length": 32,
        "image_backbone": "efficientnet_b0",
        "tabular_dim": len(TABULAR_FEATURE_NAMES),
        "target_name": "log_views"
    }
    cfg_path = artifacts_dir / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return cfg_path


def prepare_datasets(
    train_csv: str = "data/raw/train.csv",
    test_csv: str = "data/raw/test.csv",
    manifest_csv: str = "data/processed/thumbnail_manifest.csv",
    processed_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    val_size: float = 0.20,
    random_state: int = 42
):
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    art_path = Path(artifacts_dir)

    # 1. Load and filter
    raw_train = pd.read_csv(train_csv)
    raw_test = pd.read_csv(test_csv)
    manifest = pd.read_csv(manifest_csv)

    clean_train_full = filter_valid_samples(raw_train, manifest)
    raw_test["thumbnail_path"] = raw_test["video_id"].apply(lambda vid: f"data/thumbnails/{vid}.jpg")
    print(f"Retained {len(clean_train_full):,} valid training rows (dropped {len(raw_train) - len(clean_train_full):,}).")

    # 2. Group split
    train_split, val_split = split_by_channel(clean_train_full, val_size=val_size, random_state=random_state)
    print(f"Split complete: Train={len(train_split):,} rows, Val={len(val_split):,} rows (0 channel leakage).")

    # 3. Fit scaler on train split only
    scaler_stats = fit_tabular_scaler(train_split)
    cfg_file = save_config(art_path, scaler_stats)
    print(f"Saved scaler parameters to {cfg_file}")

    # 4. Save clean processed CSVs
    train_split.to_csv(proc_path / "train_clean.csv", index=False)
    val_split.to_csv(proc_path / "val_clean.csv", index=False)
    raw_test.to_csv(proc_path / "test_clean.csv", index=False)
    print(f"Saved clean datasets to {proc_path}/ (train_clean.csv, val_clean.csv, test_clean.csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data, group split, and fit scaler.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--manifest-path", default="data/processed/thumbnail_manifest.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    prepare_datasets(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        manifest_csv=args.manifest_path,
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        val_size=args.val_size,
        random_state=args.random_state
    )
