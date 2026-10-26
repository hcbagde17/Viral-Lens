"""
PyTorch Dataset and DataLoader abstractions for Viral-Lens.
Provides clean, high-performance batching for multimodal deep learning.
"""
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from core.features import (
    compute_tabular_raw_batch,
    standardize_tabular,
    clean_title,
    TRAIN_IMAGE_TRANSFORM,
    EVAL_IMAGE_TRANSFORM
)
from core.image_utils import (
    load_image_for_model,
    MODEL_IMAGE_SIZE
)


class ViralLensDataset(Dataset):
    """
    Multimodal PyTorch Dataset for YouTube virality modeling.
    Loads and standardizes thumbnail images, multilingual titles, tabular metadata, and target log-views.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        scaler_stats: Optional[Dict] = None,
        is_train: bool = False,
        image_dir: Union[str, Path] = "data/thumbnails"
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.image_dir = Path(image_dir)
        self.transform = TRAIN_IMAGE_TRANSFORM if is_train else EVAL_IMAGE_TRANSFORM

        # Vectorized pre-computation of tabular features for high training throughput
        raw_tabular = compute_tabular_raw_batch(self.df)
        scaled_tabular = standardize_tabular(raw_tabular, scaler_stats=scaler_stats)
        self.tabular_tensors = torch.from_numpy(scaled_tabular).float()

        # Targets (available during training/validation)
        if "target_log_views" in self.df.columns:
            self.targets = torch.tensor(
                self.df["target_log_views"].values, dtype=torch.float32
            )
        else:
            self.targets = None

        self.titles = self.df["title"].astype(str).tolist()
        self.video_ids = self.df["video_id"].astype(str).tolist()

        # Resolve image paths
        if "thumbnail_path" in self.df.columns:
            self.image_paths = [Path(p) for p in self.df["thumbnail_path"]]
        else:
            self.image_paths = [self.image_dir / f"{vid}.jpg" for vid in self.video_ids]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 1. Image loading and augmentation
        img_path = self.image_paths[idx]
        pil_img = load_image_for_model(
            img_path,
            target_size=MODEL_IMAGE_SIZE,
            fallback_to_placeholder=True
        )
        img_tensor = self.transform(pil_img)

        # 2. Text tokenization
        text = clean_title(self.titles[idx])
        tokens = self.tokenizer(
            text,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # 3. Tabular features
        tabular_tensor = self.tabular_tensors[idx]

        item = {
            "image": img_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tabular": tabular_tensor,
            "video_id": self.video_ids[idx]
        }

        if self.targets is not None:
            item["target"] = self.targets[idx]

        return item


def get_dataloader(
    df: pd.DataFrame,
    tokenizer,
    scaler_stats: Optional[Dict] = None,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    is_train: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Factory helper to create a PyTorch DataLoader from a dataframe.
    """
    if shuffle is None:
        shuffle = is_train

    dataset = ViralLensDataset(
        df=df,
        tokenizer=tokenizer,
        scaler_stats=scaler_stats,
        is_train=is_train
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
