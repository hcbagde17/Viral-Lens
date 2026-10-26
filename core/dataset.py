from pathlib import Path
from typing import Dict, Optional, Union
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
from core.image_utils import load_image_for_model


class ViralLensDataset(Dataset):
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

        raw_tabular = compute_tabular_raw_batch(self.df)
        scaled_tabular = standardize_tabular(raw_tabular, scaler_stats=scaler_stats)
        self.tabular_tensors = torch.from_numpy(scaled_tabular).float()

        if "target_log_views" in self.df.columns:
            self.targets = torch.tensor(
                self.df["target_log_views"].values, dtype=torch.float32
            )
        else:
            self.targets = None

        self.titles = self.df["title"].astype(str).tolist()
        self.video_ids = self.df["video_id"].astype(str).tolist()

        if "thumbnail_path" in self.df.columns:
            self.image_paths = [Path(p) for p in self.df["thumbnail_path"]]
        else:
            self.image_paths = [self.image_dir / f"{vid}.jpg" for vid in self.video_ids]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pil_img = load_image_for_model(
            self.image_paths[idx],
            target_size=None,
            fallback_to_placeholder=True
        )
        img_tensor = self.transform(pil_img)

        tokens = self.tokenizer(
            clean_title(self.titles[idx]),
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "image": img_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "tabular": self.tabular_tensors[idx],
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
