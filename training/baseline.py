import os
import sys
import json
from pathlib import Path
import joblib.externals.loky.backend.context as loky_context
loky_context._count_physical_cores = lambda: (os.cpu_count() or 4, None)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.features import compute_tabular_raw_batch, standardize_tabular


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }


def load_data_and_scaler(
    train_path: Path, val_path: Path, config_path: Path
):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    scaler_stats = {
        "mean": config["scaler_mean"],
        "std": config["scaler_std"]
    }

    x_train = standardize_tabular(compute_tabular_raw_batch(train_df), scaler_stats)
    y_train = train_df["target_log_views"].values.astype(np.float32)

    x_val = standardize_tabular(compute_tabular_raw_batch(val_df), scaler_stats)
    y_val = val_df["target_log_views"].values.astype(np.float32)

    return x_train, y_train, x_val, y_val


def run_baselines(
    train_path: str = "data/processed/train_clean.csv",
    val_path: str = "data/processed/val_clean.csv",
    config_path: str = "artifacts/config.json",
    output_path: str = "artifacts/baseline_results.json"
) -> dict:
    x_train, y_train, x_val, y_val = load_data_and_scaler(Path(train_path), Path(val_path), Path(config_path))

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=42, max_iter=200, early_stopping=True
        )
    }

    results = {}
    print("=" * 65)
    print(f"{'Model':<24} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        metrics = compute_metrics(y_val, preds)
        results[name] = metrics
        print(
            f"{name:<24} | {metrics['rmse']:<10.4f} | "
            f"{metrics['mae']:<10.4f} | {metrics['r2']:<10.4f}"
        )

    print("=" * 65)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {out_file}")
    return results


if __name__ == "__main__":
    run_baselines()
