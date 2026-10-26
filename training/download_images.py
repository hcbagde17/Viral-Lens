"""
Thumbnail Downloader and Validation Pipeline for Viral-Lens (Phase 3).
Downloads, validates, resizes, and saves YouTube thumbnails with multi-threading,
retries, Pillow byte verification, resumability, and placeholder generation for test items.
"""
import os
import io
import sys
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.image_utils import (
    validate_image_bytes,
    validate_image_file,
    save_thumbnail,
    create_placeholder_image,
    DEFAULT_MAX_DIM,
    DEFAULT_MIN_WIDTH
)


def fetch_and_validate(
    url: str,
    out_path: Path,
    min_width: int = DEFAULT_MIN_WIDTH,
    max_dim: int = DEFAULT_MAX_DIM,
    timeout: float = 10.0,
    retries: int = 3
) -> tuple[str, str, int, int]:
    """
    Downloads image from URL with retries, validates bytes with Pillow,
    resizes to max_dim, and saves as JPEG.
    
    Returns: (status, reason, width, height)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    last_reason = "unknown_error"
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 404:
                return "failed", "http_404", 0, 0
            if resp.status_code != 200:
                last_reason = f"http_{resp.status_code}"
                time.sleep(1.0 * (attempt + 1))
                continue
            
            raw_bytes = resp.content
            is_valid, reason, img = validate_image_bytes(raw_bytes, min_width=min_width)
            if not is_valid:
                return "failed", reason, 0, 0
            
            w, h = save_thumbnail(img, out_path, max_dim=max_dim, quality=90)
            return "ok", "", w, h
            
        except requests.RequestException as e:
            last_reason = f"network_{type(e).__name__}"
            time.sleep(1.0 * (attempt + 1))
        except Exception as e:
            last_reason = f"error_{type(e).__name__}"
            time.sleep(1.0 * (attempt + 1))

    return "failed", last_reason, 0, 0


def process_single_item(
    row: dict,
    out_dir: Path,
    min_width: int = DEFAULT_MIN_WIDTH,
    max_dim: int = DEFAULT_MAX_DIM,
    timeout: float = 10.0,
    retries: int = 3
) -> dict:
    """
    Processes one video: checks existing file for resumability, downloads if needed,
    and falls back to placeholder for test failures.
    """
    video_id = str(row["video_id"]).strip()
    url = str(row["thumbnail_url"]).strip()
    split = str(row.get("split", "train")).strip()
    out_path = out_dir / f"{video_id}.jpg"
    
    # 1. Resumability: check if valid image already exists on disk
    if out_path.is_file() and out_path.stat().st_size > 0:
        is_valid, reason, existing_img = validate_image_file(out_path, min_width=min_width)
        if is_valid and existing_img is not None:
            return {
                "video_id": video_id,
                "split": split,
                "status": "ok",
                "reason": "cached",
                "width": existing_img.width,
                "height": existing_img.height,
                "file_path": str(out_path)
            }
        # If file on disk is corrupt or invalid, we will re-download it
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

    # 2. Download and validate
    status, reason, width, height = fetch_and_validate(
        url=url,
        out_path=out_path,
        min_width=min_width,
        max_dim=max_dim,
        timeout=timeout,
        retries=retries
    )
    
    # 3. For test failures, generate and save a neutral gray placeholder
    # so test inference never crashes due to a missing file
    if status == "failed" and split == "test":
        placeholder = create_placeholder_image(size=(max_dim, max_dim))
        save_thumbnail(placeholder, out_path, max_dim=max_dim, quality=90)
        return {
            "video_id": video_id,
            "split": split,
            "status": "failed",
            "reason": f"{reason}_placeholder_used",
            "width": max_dim,
            "height": max_dim,
            "file_path": str(out_path)
        }

    return {
        "video_id": video_id,
        "split": split,
        "status": status,
        "reason": reason,
        "width": width,
        "height": height,
        "file_path": str(out_path) if status == "ok" else ""
    }


def download_all(
    train_csv: str = "data/raw/train.csv",
    test_csv: str = "data/raw/test.csv",
    out_dir: str = "data/thumbnails",
    manifest_path: str = "data/processed/thumbnail_manifest.csv",
    workers: int = 16,
    timeout: float = 10.0,
    retries: int = 3,
    min_width: int = DEFAULT_MIN_WIDTH,
    max_dim: int = DEFAULT_MAX_DIM,
    limit: int = 0
):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    manifest_path_obj = Path(manifest_path)
    manifest_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {train_csv} and {test_csv}...")
    train_df = pd.read_csv(train_csv)
    train_df["split"] = "train"
    
    test_df = pd.read_csv(test_csv)
    test_df["split"] = "test"
    
    combined = pd.concat([
        train_df[["video_id", "thumbnail_url", "split"]],
        test_df[["video_id", "thumbnail_url", "split"]]
    ], ignore_index=True)
    
    if limit > 0:
        combined = combined.head(limit)
        print(f"DEBUG MODE: Limiting run to {limit} items.")
        
    records = combined.to_dict(orient="records")
    total_items = len(records)
    print(f"Total items to process: {total_items:,} ({len(train_df):,} train, {len(test_df):,} test)")
    print(f"Using {workers} worker threads. Output directory: {out_dir_path}")
    
    # Read existing manifest if resumable
    manifest_records = []
    already_done_ids = set()
    if manifest_path_obj.is_file():
        try:
            existing_manifest = pd.read_csv(manifest_path_obj)
            # Find IDs that are ok or successfully handled
            for r in existing_manifest.to_dict(orient="records"):
                v_id = str(r["video_id"]).strip()
                target_file = out_dir_path / f"{v_id}.jpg"
                if target_file.is_file() and target_file.stat().st_size > 0:
                    manifest_records.append(r)
                    already_done_ids.add(v_id)
            print(f"Resuming: found {len(already_done_ids):,} valid items already processed in manifest.")
        except Exception as e:
            print(f"Could not load existing manifest ({e}), starting fresh.")
            
    items_to_process = [r for r in records if str(r["video_id"]).strip() not in already_done_ids]
    print(f"Items remaining to download / check: {len(items_to_process):,}")

    new_results = []
    start_time = time.time()
    
    if items_to_process:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    process_single_item,
                    item,
                    out_dir_path,
                    min_width,
                    max_dim,
                    timeout,
                    retries
                ): item for item in items_to_process
            }
            
            with tqdm(total=len(items_to_process), desc="Downloading thumbnails") as pbar:
                for future in as_completed(future_to_item):
                    res = future.result()
                    new_results.append(res)
                    pbar.update(1)
                    
                    # Periodic manifest checkpoint every 500 items
                    if len(new_results) % 500 == 0:
                        temp_df = pd.DataFrame(manifest_records + new_results)
                        temp_df.to_csv(manifest_path_obj, index=False)
                        
    all_results = manifest_records + new_results
    final_manifest = pd.DataFrame(all_results)
    final_manifest.to_csv(manifest_path_obj, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nFinished in {elapsed:.2f} seconds.")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("THUMBNAIL DOWNLOAD & VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total entries: {len(final_manifest):,}")
    
    for split_name in ["train", "test"]:
        sub = final_manifest[final_manifest["split"] == split_name]
        total_sub = len(sub)
        ok_count = (sub["status"] == "ok").sum()
        failed_count = (sub["status"] == "failed").sum()
        ok_pct = (ok_count / total_sub * 100) if total_sub > 0 else 0
        fail_rate = (failed_count / total_sub * 100) if total_sub > 0 else 0
        print(f"\n--- {split_name.upper()} SET ({total_sub:,} items) ---")
        print(f"  OK (Valid & Saved) : {ok_count:,} ({ok_pct:.2f}%)")
        print(f"  Failed / Invalid   : {failed_count:,} ({fail_rate:.2f}%)")
        if failed_count > 0:
            print("  Failure reasons breakdown:")
            reasons = sub[sub["status"] == "failed"]["reason"].value_counts()
            for r_name, r_cnt in reasons.items():
                print(f"    - {r_name}: {r_cnt:,}")
                
    print("=" * 60)
    print(f"Manifest written to: {manifest_path_obj}")
    return final_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate YouTube thumbnails.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="data/thumbnails")
    parser.add_argument("--manifest-path", default="data/processed/thumbnail_manifest.csv")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM)
    parser.add_argument("--limit", type=int, default=0, help="Test run on first N rows")
    args = parser.parse_args()

    download_all(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        manifest_path=args.manifest_path,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        min_width=args.min_width,
        max_dim=args.max_dim,
        limit=args.limit
    )
