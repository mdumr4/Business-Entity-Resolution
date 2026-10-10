"""Hugging Face Hub Asset Downloader for Business Entity Resolution.

Allows downloading pre-trained models, extracted feature parquets, or raw data
directly from Hugging Face Model/Dataset Hub repositories with progress tracking.

Usage:
  python scripts/download_assets.py --models
  python scripts/download_assets.py --features
  python scripts/download_assets.py --all
"""

import argparse
import os
import sys

DEFAULT_MODEL_REPO = "umar-amazon-ml-2026/entity-resolution-models"
DEFAULT_DATA_REPO = "umar-amazon-ml-2026/entity-resolution-dataset"

MODEL_FILES = [
    ("l2_pruner_xgb.json", "output/l2_pruner_xgb.json"),
    ("phase3_ranker_xgb.json", "output/phase3_ranker_xgb.json"),
    ("phase3_threshold.json", "output/phase3_threshold.json"),
]

FEATURE_FILES = [
    ("ranker_features_train.parquet", "output/ranker_features_train.parquet"),
    ("ranker_features_test.parquet", "output/ranker_features_test.parquet"),
]


def download_hf_file(repo_id: str, filename: str, local_path: str, repo_type: str = "model"):
    """Download a single file from Hugging Face Hub using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("\n[ERROR] `huggingface_hub` package is required to download assets.")
        print("Please install it via: pip install huggingface_hub\n")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    print(f"Downloading {filename} from {repo_id} ({repo_type})...")
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=os.path.dirname(os.path.abspath(local_path)),
            local_dir_use_symlinks=False,
        )
        print(f"  -> Saved to {local_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not download {filename} from {repo_id}: {e}")
        print("If you haven't uploaded artifacts to this repo yet, you can train locally or specify --repo-id.\n")


def main():
    parser = argparse.ArgumentParser(description="Download Entity Resolution Assets from Hugging Face Hub")
    parser.add_argument("--models", action="store_true", help="Download trained L2 and L3 XGBoost model weights (<2MB)")
    parser.add_argument("--features", action="store_true", help="Download precomputed feature parquets (~1.5GB)")
    parser.add_argument("--all", action="store_true", help="Download all available assets")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="Hugging Face Model repo ID")
    parser.add_argument("--data-repo", default=DEFAULT_DATA_REPO, help="Hugging Face Dataset repo ID")
    args = parser.parse_args()

    if not (args.models or args.features or args.all):
        parser.print_help()
        print("\nPlease specify at least one asset group to download (e.g. --models or --all).")
        return

    if args.models or args.all:
        print("=== Downloading Trained Models ===")
        for remote_file, local_path in MODEL_FILES:
            download_hf_file(args.model_repo, remote_file, local_path, repo_type="model")

    if args.features or args.all:
        print("\n=== Downloading Precomputed Feature Parquets ===")
        for remote_file, local_path in FEATURE_FILES:
            download_hf_file(args.data_repo, remote_file, local_path, repo_type="dataset")

    print("\nAsset download process complete.")


if __name__ == "__main__":
    main()
