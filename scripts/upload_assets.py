"""Hugging Face Hub Asset Uploader for Business Entity Resolution.

Allows the project author to push trained models, extracted feature parquets,
or dataset files to their Hugging Face Model or Dataset repository.

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login  (or pass --token)

Usage:
  # Upload models (<2MB) to your model repo
  python scripts/upload_assets.py --models --repo-id <your-hf-username>/entity-resolution-models

  # Upload feature parquets (~1.5GB) to your dataset repo
  python scripts/upload_assets.py --features --repo-id <your-hf-username>/entity-resolution-dataset
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_FILES = [
    ("output/l2_pruner_xgb.json", "l2_pruner_xgb.json"),
    ("output/phase3_ranker_xgb.json", "phase3_ranker_xgb.json"),
    ("output/phase3_threshold.json", "phase3_threshold.json"),
]

FEATURE_FILES = [
    ("output/ranker_features_train.parquet", "ranker_features_train.parquet"),
    ("output/ranker_features_test.parquet", "ranker_features_test.parquet"),
]


def upload_files(repo_id: str, file_pairs: list, repo_type: str = "model", token: str = None):
    """Upload files to a Hugging Face repository."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[ERROR] huggingface_hub is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Ensure repository exists
    try:
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Verified repository: https://huggingface.co/{'datasets/' if repo_type=='dataset' else ''}{repo_id}")
    except Exception as e:
        print(f"[WARNING] Could not verify/create repo {repo_id}: {e}")

    for local_path, remote_name in file_pairs:
        if not os.path.exists(local_path):
            print(f"[SKIP] Local file not found: {local_path}")
            continue

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {local_path} ({size_mb:.1f} MB) -> {remote_name}...")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            print(f"  -> Uploaded {remote_name} successfully!")
        except Exception as e:
            print(f"  -> [FAILED] Upload of {remote_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Upload Entity Resolution Assets to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="Your Hugging Face repository ID (e.g. username/repo-name)")
    parser.add_argument("--models", action="store_true", help="Upload trained L2 & L3 XGBoost model files (<2MB)")
    parser.add_argument("--features", action="store_true", help="Upload extracted feature parquets (~1.5GB)")
    parser.add_argument("--token", default=None, help="Hugging Face API token (or use `huggingface-cli login`)")
    args = parser.parse_args()

    if not (args.models or args.features):
        parser.print_help()
        print("\nPlease specify at least one upload target: --models or --features.")
        return

    if args.models:
        print(f"\n=== Uploading Models to https://huggingface.co/{args.repo_id} ===")
        upload_files(args.repo_id, MODEL_FILES, repo_type="model", token=args.token)

    if args.features:
        print(f"\n=== Uploading Features to https://huggingface.co/datasets/{args.repo_id} ===")
        upload_files(args.repo_id, FEATURE_FILES, repo_type="dataset", token=args.token)

    print("\nUpload process finished.")


if __name__ == "__main__":
    main()
