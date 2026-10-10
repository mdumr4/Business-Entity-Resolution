"""Hugging Face Hub Asset Uploader & Bucket Sync for Business Entity Resolution.

Supports syncing to your Hugging Face Storage Bucket using native `hf sync`:
  hf://buckets/mdumr4/Business-Entity-Resolution

Usage:
  # Sync golden artifacts (models + feature parquets, ~1.5GB) to your HF Bucket:
  python scripts/upload_assets.py --bucket mdumr4/Business-Entity-Resolution

  # Or sync using native hf CLI directly:
  hf sync output hf://buckets/mdumr4/Business-Entity-Resolution --include "*.parquet" --include "*.json"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_BUCKET = "hf://buckets/mdumr4/Business-Entity-Resolution"


def sync_to_bucket(bucket_uri: str = DEFAULT_BUCKET, dry_run: bool = False):
    """Sync feature parquets, models, and results to the Hugging Face Bucket."""
    if not bucket_uri.startswith("hf://buckets/"):
        bucket_uri = f"hf://buckets/{bucket_uri}"

    cmd = [
        "hf", "sync", "output", bucket_uri,
        "--include", "*.parquet",
        "--include", "*.json",
        "--include", "matching_results.tsv",
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n[HF Bucket Sync] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[HINT] If not logged in, run: hf auth login")
        sys.exit(result.returncode)
    print("\nBucket sync completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Upload Entity Resolution Assets to Hugging Face")
    parser.add_argument(
        "--bucket",
        default="mdumr4/Business-Entity-Resolution",
        help="Hugging Face Bucket name or URI (default: mdumr4/Business-Entity-Resolution)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview files to be uploaded without uploading")
    args = parser.parse_args()

    sync_to_bucket(bucket_uri=args.bucket, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
