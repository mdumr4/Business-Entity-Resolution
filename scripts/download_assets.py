"""Hugging Face Hub Asset Downloader for Business Entity Resolution.

Supports downloading pre-trained models, extracted feature parquets, or raw data
directly from your Hugging Face Storage Bucket (`mdumr4/Business-Entity-Resolution`)
or public repositories.

Usage:
  # Sync all golden assets from the Hugging Face Bucket:
  python scripts/download_assets.py --bucket mdumr4/Business-Entity-Resolution

  # Or using native hf CLI:
  hf sync hf://buckets/mdumr4/Business-Entity-Resolution output
"""

import argparse
import os
import subprocess
import sys

DEFAULT_BUCKET = "hf://buckets/mdumr4/Business-Entity-Resolution"


def sync_from_bucket(bucket_uri: str = DEFAULT_BUCKET, target_dir: str = "output"):
    """Download artifacts from the Hugging Face Bucket using `hf sync`."""
    if not bucket_uri.startswith("hf://buckets/"):
        bucket_uri = f"hf://buckets/{bucket_uri}"

    os.makedirs(target_dir, exist_ok=True)
    cmd = ["hf", "sync", bucket_uri, target_dir]
    print(f"\n[HF Bucket Sync] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[ERROR] `hf sync` failed. Ensure you are logged in via: hf auth login")
        sys.exit(result.returncode)
    print(f"\nSynced assets from {bucket_uri} into {target_dir}/ successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download Entity Resolution Assets from Hugging Face")
    parser.add_argument(
        "--bucket",
        default="mdumr4/Business-Entity-Resolution",
        help="Hugging Face Bucket name or URI (default: mdumr4/Business-Entity-Resolution)",
    )
    parser.add_argument("--target-dir", default="output", help="Target local directory (default: output)")
    args = parser.parse_args()

    sync_from_bucket(bucket_uri=args.bucket, target_dir=args.target_dir)


if __name__ == "__main__":
    main()
