"""Dual-Lane FAISS HNSW Index Builder for Entity Resolution.

Encodes candidate business records across two independent representation lanes:
  1. Name Lane: normalized business_name
  2. Address Lane: normalized business_address + country

Constructs memory-efficient HNSW Flat indices with unit-normalized BGE-M3
embeddings, allowing O(log N) approximate nearest neighbor retrieval across
10+ million entities.
"""

import argparse
import gc
import os
import time
from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from src.core.text import norm_series


def build_lane_index(
    texts: List[str],
    model: SentenceTransformer,
    index_output_path: str,
    d: int = 1024,
    M: int = 32,
    ef_construction: int = 200,
    chunk_size: int = 1_000_000,
    batch_size: int = 2048,
):
    """Build a single HNSW index for a text lane in streaming chunks."""
    os.makedirs(os.path.dirname(os.path.abspath(index_output_path)), exist_ok=True)
    print(f"Initializing HNSW index (d={d}, M={M}, efConstruction={ef_construction})...")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction

    total = len(texts)
    start_time = time.time()

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        print(f"  Encoding chunk [{chunk_start:,}:{chunk_end:,}] of {total:,}...")

        chunk_emb = model.encode(
            texts[chunk_start:chunk_end],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        index.add(chunk_emb.cpu().numpy().astype(np.float32))

        del chunk_emb
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Lane built in {time.time() - start_time:.1f}s. Saving to {index_output_path}...")
    faiss.write_index(index, index_output_path)
    print("Saved successfully.\n")


def build_all_indices(
    data_dir: str = "dataset",
    output_dir: str = "output/faiss_train",
    split: str = "train",
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 2048,
    device: Optional[str] = None,
    skip_existing: bool = True,
):
    """Load candidates and build both Name and Address FAISS indices."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    name_idx_path = os.path.join(output_dir, "candidates_name_hnsw.index")
    addr_idx_path = os.path.join(output_dir, "candidates_addr_hnsw.index")

    if skip_existing and os.path.exists(name_idx_path) and os.path.exists(addr_idx_path):
        print(f"Both indices already exist in {output_dir}. Skipping.")
        return

    s2_path = os.path.join(data_dir, split, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.tsv")

    print(f"Loading candidate entities from {s2_path} and {s3_path}...")
    s2 = pl.read_csv(s2_path, separator="\t")
    s3 = pl.read_csv(s3_path, separator="\t")
    candidates = pl.concat([s2, s3])
    del s2, s3
    gc.collect()

    print("Normalizing candidate texts...")
    c_names = norm_series(candidates["business_name"]).to_list()
    c_addrs = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_list()
    del candidates
    gc.collect()

    print(f"Loading {model_name} on {device} (FP16)...")
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )

    # 1. Name Lane
    if not (skip_existing and os.path.exists(name_idx_path)):
        print("=== Building Name Lane Index ===")
        build_lane_index(c_names, model, name_idx_path, batch_size=batch_size)
    else:
        print(f"Name lane index already exists at {name_idx_path}.")

    del c_names
    gc.collect()

    # 2. Address Lane
    if not (skip_existing and os.path.exists(addr_idx_path)):
        print("=== Building Address Lane Index ===")
        build_lane_index(c_addrs, model, addr_idx_path, batch_size=batch_size)
    else:
        print(f"Address lane index already exists at {addr_idx_path}.")

    del c_addrs, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Build Dual-Lane FAISS HNSW Indices")
    parser.add_argument("--data-dir", default="dataset", help="Root data directory containing split folders")
    parser.add_argument("--output-dir", default="output/faiss_train", help="Target directory for FAISS indices")
    parser.add_argument("--split", default="train", help="Dataset split (train or test)")
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Hugging Face embedding model name")
    parser.add_argument("--batch-size", type=int, default=2048, help="Encoding batch size")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild even if indices exist")
    args = parser.parse_args()

    build_all_indices(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        skip_existing=not args.force_rebuild,
    )


if __name__ == "__main__":
    main()
