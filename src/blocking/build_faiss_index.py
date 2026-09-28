import faiss
import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
import time
import torch
from sentence_transformers import SentenceTransformer

def norm(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def build_lane_index(texts: list, model: SentenceTransformer, index_output_path: str, d: int = 1024, M: int = 32, efConstruction: int = 200):
    print(f"Initializing HNSW index (M={M}, efConstruction={efConstruction})...")
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = efConstruction
    
    total = len(texts)
    chunk_size = 1_000_000
    
    start_time = time.time()
    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        print(f"\nEncoding Chunk [{chunk_start}:{chunk_end}] of {total}...")
        
        # Encode on GPU (FP16)
        chunk_emb = model.encode(texts[chunk_start:chunk_end], batch_size=2048, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
        
        # Move to CPU, cast to FP32, and add to FAISS
        print("Adding chunk to FAISS index...")
        index.add(chunk_emb.cpu().numpy().astype(np.float32))
        
        # Free VRAM
        del chunk_emb
        import gc
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
    end_time = time.time()
    print(f"Lane index built in {end_time - start_time:.2f} seconds.")
    
    print(f"Saving index to {index_output_path}...")
    faiss.write_index(index, index_output_path)
    print("Done.\n")

def main(split):
    output_dir = f"output/faiss_{split}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading Candidates from {split} TSV...")
    s2 = pl.read_csv(f"dataset/{split}/{split}_source2.tsv", separator="\t")
    s3 = pl.read_csv(f"dataset/{split}/{split}_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])
    
    print("Normalizing Text Data...")
    c_names = norm(candidates["business_name"]).to_list()
    # Mirroring the exact concatenation used in bge_rrf_blocker.py
    c_addrs = (norm(candidates["business_address"]) + " " + norm(candidates["country"])).to_list()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE-M3 model on {device} (FP16 optimized)...")
    model = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("=== Skipping Name Lane (already built) ===")
    # build_lane_index(c_names, model, f"{output_dir}/candidates_name_hnsw.index")
    
    del c_names
    import gc
    gc.collect()
    
    print("=== Encoding and Building Address Lane ===")
    build_lane_index(c_addrs, model, f"{output_dir}/candidates_addr_hnsw.index")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args.split)
