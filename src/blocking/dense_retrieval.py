import polars as pl
import torch
import gc
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def create_text_field(df: pl.DataFrame) -> pl.DataFrame:
    """Combines business_name and business_address for the embedding model."""
    return df.with_columns(
        pl.concat_str(
            [
                pl.col("business_name").fill_null(""),
                pl.lit(" | "),
                pl.col("business_address").fill_null("")
            ]
        ).alias("text_for_embedding")
    )

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(data_dir: str, output_dir: str):
    set_seed(42)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    # We use BGE-m3 for exceptional multilingual zero-shot accuracy
    model_name = "BAAI/bge-m3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name} on {device}")
    
    # Use float16 to halve VRAM/RAM usage while retaining 99.9% accuracy
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    # 1. Process Source 1 (Queries)
    print("Loading Source 1 (Queries)...")
    s1 = create_text_field(pl.read_csv(data_path / "train_source1.tsv", separator="\t"))
    s1_texts = s1["text_for_embedding"].to_list()
    
    print(f"Extracting embeddings for {len(s1_texts)} S1 rows...")
    q_embeddings = model.encode(s1_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    q_embeddings = q_embeddings.half() # Cast to float16
    
    TOP_K = 20
    num_queries = q_embeddings.shape[0]
    
    # Global Top-K memory buffers (Keeps RAM usage extremely low!)
    global_topk_scores = torch.full((num_queries, TOP_K), -float('inf'), dtype=torch.float16, device=device)
    global_topk_indices = torch.zeros((num_queries, TOP_K), dtype=torch.long, device=device)
    
    # 2. Process Candidates (S2 + S3)
    print("Loading Source 2 and 3 Entity IDs...")
    s2 = create_text_field(pl.read_csv(data_path / "train_source2.tsv", separator="\t"))
    s3 = create_text_field(pl.read_csv(data_path / "train_source3.tsv", separator="\t"))
    candidates_df = pl.concat([s2, s3])
    candidate_ids = candidates_df["entity_id"].to_numpy()
    candidate_texts = candidates_df["text_for_embedding"].to_list()
    
    del s2, s3, candidates_df
    gc.collect()
    
    # Chunking prevents the 40GB RAM OOM explosion
    CHUNK_SIZE = 1_000_000 
    total_candidates = len(candidate_texts)
    print(f"Processing {total_candidates} candidates in chunks of {CHUNK_SIZE} (Memory Safe Mode)...")
    
    for chunk_start in range(0, total_candidates, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, total_candidates)
        print(f"\n--- Processing Candidate Chunk [{chunk_start}:{chunk_end}] ---")
        
        chunk_texts = candidate_texts[chunk_start:chunk_end]
        c_embeddings = model.encode(chunk_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
        c_embeddings = c_embeddings.half()
        
        print("Computing similarity matrix...")
        # CRITICAL VRAM FIX: Changed from 50,000 to 1,000 so the intermediate sim_scores matrix only takes ~2GB of VRAM instead of 100GB+
        QUERY_BATCH = 1000
        for q_start in tqdm(range(0, num_queries, QUERY_BATCH), desc="Scoring"):
            q_end = min(q_start + QUERY_BATCH, num_queries)
            
            # Exact Matrix Multiplication (Cosine Similarity)
            sim_scores = torch.matmul(q_embeddings[q_start:q_end], c_embeddings.T)
            scores, indices = torch.topk(sim_scores, k=min(TOP_K, sim_scores.shape[1]), dim=1)
            
            absolute_indices = indices + chunk_start
            
            combined_scores = torch.cat([global_topk_scores[q_start:q_end], scores], dim=1)
            combined_indices = torch.cat([global_topk_indices[q_start:q_end], absolute_indices], dim=1)
            
            new_topk_scores, topk_idx = torch.topk(combined_scores, k=TOP_K, dim=1)
            new_topk_indices = torch.gather(combined_indices, 1, topk_idx)
            
            global_topk_scores[q_start:q_end] = new_topk_scores
            global_topk_indices[q_start:q_end] = new_topk_indices
            
        del c_embeddings, sim_scores
        torch.cuda.empty_cache()
        gc.collect()
        
    print("\nExtraction and Search Complete! Formatting results...")
    results = []
    global_topk_indices_cpu = global_topk_indices.cpu().numpy()
    
    for i, s1_id in enumerate(tqdm(s1["entity_id"].to_list(), desc="Formatting")):
        winning_indices = global_topk_indices_cpu[i]
        retrieved_ids = candidate_ids[winning_indices]
        
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(retrieved_ids)
        })
        
    final_df = pl.DataFrame(results)
    output_file = out_path / "candidate_pairs_dense.tsv"
    final_df.write_csv(output_file, separator="\t")
    print(f"Success! Saved dense candidates to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset/train")
    parser.add_argument("--output-dir", type=str, default="../../output")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
