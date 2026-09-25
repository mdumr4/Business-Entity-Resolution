import polars as pl
import torch
import gc
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_lane(model, q_texts, c_texts, top_k, chunk_size=1_000_000, query_batch=1000):
    """Encodes and searches a single text lane (e.g., Name only or Address only)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_queries = len(q_texts)
    
    print(f"Encoding {num_queries} queries...")
    q_embeddings = model.encode(q_texts, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
    
    global_topk_scores = torch.full((num_queries, top_k), -float('inf'), device=device, dtype=torch.float16)
    global_topk_indices = torch.zeros((num_queries, top_k), device=device, dtype=torch.long)
    
    total_candidates = len(c_texts)
    print(f"Processing {total_candidates} candidates in chunks of {chunk_size}...")
    
    for chunk_start in range(0, total_candidates, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_candidates)
        print(f"\n--- Processing Candidate Chunk [{chunk_start}:{chunk_end}] ---")
        
        chunk_texts = c_texts[chunk_start:chunk_end]
        c_embeddings = model.encode(chunk_texts, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
        
        print("Computing similarity matrix...")
        for q_start in tqdm(range(0, num_queries, query_batch), desc="Scoring"):
            q_end = min(q_start + query_batch, num_queries)
            
            sim_scores = torch.matmul(q_embeddings[q_start:q_end], c_embeddings.T)
            scores, indices = torch.topk(sim_scores, k=min(top_k, sim_scores.shape[1]), dim=1)
            absolute_indices = indices + chunk_start
            
            combined_scores = torch.cat([global_topk_scores[q_start:q_end], scores], dim=1)
            combined_indices = torch.cat([global_topk_indices[q_start:q_end], absolute_indices], dim=1)
            
            new_topk_scores, topk_idx = torch.topk(combined_scores, k=top_k, dim=1)
            new_topk_indices = torch.gather(combined_indices, 1, topk_idx)
            
            global_topk_scores[q_start:q_end] = new_topk_scores
            global_topk_indices[q_start:q_end] = new_topk_indices
            
        del c_embeddings, sim_scores
        torch.cuda.empty_cache()
        gc.collect()
        
    return global_topk_indices.cpu().numpy()

def norm(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def main(data_dir: str, output_dir: str, top_k: int):
    set_seed(42)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    model_name = "BAAI/bge-m3"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    prefix = "test" if "test" in data_dir.lower() else "train"
    print(f"Loading Source 1 ({prefix})...")
    s1 = pl.read_csv(data_path / f"{prefix}_source1.tsv", separator="\t")
    
    print("Loading Candidates...")
    s2 = pl.read_csv(data_path / f"{prefix}_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_path / f"{prefix}_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])
    
    s1_names = norm(s1["business_name"]).to_list()
    s1_addrs = norm(s1["business_address"]).to_list()
    c_names = norm(candidates["business_name"]).to_list()
    c_addrs = norm(candidates["business_address"]).to_list()
    c_ids = candidates["entity_id"].to_numpy()
    
    print("\n==============================")
    print("     LANE 1: NAME SEARCH      ")
    print("==============================")
    name_indices = run_lane(model, s1_names, c_names, top_k=top_k)
    
    print("\n==============================")
    print("    LANE 2: ADDRESS SEARCH    ")
    print("==============================")
    addr_indices = run_lane(model, s1_addrs, c_addrs, top_k=top_k)
    
    print("\nFormatting and Unioning Results...")
    results = []
    
    s1_ids = s1["entity_id"].to_list()
    for i in tqdm(range(len(s1_ids)), desc="Merging"):
        # Union the candidates from Name and Address
        n_cands = set(c_ids[name_indices[i]])
        a_cands = set(c_ids[addr_indices[i]])
        merged = list(n_cands.union(a_cands))
        
        results.append({
            "source1_entity_id": s1_ids[i], 
            "candidate_entity_ids": ",".join(merged)
        })
        
    final_df = pl.DataFrame(results)
    output_file = out_path / "candidate_pairs_dense_two_lane.tsv"
    final_df.write_csv(output_file, separator="\t")
    
    # Maximum candidates per query is top_k * 2 (e.g. 10 * 2 = 20)
    print(f"\nSuccess! Saved Two-Lane Dense candidates (Max {top_k*2} per query) to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10, help="Top K to pull per lane (Name and Address). Max candidates = Top K * 2.")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.top_k)
