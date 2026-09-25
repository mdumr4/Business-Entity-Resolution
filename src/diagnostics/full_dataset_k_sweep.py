import polars as pl
import torch
import gc
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def norm(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def main(data_dir: str, token_join_path: str, max_k: int = 30):
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path(data_dir)
    
    print("=== FULL DATASET K-SWEEP DIAGNOSTIC (2.2M vs 10.3M) ===")
    
    # 1. Load Ground Truth
    print("Loading Ground Truth...")
    gt = pl.read_csv(data_path / "train_ground_truth.tsv", separator="\t")
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # Create a fast lookup: S1_ID -> set(True Matches)
    s1_to_gt = gt_exploded.group_by("source1_entity_id").agg(
        pl.col("matched_entity_ids")
    ).to_dict(as_series=False)
    # dict structure: s1_to_gt['source1_entity_id'] -> list of ids, s1_to_gt['matched_entity_ids'] -> list of lists
    gt_dict = {k: set(v) for k, v in zip(s1_to_gt['source1_entity_id'], s1_to_gt['matched_entity_ids'])}
    
    # 2. Load Text Data
    print("Loading Source 1 and Candidates...")
    s1 = pl.read_csv(data_path / "train_source1.tsv", separator="\t")
    
    # Take a random sample of 50,000 queries
    print("Sampling 50,000 random queries from S1...")
    s1 = s1.sample(n=50000, seed=42)
    
    s2 = pl.read_csv(data_path / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_path / "train_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])
    
    s1_ids = s1["entity_id"].to_numpy()
    c_ids = candidates["entity_id"].to_numpy()
    
    s1_names = norm(s1["business_name"]).to_list()
    s1_addrs = norm(s1["business_address"]).to_list()
    s1_comb = [n + " " + a for n, a in zip(s1_names, s1_addrs)]
    
    c_names = norm(candidates["business_name"]).to_list()
    c_addrs = norm(candidates["business_address"]).to_list()
    c_comb = [n + " " + a for n, a in zip(c_names, c_addrs)]
    
    num_queries = len(s1_ids)
    total_candidates = len(c_ids)
    
    # 3. Load Token Join Pairs
    print(f"Loading Token Join output from {token_join_path}...")
    tj_df = pl.read_csv(token_join_path, separator="\t")
    # Filter Token Join candidates to only those in our 50k sample
    tj_df = tj_df.filter(pl.col("source1_entity_id").is_in(s1["entity_id"]))
    tj_df = tj_df.with_columns(pl.col("candidate_entity_ids").str.split(",")).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
    tj_agg = tj_df.group_by("source1_entity_id").agg(pl.col("candidate_entity_ids")).to_dict(as_series=False)
    tj_dict = {k: set(v) for k, v in zip(tj_agg['source1_entity_id'], tj_agg['candidate_entity_ids'])}
    
    # 4. Dense Retrieval setup
    model_name = "BAAI/bge-m3"
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("Encoding S1 Queries (Name, Address, Combined)...")
    q_emb_n = model.encode(s1_names, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
    q_emb_a = model.encode(s1_addrs, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
    q_emb_c = model.encode(s1_comb, batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
    
    # Rolling Top K buffers
    def init_buffers():
        scores = torch.full((num_queries, max_k), -float('inf'), device=device, dtype=torch.float16)
        indices = torch.zeros((num_queries, max_k), device=device, dtype=torch.long)
        return scores, indices
        
    topk_s_n, topk_i_n = init_buffers()
    topk_s_a, topk_i_a = init_buffers()
    topk_s_c, topk_i_c = init_buffers()
    
    chunk_size = 1_000_000
    query_batch = 1000
    
    print(f"Starting Dense Search over {total_candidates} candidates...")
    for chunk_start in range(0, total_candidates, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_candidates)
        print(f"\n--- Chunk [{chunk_start}:{chunk_end}] ---")
        
        c_emb_n = model.encode(c_names[chunk_start:chunk_end], batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
        c_emb_a = model.encode(c_addrs[chunk_start:chunk_end], batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
        c_emb_c = model.encode(c_comb[chunk_start:chunk_end], batch_size=1024, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True).half()
        
        for q_start in tqdm(range(0, num_queries, query_batch), desc="Scoring"):
            q_end = min(q_start + query_batch, num_queries)
            
            def update_topk(q_emb, c_emb, g_scores, g_indices):
                sim = torch.matmul(q_emb[q_start:q_end], c_emb.T)
                scores, indices = torch.topk(sim, k=min(max_k, sim.shape[1]), dim=1)
                abs_indices = indices + chunk_start
                
                comb_scores = torch.cat([g_scores[q_start:q_end], scores], dim=1)
                comb_indices = torch.cat([g_indices[q_start:q_end], abs_indices], dim=1)
                
                new_scores, topk_idx = torch.topk(comb_scores, k=max_k, dim=1)
                new_indices = torch.gather(comb_indices, 1, topk_idx)
                
                g_scores[q_start:q_end] = new_scores
                g_indices[q_start:q_end] = new_indices

            update_topk(q_emb_n, c_emb_n, topk_s_n, topk_i_n)
            update_topk(q_emb_a, c_emb_a, topk_s_a, topk_i_a)
            update_topk(q_emb_c, c_emb_c, topk_s_c, topk_i_c)
            
        del c_emb_n, c_emb_a, c_emb_c
        torch.cuda.empty_cache()
        gc.collect()

    # Move indices to CPU for evaluation
    idx_n = topk_i_n.cpu().numpy()
    idx_a = topk_i_a.cpu().numpy()
    idx_c = topk_i_c.cpu().numpy()
    
    del topk_s_n, topk_i_n, topk_s_a, topk_i_a, topk_s_c, topk_i_c
    torch.cuda.empty_cache()
    
    # 5. Evaluate Recall
    print("\nCalculating Final Recall Sweeps...")
    k_values = [1, 5, 10, 15, 20, 25, 30]
    results = {k: {"name": 0, "addr": 0, "comb": 0, "gu_two_lane": 0, "gu_combined": 0} for k in k_values}
    
    total_true_pairs = gt_exploded.shape[0]
    hits_tj = 0
    
    for i in tqdm(range(num_queries), desc="Evaluating"):
        s1_id = s1_ids[i]
        if s1_id not in gt_dict:
            continue
            
        true_cands = gt_dict[s1_id]
        tj_cands = tj_dict.get(s1_id, set())
        
        # Check if true candidate is in token join
        for tc in true_cands:
            if tc in tj_cands:
                hits_tj += 1
                
        for k in k_values:
            # Map index array to actual candidate IDs
            cands_n = set(c_ids[idx_n[i, :k]])
            cands_a = set(c_ids[idx_a[i, :k]])
            cands_c = set(c_ids[idx_c[i, :k]])
            
            union_two_lane = cands_n.union(cands_a)
            gu_two_lane = union_two_lane.union(tj_cands)
            gu_combined = cands_c.union(tj_cands)
            
            for tc in true_cands:
                if tc in cands_n: results[k]["name"] += 1
                if tc in cands_a: results[k]["addr"] += 1
                if tc in union_two_lane: results[k]["comb"] += 1
                if tc in gu_two_lane: results[k]["gu_two_lane"] += 1
                if tc in gu_combined: results[k]["gu_combined"] += 1

    print("\n=======================================================")
    print(f"Tested on {num_queries:,} S1 queries against {total_candidates:,} candidates.")
    print(f"Total True Pairs (Ground Truth): {total_true_pairs:,}")
    print("=======================================================")
    print("K     | Name Only    | Address Only   | Combined (Max) ")
    print("-------------------------------------------------------")
    for k in k_values:
        rn = results[k]["name"] / total_true_pairs
        ra = results[k]["addr"] / total_true_pairs
        rc = results[k]["comb"] / total_true_pairs
        print(f"Top {k:<2}|      {rn:>6.2%} |        {ra:>6.2%} |         {rc:>6.2%}")
    print("=======================================================\n")
    
    print(f"Token Join Only (Rare Words):           {hits_tj/total_true_pairs:>7.2%}")
    
    print("\n=======================================================")
    print("             GRAND UNION PERFORMANCE                   ")
    print("=======================================================")
    print("K     | GU (Two-Lane + TJ) | GU (Combined + TJ) ")
    print("-------------------------------------------------------")
    for k in k_values:
        gu_2l = results[k]["gu_two_lane"] / total_true_pairs
        gu_c = results[k]["gu_combined"] / total_true_pairs
        print(f"Top {k:<2}|             {gu_2l:>6.2%} |             {gu_c:>6.2%}")
    print("=======================================================")
    
    print(f"\nExecution Time: {(time.time() - start_time) / 3600:.2f} Hours.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dataset/train")
    parser.add_argument("--token-join", type=str, default="output/candidate_pairs_token_join.tsv")
    args = parser.parse_args()
    main(args.data_dir, args.token_join)
