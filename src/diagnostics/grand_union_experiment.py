import argparse
import polars as pl
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
from collections import defaultdict

def main(model_name: str):
    print("=== GRAND UNION EXPERIMENT: DENSE + TOKEN JOIN ===")
    data_dir = Path("dataset/train")
    
    print("Loading Ground Truth...")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # 5000 sample
    gt_sample = gt_exploded.head(5000)
    s1_ids = gt_sample["source1_entity_id"].to_list()
    true_cand_ids = gt_sample["matched_entity_ids"].to_list()
    
    print("Loading Raw Text Data...")
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_dir / "train_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])
    
    s1_target = s1.filter(pl.col("entity_id").is_in(s1_ids))
    s1_target = pl.DataFrame({"entity_id": s1_ids}).join(s1_target, on="entity_id", how="left")
    
    cands_true = candidates.filter(pl.col("entity_id").is_in(true_cand_ids))
    cands_true = pl.DataFrame({"entity_id": true_cand_ids}).join(cands_true, on="entity_id", how="left")
    
    cands_noise = candidates.sample(n=100000, seed=42)
    cands_pool = pl.concat([cands_true, cands_noise])
    
    def norm(col):
        return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")
        
    s1_names = norm(s1_target["business_name"]).to_list()
    s1_addrs = norm(s1_target["business_address"]).to_list()
    c_names = norm(cands_pool["business_name"]).to_list()
    c_addrs = norm(cands_pool["business_address"]).to_list()
    
    # Combined for Single Lane test
    s1_combined = [n + " " + a for n, a in zip(s1_names, s1_addrs)]
    c_combined = [n + " " + a for n, a in zip(c_names, c_addrs)]
    
    c_ids_pool = cands_pool["entity_id"].to_list()
    
    # -------------------------------------------------------------
    # 1. RUN DENSE RETRIEVAL (Name, Address, and Combined)
    # -------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("Encoding Name Lane...")
    s1_n_emb = model.encode(s1_names, convert_to_tensor=True, normalize_embeddings=True)
    c_n_emb = model.encode(c_names, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    sim_n = torch.matmul(s1_n_emb.float(), c_n_emb.float().T)
    
    print("Encoding Address Lane...")
    s1_a_emb = model.encode(s1_addrs, convert_to_tensor=True, normalize_embeddings=True)
    c_a_emb = model.encode(c_addrs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    sim_a = torch.matmul(s1_a_emb.float(), c_a_emb.float().T)
    
    print("Encoding Single Lane (Name + Address combined string)...")
    s1_c_emb = model.encode(s1_combined, convert_to_tensor=True, normalize_embeddings=True)
    c_c_emb = model.encode(c_combined, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    sim_c = torch.matmul(s1_c_emb.float(), c_c_emb.float().T)
    
    # -------------------------------------------------------------
    # 2. RUN TOKEN JOIN (Inverted Index)
    # -------------------------------------------------------------
    print("\nBuilding Token Join Inverted Index...")
    # Calculate word frequencies in candidate pool
    word_freq = defaultdict(int)
    c_tokens_list = []
    for text in c_combined:
        tokens = set(text.split())
        c_tokens_list.append(tokens)
        for t in tokens:
            word_freq[t] += 1
            
    # Keep rare tokens (appear in less than 50 documents)
    inverted_index = defaultdict(set)
    for c_idx, tokens in enumerate(c_tokens_list):
        for t in tokens:
            if word_freq[t] <= 50:
                inverted_index[t].add(c_idx)
                
    # -------------------------------------------------------------
    # 3. EVALUATE RECALL (Top K Sweeps)
    # -------------------------------------------------------------
    print("Calculating Recall Sweeps...")
    
    k_values = [1, 5, 10, 15, 20, 25, 30]
    results = {}
    
    total = len(s1_ids)
    
    # Calculate Token Join Only first
    hits_token_join = 0
    tj_cands_per_query = []
    for i in range(total):
        tj_cands = set()
        s1_tokens = set(s1_combined[i].split())
        for t in s1_tokens:
            if t in inverted_index:
                tj_cands.update(inverted_index[t])
        tj_cands_per_query.append(tj_cands)
        if i in tj_cands:
            hits_token_join += 1
            
    for k in k_values:
        _, top_n_idx = torch.topk(sim_n, k=k, dim=1)
        _, top_a_idx = torch.topk(sim_a, k=k, dim=1)
        _, top_c_idx = torch.topk(sim_c, k=k, dim=1)
        
        hits_n = 0
        hits_a = 0
        hits_combined = 0
        hits_grand = 0
        
        for i in range(total):
            true_idx = i
            n_cands = set(top_n_idx[i].tolist())
            a_cands = set(top_a_idx[i].tolist())
            c_cands = set(top_c_idx[i].tolist())
            
            if true_idx in n_cands: hits_n += 1
            if true_idx in a_cands: hits_a += 1
            
            # Union of Name and Address (Combined Max)
            union_cands = n_cands.union(a_cands)
            if true_idx in union_cands: hits_combined += 1
            
            # Grand Union (Two-Lane Dense + Token Join)
            grand_cands = union_cands.union(tj_cands_per_query[i])
            if true_idx in grand_cands: hits_grand += 1
            
        results[k] = {
            "name": hits_n / total,
            "addr": hits_a / total,
            "combined": hits_combined / total,
            "grand": hits_grand / total
        }

    print("\n=============================================")
    print(f"Tested on {total} queries against {len(c_ids_pool)} candidates.")
    print("K     | Name Only    | Address Only   | Combined (Max) ")
    print("-------------------------------------------------------")
    for k in k_values:
        print(f"Top {k:<2}|      {results[k]['name']:>6.2%} |        {results[k]['addr']:>6.2%} |         {results[k]['combined']:>6.2%}")
    print("=============================================\n")
    
    print(f"3. Token Join Only (Rare Words):          {hits_token_join/total:>7.2%}")
    
    # We report the Grand Union for the highest K value (K=30)
    print(f"4. GRAND UNION (Two-Lane K=30 + Token Join): {results[30]['grand']:>7.2%}")
    print("=============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()
    main(args.model_name)
