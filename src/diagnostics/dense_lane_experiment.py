import argparse
import polars as pl
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

def evaluate_top_k(sim_scores, true_indices, top_k):
    _, top_k_indices = torch.topk(sim_scores, k=top_k, dim=1)
    hits = 0
    for i, true_idx in enumerate(true_indices):
        if true_idx in top_k_indices[i]:
            hits += 1
    return hits / len(true_indices)

def main(model_name: str, strip_numbers: bool):
    print("=== Dense Retrieval Two-Lane Experiment ===")
    if strip_numbers:
        print(">>> WARNING: STRIPPING ALL NUMBERS FROM TEXT TO TEST THEORY <<<")
        
    data_dir = Path("dataset/train")
    
    print("Loading Ground Truth...")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
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
        # 1. Lowercase and remove punctuation
        col_norm = col.str.to_lowercase().str.replace_all(r"[^\w\s]", "")
        # 2. Strip digits if flag is set
        if strip_numbers:
            col_norm = col_norm.str.replace_all(r"\d+", "")
        return col_norm.fill_null("")
        
    s1_names = norm(s1_target["business_name"]).to_list()
    s1_addrs = norm(s1_target["business_address"]).to_list()
    c_names = norm(cands_pool["business_name"]).to_list()
    c_addrs = norm(cands_pool["business_address"]).to_list()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("\n--- LANE 1: NAME ENCODING ---")
    start = time.time()
    s1_name_emb = model.encode(s1_names, convert_to_tensor=True, normalize_embeddings=True)
    c_name_emb = model.encode(c_names, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    print(f"Name Encoding took {time.time() - start:.2f}s")
    
    print("\n--- LANE 2: ADDRESS ENCODING ---")
    start = time.time()
    s1_addr_emb = model.encode(s1_addrs, convert_to_tensor=True, normalize_embeddings=True)
    c_addr_emb = model.encode(c_addrs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    print(f"Address Encoding took {time.time() - start:.2f}s")
    
    print("\nCalculating Similarity Matrices...")
    name_sims = torch.matmul(s1_name_emb.float(), c_name_emb.float().T)
    addr_sims = torch.matmul(s1_addr_emb.float(), c_addr_emb.float().T)
    
    # We use MAX here since we proved Average drops recall heavily on missing addresses
    combined_sims = torch.max(name_sims, addr_sims)
    
    true_indices = list(range(len(s1_ids)))
    
    print("\n=============================================")
    print(f"      DENSE RETRIEVAL: {model_name}     ")
    if strip_numbers:
        print("      >>> NUMBERS STRIPPED FROM TEXT <<<")
    print("=============================================")
    print(f"Tested on {len(s1_ids)} queries against {cands_pool.shape[0]} candidates.")
    print(f"{'K':<5} | {'Name Only':<12} | {'Address Only':<14} | {'Combined (Max)':<15}")
    print("-" * 55)
    
    for k in [1, 5, 10, 15, 20]:
        n_recall = evaluate_top_k(name_sims, true_indices, k)
        a_recall = evaluate_top_k(addr_sims, true_indices, k)
        c_recall = evaluate_top_k(combined_sims, true_indices, k)
        print(f"Top {k:<2}| {n_recall:>11.2%} | {a_recall:>13.2%} | {c_recall:>14.2%}")
    print("=============================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    parser.add_argument("--strip-numbers", action="store_true", help="Remove all digits from text before encoding")
    args = parser.parse_args()
    main(args.model_name, args.strip_numbers)
