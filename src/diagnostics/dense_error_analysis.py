import argparse
import polars as pl
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def main(model_name: str):
    print(f"=== Dense Retrieval Error Analysis: {model_name} ===")
    data_dir = Path("dataset/train")
    
    print("Loading Ground Truth...")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    gt_sample = gt_exploded.head(2000)
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
    
    cands_noise = candidates.sample(n=50000, seed=42)
    cands_pool = pl.concat([cands_true, cands_noise])
    
    s1_orig_names = s1_target["business_name"].fill_null("").to_list()
    s1_orig_addrs = s1_target["business_address"].fill_null("").to_list()
    c_orig_names = cands_pool["business_name"].fill_null("").to_list()
    c_orig_addrs = cands_pool["business_address"].fill_null("").to_list()
    
    def norm(col):
        return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")
        
    s1_names = norm(s1_target["business_name"]).to_list()
    s1_addrs = norm(s1_target["business_address"]).to_list()
    c_names = norm(cands_pool["business_name"]).to_list()
    c_addrs = norm(cands_pool["business_address"]).to_list()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    
    print("Encoding Names...")
    s1_name_emb = model.encode(s1_names, convert_to_tensor=True, normalize_embeddings=True)
    c_name_emb = model.encode(c_names, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    
    print("Encoding Addresses...")
    s1_addr_emb = model.encode(s1_addrs, convert_to_tensor=True, normalize_embeddings=True)
    c_addr_emb = model.encode(c_addrs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
    
    name_sims = torch.matmul(s1_name_emb.float(), c_name_emb.float().T)
    addr_sims = torch.matmul(s1_addr_emb.float(), c_addr_emb.float().T)
    
    # We use Union Logic here: Take max instead of average!
    # Because if they match perfectly in one lane, we don't want missing data in the other lane to destroy the score.
    combined_sims = torch.max(name_sims, addr_sims)
    
    _, top20_indices = torch.topk(combined_sims, k=20, dim=1)
    
    examples_found_missed = 0
    missed_examples = []
    
    for i in range(len(s1_ids)):
        true_cand_idx = i
        rank = -1
        for r, c_idx in enumerate(top20_indices[i]):
            if c_idx.item() == true_cand_idx:
                rank = r + 1
                break
                
        # Completely Missed (Rank > 20)
        if rank == -1 and examples_found_missed < 5:
            missed_examples.append(i)
            examples_found_missed += 1
            
        if examples_found_missed >= 5:
            break

    print("\n=========================================================")
    print(f" THE 1.5%: WHY {model_name} COMPLETELY FAILED ")
    print("=========================================================")
    
    if len(missed_examples) == 0:
        print("Wow! No examples missed the Top 20 in this sample size!")
    
    for idx, i in enumerate(missed_examples):
        true_cand_idx = i
        print(f"\n--- FATAL MISS EXAMPLE {idx + 1} ---")
        print(f"QUERY S1:")
        print(f"  Name:    {s1_orig_names[i]}")
        print(f"  Address: {s1_orig_addrs[i]}\n")
        
        print("THE FALSE POSITIVES THAT TOOK THE TOP SPOTS (Top 3):")
        for r in range(3):
            c_idx = top20_indices[i][r].item()
            n_score = name_sims[i][c_idx].item()
            a_score = addr_sims[i][c_idx].item()
            print(f"  #{r+1} (NameSim: {n_score:.2f} | AddrSim: {a_score:.2f}) | Name: {c_orig_names[c_idx]} | Addr: {c_orig_addrs[c_idx]}")
            
        n_true = name_sims[i][true_cand_idx].item()
        a_true = addr_sims[i][true_cand_idx].item()
        print(f"\nTHE TRUE MATCH (Missed):")
        print(f"  (NameSim: {n_true:.2f} | AddrSim: {a_true:.2f}) | Name: {c_orig_names[true_cand_idx]} | Addr: {c_orig_addrs[true_cand_idx]}")
        print("---------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="BAAI/bge-m3")
    args = parser.parse_args()
    main(args.model_name)
