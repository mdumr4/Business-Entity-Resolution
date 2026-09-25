import polars as pl
import argparse
from pathlib import Path
import time
import os

def calculate_recall(gt_exploded, candidates_path, total_true_matches, name):
    if not os.path.exists(candidates_path):
        return f"{name:<15} | File Not Found"
        
    candidates = pl.read_csv(candidates_path, separator="\t")
    
    evaluation_df = gt_exploded.join(candidates, left_on="source1_entity_id", right_on="source1_entity_id", how="left")
    evaluation_df = evaluation_df.with_columns(pl.col("candidate_entity_ids").fill_null(""))
    
    hits = evaluation_df.filter(
        pl.col("candidate_entity_ids").str.contains(pl.col("matched_entity_ids"))
    ).shape[0]
    
    recall = hits / total_true_matches
    return f"{name:<15} | {recall:>7.2%} ({hits:,}/{total_true_matches:,})"

def main(dense_path: str, token_path: str, grand_path: str):
    print("=== FINAL PHASE 1 RECALL CALCULATOR (FULL DATASET) ===")
    
    data_dir = Path("dataset/train")
    start = time.time()
    
    print("Loading Ground Truth...")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    total_true_matches = gt_exploded.shape[0]
    
    print("\nEvaluating Dense Retrieval Only...")
    dense_stats = calculate_recall(gt_exploded, dense_path, total_true_matches, "Dense Only")
    
    print("Evaluating Token Join Only...")
    token_stats = calculate_recall(gt_exploded, token_path, total_true_matches, "Token Join Only")
    
    print("Evaluating Grand Union...")
    grand_stats = calculate_recall(gt_exploded, grand_path, total_true_matches, "Grand Union")
    
    print("\n=======================================================")
    print("         FULL DATASET RECALL REPORT (2.2M Queries)     ")
    print("=======================================================")
    print(f"Total True Pairs (Ground Truth): {total_true_matches:,}")
    print("-" * 55)
    print("Method          | Recall")
    print("-" * 55)
    print(dense_stats)
    print(token_stats)
    print(grand_stats)
    print("=======================================================")
    print(f"Completed in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=str, default="output/candidate_pairs_dense_two_lane.tsv")
    parser.add_argument("--token", type=str, default="output/candidate_pairs_token_join.tsv")
    parser.add_argument("--grand", type=str, default="output/candidate_pairs_grand_union.tsv")
    args = parser.parse_args()
    main(args.dense, args.token, args.grand)
