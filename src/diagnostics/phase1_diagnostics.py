import polars as pl
from pathlib import Path
import json

def calculate_recall(candidates_file: Path, ground_truth_file: Path, channel_name: str) -> dict:
    if not candidates_file.exists():
        print(f"Skipping {channel_name}: {candidates_file} not found.")
        return None
        
    print(f"Evaluating {channel_name} Blocking Channel...")
    gt = pl.read_csv(ground_truth_file, separator="\t")
    cands = pl.read_csv(candidates_file, separator="\t")
    
    # Explode candidates
    cands = cands.with_columns(
        pl.col("candidate_entity_ids").str.split(",")
    ).explode("candidate_entity_ids")
    
    cands = cands.filter(pl.col("candidate_entity_ids") != "")
    
    # Explode ground truth
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids")
    
    # Rename for joining
    cands = cands.rename({"candidate_entity_ids": "candidate_id"})
    gt_exploded = gt_exploded.rename({"matched_entity_ids": "true_id"})
    
    total_true_matches = gt_exploded.shape[0]
    total_candidates_retrieved = cands.shape[0]
    
    # Inner join to find hits
    hits = cands.join(
        gt_exploded,
        left_on=["source1_entity_id", "candidate_id"],
        right_on=["source1_entity_id", "true_id"],
        how="inner"
    )
    
    total_hits = hits.shape[0]
    
    recall = total_hits / total_true_matches if total_true_matches > 0 else 0
    precision = total_hits / total_candidates_retrieved if total_candidates_retrieved > 0 else 0
    
    stats = {
        "channel": channel_name,
        "total_true_pairs": total_true_matches,
        "total_candidates_retrieved": total_candidates_retrieved,
        "true_matches_found": total_hits,
        "recall": recall,
        "precision": precision
    }
    
    print(f"[{channel_name}] Recall: {recall:.2%} | Precision: {precision:.2%} | Candidates: {total_candidates_retrieved}")
    return stats

def main():
    data_dir = Path("dataset/train")
    output_dir = Path("output")
    gt_file = data_dir / "train_ground_truth.tsv"
    
    results = []
    
    exact = calculate_recall(output_dir / "candidate_pairs_exact.tsv", gt_file, "Exact Match")
    if exact: results.append(exact)
    
    lexical = calculate_recall(output_dir / "candidate_pairs_lexical.tsv", gt_file, "Lexical (Legacy)")
    if lexical: results.append(lexical)
    
    token_join = calculate_recall(output_dir / "candidate_pairs_token_join.tsv", gt_file, "Two-Lane Token Join")
    if token_join: results.append(token_join)
    
    # We can also evaluate MiniLM if it exists in the output dir (it might be from the test run though!)
    # Actually, the user ran MiniLM on TEST. Local output dir might not have MiniLM for train.
    
    report_dir = Path("results/phase1_diagnostics")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "phase1_diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nPhase 1 Diagnostics saved to {report_path}")

if __name__ == "__main__":
    main()
