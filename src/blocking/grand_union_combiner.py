import polars as pl
from pathlib import Path
import argparse
import time

def main(dense_path: str, token_join_path: str, output_path: str):
    print(f"=== THE GRAND UNION: Combining Candidates ===")
    
    start = time.time()
    
    print(f"Loading Dense Candidates from: {dense_path}...")
    try:
        dense_cands = pl.read_csv(dense_path, separator="\t")
        print(f"  -> Found {dense_cands.shape[0]:,} Dense queries")
        # Explode comma-separated candidates
        dense_cands = dense_cands.with_columns(
            pl.col("candidate_entity_ids").str.split(",")
        ).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
    except Exception as e:
        print(f"Error loading Dense Candidates: {e}")
        return

    print(f"Loading Token Join Candidates from: {token_join_path}...")
    try:
        token_cands = pl.read_csv(token_join_path, separator="\t")
        print(f"  -> Found {token_cands.shape[0]:,} Token Join queries")
        # Explode comma-separated candidates
        token_cands = token_cands.with_columns(
            pl.col("candidate_entity_ids").str.split(",")
        ).explode("candidate_entity_ids").filter(pl.col("candidate_entity_ids") != "")
    except Exception as e:
        print(f"Error loading Token Join Candidates: {e}")
        return
        
    print("\nUnioning and deduplicating...")
    dense_cands = dense_cands.select(["source1_entity_id", pl.col("candidate_entity_ids").alias("candidate_entity_id")])
    token_cands = token_cands.select(["source1_entity_id", pl.col("candidate_entity_ids").alias("candidate_entity_id")])
    
    # Vertically stack and find unique pairs
    grand_union = pl.concat([dense_cands, token_cands]).unique(subset=["source1_entity_id", "candidate_entity_id"])
    
    print(f"\nSUCCESS! Grand Union generated {grand_union.shape[0]:,} total unique pairs.")
    
    print("Formatting output into comma-separated lists for Phase 2...")
    final_df = grand_union.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").str.join(",")
    ).rename({"candidate_entity_id": "candidate_entity_ids"})
    
    print(f"\nSaving final candidate pool to: {output_path}...")
    # Ensure output dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.write_csv(output_path, separator="\t")
    
    print(f"Finished in {time.time() - start:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=str, default="output/candidate_pairs_dense_two_lane.tsv")
    parser.add_argument("--token", type=str, default="output/candidate_pairs_token_join.tsv")
    parser.add_argument("--output", type=str, default="output/candidate_pairs_grand_union.tsv")
    args = parser.parse_args()
    main(args.dense, args.token, args.output)
