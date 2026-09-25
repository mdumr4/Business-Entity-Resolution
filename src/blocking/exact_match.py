import polars as pl
import argparse
from pathlib import Path

def main(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("Loading data for Exact Match Blocking...")
    s1 = pl.read_csv(data_path / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_path / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_path / "train_source3.tsv", separator="\t")
    
    # Simple normalization for exact matching
    def normalize(df):
        return df.with_columns([
            pl.col("business_name").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "").alias("norm_name"),
            pl.col("business_address").str.to_lowercase().str.replace_all(r"[^a-z0-9]", "").alias("norm_addr")
        ])
        
    s1 = normalize(s1)
    candidates = normalize(pl.concat([s2, s3]))
    
    print("Performing Exact Match Joins...")
    
    # Match on Exact Name OR Exact Address (if not null)
    # Join on Name
    name_matches = s1.join(
        candidates.filter(pl.col("norm_name").is_not_null() & (pl.col("norm_name") != "")),
        on="norm_name", 
        how="inner"
    ).select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ])
    
    # Join on Address
    addr_matches = s1.join(
        candidates.filter(pl.col("norm_addr").is_not_null() & (pl.col("norm_addr") != "")),
        on="norm_addr", 
        how="inner"
    ).select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ])
    
    # Combine matches
    all_matches = pl.concat([name_matches, addr_matches]).unique()
    
    print("Formatting output...")
    # Group by S1 ID to create comma-separated candidate lists
    final_df = all_matches.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").str.join(",")
    ).rename({"candidate_entity_id": "candidate_entity_ids"})
    
    # Add queries that had no exact matches (with empty candidate strings)
    missing = s1.filter(~pl.col("entity_id").is_in(final_df["source1_entity_id"]))
    missing_df = missing.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.lit("").alias("candidate_entity_ids")
    )
    
    final_df = pl.concat([final_df, missing_df])
    
    output_file = out_path / "candidate_pairs_exact.tsv"
    final_df.write_csv(output_file, separator="\t")
    print(f"Done! Saved exact match candidates to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset/train")
    parser.add_argument("--output-dir", type=str, default="../../output")
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
