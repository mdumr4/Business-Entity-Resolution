import polars as pl
import argparse
from pathlib import Path
import time

def build_rare_token_blocker(s1: pl.DataFrame, candidates: pl.DataFrame, max_freq: int = 50) -> pl.DataFrame:
    
    # ---------------------------
    # LANE 1: NAME TOKENS
    # ---------------------------
    print("Tokenizing Names...")
    all_names = pl.concat([
        s1.select(["entity_id", "business_name"]),
        candidates.select(["entity_id", "business_name"])
    ])
    
    name_tokens = all_names.with_columns(
        pl.col("business_name").str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").alias("token")
    ).explode("token").filter(
        (pl.col("token").is_not_null()) & (pl.col("token").str.len_chars() >= 3)
    )
    
    print("Calculating rare Name tokens...")
    name_freq = name_tokens.group_by("token").agg(pl.col("entity_id").n_unique().alias("doc_count"))
    rare_name_tokens = name_freq.filter(pl.col("doc_count") <= max_freq)
    
    print(f"Found {rare_name_tokens.shape[0]} unique rare Name tokens.")
    
    s1_name_rare = name_tokens.filter(pl.col("entity_id").is_in(s1["entity_id"])).join(rare_name_tokens, on="token", how="inner")
    cand_name_rare = name_tokens.filter(pl.col("entity_id").is_in(candidates["entity_id"])).join(rare_name_tokens, on="token", how="inner")
    
    print("Joining S1 and Candidates on Rare Name Tokens...")
    name_matches = s1_name_rare.join(
        cand_name_rare, on="token", how="inner"
    ).select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).unique()
    
    # ---------------------------
    # LANE 2: ADDRESS TOKENS
    # ---------------------------
    print("\nTokenizing Addresses...")
    all_addrs = pl.concat([
        s1.select(["entity_id", "business_address"]),
        candidates.select(["entity_id", "business_address"])
    ])
    
    addr_tokens = all_addrs.with_columns(
        pl.col("business_address").str.to_lowercase().str.replace_all(r"[^\w\s]", "").str.split(" ").alias("token")
    ).explode("token").filter(
        (pl.col("token").is_not_null()) & (pl.col("token").str.len_chars() >= 3)
    )
    
    print("Calculating rare Address tokens...")
    addr_freq = addr_tokens.group_by("token").agg(pl.col("entity_id").n_unique().alias("doc_count"))
    rare_addr_tokens = addr_freq.filter(pl.col("doc_count") <= max_freq)
    
    print(f"Found {rare_addr_tokens.shape[0]} unique rare Address tokens.")
    
    s1_addr_rare = addr_tokens.filter(pl.col("entity_id").is_in(s1["entity_id"])).join(rare_addr_tokens, on="token", how="inner")
    cand_addr_rare = addr_tokens.filter(pl.col("entity_id").is_in(candidates["entity_id"])).join(rare_addr_tokens, on="token", how="inner")
    
    print("Joining S1 and Candidates on Rare Address Tokens...")
    addr_matches = s1_addr_rare.join(
        cand_addr_rare, on="token", how="inner"
    ).select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("entity_id_right").alias("candidate_entity_id")
    ]).unique()
    
    # ---------------------------
    # UNION AND FORMAT
    # ---------------------------
    print("\nUnioning Name and Address Candidates...")
    all_matches = pl.concat([name_matches, addr_matches]).unique()
    
    print(f"Total Unique Candidate Pairs Generated: {all_matches.shape[0]:,}")
    
    print("Formatting output...")
    final_df = all_matches.group_by("source1_entity_id").agg(
        pl.col("candidate_entity_id").str.join(",")
    ).rename({"candidate_entity_id": "candidate_entity_ids"})
    
    missing = s1.filter(~pl.col("entity_id").is_in(final_df["source1_entity_id"]))
    missing_df = missing.select(
        pl.col("entity_id").alias("source1_entity_id"),
        pl.lit("").alias("candidate_entity_ids")
    )
    
    final_df = pl.concat([final_df, missing_df])
    return final_df

def main(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("Loading datasets for Two-Lane Token-Join Blocking...")
    prefix = "test" if "test" in data_dir.lower() else "train"
    s1 = pl.read_csv(data_path / f"{prefix}_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_path / f"{prefix}_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_path / f"{prefix}_source3.tsv", separator="\t")
    
    candidates = pl.concat([s2, s3])
    
    start_time = time.time()
    result_df = build_rare_token_blocker(s1, candidates, max_freq=50)
    print(f"Blocking completed in {time.time() - start_time:.2f} seconds.")
    
    out_file = out_path / "candidate_pairs_token_join.tsv"
    result_df.write_csv(out_file, separator="\t")
    print(f"Saved Token-Join candidates to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
