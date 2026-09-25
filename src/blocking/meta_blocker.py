import polars as pl
import argparse
from pathlib import Path

def merge_candidates(candidates_series: pl.Series) -> str:
    """Takes a list of comma-separated candidate strings and merges them into a unique list."""
    all_cands = set()
    for c_str in candidates_series:
        if c_str:
            all_cands.update(c_str.split(","))
    return ",".join(all_cands)

def main(dense_file: str, exact_file: str, lexical_file: str, output_dir: str):
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("Loading candidate pools from all 3 blocking channels...")
    
    dense_df = pl.read_csv(dense_file, separator="\t").rename({"candidate_entity_ids": "dense_cands"})
    exact_df = pl.read_csv(exact_file, separator="\t").rename({"candidate_entity_ids": "exact_cands"})
    lexical_df = pl.read_csv(lexical_file, separator="\t").rename({"candidate_entity_ids": "lexical_cands"})
    
    print("Joining candidates on source1_entity_id...")
    merged = dense_df.join(exact_df, on="source1_entity_id", how="full", coalesce=True)
    merged = merged.join(lexical_df, on="source1_entity_id", how="full", coalesce=True)
    
    merged = merged.with_columns([
        pl.col("dense_cands").fill_null(""),
        pl.col("exact_cands").fill_null(""),
        pl.col("lexical_cands").fill_null("")
    ])
    
    print("Exploding and generating Meta-Blocking features...")
    df_dense = merged.select(["source1_entity_id", "dense_cands"]).with_columns(
        pl.col("dense_cands").str.split(",").alias("candidate")
    ).explode("candidate").filter(pl.col("candidate") != "").with_columns(pl.lit(1).alias("found_in_dense"))
    
    df_exact = merged.select(["source1_entity_id", "exact_cands"]).with_columns(
        pl.col("exact_cands").str.split(",").alias("candidate")
    ).explode("candidate").filter(pl.col("candidate") != "").with_columns(pl.lit(1).alias("found_in_exact"))
    
    df_lexical = merged.select(["source1_entity_id", "lexical_cands"]).with_columns(
        pl.col("lexical_cands").str.split(",").alias("candidate")
    ).explode("candidate").filter(pl.col("candidate") != "").with_columns(pl.lit(1).alias("found_in_lexical"))
    
    all_pairs = df_dense.join(df_exact, on=["source1_entity_id", "candidate"], how="full", coalesce=True)
    all_pairs = all_pairs.join(df_lexical, on=["source1_entity_id", "candidate"], how="full", coalesce=True)
    
    all_pairs = all_pairs.with_columns([
        pl.col("found_in_dense").fill_null(0),
        pl.col("found_in_exact").fill_null(0),
        pl.col("found_in_lexical").fill_null(0)
    ]).rename({"candidate": "candidate_entity_id"})
    
    out_file = out_path / "meta_blocking_pairs.tsv"
    all_pairs.write_csv(out_file, separator="\t")
    
    print(f"Done! Master Candidate Pool generated: {all_pairs.shape[0]} unique pairs.")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=str, default="output/candidate_pairs_dense_minilm.tsv")
    parser.add_argument("--exact", type=str, default="output/candidate_pairs_exact.tsv")
    parser.add_argument("--lexical", type=str, default="output/candidate_pairs_lexical.tsv")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    
    main(args.dense, args.exact, args.lexical, args.output_dir)
