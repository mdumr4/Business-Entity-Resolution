import polars as pl
from pathlib import Path

def main():
    print("Empirically testing the 'Shopping Mall' and 'Same Name' hypotheses on the training data...")
    
    data_dir = Path("dataset/train")
    
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_dir / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    
    candidates = pl.concat([s2, s3])
    
    # Explode ground truth to get 1-to-1 true pairs
    gt_pairs = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
    
    # Join the actual text data for S1 and Candidate
    df = gt_pairs.join(
        s1.select(["entity_id", "business_name", "business_address"]).rename({
            "business_name": "s1_name",
            "business_address": "s1_address"
        }),
        left_on="source1_entity_id", right_on="entity_id", how="inner"
    ).join(
        candidates.select(["entity_id", "business_name", "business_address"]).rename({
            "business_name": "cand_name",
            "business_address": "cand_address"
        }),
        left_on="matched_entity_ids", right_on="entity_id", how="inner"
    )
    
    df = df.with_columns([
        pl.col("s1_name").str.to_lowercase().str.strip_chars().alias("s1_name_clean"),
        pl.col("cand_name").str.to_lowercase().str.strip_chars().alias("cand_name_clean"),
        pl.col("s1_address").str.to_lowercase().str.strip_chars().alias("s1_address_clean"),
        pl.col("cand_address").str.to_lowercase().str.strip_chars().alias("cand_address_clean")
    ])
    
    total_true_matches = df.shape[0]
    
    # 1. How many true matches have identical names but DIFFERENT addresses?
    same_name_diff_addr = df.filter(
        (pl.col("s1_name_clean") == pl.col("cand_name_clean")) &
        (pl.col("s1_address_clean") != pl.col("cand_address_clean")) &
        (pl.col("s1_name_clean").is_not_null()) & 
        (pl.col("s1_address_clean").is_not_null()) &
        (pl.col("cand_address_clean").is_not_null())
    )
    
    # 2. How many true matches have identical addresses but DIFFERENT names?
    same_addr_diff_name = df.filter(
        (pl.col("s1_address_clean") == pl.col("cand_address_clean")) &
        (pl.col("s1_name_clean") != pl.col("cand_name_clean")) &
        (pl.col("s1_address_clean").is_not_null()) &
        (pl.col("s1_name_clean").is_not_null()) &
        (pl.col("cand_name_clean").is_not_null())
    )
    
    print("\n=== EMPIRICAL DATA FACTS (TRUE MATCHES) ===")
    print(f"Total True Matches Analyzed: {total_true_matches:,}")
    print(f"\n1. True Matches with EXACT Same Name but DIFFERENT Address: {same_name_diff_addr.shape[0]:,}")
    print(f"   (That's {(same_name_diff_addr.shape[0]/total_true_matches)*100:.1f}% of all true matches)")
    
    if same_name_diff_addr.shape[0] > 0:
        print("\n   Example:")
        ex = same_name_diff_addr.row(0, named=True)
        print(f"   S1:   {ex['s1_name']} | {ex['s1_address']}")
        print(f"   Cand: {ex['cand_name']} | {ex['cand_address']}")
        
    print(f"\n2. True Matches with EXACT Same Address but DIFFERENT Name: {same_addr_diff_name.shape[0]:,}")
    print(f"   (That's {(same_addr_diff_name.shape[0]/total_true_matches)*100:.1f}% of all true matches)")
    
    if same_addr_diff_name.shape[0] > 0:
        print("\n   Example:")
        ex = same_addr_diff_name.row(0, named=True)
        print(f"   S1:   {ex['s1_name']} | {ex['s1_address']}")
        print(f"   Cand: {ex['cand_name']} | {ex['cand_address']}")
        
if __name__ == "__main__":
    main()
