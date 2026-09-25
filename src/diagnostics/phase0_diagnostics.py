import polars as pl
from pathlib import Path
import json

def run_phase0_diagnostics(data_dir: str, output_dir: str):
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    s1 = pl.read_csv(data_path / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(data_path / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_path / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(data_path / "train_ground_truth.tsv", separator="\t")
    
    # ---------------------------------------------------------
    # 1. Missing-field rates by country/source
    # ---------------------------------------------------------
    print("Computing missing-field rates...")
    def get_missing_rates(df, source_name):
        # Count total rows per country
        counts = df.group_by("country").agg(pl.len().alias("total_rows"))
        
        # Count nulls per country
        nulls = df.group_by("country").agg([
            pl.col("business_name").is_null().sum().alias("name_nulls"),
            pl.col("business_address").is_null().sum().alias("addr_nulls")
        ])
        
        joined = counts.join(nulls, on="country")
        return joined.with_columns([
            (pl.col("name_nulls") / pl.col("total_rows")).alias("name_missing_pct"),
            (pl.col("addr_nulls") / pl.col("total_rows")).alias("addr_missing_pct")
        ]).to_dicts()

    missing_rates = {
        "S1": get_missing_rates(s1, "S1"),
        "S2": get_missing_rates(s2, "S2"),
        "S3": get_missing_rates(s3, "S3")
    }

    # ---------------------------------------------------------
    # 2. Match-count and singleton-rate distribution
    # ---------------------------------------------------------
    print("Computing match distributions...")
    # Join GT with S1 to get the country of the query
    gt_with_country = gt.join(s1.select(["entity_id", "country"]), left_on="source1_entity_id", right_on="entity_id", how="left")
    
    # Explode the true matches to see source-pairs
    exploded_gt = gt_with_country.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids")
    
    # Figure out if target is S2 or S3
    exploded_gt = exploded_gt.with_columns(
        pl.when(pl.col("matched_entity_ids").str.starts_with("S2")).then(pl.lit("S1-S2"))
        .otherwise(pl.lit("S1-S3")).alias("source_pair")
    )
    
    # Group by S1 Entity to get match counts
    match_counts = exploded_gt.group_by(["source1_entity_id", "country"]).agg([
        pl.len().alias("total_matches"),
        (pl.col("source_pair") == "S1-S2").sum().alias("s2_matches"),
        (pl.col("source_pair") == "S1-S3").sum().alias("s3_matches")
    ])
    
    distribution_stats = match_counts.group_by("country").agg([
        pl.len().alias("total_s1_entities_with_matches"),
        (pl.col("total_matches") == 1).sum().alias("singleton_count"),
        pl.col("total_matches").median().alias("median_matches"),
        pl.col("s2_matches").sum().alias("total_s2_matches"),
        pl.col("s3_matches").sum().alias("total_s3_matches")
    ]).to_dicts()

    # ---------------------------------------------------------
    # 3. One-record-one-business assumption check
    # ---------------------------------------------------------
    print("Checking one-record-one-business assumption...")
    # Does any true S2/S3 id appear under two different S1 rows?
    target_counts = exploded_gt.group_by("matched_entity_ids").agg(
        pl.col("source1_entity_id").n_unique().alias("unique_s1_parents")
    )
    
    violations = target_counts.filter(pl.col("unique_s1_parents") > 1)
    violation_count = violations.shape[0]
    is_assumption_safe = violation_count == 0

    # ---------------------------------------------------------
    # Save Report
    # ---------------------------------------------------------
    report = {
        "missing_field_rates": missing_rates,
        "match_distribution": distribution_stats,
        "one_record_one_business": {
            "is_safe_to_apply_phase4_conflict_resolution": is_assumption_safe,
            "target_entities_with_multiple_s1_parents": violation_count
        }
    }
    
    report_path = out_path / "phase0_diagnostic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Phase 0 Diagnostics completed! Saved to {report_path}")

if __name__ == "__main__":
    run_phase0_diagnostics("dataset/train", "results/phase0_diagnostics")
