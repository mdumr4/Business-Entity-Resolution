import polars as pl
import argparse
from pathlib import Path
import json

def profile_dataset(data_dir: str, output_dir: str):
    """
    Production-level profiling of the Entity Resolution dataset using Polars.
    Polars is used for memory efficiency and speed on large tabular datasets.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    
    print("Starting Data Profiling...")
    
    # 1. Row counts and Country Distributions using LazyFrames (out-of-core)
    sources = ["source1", "source2", "source3"]
    stats = {}
    
    for src in sources:
        print(f"Scanning train_{src}.tsv...")
        df_lazy = pl.scan_csv(data_path / f"train_{src}.tsv", separator="\t")
        
        # Count total rows
        total_rows = df_lazy.select(pl.len()).collect().item()
        
        # Count unique countries
        country_dist = (
            df_lazy.group_by("country")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .collect()
        )
        
        # Missing values analysis
        null_counts = df_lazy.select(pl.all().is_null().sum()).collect().to_dicts()[0]
        
        stats[src] = {
            "total_rows": total_rows,
            "country_distribution": country_dist.to_dicts(),
            "null_values": null_counts
        }
        print(f"  {src}: {total_rows} rows")
        
    # 2. Ground Truth Analysis (Singleton rates, max matches)
    print("Scanning train_ground_truth.tsv...")
    gt_lazy = pl.scan_csv(data_path / "train_ground_truth.tsv", separator="\t")
    
    # Process ground truth: calculate number of matches by splitting the string
    gt_processed = gt_lazy.with_columns(
        pl.when(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
        .then(0)
        .otherwise(
            # Count commas + 1 to get number of items
            pl.col("matched_entity_ids").str.count_matches(",") + 1
        )
        .alias("match_count")
    ).collect()
    
    total_s1_in_gt = gt_processed.height
    singletons = gt_processed.filter(pl.col("match_count") == 0).height
    singleton_rate = singletons / total_s1_in_gt if total_s1_in_gt > 0 else 0
    
    match_distribution = (
        gt_processed.group_by("match_count")
        .agg(pl.len().alias("frequency"))
        .sort("match_count")
    )
    
    stats["ground_truth"] = {
        "total_s1_entities": total_s1_in_gt,
        "singletons_count": singletons,
        "singleton_rate": singleton_rate,
        "max_matches_for_single_entity": gt_processed["match_count"].max(),
        "match_distribution": match_distribution.to_dicts()[:10] # top 10 frequencies
    }
    
    print(f"  Singletons: {singletons} ({singleton_rate:.1%})")
    print(f"  Max matches for one entity: {stats['ground_truth']['max_matches_for_single_entity']}")
    
    # Save stats to JSON for downstream tasks
    report_path = out_path / "eda_profile_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"Profiling complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the ER Dataset")
    parser.add_argument("--data-dir", type=str, default="../../dataset/train", help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="../../output", help="Path to save reports")
    args = parser.parse_args()
    
    profile_dataset(args.data_dir, args.output_dir)
