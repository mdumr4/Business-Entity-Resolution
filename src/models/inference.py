import polars as pl
import xgboost as xgb
import argparse
from pathlib import Path
import sys

# Import our feature calculation from the reranker script
from xgboost_reranker import calculate_features

def build_test_dataset(data_dir: Path, candidates_file: Path) -> pl.DataFrame:
    print("Loading Test Candidates...")
    candidates_df = pl.read_csv(candidates_file, separator="\t")
    
    # Explode the comma separated candidate IDs into distinct rows
    pairs = candidates_df.with_columns(
        pl.col("candidate_entity_ids").str.split(",")
    ).explode("candidate_entity_ids").rename({"candidate_entity_ids": "candidate_entity_id"})
    
    # Remove empty rows
    pairs = pairs.filter(pl.col("candidate_entity_id") != "")
    
    print("Joining original text features for Test Set...")
    # NOTE: Assuming test set has the same file names but prefixed/placed in a test folder
    # Adjust file names if the hackathon provides them differently (e.g., 'test_source1.tsv')
    s1_path = data_dir / "test_source1.tsv"
    if not s1_path.exists():
        # Fallback to train if we are just testing the inference script locally
        s1_path = data_dir / "train_source1.tsv"
        print(f"Warning: test_source1.tsv not found. Using {s1_path.name} for testing purposes.")
        
    s1 = pl.read_csv(s1_path, separator="\t").select([
        pl.col("entity_id"), 
        pl.col("business_name").str.to_lowercase().alias("s1_norm_name"), 
        pl.col("business_address").str.to_lowercase().alias("s1_norm_address")
    ])
    
    s2_path = data_dir / "test_source2.tsv"
    if not s2_path.exists(): s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "test_source3.tsv"
    if not s3_path.exists(): s3_path = data_dir / "train_source3.tsv"
    
    s2 = pl.read_csv(s2_path, separator="\t")
    s3 = pl.read_csv(s3_path, separator="\t")
    
    all_candidates = pl.concat([s2, s3]).select([
        pl.col("entity_id"), 
        pl.col("business_name").str.to_lowercase().alias("candidate_norm_name"), 
        pl.col("business_address").str.to_lowercase().alias("candidate_norm_address")
    ])
    
    pairs = pairs.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    pairs = pairs.join(all_candidates, left_on="candidate_entity_id", right_on="entity_id", how="inner")
    
    return calculate_features(pairs)

def run_inference(test_df: pl.DataFrame, model_dir: Path, output_dir: Path, threshold: float = 0.5):
    model_path = model_dir / "xgb_reranker.json"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    print("Loading XGBoost Model...")
    model = xgb.Booster()
    model.load_model(model_path)
    
    features = ["feat_name_jw", "feat_name_lev", "feat_addr_jw", "feat_addr_lev"]
    X = test_df.select(features).to_numpy()
    dtest = xgb.DMatrix(X)
    
    print("Scoring candidates...")
    scores = model.predict(dtest)
    
    # Attach scores to the dataframe
    scored_df = test_df.with_columns(pl.Series("match_score", scores))
    
    # Phase 4 Logic: Filtering and Top-1 Selection
    print(f"Filtering out predictions below threshold {threshold}...")
    filtered_df = scored_df.filter(pl.col("match_score") >= threshold)
    
    print("Selecting the highest scoring candidate for each S1 Entity...")
    # Sort by score descending, then group by source 1 and take the top 1
    final_predictions = (
        filtered_df
        .sort(["source1_entity_id", "match_score"], descending=[False, True])
        .group_by("source1_entity_id")
        .head(1)
    )
    
    # Format for hackathon submission (usually a CSV with two columns)
    submission = final_predictions.select([
        pl.col("source1_entity_id"),
        pl.col("candidate_entity_id")
    ])
    
    sub_path = output_dir / "submission.csv"
    submission.write_csv(sub_path)
    print(f"Done! Final predictions saved to {sub_path}")
    print(f"Total Matches Predicted: {submission.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset/test", help="Path to test datasets")
    parser.add_argument("--candidates-file", type=str, required=True, help="Path to test candidate pairs TSV")
    parser.add_argument("--model-dir", type=str, default="../../models", help="Path to trained XGBoost model")
    parser.add_argument("--output-dir", type=str, default="../../output", help="Path to save submission.csv")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold to declare a match")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    
    df = build_test_dataset(Path(args.data_dir), Path(args.candidates_file))
    run_inference(df, Path(args.model_dir), out_dir, threshold=args.threshold)
