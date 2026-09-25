import polars as pl
import xgboost as xgb
import argparse
from pathlib import Path
from rapidfuzz import fuzz, distance
import numpy as np

def calculate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates pure-algorithm string distance features between S1 and candidate strings."""
    print("Calculating RapidFuzz string distance features...")
    s1_names = df["s1_norm_name"].fill_null("").to_list()
    c_names = df["candidate_norm_name"].fill_null("").to_list()
    s1_addrs = df["s1_norm_address"].fill_null("").to_list()
    c_addrs = df["candidate_norm_address"].fill_null("").to_list()
    
    name_jw, name_lev, addr_jw, addr_lev = [], [], [], []
    
    for i in range(len(s1_names)):
        name_jw.append(distance.JaroWinkler.normalized_similarity(s1_names[i], c_names[i]))
        name_lev.append(fuzz.ratio(s1_names[i], c_names[i]) / 100.0)
        
        if not s1_addrs[i] or not c_addrs[i]:
            addr_jw.append(-1.0)
            addr_lev.append(-1.0)
        else:
            addr_jw.append(distance.JaroWinkler.normalized_similarity(s1_addrs[i], c_addrs[i]))
            addr_lev.append(fuzz.ratio(s1_addrs[i], c_addrs[i]) / 100.0)
            
    return df.with_columns([
        pl.Series("feat_name_jw", name_jw),
        pl.Series("feat_name_lev", name_lev),
        pl.Series("feat_addr_jw", addr_jw),
        pl.Series("feat_addr_lev", addr_lev)
    ])

def build_training_dataset(data_dir: Path, candidates_file: Path) -> pl.DataFrame:
    print("Loading Candidates and Ground Truth...")
    # Load the generated candidates
    candidates_df = pl.read_csv(candidates_file, separator="\t")
    
    # Explode the comma separated candidate IDs into distinct rows!
    pairs = candidates_df.with_columns(
        pl.col("candidate_entity_ids").str.split(",")
    ).explode("candidate_entity_ids").rename({"candidate_entity_ids": "candidate_entity_id"})
    
    # Remove empty rows
    pairs = pairs.filter(pl.col("candidate_entity_id") != "")
    
    # Load Truth
    truth = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    # For now, let's assume truth has 'source1_entity_id' and 'source2_entity_id' (or similar).
    # Since I don't know the exact column names, we will try to infer or ask the user if this fails.
    # We will assume truth just maps a pair of IDs. We label 1 if pair is in truth, else 0.
    truth = truth.with_columns(pl.lit(1).alias("is_match"))
    
    # Assuming truth columns are something like ['s1_id', 's2_id']
    truth_cols = truth.columns
    pairs = pairs.join(
        truth, 
        left_on=["source1_entity_id", "candidate_entity_id"], 
        right_on=[truth_cols[0], truth_cols[1]], 
        how="left"
    ).with_columns(pl.col("is_match").fill_null(0))
    
    print(f"Generated {pairs.shape[0]} candidate pairs. Matches found: {pairs['is_match'].sum()}")
    
    print("Joining original text features for XGBoost...")
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t").select([
        pl.col("entity_id"), 
        pl.col("business_name").str.to_lowercase().alias("s1_norm_name"), 
        pl.col("business_address").str.to_lowercase().alias("s1_norm_address")
    ])
    
    s2 = pl.read_csv(data_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(data_dir / "train_source3.tsv", separator="\t")
    all_candidates = pl.concat([s2, s3]).select([
        pl.col("entity_id"), 
        pl.col("business_name").str.to_lowercase().alias("candidate_norm_name"), 
        pl.col("business_address").str.to_lowercase().alias("candidate_norm_address")
    ])
    
    pairs = pairs.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
    pairs = pairs.join(all_candidates, left_on="candidate_entity_id", right_on="entity_id", how="inner")
    
    return calculate_features(pairs)

def train_xgboost(train_df: pl.DataFrame, model_out: Path):
    features = ["feat_name_jw", "feat_name_lev", "feat_addr_jw", "feat_addr_lev"]
    X = train_df.select(features).to_numpy()
    y = train_df["is_match"].to_numpy()
    
    dtrain = xgb.DMatrix(X, label=y)
    
    # CRITICAL: Handle massive class imbalance (e.g. 34k matches vs 23M negatives)
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    scale_weight = num_neg / max(1, num_pos)
    print(f"Setting scale_pos_weight to {scale_weight:.2f} to handle class imbalance...")
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "scale_pos_weight": scale_weight
    }
    
    print("Training XGBoost Classifier...")
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    model_path = model_out / "xgb_reranker.json"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../../dataset/train")
    parser.add_argument("--candidates-file", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="../../models")
    args = parser.parse_args()
    
    out_dir = Path(args.model_dir)
    out_dir.mkdir(exist_ok=True)
    
    df = build_training_dataset(Path(args.data_dir), Path(args.candidates_file))
    train_xgboost(df, out_dir)
