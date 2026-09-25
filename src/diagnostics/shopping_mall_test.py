import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, distance
from pathlib import Path

def main():
    print("Empirically testing the 'Shopping Mall Trap' on the XGBoost Model...")
    
    data_dir = Path("dataset/train")
    
    # 1. Load a chunk of Source 1
    s1 = pl.read_csv(data_dir / "train_source1.tsv", separator="\t", n_rows=500000)
    
    s1 = s1.with_columns([
        pl.col("business_name").str.to_lowercase().str.strip_chars().fill_null("").alias("clean_name"),
        pl.col("business_address").str.to_lowercase().str.strip_chars().fill_null("").alias("clean_addr")
    ])
    
    # 2. Find businesses that share the exact same address
    addr_counts = s1.group_by("clean_addr").agg(
        pl.col("entity_id").alias("ids"),
        pl.col("clean_name").alias("names"),
        pl.len().alias("count")
    ).filter(
        (pl.col("count") >= 2) & 
        (pl.col("clean_addr") != "")
    )
    
    # 3. Create negative pairs (same address, different names)
    print("Creating synthetic 'Shopping Mall' pairs (Same Address, Different Businesses)...")
    synthetic_pairs = []
    
    for row in addr_counts.iter_rows(named=True):
        ids = row["ids"]
        names = row["names"]
        
        # Take the first two if they have completely different names
        if len(ids) >= 2:
            if names[0] != names[1]:
                synthetic_pairs.append({
                    "source1_entity_id": ids[0],
                    "candidate_entity_id": ids[1],
                    "s1_name": names[0],
                    "cand_name": names[1],
                    "s1_addr": row["clean_addr"],
                    "cand_addr": row["clean_addr"]
                })
        
        if len(synthetic_pairs) >= 10000:
            break
            
    df_pairs = pl.DataFrame(synthetic_pairs)
    print(f"Generated {df_pairs.shape[0]} Shopping Mall pairs.")
    
    # 4. Filter out any true matches just in case
    gt = pl.read_csv(data_dir / "train_ground_truth.tsv", separator="\t")
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").str.split(",")
    ).explode("matched_entity_ids").rename({"matched_entity_ids": "candidate_entity_id"})
    
    # Anti-join to ensure these are purely FALSE POSITIVE candidates
    df_pairs = df_pairs.join(gt_exploded, on=["source1_entity_id", "candidate_entity_id"], how="anti")
    
    print(f"Remaining pure Negative pairs: {df_pairs.shape[0]}")
    
    if df_pairs.shape[0] == 0:
        print("No pairs found.")
        return
        
    # 5. Calculate the exact features the XGBoost model expects
    print("Calculating RapidFuzz Jaro-Winkler features...")
    
    name_jw = []
    addr_jw = []
    name_lev = []
    addr_lev = []
    
    for row in df_pairs.iter_rows(named=True):
        name_jw.append(distance.JaroWinkler.normalized_similarity(row["s1_name"], row["cand_name"]))
        addr_jw.append(distance.JaroWinkler.normalized_similarity(row["s1_addr"], row["cand_addr"]))
        name_lev.append(fuzz.ratio(row["s1_name"], row["cand_name"]) / 100.0)
        addr_lev.append(fuzz.ratio(row["s1_addr"], row["cand_addr"]) / 100.0)
        
    df_pairs = df_pairs.with_columns([
        pl.Series("feat_name_jw", name_jw),
        pl.Series("feat_name_lev", name_lev),
        pl.Series("feat_addr_jw", addr_jw),
        pl.Series("feat_addr_lev", addr_lev)
    ])
    
    # 6. Score with XGBoost
    print("Loading XGBoost Model...")
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_reranker.json")
    
    features = df_pairs.select(["feat_name_jw", "feat_name_lev", "feat_addr_jw", "feat_addr_lev"]).to_numpy()
    preds = model.predict_proba(features)[:, 1]
    
    df_pairs = df_pairs.with_columns(pl.Series("xgb_score", preds))
    
    # 7. Analyze how many fell into the trap
    trap_victims = df_pairs.filter(pl.col("xgb_score") >= 0.5)
    
    print("\n=== XGBOOST SHOPPING MALL TRAP ANALYSIS ===")
    print(f"Total Negative Pairs Tested (Same Address, Different Name): {df_pairs.shape[0]}")
    print(f"Number of times XGBoost incorrectly predicted MATCH (>0.5): {trap_victims.shape[0]}")
    print(f"Failure Rate (False Positive Rate for this scenario): {(trap_victims.shape[0] / df_pairs.shape[0]) * 100:.2f}%")
    
    if trap_victims.shape[0] > 0:
        print("\nExamples of the Trap (Model predicted True, but they are completely different businesses):")
        exs = trap_victims.sort("xgb_score", descending=True).head(3)
        for row in exs.iter_rows(named=True):
            print(f"\nScore: {row['xgb_score']:.4f}")
            print(f"Business A: {row['s1_name']}")
            print(f"Business B: {row['cand_name']}")
            print(f"Address (Identical): {row['s1_addr']}")
            print(f"Name JW: {row['feat_name_jw']:.2f} | Addr JW: {row['feat_addr_jw']:.2f}")

if __name__ == "__main__":
    main()
