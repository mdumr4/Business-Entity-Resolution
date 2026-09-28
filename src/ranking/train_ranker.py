import polars as pl
import numpy as np
import xgboost as xgb
import argparse
import json
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from collections import defaultdict

def build_entity_dict(probs, labels, groups):
    by_entity = defaultdict(list)
    for p, l, g in zip(probs, labels, groups):
        by_entity[g].append((p, l))
    return by_entity

def evaluate_f05_from_dict(by_entity, threshold):
    scores = []
    for g, items in by_entity.items():
        true_count = sum(l for _, l in items)
        pred_positive = [(p, l) for p, l in items if p >= threshold]
        pred_count = len(pred_positive)
        
        if true_count == 0 and pred_count == 0:
            scores.append(1.0)
        elif pred_count == 0:
            scores.append(0.0)  # true_count > 0, nothing predicted
        else:
            tp = sum(l for _, l in pred_positive)
            precision = tp / pred_count
            recall = tp / true_count if true_count > 0 else 0.0
            if precision == 0 and recall == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
                scores.append(f05)
    return np.mean(scores)

def main():
    print("=== PHASE 3: TRAINING FINAL RANKER ===")
    
    features_file = "output/ranker_features_train.parquet"
    gt_file = "dataset/train/train_ground_truth.tsv"
    
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"Missing {features_file}. Run extract_ranker_features.py first.")
        
    print("Loading Feature Matrix...")
    df = pl.read_parquet(features_file)
    
    print("Loading Ground Truth...")
    gt = pl.read_csv(gt_file, separator="\t")
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        m = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(x.strip() for x in m.split(",") if x.strip()) if m and m.strip() else set()
    
    print("Assigning Labels...")
    qids = df["source1_entity_id"].to_list()
    cids = df["candidate_entity_id"].to_list()
    
    labels = [1 if cids[i] in gt_dict.get(qids[i], set()) else 0 for i in range(len(qids))]
        
    df = df.with_columns(pl.Series("label", labels))
    
    # Compute intra-query relative features
    print("Computing intra-query relative features...")
    df = df.with_columns([
        (pl.col("addr_cos") - pl.col("addr_cos").max().over("source1_entity_id")).alias("addr_cos_gap_from_top"),
        (pl.col("name_cos") - pl.col("name_cos").max().over("source1_entity_id")).alias("name_cos_gap_from_top"),
        pl.col("addr_cos").rank(method="ordinal", descending=True).over("source1_entity_id").alias("addr_cos_rank"),
    ])
    
    feature_cols = [
        "name_cos", "addr_cos", 
        "name_jac", "addr_jac", "num_jac", 
        "name_exact", 
        "name_tfidf", "addr_tfidf", 
        "name_lev", "addr_lev",
        "addr_cos_gap_from_top", "name_cos_gap_from_top", "addr_cos_rank"
    ]
    
    print(f"Total Pairs: {len(df)}")
    print(f"Total Positives: {sum(labels)}")
    
    print("\nSplitting Train/Tune/Holdout via GroupShuffleSplit (no query leakage)...")
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()
    groups = df["source1_entity_id"].to_numpy()
    
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_tune_idx, holdout_idx = next(gss1.split(X, y, groups))
    
    X_train_tune, y_train_tune = X[train_tune_idx], y[train_tune_idx]
    groups_train_tune = groups[train_tune_idx]
    
    # Split the remaining 80% into Train (70%) and Tune (10%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42) 
    train_idx, tune_idx = next(gss2.split(X_train_tune, y_train_tune, groups_train_tune))
    
    X_train, y_train = X_train_tune[train_idx], y_train_tune[train_idx]
    X_tune, y_tune = X_train_tune[tune_idx], y_train_tune[tune_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]
    
    groups_train = groups_train_tune[train_idx]
    groups_tune = groups_train_tune[tune_idx]
    groups_holdout = groups[holdout_idx]
    
    print(f"Train size: {len(X_train)} pairs | Positives: {y_train.sum()}")
    print(f"Tune size : {len(X_tune)} pairs | Positives: {y_tune.sum()}")
    print(f"Holdout size: {len(X_holdout)} pairs | Positives: {y_holdout.sum()}")
    
    print("\nTraining XGBoost Ranker (Precision-Biased)...")
    # We use binary:logistic because we will tune a hard threshold for F0.5
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42
    )
    
    clf.fit(
        X_train, y_train,
        eval_set=[(X_tune, y_tune)],
        verbose=50
    )
    
    print("\nExtracting Feature Importances...")
    importances = clf.feature_importances_
    for col, imp in zip(feature_cols, importances):
        print(f"  {col}: {imp:.4f}")
        
    print("Loading Countries for Stratified Evaluation...")
    s1 = pl.read_csv(f"dataset/train/train_source1.tsv", separator="\t")
    q_country_dict = {row["entity_id"]: row["country"] for row in s1.iter_rows(named=True)}
    groups_holdout_countries = np.array([q_country_dict.get(g, "unknown") for g in groups_holdout])
    
    idx_in = groups_holdout_countries == "India"
    idx_us = groups_holdout_countries == "US"
    print(f"Country match counts — India: {idx_in.sum()}, US: {idx_us.sum()}, unknown: {(groups_holdout_countries == 'unknown').sum()}")
    
    print("\n=== TUNING THRESHOLD FOR MACRO F0.5 ===")
    probs_holdout = clf.predict_proba(X_holdout)[:, 1]
    
    # Build dictionaries ONCE to avoid O(N) dict building inside the loop 86 times
    print("Building entity dictionaries for fast F0.5 sweep...")
    dict_all = build_entity_dict(probs_holdout, y_holdout, groups_holdout)
    dict_in = build_entity_dict(probs_holdout[idx_in], y_holdout[idx_in], groups_holdout[idx_in]) if idx_in.any() else {}
    dict_us = build_entity_dict(probs_holdout[idx_us], y_holdout[idx_us], groups_holdout[idx_us]) if idx_us.any() else {}
    
    best_t = 0.5
    best_f05_all = 0.0
    best_metrics = {}
    
    thresholds = np.linspace(0.1, 0.95, 86)
    
    print("Sweeping 86 thresholds...")
    from tqdm import tqdm
    for t in tqdm(thresholds):
        f05_all = evaluate_f05_from_dict(dict_all, t)
        
        if f05_all > best_f05_all:
            best_f05_all = f05_all
            best_t = t
            
            f05_in = evaluate_f05_from_dict(dict_in, t) if dict_in else 0.0
            f05_us = evaluate_f05_from_dict(dict_us, t) if dict_us else 0.0
            
            preds = (probs_holdout >= t).astype(int)
            best_metrics = {
                "f05_all": f05_all,
                "f05_in": f05_in,
                "f05_us": f05_us,
                "precision": precision_score(y_holdout, preds, zero_division=0),
                "recall": recall_score(y_holdout, preds, zero_division=0)
            }
            
    print(f"\nOptimal Aggregate Threshold: {best_t:.4f}")
    print(f"  Validation F0.5 Macro (ALL):   {best_metrics['f05_all']:.4f}")
    print(f"  Validation F0.5 Macro (India): {best_metrics['f05_in']:.4f}")
    print(f"  Validation F0.5 Macro (US):    {best_metrics['f05_us']:.4f}")
    print(f"  Validation Global Precision:   {best_metrics['precision']:.4f}")
    print(f"  Validation Global Recall:      {best_metrics['recall']:.4f}")
    
    print("\nSaving Model and Threshold...")
    clf.save_model("output/phase3_ranker_xgb.json")
    with open("output/phase3_threshold.json", "w") as f:
        json.dump({"optimal_threshold": float(best_t)}, f)
        
    print("Done! Phase 3 Training Complete.")

if __name__ == "__main__":
    main()
