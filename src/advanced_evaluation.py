import polars as pl
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import precision_score, recall_score, fbeta_score
from collections import defaultdict
import time

def evaluate_predictions(preds_dict, gt_dict, qids):
    """Calculates F0.5, Precision, Recall on a per-entity basis (Macro average)."""
    f05_scores = []
    precisions = []
    recalls = []
    
    total_singletons = 0
    correct_singletons = 0

    for qid in qids:
        true_matches = gt_dict.get(qid, set())
        pred_matches = preds_dict.get(qid, set())
        
        true_count = len(true_matches)
        pred_count = len(pred_matches)
        
        if true_count == 0:
            total_singletons += 1
            if pred_count == 0:
                correct_singletons += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        elif pred_count == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(true_matches.intersection(pred_matches))
            precision = tp / pred_count
            recall = tp / true_count
            
            precisions.append(precision)
            recalls.append(recall)
            
            if precision == 0 and recall == 0:
                f05_scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f05_scores.append(f05)

    return {
        "macro_f05": np.mean(f05_scores),
        "macro_precision": np.mean(precisions),
        "macro_recall": np.mean(recalls),
        "singleton_accuracy": correct_singletons / total_singletons if total_singletons > 0 else 0.0
    }

def main():
    print("=" * 60)
    print(" ADVANCED ML EVALUATION: ENTITY RESOLUTION PIPELINE")
    print("=" * 60)
    
    t0 = time.time()

    # Paths
    features_file = "output/ranker_features_train.parquet"
    gt_file = "dataset/train/train_ground_truth.tsv"
    model_path = "output/phase3_ranker_xgb.json"
    thresh_path = "output/phase3_threshold.json"
    s1_file = "dataset/train/train_source1.tsv"

    print("Loading extracted features for the training set...")
    df = pl.read_parquet(features_file)
    
    # Compute relative ranking features
    df = df.with_columns([
        (pl.col("addr_cos") - pl.col("addr_cos").max().over("source1_entity_id")).alias("addr_cos_gap_from_top"),
        (pl.col("name_cos") - pl.col("name_cos").max().over("source1_entity_id")).alias("name_cos_gap_from_top"),
        pl.col("addr_cos").rank(method="ordinal", descending=True).over("source1_entity_id").alias("addr_cos_rank"),
    ])

    feature_cols = [
        "name_cos", "addr_cos", "name_jac", "addr_jac", "num_jac",
        "name_exact", "name_tfidf", "addr_tfidf", "name_lev", "addr_lev",
        "addr_cos_gap_from_top", "name_cos_gap_from_top", "addr_cos_rank"
    ]
    
    X = df.select(feature_cols).to_numpy()
    qids_feat = df["source1_entity_id"].to_list()
    cids_feat = df["candidate_entity_id"].to_list()

    print("Loading Ground Truth...")
    gt_df = pl.read_csv(gt_file, separator="\t")
    gt_dict = {}
    for row in gt_df.iter_rows(named=True):
        m = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(x.strip() for x in m.split(",") if x.strip()) if m and m.strip() else set()

    print("Loading XGBoost Ranker & Optimal Threshold...")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    
    with open(thresh_path, "r") as f:
        threshold = json.load(f).get("optimal_threshold", 0.5)
    print(f"Optimal F0.5 Threshold: {threshold:.4f}")

    print("Running Inference...")
    probs = clf.predict_proba(X)[:, 1]
    
    preds_dict = defaultdict(set)
    for i in range(len(probs)):
        if probs[i] >= threshold:
            preds_dict[qids_feat[i]].add(cids_feat[i])

    print("\nFetching full Source 1 entity list for evaluation (including singletons)...")
    s1_df = pl.read_csv(s1_file, separator="\t")
    all_qids = s1_df["entity_id"].to_list()
    countries = s1_df["country"].to_list()
    country_dict = dict(zip(all_qids, countries))
    
    print("\n" + "=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    
    # 1. Global Metrics
    global_metrics = evaluate_predictions(preds_dict, gt_dict, all_qids)
    print(f"[GLOBAL] Macro F0.5 Score:     {global_metrics['macro_f05']:.4f}")
    print(f"[GLOBAL] Macro Precision:      {global_metrics['macro_precision']:.4f}")
    print(f"[GLOBAL] Macro Recall:         {global_metrics['macro_recall']:.4f}")
    print(f"[GLOBAL] Singleton Accuracy:   {global_metrics['singleton_accuracy']:.2%}")
    
    # 2. Slice by Country
    qids_india = [q for q in all_qids if country_dict.get(q) == "India"]
    qids_us = [q for q in all_qids if country_dict.get(q) == "US"]
    
    if qids_india:
        in_metrics = evaluate_predictions(preds_dict, gt_dict, qids_india)
        print(f"\n[INDIA]  Macro F0.5 Score:     {in_metrics['macro_f05']:.4f}")
    if qids_us:
        us_metrics = evaluate_predictions(preds_dict, gt_dict, qids_us)
        print(f"[US]     Macro F0.5 Score:     {us_metrics['macro_f05']:.4f}")
        
    print("=" * 60)
    print(f"Evaluation completed in {time.time() - t0:.2f} seconds.")
    print("Copy these metrics for your final report!")

if __name__ == "__main__":
    main()
