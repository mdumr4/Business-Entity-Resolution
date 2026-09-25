import xgboost as xgb
from pathlib import Path
import json

def main():
    model_path = Path("models/xgb_reranker.json")
    if not model_path.exists():
        print("Model not found!")
        return
        
    model = xgb.Booster()
    model.load_model(model_path)
    
    # Get feature importances
    gain = model.get_score(importance_type="gain")
    weight = model.get_score(importance_type="weight")
    cover = model.get_score(importance_type="cover")
    
    # In XGBoost, if features weren't explicitly named during training, they are f0, f1, etc.
    # We mapped them in inference as: ["feat_name_jw", "feat_name_lev", "feat_addr_jw", "feat_addr_lev"]
    feature_map = {
        "f0": "feat_name_jw",
        "f1": "feat_name_lev",
        "f2": "feat_addr_jw",
        "f3": "feat_addr_lev"
    }
    
    report = ["# Phase 2: Flow 1 XGBoost Model Analysis\n"]
    
    report.append("## Feature Importance (Gain)")
    report.append("Gain measures the relative contribution of each feature to the model. Higher gain = more important for decision making.\n")
    
    # Sort by gain
    sorted_gain = sorted(gain.items(), key=lambda x: x[1], reverse=True)
    for feat, g in sorted_gain:
        mapped_name = feature_map.get(feat, feat)
        report.append(f"* **{mapped_name}**: {g:.4f}")
        
    report.append("\n## Feature Frequency (Weight)")
    report.append("Weight is the number of times a feature is used to split the data across all trees.\n")
    
    sorted_weight = sorted(weight.items(), key=lambda x: x[1], reverse=True)
    for feat, w in sorted_weight:
        mapped_name = feature_map.get(feat, feat)
        report.append(f"* **{mapped_name}**: {w}")
        
    report_dir = Path("results/phase2_candidate_models")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "flow1_model_analysis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))
        
    print(f"Model analysis saved to {report_path}")
    
if __name__ == "__main__":
    main()
