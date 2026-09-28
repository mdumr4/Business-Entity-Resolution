import polars as pl
import numpy as np
import faiss
import torch
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
import time

def norm_series(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def get_numbers(s):
    return set(re.findall(r'\d+', str(s)))

def token_jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def remove_legal_suffixes(s):
    suffixes = r'\b(llc|inc|corp|ltd|limited|company|co|private|pvt|llp)\b'
    return re.sub(suffixes, "", s).strip()

def extract_features(q_n_norm, q_a_norm, c_n_norm, c_a_norm, q_addr, c_addr, name_cos, addr_cos, found_n, found_a):
    name_jaccard = token_jaccard(q_n_norm, c_n_norm)
    addr_jaccard = token_jaccard(q_a_norm, c_a_norm)
    
    q_nums = get_numbers(q_addr)
    c_nums = get_numbers(c_addr)
    # If both have numbers and they exactly match -> 1
    # If both have numbers and they mismatch -> 0
    # If one or both lack numbers -> 0.5 (neutral)
    if q_nums and c_nums:
        addr_num_match = 1.0 if q_nums == c_nums else 0.0
    else:
        addr_num_match = 0.5
        
    q_n_strip = remove_legal_suffixes(q_n_norm)
    c_n_strip = remove_legal_suffixes(c_n_norm)
    name_exact = 1.0 if (q_n_strip and q_n_strip == c_n_strip) else 0.0
    
    return [
        name_cos,
        addr_cos,
        float(found_n),
        float(found_a),
        name_jaccard,
        addr_jaccard,
        addr_num_match,
        name_exact
    ]

def main():
    print("=== PHASE 1 - STEP 2: L2 PRUNER TRAINING ===")
    
    print("\n1. Sampling a FRESH 50k Queries (seed=43) for Honest Validation...")
    s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    # Fresh draw, not seed=42
    s1_sample = s1.sample(n=50000, seed=43)
    
    q_ids = s1_sample["entity_id"].to_list()
    q_names = s1_sample["business_name"].fill_null("").to_list()
    q_addrs = (s1_sample["business_address"].fill_null("") + " " + s1_sample["country"].fill_null("")).to_list()
    q_names_norm = norm_series(s1_sample["business_name"]).to_list()
    q_addrs_norm = (norm_series(s1_sample["business_address"]) + " " + norm_series(s1_sample["country"])).to_list()
    q_countries = s1_sample["country"].fill_null("unknown").to_list()
    
    print("\n2. Encoding Queries (FP16)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16})
    
    q_n_emb = model.encode(q_names_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    q_a_emb = model.encode(q_addrs_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    
    print("\n3. Retrieving Top-50 via FAISS HNSW...")
    import time
    start_time = time.time()
    
    index_n = faiss.read_index("output/faiss/candidates_name_hnsw.index")
    index_a = faiss.read_index("output/faiss/candidates_addr_hnsw.index")
    index_n.hnsw.efSearch = 128
    index_a.hnsw.efSearch = 128
    
    K_L1 = 50
    scores_n, idx_n = index_n.search(q_n_emb.cpu().numpy().astype(np.float32), K_L1)
    scores_a, idx_a = index_a.search(q_a_emb.cpu().numpy().astype(np.float32), K_L1)
    
    print(f"   -> FAISS Search completed in {time.time() - start_time:.2f} seconds.")
    print(f"   -> Raw FAISS scores_n Min: {scores_n.min():.4f} | Max: {scores_n.max():.4f}")
    
    # FAISS IndexHNSWFlat defaults to METRIC_L2. For unit-normalized vectors:
    # L2^2 = 2 - 2 * cos(theta)  =>  cos(theta) = 1 - (L2^2 / 2)
    scores_n = 1.0 - (scores_n / 2.0)
    scores_a = 1.0 - (scores_a / 2.0)
    print(f"   -> Converted to Cosine  Min: {scores_n.min():.4f} | Max: {scores_n.max():.4f}")
    
    del q_n_emb, q_a_emb, index_n, index_a, model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    print("\n3.5. Loading Candidates for Feature Extraction (After freeing FAISS RAM)...")
    s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])
    
    c_id_to_idx = {cid: i for i, cid in enumerate(candidates["entity_id"].to_list())}
    c_addrs_arr = (candidates["business_address"].fill_null("") + " " + candidates["country"].fill_null("")).to_numpy()
    c_ids_arr = candidates["entity_id"].to_numpy()
    
    print("\nPrecomputing candidate normalizations...")
    c_names_norm = norm_series(candidates["business_name"]).to_numpy()
    c_addrs_norm = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_numpy()
    
    # SMOKE TEST: Check row alignment for an easy match
    smoke_idx = 0
    smoke_qid = q_ids[smoke_idx]
    print(f"\n   -> SMOKE TEST: QID '{smoke_qid}' | Query Name: {q_names[smoke_idx]}")
    print(f"      Top 3 Retrieved by FAISS Name Lane:")
    for k in range(3):
        cid_idx = idx_n[smoke_idx][k]
        score = scores_n[smoke_idx][k]
        print(f"         Rank {k+1}: CID '{c_ids_arr[cid_idx]}' | Cosine: {score:.4f} | Name: {c_names_norm[cid_idx]}")
        
    print("\nLoading Ground Truth...")
    gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")
    gt_dict = {}
    for row in gt.iter_rows(named=True):
        m = row["matched_entity_ids"]
        gt_dict[row["source1_entity_id"]] = set(x.strip() for x in m.split(",") if x.strip()) if m and m.strip() else set()
        
    print("\n4. Extracting L2 Features...")
    X, y, groups = [], [], []
    
    # We will build dataset efficiently
    for i in tqdm(range(len(q_ids)), desc="Feature Extraction"):
        qid = q_ids[i]
        q_n_norm = q_names_norm[i]
        q_a_norm = q_addrs_norm[i]
        q_addr = q_addrs[i]
        true_matches = gt_dict.get(qid, set())
        
        # Merge Top K from both lanes
        cands = {} # cid -> [name_cos, addr_cos, found_n, found_a]
        
        for j in range(K_L1):
            cn_id = c_ids_arr[idx_n[i][j]]
            ca_id = c_ids_arr[idx_a[i][j]]
            
            if cn_id not in cands:
                cands[cn_id] = [scores_n[i][j], 0.0, 1, 0]
            else:
                cands[cn_id][0] = max(cands[cn_id][0], scores_n[i][j])
                cands[cn_id][2] = 1
                
            if ca_id not in cands:
                cands[ca_id] = [0.0, scores_a[i][j], 0, 1]
            else:
                cands[ca_id][1] = max(cands[ca_id][1], scores_a[i][j])
                cands[ca_id][3] = 1
                
        for cid, (n_cos, a_cos, fn, fa) in cands.items():
            c_idx = c_id_to_idx.get(cid)
            if c_idx is None: continue
            
            c_n_norm = c_names_norm[c_idx]
            c_a_norm = c_addrs_norm[c_idx]
            c_addr = c_addrs_arr[c_idx]
            
            feats = extract_features(q_n_norm, q_a_norm, c_n_norm, c_a_norm, q_addr, c_addr, n_cos, a_cos, fn, fa)
            
            X.append(feats)
            y.append(1 if cid in true_matches else 0)
            groups.append(i) # Group by Query ID (0 to 49999)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    groups = np.array(groups, dtype=np.int32)
    
    print(f"\nExtracted {len(X)} candidate pairs. Positives: {y.sum()}")
    
    print("\n5. Splitting (Train / Tune / Holdout)...")
    # First split: Train (60%), Temp (40%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y, groups))
    
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp, y_temp, groups_temp = X[temp_idx], y[temp_idx], groups[temp_idx]
    
    # Second split on Temp: Tune (20%), Holdout (20%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    tune_idx, holdout_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    
    X_tune, y_tune, groups_tune = X_temp[tune_idx], y_temp[tune_idx], groups_temp[tune_idx]
    X_holdout, y_holdout, groups_holdout = X_temp[holdout_idx], y_temp[holdout_idx], groups_temp[holdout_idx]
    
    print(f"Train size: {len(X_train)} | Tune size: {len(X_tune)} | Holdout size: {len(X_holdout)}")
    
    print("\n6. Training XGBoost Pruner (Depth 3)...")
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        objective="binary:logistic",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=10.0 # Focus on recall
    )
    clf.fit(X_train, y_train)
    
    print("\nFeature Importances:")
    f_names = ["name_cos", "addr_cos", "found_n", "found_a", "name_jaccard", "addr_jaccard", "addr_num_match", "name_exact"]
    for name, imp in zip(f_names, clf.feature_importances_):
        print(f"  {name}: {imp:.4f}")
        
    print("\nDone. Saving model to output/l2_pruner_xgb.json")
    clf.save_model("output/l2_pruner_xgb.json")
    
    print("\n==================================================")
    print("7. SWEEPING L2 PRUNER THRESHOLDS (HOLDOUT SET)")
    print("==================================================")
    
    probs = clf.predict_proba(X_holdout)[:, 1]
    unique_groups = np.unique(groups_holdout)
    
    # Map group -> list of (prob, label)
    holdout_dict = {g: [] for g in unique_groups}
    for i in range(len(probs)):
        holdout_dict[groups_holdout[i]].append((probs[i], y_holdout[i]))
        
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for country_filter in ["ALL", "India", "US"]:
        print(f"\n--- {country_filter} ---")
        print(f"{'Threshold':<10} | {'Abs Recall':<12} | {'Rel Recall':<12} | {'Avg Cands':<10} | {'Correct-Empty':<15} | {'False-Zero':<10}")
        print("-" * 80)
        
        for t in thresholds:
            total_gt_pairs = 0
            total_l1_pairs = 0
            total_hits = 0
            total_cands = 0
            
            correct_empty = 0
            false_zero = 0
            
            zero_match_queries = 0
            has_match_queries = 0
            query_count = 0
            
            for g in unique_groups:
                qid = q_ids[g]
                q_country = q_countries[g]
                if country_filter != "ALL" and q_country != country_filter:
                    continue
                    
                query_count += 1
                items = holdout_dict[g]
                
                gt_count = len(gt_dict.get(qid, set()))
                l1_count = sum(1 for p, l in items if l == 1)
                
                total_gt_pairs += gt_count
                total_l1_pairs += l1_count
                
                if gt_count == 0: zero_match_queries += 1
                else: has_match_queries += 1
                
                kept = [l for p, l in items if p >= t]
                total_cands += len(kept)
                hits = sum(1 for l in kept if l == 1)
                total_hits += hits
                
                if len(kept) == 0:
                    if gt_count == 0: correct_empty += 1
                    else: false_zero += 1
                        
            abs_rec = (total_hits / max(1, total_gt_pairs)) * 100
            rel_rec = (total_hits / max(1, total_l1_pairs)) * 100
            avg_c = total_cands / max(1, query_count)
            ce_rate = (correct_empty / max(1, zero_match_queries)) * 100
            fz_rate = (false_zero / max(1, has_match_queries)) * 100
            
            print(f"{t:<10.2f} | {abs_rec:<10.2f}% | {rel_rec:<10.2f}% | {avg_c:<9.2f} | {ce_rate:<13.2f}% | {fz_rate:<9.2f}%")

    print("\n==================================================")
    print("8. DECOY SPOT-CHECK (Dense Trap FPs)")
    print("==================================================")
    decoys = [
        ("S1-870151764", "S3-119800890"), ("S1-634558596", "S3-191804096"),
        ("S1-237329659", "S2-593841434"), ("S1-114653583", "S3-311653660"),
        ("S1-672885872", "S3-386926575"), ("S1-262668912", "S2-771795139"),
        ("S1-894573685", "S3-392303769"), ("S1-881753450", "S2-484194626"),
        ("S1-578839908", "S2-183034903"), ("S1-383156502", "S3-807046580"),
        ("S1-449204731", "S2-594475836"), ("S1-320688624", "S2-973048106"),
        ("S1-456059199", "S3-535852196"), ("S1-339374125", "S2-278707565"),
        ("S1-878607831", "S2-832410410"), ("S1-855736486", "S2-692350025"),
        ("S1-478308388", "S2-448797717"), ("S1-691432826", "S3-532574167"),
        ("S1-145037624", "S3-815190267"), ("S1-999703273", "S3-770870491")
    ]
    
    # We must fetch the exact text for these to compute lexical features
    s1_df = pl.read_csv("dataset/train/train_source1.tsv", separator="\t")
    q_dict = {}
    for row in s1_df.filter(pl.col("entity_id").is_in([q for q,c in decoys])).iter_rows(named=True):
        q_dict[row["entity_id"]] = (row["business_name"] or "", (row["business_address"] or "") + " " + (row["country"] or ""))
        
    c_dict = {}
    s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t")
    s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t")
    c_all = pl.concat([s2, s3])
    for row in c_all.filter(pl.col("entity_id").is_in([c for q,c in decoys])).iter_rows(named=True):
        c_dict[row["entity_id"]] = (row["business_name"] or "", (row["business_address"] or "") + " " + (row["country"] or ""))
        
    print("Scoring known decoys...")
    for qid, cid in decoys:
        q_name, q_addr = q_dict.get(qid, ("", ""))
        c_name, c_addr = c_dict.get(cid, ("", ""))
        q_n_norm = norm_series(pl.Series([q_name]))[0]
        q_a_norm = norm_series(pl.Series([q_addr]))[0]
        c_n_norm = norm_series(pl.Series([c_name]))[0]
        c_a_norm = norm_series(pl.Series([c_addr]))[0]
        feats = extract_features(q_n_norm, q_a_norm, c_n_norm, c_a_norm, q_addr, c_addr, 0.99, 0.99, 1, 1)
        prob = clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1]
        print(f"Prob: {prob:.4f} | Feats: {[round(f, 4) for f in feats]} | {q_addr[:40]}... vs {c_addr[:40]}...")

if __name__ == "__main__":
    main()
