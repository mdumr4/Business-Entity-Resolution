"""
Unified Test Inference Pipeline
================================
Merges: generate_candidate_pairs + extract_ranker_features + generate_submission
into a single script with hard checkpoints at every stage.

Assumes: FAISS test indices are pre-built in output/faiss_test/

Checkpoints:
  1. output/q_emb_cache_test.npz       — query embeddings (27 min GPU work)
  2. output/candidate_pairs_test.tsv    — L2-pruned candidate set
  3. output/ranker_features_test.parquet — rich pairwise features
  Final: output/matching_results.tsv    — submission file
"""

import polars as pl
import numpy as np
import math
import re
import os
import json
import gzip
import gc
import time
import faiss
import torch
import xgboost as xgb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from collections import defaultdict

# ── Shared Utilities ────────────────────────────────────────────

def norm_series(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def get_numbers(s):
    return set(re.findall(r'\d+', str(s)))

def remove_legal_suffixes(s):
    suffixes = r'\b(llc|inc|corp|ltd|limited|company|co|private|pvt|llp|sarl|sas|sa|eurl|sci|snc)\b'
    return re.sub(suffixes, "", s).strip()

def token_jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def numeric_jaccard(q_nums, c_nums):
    if not q_nums and not c_nums: return 0.5
    if not q_nums or not c_nums: return 0.0
    return len(q_nums.intersection(c_nums)) / len(q_nums.union(c_nums))

def build_idf_dict(text_list):
    print("Building IDF dictionary...")
    df_dict = defaultdict(int)
    N = 0
    for text in tqdm(text_list, desc="IDF Counting"):
        tokens = set(text.split())
        for t in tokens:
            df_dict[t] += 1
        N += 1
    idf_dict = {}
    default_idf = math.log((N + 1) / 1.0) + 1.0
    for w, df in df_dict.items():
        idf_dict[w] = math.log((N + 1) / (df + 1)) + 1.0
    return idf_dict, default_idf

def tfidf_cosine(s1, s2, idf_dict, default_idf):
    tokens1 = s1.split()
    tokens2 = s2.split()
    if not tokens1 or not tokens2: return 0.0
    vec1 = defaultdict(float)
    for t in tokens1: vec1[t] += idf_dict.get(t, default_idf)
    vec2 = defaultdict(float)
    for t in tokens2: vec2[t] += idf_dict.get(t, default_idf)
    dot = sum(vec1[t] * vec2[t] for t in set(vec1.keys()).intersection(set(vec2.keys())))
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)


def main():
    split = "train"
    print("=" * 60)
    print("  UNIFIED TEST INFERENCE PIPELINE")
    print("=" * 60)

    # ── Paths ───────────────────────────────────────────────────
    faiss_dir       = f"output/faiss_{split}"
    emb_cache       = f"output/q_emb_cache_{split}.npz"
    pairs_file      = f"output/candidate_pairs_{split}.tsv"
    features_file   = f"output/ranker_features_{split}.parquet"
    model_path      = "output/phase3_ranker_xgb.json"
    thresh_path     = "output/phase3_threshold.json"
    pruner_path     = "output/l2_pruner_xgb.json"
    final_out       = "output/matching_results.tsv"
    candidate_out   = "output/candidate_pairs.tsv"

    # ── Pre-flight Checks ───────────────────────────────────────
    for idx_file in ["candidates_name_hnsw.index", "candidates_addr_hnsw.index"]:
        path = f"{faiss_dir}/{idx_file}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found: {path}. Run build_faiss_index.py --split test first.")

    if not os.path.exists(pruner_path):
        raise FileNotFoundError(f"L2 Pruner not found: {pruner_path}. Train it first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Ranker model not found: {model_path}. Train it first.")

    # ================================================================
    # STAGE 0: LOAD ALL TEXT DATA (always runs, cheap ~15s)
    # ================================================================
    print("\n[STAGE 0] Loading and normalizing text data...")
    t0 = time.time()

    s1 = pl.read_csv(f"dataset/{split}/{split}_source1.tsv", separator="\t")
    s2 = pl.read_csv(f"dataset/{split}/{split}_source2.tsv", separator="\t")
    s3 = pl.read_csv(f"dataset/{split}/{split}_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])

    q_ids       = s1["entity_id"].to_list()
    q_names_norm = norm_series(s1["business_name"]).to_list()
    q_addrs_norm = (norm_series(s1["business_address"]) + " " + norm_series(s1["country"])).to_list()

    c_ids_full   = candidates["entity_id"].to_list()
    c_names_norm = norm_series(candidates["business_name"]).to_list()
    c_addrs_norm = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_list()

    c_id_to_idx = {cid: i for i, cid in enumerate(c_ids_full)}
    q_dict = {q_ids[i]: (q_names_norm[i], q_addrs_norm[i]) for i in range(len(q_ids))}
    c_dict = {c_ids_full[i]: (c_names_norm[i], c_addrs_norm[i]) for i in range(len(c_ids_full))}

    # Pre-compute expensive per-candidate data
    c_nums    = [get_numbers(a) for a in c_addrs_norm]
    c_n_strip = [remove_legal_suffixes(n) for n in c_names_norm]

    del s1, s2, s3, candidates; gc.collect()
    print(f"  Loaded {len(q_ids)} queries, {len(c_ids_full)} candidates in {time.time()-t0:.1f}s")

    # ================================================================
    # STAGE 1: ENCODE QUERIES (checkpoint: q_emb_cache_test.npz)
    # ================================================================
    if os.path.exists(emb_cache):
        print(f"\n[STAGE 1] SKIPPED — loading cached embeddings from {emb_cache}")
        data = np.load(emb_cache, allow_pickle=True)
        uq_list = data['uq_list'].tolist()
        q_emb_n = data['q_emb_n']
        q_emb_a = data['q_emb_a']
        model = None  # Don't need BGE-M3 if cache exists
    else:
        print(f"\n[STAGE 1] Encoding {len(q_ids)} queries on GPU...")
        t1 = time.time()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading BGE-M3 on {device} (FP16)...")
        model = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16})

        uq_list = q_ids  # All source1 entities are unique
        q_emb_n = model.encode(q_names_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)
        q_emb_a = model.encode(q_addrs_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)

        print(f"  Saving checkpoint → {emb_cache}")
        np.savez(emb_cache, uq_list=np.array(uq_list), q_emb_n=q_emb_n, q_emb_a=q_emb_a)
        print(f"  Stage 1 complete in {(time.time()-t1)/60:.1f} min")

    q_emb_n_dict = {uq_list[i]: q_emb_n[i] for i in range(len(uq_list))}
    q_emb_a_dict = {uq_list[i]: q_emb_a[i] for i in range(len(uq_list))}

    # Free the model if we loaded it — FAISS search uses raw numpy, not the model
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ================================================================
    # STAGE 2: FAISS SEARCH + L2 PRUNING (checkpoint: candidate_pairs_test.tsv)
    # ================================================================
    if os.path.exists(pairs_file):
        print(f"\n[STAGE 2] SKIPPED — {pairs_file} already exists")
    else:
        print(f"\n[STAGE 2] FAISS search + L2 pruning...")
        t2 = time.time()

        print("  Loading FAISS indices (MMAP)...")
        index_n = faiss.read_index(f"{faiss_dir}/candidates_name_hnsw.index", faiss.IO_FLAG_MMAP)
        index_a = faiss.read_index(f"{faiss_dir}/candidates_addr_hnsw.index", faiss.IO_FLAG_MMAP)
        index_n.hnsw.efSearch = 128
        index_a.hnsw.efSearch = 128

        print("  Loading L2 Pruner...")
        pruner = xgb.XGBClassifier()
        pruner.load_model(pruner_path)

        K_L1 = 50
        chunk_size = 50000
        c_ids_arr = np.array(c_ids_full)

        with open(pairs_file, "w", encoding="utf-8") as f_out:
            f_out.write("source1_entity_id\tcandidate_entity_ids\n")

            for chunk_start in tqdm(range(0, len(q_ids), chunk_size), desc="FAISS+Prune"):
                chunk_end = min(chunk_start + chunk_size, len(q_ids))

                # Use cached embeddings (FP16 → FP32 for FAISS)
                chunk_emb_n = np.array([q_emb_n_dict[q_ids[i]] for i in range(chunk_start, chunk_end)], dtype=np.float32)
                chunk_emb_a = np.array([q_emb_a_dict[q_ids[i]] for i in range(chunk_start, chunk_end)], dtype=np.float32)

                scores_n, idx_n = index_n.search(chunk_emb_n, K_L1)
                scores_a, idx_a = index_a.search(chunk_emb_a, K_L1)

                # L2² → cosine for normalized vectors
                scores_n = 1.0 - (scores_n / 2.0)
                scores_a = 1.0 - (scores_a / 2.0)

                for i in range(chunk_end - chunk_start):
                    qi = chunk_start + i
                    qid = q_ids[qi]
                    qnn = q_names_norm[qi]
                    qan = q_addrs_norm[qi]
                    qnums = get_numbers(qan)
                    qns = remove_legal_suffixes(qnn)

                    # Merge lanes
                    cands = {}
                    for j in range(K_L1):
                        cn_id = c_ids_arr[idx_n[i][j]]
                        ca_id = c_ids_arr[idx_a[i][j]]
                        if cn_id not in cands: cands[cn_id] = [float(scores_n[i][j]), 0.0, 1, 0]
                        else: cands[cn_id][0] = max(cands[cn_id][0], float(scores_n[i][j])); cands[cn_id][2] = 1
                        if ca_id not in cands: cands[ca_id] = [0.0, float(scores_a[i][j]), 0, 1]
                        else: cands[ca_id][1] = max(cands[ca_id][1], float(scores_a[i][j])); cands[ca_id][3] = 1

                    # L2 Pruner
                    X_l2 = []
                    cid_list = []
                    for cid, (n_cos, a_cos, fn, fa) in cands.items():
                        c_idx = c_id_to_idx.get(cid)
                        if c_idx is None: continue
                        cnn = c_names_norm[c_idx]
                        can = c_addrs_norm[c_idx]
                        cn = c_nums[c_idx]
                        cns = c_n_strip[c_idx]
                        X_l2.append([
                            n_cos, a_cos, float(fn), float(fa),
                            token_jaccard(qnn, cnn), token_jaccard(qan, can),
                            1.0 if qnums == cn else (0.5 if (not qnums and not cn) else 0.0),
                            1.0 if (qns and qns == cns) else 0.0
                        ])
                        cid_list.append(cid)

                    if X_l2:
                        probs = pruner.predict_proba(np.array(X_l2, dtype=np.float32))[:, 1]
                        final_cands = [cid_list[k] for k in range(len(cid_list)) if probs[k] >= 0.05]
                    else:
                        final_cands = []

                    f_out.write(f"{qid}\t{','.join(final_cands)}\n")

        del index_n, index_a, pruner, c_ids_arr
        gc.collect()
        print(f"  Stage 2 complete in {(time.time()-t2)/60:.1f} min → {pairs_file}")

    # ================================================================
    # STAGE 3: RICH FEATURE EXTRACTION (checkpoint: ranker_features_test.parquet)
    # ================================================================
    if os.path.exists(features_file):
        print(f"\n[STAGE 3] SKIPPED — {features_file} already exists")
    else:
        print(f"\n[STAGE 3] Extracting rich pairwise features...")
        t3 = time.time()

        # Build IDF
        idf_n, def_n = build_idf_dict(c_names_norm)
        idf_a, def_a = build_idf_dict(c_addrs_norm)

        # Load FAISS for reconstruction (MMAP = 0 RAM)
        print("  Loading FAISS indices for embedding reconstruction (MMAP)...")
        index_n = faiss.read_index(f"{faiss_dir}/candidates_name_hnsw.index", faiss.IO_FLAG_MMAP)
        index_a = faiss.read_index(f"{faiss_dir}/candidates_addr_hnsw.index", faiss.IO_FLAG_MMAP)

        # Parse candidate pairs
        queries_to_process = []
        with open(pairs_file, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip("\n").split("\t")
                if len(parts) < 2: continue
                qid = parts[0]
                cids = parts[1].split(",") if parts[1] else []
                if cids:
                    queries_to_process.append((qid, cids))

        print(f"  Processing {len(queries_to_process)} queries with candidates...")

        csv_temp = f"output/temp_features_{split}.csv"
        with open(csv_temp, 'w') as f_out:
            f_out.write("source1_entity_id,candidate_entity_id,name_cos,addr_cos,name_jac,addr_jac,num_jac,name_exact,name_tfidf,addr_tfidf,name_lev,addr_lev\n")

            for qid, cids in tqdm(queries_to_process, desc="Features"):
                qnn, qan = q_dict.get(qid, ("", ""))
                qnums = get_numbers(qan)
                qns = remove_legal_suffixes(qnn)
                qen = q_emb_n_dict[qid]
                qea = q_emb_a_dict[qid]

                for cid in cids:
                    cnn, can = c_dict.get(cid, ("", ""))
                    cnums = get_numbers(can)
                    cns = remove_legal_suffixes(cnn)

                    row_idx = c_id_to_idx.get(cid)
                    if row_idx is None: continue
                    cen = index_n.reconstruct(int(row_idx))
                    cea = index_a.reconstruct(int(row_idx))

                    name_cos = float(np.dot(qen, cen))
                    addr_cos = float(np.dot(qea, cea))
                    name_jac = token_jaccard(qnn, cnn)
                    addr_jac = token_jaccard(qan, can)
                    num_jac  = numeric_jaccard(qnums, cnums)
                    name_exact = 1.0 if (qns and qns == cns) else 0.0
                    name_tfidf = tfidf_cosine(qnn, cnn, idf_n, def_n)
                    addr_tfidf = tfidf_cosine(qan, can, idf_a, def_a)
                    name_lev = fuzz.ratio(qnn, cnn) / 100.0
                    addr_lev = fuzz.ratio(qan, can) / 100.0

                    f_out.write(f"{qid},{cid},{name_cos:.4f},{addr_cos:.4f},{name_jac:.4f},{addr_jac:.4f},{num_jac:.4f},{name_exact:.1f},{name_tfidf:.4f},{addr_tfidf:.4f},{name_lev:.4f},{addr_lev:.4f}\n")

        print("  Converting CSV → Parquet...")
        df = pl.read_csv(csv_temp)
        df.write_parquet(features_file)
        os.remove(csv_temp)
        print(f"  Stage 3 complete in {(time.time()-t3)/60:.1f} min → {features_file} ({len(df)} rows)")
        del df, index_n, index_a, idf_n, idf_a
        gc.collect()

    # ================================================================
    # STAGE 4: FINAL RANKING + SUBMISSION (always runs, ~2 min)
    # ================================================================
    print(f"\n[STAGE 4] Applying ranker and generating submission...")
    t4 = time.time()

    df = pl.read_parquet(features_file)
    print(f"  Loaded {len(df)} candidate pairs")

    # Relative rank features
    print("  Computing intra-query relative features...")
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

    X_test = df.select(feature_cols).to_numpy()
    qids = df["source1_entity_id"].to_numpy()
    cids = df["candidate_entity_id"].to_numpy()

    print("  Loading ranker model...")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)

    with open(thresh_path, "r") as f:
        best_thresh = json.load(f).get("optimal_threshold", 0.63)
    print(f"  Optimal threshold: {best_thresh:.4f}")

    probs = clf.predict_proba(X_test)[:, 1]

    # Group ALL matches above threshold per query (not just the best one!)
    print("  Collecting matches above threshold...")
    matches_by_query = defaultdict(list)
    for i in range(len(qids)):
        if probs[i] >= best_thresh:
            matches_by_query[qids[i]].append(cids[i])

    # Load full query list to ensure every Source1 entity has a row
    s1_test = pl.read_csv(f"dataset/{split}/{split}_source1.tsv", separator="\t")
    all_qids = s1_test["entity_id"].to_list()

    matched_count = 0
    with open(final_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in all_qids:
            matched = matches_by_query.get(qid, [])
            f.write(f"{qid}\t{','.join(matched)}\n")
            if matched:
                matched_count += 1

    print(f"  Wrote {len(all_qids)} rows to {final_out}")
    print(f"  Entities with ≥1 match: {matched_count}")
    print(f"  Entities with 0 matches (singletons): {len(all_qids) - matched_count}")

    # Also copy candidate_pairs for the submission ZIP
    import shutil
    if os.path.exists(pairs_file):
        shutil.copy(pairs_file, candidate_out)
        print(f"  Copied {pairs_file} → {candidate_out}")

    print(f"\n  Stage 4 complete in {(time.time()-t4)/60:.1f} min")
    print("=" * 60)
    print(f"  DONE! Submission ready at: {final_out}")
    print(f"  Validate with: python utils/validate_submission.py --matching {final_out} --candidate {candidate_out} --test-dir dataset/{split}")
    print("=" * 60)

if __name__ == "__main__":
    main()
