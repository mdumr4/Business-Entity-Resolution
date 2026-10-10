"""Unified Entity Resolution Inference Engine.

Executes the full 3-stage cascading inference funnel with persistent checkpointing:
  Stage 0: Multilingual text loading & Polars normalization
  Stage 1: Query encoding via BGE-M3 (or checkpointed cache)
  Stage 2: Memory-mapped FAISS dual-lane search & L2 pruner filtering (candidate_pairs.tsv)
  Stage 3: Rich pairwise feature extraction (FAISS on-the-fly reconstruction)
  Stage 4: L3 XGBoost ranking, optimal thresholding, and global one-record-one-business resolution
"""

import argparse
import gc
import json
import os
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional
import faiss
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.core.similarity import (
    build_idf_dict,
    l2_sq_to_cosine,
    levenshtein_ratio,
    numeric_jaccard,
    tfidf_cosine,
    token_jaccard,
)
from src.core.text import get_numbers, norm_series, remove_legal_suffixes
from src.models.l2_pruner import L2Pruner
from src.models.l3_ranker import ALL_RANKER_FEATURES, L3Ranker


def run_pipeline(
    data_dir: str = "dataset",
    split: str = "test",
    faiss_dir: Optional[str] = None,
    output_dir: str = "output",
    output_matches_file: Optional[str] = None,
    output_candidates_file: Optional[str] = None,
    l2_model_path: str = "output/l2_pruner_xgb.json",
    l3_model_path: str = "output/phase3_ranker_xgb.json",
    threshold_path: str = "output/phase3_threshold.json",
    device: Optional[str] = None,
    k_l1: int = 50,
    l2_threshold: float = 0.05,
):
    """Run the complete entity resolution inference pipeline."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if faiss_dir is None:
        faiss_dir = os.path.join(output_dir, f"faiss_{split}")

    os.makedirs(output_dir, exist_ok=True)
    emb_cache = os.path.join(output_dir, f"q_emb_cache_{split}.npz")
    pairs_file = os.path.join(output_dir, f"candidate_pairs_{split}.tsv")
    features_file = os.path.join(output_dir, f"ranker_features_{split}.parquet")

    final_matches_out = output_matches_file or os.path.join(output_dir, "matching_results.tsv")
    final_candidates_out = output_candidates_file or os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 65)
    print(f"  SCALABLE BUSINESS ENTITY RESOLUTION PIPELINE ({split.upper()})")
    print("=" * 65)
    print(f"Data Dir   : {data_dir}/{split}")
    print(f"FAISS Dir  : {faiss_dir}")
    print(f"Device     : {device}")
    print(f"Output     : {final_matches_out}")

    # ================================================================
    # STAGE 0: LOAD & NORMALIZE TEXT DATA
    # ================================================================
    print("\n[STAGE 0] Loading and normalizing entities...")
    t0 = time.time()

    s1_path = os.path.join(data_dir, split, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, split, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.tsv")

    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Missing Source 1 queries at {s1_path}")

    s1 = pl.read_csv(s1_path, separator="\t")
    s2 = pl.read_csv(s2_path, separator="\t")
    s3 = pl.read_csv(s3_path, separator="\t")
    candidates = pl.concat([s2, s3])

    q_ids = s1["entity_id"].to_list()
    q_names_norm = norm_series(s1["business_name"]).to_list()
    q_addrs_norm = (norm_series(s1["business_address"]) + " " + norm_series(s1["country"])).to_list()

    c_ids_full = candidates["entity_id"].to_list()
    c_names_norm = norm_series(candidates["business_name"]).to_list()
    c_addrs_norm = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_list()

    c_id_to_idx = {cid: i for i, cid in enumerate(c_ids_full)}
    q_dict = {q_ids[i]: (q_names_norm[i], q_addrs_norm[i]) for i in range(len(q_ids))}
    c_dict = {c_ids_full[i]: (c_names_norm[i], c_addrs_norm[i]) for i in range(len(c_ids_full))}

    c_nums = [get_numbers(a) for a in c_addrs_norm]
    c_n_strip = [remove_legal_suffixes(n) for n in c_names_norm]

    del s1, s2, s3, candidates
    gc.collect()
    print(f"  Loaded {len(q_ids):,} queries, {len(c_ids_full):,} candidates in {time.time()-t0:.1f}s")

    # ================================================================
    # STAGE 1: ENCODE QUERIES
    # ================================================================
    if os.path.exists(emb_cache):
        print(f"\n[STAGE 1] SKIPPED — loading cached query embeddings from {emb_cache}")
        cached = np.load(emb_cache, allow_pickle=True)
        uq_list = cached["uq_list"].tolist()
        q_emb_n = cached["q_emb_n"]
        q_emb_a = cached["q_emb_a"]
        del cached
    else:
        print(f"\n[STAGE 1] Encoding {len(q_ids):,} queries on {device}...")
        t1 = time.time()
        model = SentenceTransformer(
            "BAAI/bge-m3",
            device=device,
            model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
        )
        uq_list = q_ids
        q_emb_n = model.encode(q_names_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)
        q_emb_a = model.encode(q_addrs_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)

        np.savez(emb_cache, uq_list=np.array(uq_list), q_emb_n=q_emb_n, q_emb_a=q_emb_a)
        print(f"  Stage 1 complete in {(time.time()-t1)/60:.1f} min → {emb_cache}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    q_emb_n_dict = {uq_list[i]: q_emb_n[i] for i in range(len(uq_list))}
    q_emb_a_dict = {uq_list[i]: q_emb_a[i] for i in range(len(uq_list))}
    del q_emb_n, q_emb_a
    gc.collect()

    # ================================================================
    # STAGE 2: FAISS SEARCH + L2 PRUNING
    # ================================================================
    if os.path.exists(pairs_file):
        print(f"\n[STAGE 2] SKIPPED — {pairs_file} already exists")
    else:
        print(f"\n[STAGE 2] Running L1 FAISS retrieval + L2 pruning...")
        t2 = time.time()

        name_idx_p = os.path.join(faiss_dir, "candidates_name_hnsw.index")
        addr_idx_p = os.path.join(faiss_dir, "candidates_addr_hnsw.index")
        if not (os.path.exists(name_idx_p) and os.path.exists(addr_idx_p)):
            raise FileNotFoundError(f"Missing FAISS indices in {faiss_dir}. Build them first.")

        index_n = faiss.read_index(name_idx_p, faiss.IO_FLAG_MMAP)
        index_a = faiss.read_index(addr_idx_p, faiss.IO_FLAG_MMAP)
        index_n.hnsw.efSearch = 128
        index_a.hnsw.efSearch = 128

        pruner = L2Pruner()
        pruner.load_model(l2_model_path)

        chunk_size = 50000
        c_ids_arr = np.array(c_ids_full)

        with open(pairs_file, "w", encoding="utf-8") as f_out:
            f_out.write("source1_entity_id\tcandidate_entity_ids\n")
            for chunk_start in tqdm(range(0, len(q_ids), chunk_size), desc="L1+L2 Funnel"):
                chunk_end = min(chunk_start + chunk_size, len(q_ids))
                chunk_emb_n = np.array([q_emb_n_dict[q_ids[i]] for i in range(chunk_start, chunk_end)], dtype=np.float32)
                chunk_emb_a = np.array([q_emb_a_dict[q_ids[i]] for i in range(chunk_start, chunk_end)], dtype=np.float32)

                scores_n, idx_n = index_n.search(chunk_emb_n, k_l1)
                scores_a, idx_a = index_a.search(chunk_emb_a, k_l1)
                scores_n = l2_sq_to_cosine(scores_n)
                scores_a = l2_sq_to_cosine(scores_a)

                for i in range(chunk_end - chunk_start):
                    qi = chunk_start + i
                    qid = q_ids[qi]
                    qnn, qan = q_names_norm[qi], q_addrs_norm[qi]
                    qnums = get_numbers(qan)
                    qns = remove_legal_suffixes(qnn)

                    cands = {}
                    for j in range(k_l1):
                        cn_id = c_ids_arr[idx_n[i][j]]
                        ca_id = c_ids_arr[idx_a[i][j]]
                        if cn_id not in cands:
                            cands[cn_id] = [float(scores_n[i][j]), 0.0, 1, 0]
                        else:
                            cands[cn_id][0] = max(cands[cn_id][0], float(scores_n[i][j]))
                            cands[cn_id][2] = 1

                        if ca_id not in cands:
                            cands[ca_id] = [0.0, float(scores_a[i][j]), 0, 1]
                        else:
                            cands[ca_id][1] = max(cands[ca_id][1], float(scores_a[i][j]))
                            cands[ca_id][3] = 1

                    X_l2 = []
                    cid_list = []
                    for cid, (n_cos, a_cos, fn, fa) in cands.items():
                        c_idx = c_id_to_idx.get(cid)
                        if c_idx is None:
                            continue
                        feats = L2Pruner.extract_pair_features(
                            q_name_norm=qnn,
                            q_addr_norm=qan,
                            c_name_norm=c_names_norm[c_idx],
                            c_addr_norm=c_addrs_norm[c_idx],
                            q_nums=qnums,
                            c_nums=c_nums[c_idx],
                            q_name_stripped=qns,
                            c_name_stripped=c_n_strip[c_idx],
                            name_cos=n_cos,
                            addr_cos=a_cos,
                            found_in_name_lane=bool(fn),
                            found_in_addr_lane=bool(fa),
                        )
                        X_l2.append(feats)
                        cid_list.append(cid)

                    if X_l2:
                        final_cands = pruner.filter_candidates(
                            np.array(X_l2, dtype=np.float32), cid_list, threshold=l2_threshold
                        )
                    else:
                        final_cands = []

                    f_out.write(f"{qid}\t{','.join(final_cands)}\n")

        del index_n, index_a, pruner, c_ids_arr
        gc.collect()
        print(f"  Stage 2 complete in {(time.time()-t2)/60:.1f} min → {pairs_file}")

    # ================================================================
    # STAGE 3: RICH FEATURE EXTRACTION
    # ================================================================
    if os.path.exists(features_file):
        print(f"\n[STAGE 3] SKIPPED — {features_file} already exists")
    else:
        print(f"\n[STAGE 3] Extracting rich pairwise features...")
        t3 = time.time()

        idf_n, def_n = build_idf_dict(c_names_norm)
        idf_a, def_a = build_idf_dict(c_addrs_norm)

        index_n = faiss.read_index(os.path.join(faiss_dir, "candidates_name_hnsw.index"), faiss.IO_FLAG_MMAP)
        index_a = faiss.read_index(os.path.join(faiss_dir, "candidates_addr_hnsw.index"), faiss.IO_FLAG_MMAP)

        queries_to_process = []
        with open(pairs_file, "r", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.strip("\n").split("\t")
                if len(parts) >= 2 and parts[1]:
                    queries_to_process.append((parts[0], parts[1].split(",")))

        csv_temp = os.path.join(output_dir, f"temp_features_{split}.csv")
        with open(csv_temp, "w", encoding="utf-8") as f_out:
            f_out.write(
                "source1_entity_id,candidate_entity_id,name_cos,addr_cos,name_jac,addr_jac,num_jac,name_exact,name_tfidf,addr_tfidf,name_lev,addr_lev\n"
            )
            for qid, cids in tqdm(queries_to_process, desc="L3 Features"):
                qnn, qan = q_dict.get(qid, ("", ""))
                qnums = get_numbers(qan)
                qns = remove_legal_suffixes(qnn)
                qen, qea = q_emb_n_dict[qid], q_emb_a_dict[qid]

                for cid in cids:
                    cnn, can = c_dict.get(cid, ("", ""))
                    cnums = get_numbers(can)
                    cns = remove_legal_suffixes(cnn)
                    row_idx = c_id_to_idx.get(cid)
                    if row_idx is None:
                        continue

                    cen = index_n.reconstruct(int(row_idx))
                    cea = index_a.reconstruct(int(row_idx))

                    name_cos = float(np.dot(qen, cen))
                    addr_cos = float(np.dot(qea, cea))
                    name_jac = token_jaccard(qnn, cnn)
                    addr_jac = token_jaccard(qan, can)
                    num_jac = numeric_jaccard(qnums, cnums)
                    name_exact = 1.0 if (qns and qns == cns) else 0.0
                    name_tfidf = tfidf_cosine(qnn, cnn, idf_n, def_n)
                    addr_tfidf = tfidf_cosine(qan, can, idf_a, def_a)
                    name_lev = levenshtein_ratio(qnn, cnn)
                    addr_lev = levenshtein_ratio(qan, can)

                    f_out.write(
                        f"{qid},{cid},{name_cos:.4f},{addr_cos:.4f},{name_jac:.4f},{addr_jac:.4f},{num_jac:.4f},{name_exact:.1f},{name_tfidf:.4f},{addr_tfidf:.4f},{name_lev:.4f},{addr_lev:.4f}\n"
                    )

        df = pl.read_csv(csv_temp)
        df.write_parquet(features_file)
        os.remove(csv_temp)
        del df, index_n, index_a, idf_n, idf_a
        gc.collect()
        print(f"  Stage 3 complete in {(time.time()-t3)/60:.1f} min → {features_file}")

    # ================================================================
    # STAGE 4: RANKING, THRESHOLDING & RESOLUTION
    # ================================================================
    print(f"\n[STAGE 4] Applying ranker & resolving entity matches...")
    t4 = time.time()

    ranker = L3Ranker()
    ranker.load_model(l3_model_path, threshold_path)

    df_feats = pl.read_parquet(features_file)
    df_feats = L3Ranker.add_relative_features(df_feats)

    X_test = df_feats.select(ALL_RANKER_FEATURES).to_numpy()
    qids_arr = df_feats["source1_entity_id"].to_list()
    cids_arr = df_feats["candidate_entity_id"].to_list()

    probs = ranker.predict_proba(X_test)
    resolved_matches = ranker.predict_matches(
        query_ids=qids_arr,
        candidate_ids=cids_arr,
        probs=probs,
        all_query_universe=q_ids,
        enforce_one_owner=True,
    )

    matched_count = 0
    with open(final_matches_out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for qid in q_ids:
            matches = resolved_matches.get(qid, [])
            f.write(f"{qid}\t{','.join(matches)}\n")
            if matches:
                matched_count += 1

    # Keep candidate_pairs synced
    if os.path.exists(pairs_file) and pairs_file != final_candidates_out:
        shutil.copy(pairs_file, final_candidates_out)

    print(f"  Total Source 1 entities processed : {len(q_ids):,}")
    print(f"  Entities with ≥1 match             : {matched_count:,}")
    print(f"  Singletons (0 matches)            : {len(q_ids) - matched_count:,}")
    print(f"  Matches written to                : {final_matches_out}")
    print(f"  Stage 4 completed in {(time.time()-t4)/60:.1f} min")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="End-to-End Entity Resolution Inference")
    parser.add_argument("--data-dir", default="dataset", help="Root dataset directory")
    parser.add_argument("--split", default="train", help="Split to run on (train or test)")
    parser.add_argument("--faiss-dir", default=None, help="Directory containing pre-built FAISS indices")
    parser.add_argument("--output-dir", default="output", help="Directory for intermediate checkpoints and outputs")
    parser.add_argument("--output-matches", default=None, help="Path for final matching_results.tsv")
    parser.add_argument("--output-candidates", default=None, help="Path for candidate_pairs.tsv")
    parser.add_argument("--l2-model", default="output/l2_pruner_xgb.json", help="Path to L2 Pruner model")
    parser.add_argument("--l3-model", default="output/phase3_ranker_xgb.json", help="Path to L3 Ranker model")
    parser.add_argument("--threshold-file", default="output/phase3_threshold.json", help="Path to threshold JSON")
    parser.add_argument("--device", default=None, help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        split=args.split,
        faiss_dir=args.faiss_dir,
        output_dir=args.output_dir,
        output_matches_file=args.output_matches,
        output_candidates_file=args.output_candidates,
        l2_model_path=args.l2_model,
        l3_model_path=args.l3_model,
        threshold_path=args.threshold_file,
        device=args.device,
    )


if __name__ == "__main__":
    main()
