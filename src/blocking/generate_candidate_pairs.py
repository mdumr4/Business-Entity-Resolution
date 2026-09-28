import polars as pl
import numpy as np
import faiss
import torch
import re
import os
import argparse
import json
import gzip
import gc
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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
    suffixes = r'\b(llc|inc|corp|ltd|limited|company|co|private|pvt|llp|sarl|sas|sa|eurl|sci|snc)\b'
    return re.sub(suffixes, "", s).strip()

def extract_features(q_n_norm, q_a_norm, c_n_norm, c_a_norm, q_nums, c_nums, q_n_strip, c_n_strip, name_cos, addr_cos, found_n, found_a):
    name_jaccard = token_jaccard(q_n_norm, c_n_norm)
    addr_jaccard = token_jaccard(q_a_norm, c_a_norm)

    if q_nums and c_nums:
        addr_num_match = 1.0 if q_nums == c_nums else 0.0
    else:
        addr_num_match = 0.5

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

def main(threshold=0.05, split="train", skip_phase_a=False):
    print(f"=== GENERATING FINAL CANDIDATE PAIRS ({split.upper()}) ===")
    print(f"L2 Pruner Threshold: {threshold}\n")

    out_file = f"output/candidate_pairs_{split}.tsv"
    temp_file = f"output/l1_temp_{split}.jsonl.gz"

    # We always need the queries for both phases
    print("Loading Source 1 Queries...")
    s1 = pl.read_csv(f"dataset/{split}/{split}_source1.tsv", separator="\t")
    q_ids = s1["entity_id"].to_list()
    q_names_raw = s1["business_name"].fill_null("").to_list()
    q_addrs_raw = (s1["business_address"].fill_null("") + " " + s1["country"].fill_null("")).to_list()
    
    q_names_norm = norm_series(s1["business_name"]).to_list()
    q_addrs_norm = (norm_series(s1["business_address"]) + " " + norm_series(s1["country"])).to_list()
    
    if not skip_phase_a:
        print("\n=== PHASE A: L1 FAISS RETRIEVAL ===")
        print("Loading Candidate IDs to map FAISS integer indices back to strings...")
        s2 = pl.read_csv(f"dataset/{split}/{split}_source2.tsv", separator="\t")
        s3 = pl.read_csv(f"dataset/{split}/{split}_source3.tsv", separator="\t")
        c_ids_arr = pl.concat([s2.select("entity_id"), s3.select("entity_id")])["entity_id"].to_numpy()
        del s2, s3
        gc.collect()

        print("\nLoading FAISS Indices...")
        faiss_dir = f"output/faiss_{split}" if split != "train" else "output/faiss"
        index_n = faiss.read_index(f"{faiss_dir}/candidates_name_hnsw.index", faiss.IO_FLAG_MMAP)
        index_a = faiss.read_index(f"{faiss_dir}/candidates_addr_hnsw.index", faiss.IO_FLAG_MMAP)
        index_n.hnsw.efSearch = 128
        index_a.hnsw.efSearch = 128

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading BGE-M3 model on {device} (FP16)...")
        model = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16})

        K_L1 = 50
        chunk_size = 50000

        print(f"\nProcessing {len(q_ids)} Queries in chunks of {chunk_size}...")
        start_time = time.time()

        # Open temp file to stream L1 outputs
        with gzip.open(temp_file, "wt", encoding="utf-8") as f:
            for chunk_start in range(0, len(q_ids), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(q_ids))
                print(f"\n--- Chunk [{chunk_start}:{chunk_end}] ---")

                chunk_n_norm = q_names_norm[chunk_start:chunk_end]
                chunk_a_norm = q_addrs_norm[chunk_start:chunk_end]

                q_n_emb = model.encode(chunk_n_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
                q_a_emb = model.encode(chunk_a_norm, batch_size=2048, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)

                scores_n, idx_n = index_n.search(q_n_emb.cpu().numpy().astype(np.float32), K_L1)
                scores_a, idx_a = index_a.search(q_a_emb.cpu().numpy().astype(np.float32), K_L1)

                # Convert L2^2 to Cosine Similarity
                scores_n = 1.0 - (scores_n / 2.0)
                scores_a = 1.0 - (scores_a / 2.0)

                del q_n_emb, q_a_emb; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

                # Map indices to CIDs and merge lanes
                for i in range(len(chunk_n_norm)):
                    cands = {}
                    for j in range(K_L1):
                        cn_id = c_ids_arr[idx_n[i][j]]
                        ca_id = c_ids_arr[idx_a[i][j]]

                        if cn_id not in cands: cands[cn_id] = [float(scores_n[i][j]), 0.0, 1, 0]
                        else: cands[cn_id][0] = float(max(cands[cn_id][0], scores_n[i][j])); cands[cn_id][2] = 1

                        if ca_id not in cands: cands[ca_id] = [0.0, float(scores_a[i][j]), 0, 1]
                        else: cands[ca_id][1] = float(max(cands[ca_id][1], scores_a[i][j])); cands[ca_id][3] = 1

                    # Write directly to JSONL to save RAM
                    f.write(json.dumps(cands) + "\n")

        # Free the 94GB of FAISS Indices and the Model
        del index_n, index_a, model, c_ids_arr
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"\nPhase A completed in {(time.time() - start_time) / 60:.2f} minutes.")
    else:
        print(f"\n=== SKIPPING PHASE A ===")
        print(f"Using existing temp file: {temp_file}")


    # ---------------------------------------------------------
    # PHASE B: L2 PRUNING (Heavy RAM for Text Strings)
    # ---------------------------------------------------------
    print("\n=== PHASE B: L2 PRUNING ===")
    print("Loading 10.3M Candidate Strings for Feature Extraction...")
    s2 = pl.read_csv(f"dataset/{split}/{split}_source2.tsv", separator="\t")
    s3 = pl.read_csv(f"dataset/{split}/{split}_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])

    c_id_to_idx = {cid: i for i, cid in enumerate(candidates["entity_id"].to_list())}
    c_names_raw = candidates["business_name"].fill_null("").to_list()
    c_addrs_raw = (candidates["business_address"].fill_null("") + " " + candidates["country"].fill_null("")).to_list()

    print("Pre-computing normalized candidate strings...")
    c_n_norm = norm_series(candidates["business_name"]).to_list()
    c_a_norm = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_list()
    c_nums = [get_numbers(a) for a in c_addrs_raw]
    c_n_strip = [remove_legal_suffixes(n) for n in c_n_norm]
    del candidates, s2, s3, c_names_raw, c_addrs_raw
    gc.collect()

    clf = xgb.XGBClassifier()
    clf.load_model("output/l2_pruner_xgb.json")

    print("\nExecuting Phase B Smoke Test...")
    with gzip.open(temp_file, "rt", encoding="utf-8") as f_smoke:
        first_line = f_smoke.readline()
        if first_line:
            cands_smoke = json.loads(first_line)
            smoke_qid = q_ids[0]
            print(f"   -> SMOKE TEST: QID '{smoke_qid}' | Query Name: {q_names_raw[0]}")
            sorted_cands = sorted(cands_smoke.items(), key=lambda x: x[1][0], reverse=True)[:3]
            for rank, (cid, feats) in enumerate(sorted_cands):
                c_idx = c_id_to_idx.get(cid)
                c_name = c_n_norm[c_idx] if c_idx is not None else "UNKNOWN"
                print(f"      Rank {rank+1}: CID '{cid}' | Name_Cos: {feats[0]:.4f} | Name: {c_name}")
    print("Smoke Test Complete.\n")

    # Write header for output TSV
    with open(out_file, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

    print(f"Extracting features and pruning with threshold {threshold}...")
    start_time = time.time()

    with gzip.open(temp_file, "rt", encoding="utf-8") as f_in, open(out_file, "a", encoding="utf-8") as f_out:
        # We iterate over queries one by one
        for i, line in tqdm(enumerate(f_in), total=len(q_ids), desc="Pruning"):
            qid = q_ids[i]
            cands = json.loads(line)

            # We already have Polars pre-normalized strings!
            qnn = q_names_norm[i]
            qan = q_addrs_norm[i]
            # get_numbers and suffixes still need string logic
            qnums = get_numbers(q_addrs_raw[i])
            qns = remove_legal_suffixes(qnn)

            X = []
            cid_list = []

            for cid, (n_cos, a_cos, fn, fa) in cands.items():
                c_idx = c_id_to_idx.get(cid)
                if c_idx is None: continue

                feats = extract_features(
                    qnn, qan, c_n_norm[c_idx], c_a_norm[c_idx],
                    qnums, c_nums[c_idx], qns, c_n_strip[c_idx],
                    n_cos, a_cos, fn, fa
                )
                X.append(feats)
                cid_list.append(cid)

            if X:
                probs = clf.predict_proba(np.array(X, dtype=np.float32))[:, 1]
                final_cands = [cid_list[k] for k in range(len(cid_list)) if probs[k] >= threshold]
            else:
                final_cands = []

            f_out.write(f"{qid}\t{','.join(final_cands)}\n")

    print(f"\nDone! Phase B finished in {(time.time() - start_time) / 60:.2f} minutes.")
    print(f"Final Candidate Pairs saved to: {out_file}")

    # Cleanup temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.05, help="L2 Pruner probability threshold")
    parser.add_argument("--split", type=str, default="train", help="train or test")
    parser.add_argument("--skip-phase-a", action="store_true", help="Skip FAISS retrieval and use existing temp file")
    args = parser.parse_args()
    main(args.threshold, args.split, args.skip_phase_a)
