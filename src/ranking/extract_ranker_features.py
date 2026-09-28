import polars as pl
import numpy as np
import math
import re
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from rapidfuzz import fuzz
from collections import defaultdict
import gc
import faiss

def norm_series(col):
    return col.str.to_lowercase().str.replace_all(r"[^\w\s]", "").fill_null("")

def get_numbers(s):
    return set(re.findall(r'\d+', str(s)))

def remove_legal_suffixes(s):
    suffixes = r'\b(llc|inc|corp|ltd|limited|company|co|private|pvt|llp|sarl|sas|sa|eurl|sci|snc)\b'
    return re.sub(suffixes, "", s).strip()

def build_idf_dict(text_series):
    """Builds a manual IDF dictionary to avoid massive scipy sparse matrix RAM overhead"""
    print("Building IDF dictionary...")
    df_dict = defaultdict(int)
    N = 0
    for text in tqdm(text_series, desc="IDF Counting"):
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

    if not tokens1 or not tokens2:
        return 0.0

    vec1 = defaultdict(float)
    for t in tokens1: vec1[t] += idf_dict.get(t, default_idf)

    vec2 = defaultdict(float)
    for t in tokens2: vec2[t] += idf_dict.get(t, default_idf)

    dot = sum(vec1[t] * vec2[t] for t in set(vec1.keys()).intersection(set(vec2.keys())))
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def numeric_jaccard(q_nums, c_nums):
    if not q_nums and not c_nums: return 0.5 # Neutral
    if not q_nums or not c_nums: return 0.0
    return len(q_nums.intersection(c_nums)) / len(q_nums.union(c_nums))

def token_jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def main(split="train"):
    print(f"=== PHASE 3: EXTRACTING RANKER FEATURES ({split.upper()}) ===")

    pairs_file = f"output/candidate_pairs_{split}.tsv"
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Missing {pairs_file}. Run Phase 1 generation first.")

    out_file = f"output/ranker_features_{split}.parquet"

    # 1. Load Texts
    print("Loading text data...")
    s1 = pl.read_csv(f"dataset/{split}/{split}_source1.tsv", separator="\t")
    
    q_names_norm = norm_series(s1["business_name"]).to_list()
    q_addrs_norm = (norm_series(s1["business_address"]) + " " + norm_series(s1["country"])).to_list()
    
    q_dict = {}
    for i, row in enumerate(s1.iter_rows(named=True)):
        q_dict[row["entity_id"]] = (q_names_norm[i], q_addrs_norm[i])
    del s1; gc.collect()

    s2 = pl.read_csv(f"dataset/{split}/{split}_source2.tsv", separator="\t")
    s3 = pl.read_csv(f"dataset/{split}/{split}_source3.tsv", separator="\t")
    candidates = pl.concat([s2, s3])

    # Maintain exact row order used during faiss index building
    c_ids_full = candidates["entity_id"].to_list()
    cid_to_row = {cid: i for i, cid in enumerate(c_ids_full)}

    c_names_norm = norm_series(candidates["business_name"]).to_list()
    c_addrs_norm = (norm_series(candidates["business_address"]) + " " + norm_series(candidates["country"])).to_list()
    
    c_dict = {}
    for i, row in enumerate(candidates.iter_rows(named=True)):
        c_dict[row["entity_id"]] = (c_names_norm[i], c_addrs_norm[i])
        
    c_name_corpus = c_names_norm
    c_addr_corpus = c_addrs_norm
    del candidates, s2, s3; gc.collect()

    # 2. Build IDF Dictionaries
    idf_n, def_n = build_idf_dict(c_name_corpus)
    idf_a, def_a = build_idf_dict(c_addr_corpus)
    del c_name_corpus, c_addr_corpus; gc.collect()

    # 3. Load BGE-M3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BGE-M3 model on {device} (FP16)...")
    model = SentenceTransformer("BAAI/bge-m3", device=device, model_kwargs={"torch_dtype": torch.float16})

    # 4. Parse Pairs & Identify Unique Entities
    print("Parsing candidate pairs...")
    queries_to_process = []
    unique_q = set()
    unique_c = set()

    with open(pairs_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip("\n").split("\t")
            if len(parts) < 2: continue
            qid = parts[0]
            cids = parts[1].split(",") if parts[1] else []
            if not cids: continue

            queries_to_process.append((qid, cids))
            unique_q.add(qid)
            unique_c.update(cids)

    print(f"Found {len(queries_to_process)} queries with at least 1 candidate.")
    print(f"Unique Queries: {len(unique_q)} | Unique Candidates: {len(unique_c)}")

    cache_file = f"output/q_emb_cache_{split}.npz"
    if os.path.exists(cache_file):
        print(f"Loading cached query embeddings from {cache_file}...")
        data = np.load(cache_file, allow_pickle=True)
        uq_list = data['uq_list'].tolist()
        q_emb_n = data['q_emb_n']
        q_emb_a = data['q_emb_a']
    else:
        print("Encoding unique QUERY entities (fresh, GPU)...")
        uq_list = list(unique_q)
        uq_n = [q_dict[q][0] for q in uq_list]
        uq_a = [q_dict[q][1] for q in uq_list]

        q_emb_n = model.encode(uq_n, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)
        q_emb_a = model.encode(uq_a, batch_size=2048, show_progress_bar=True, normalize_embeddings=True).astype(np.float16)
        
        print(f"Saving query embeddings to {cache_file}...")
        np.savez(cache_file, uq_list=np.array(uq_list), q_emb_n=q_emb_n, q_emb_a=q_emb_a)

    q_emb_n_dict = {uq_list[i]: q_emb_n[i] for i in range(len(uq_list))}
    q_emb_a_dict = {uq_list[i]: q_emb_a[i] for i in range(len(uq_list))}

    del model, q_emb_n, q_emb_a; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("Loading FAISS indices for on-the-fly reconstruction...")
    faiss_dir = f"output/faiss_{split}" if split != "train" else "output/faiss"
    index_n = faiss.read_index(f"{faiss_dir}/candidates_name_hnsw.index", faiss.IO_FLAG_MMAP)
    index_a = faiss.read_index(f"{faiss_dir}/candidates_addr_hnsw.index", faiss.IO_FLAG_MMAP)

    # 6. Feature Extraction Loop
    print("Extracting rich pairwise features (reconstructing on-the-fly to save RAM)...")

    csv_temp = f"output/temp_features_{split}.csv"
    with open(csv_temp, 'w') as f_out:
        f_out.write("source1_entity_id,candidate_entity_id,name_cos,addr_cos,name_jac,addr_jac,num_jac,name_exact,name_tfidf,addr_tfidf,name_lev,addr_lev\n")
        
        for qid, cids in tqdm(queries_to_process, desc="Pairs"):
            # We already pre-normalized these via Polars, so they are perfectly preserved
            qnn, qan = q_dict.get(qid, ("", ""))
            
            # We still need raw strings for get_numbers, so we do a quick hack (though stripped works ok for digits)
            qnums = get_numbers(qan)
            qns = remove_legal_suffixes(qnn)

            qen = q_emb_n_dict[qid]
            qea = q_emb_a_dict[qid]

            for cid in cids:
                cnn, can = c_dict.get(cid, ("", ""))
                cnums = get_numbers(can)
                cns = remove_legal_suffixes(cnn)

                # Reconstruct on the fly to prevent 80GB OOM crash!
                row_idx = int(cid_to_row[cid])
                cen = index_n.reconstruct(row_idx)
                cea = index_a.reconstruct(row_idx)

                # Dense
                name_cos = float(np.dot(qen, cen))
                addr_cos = float(np.dot(qea, cea))

                # Lexical
                name_jac = token_jaccard(qnn, cnn)
                addr_jac = token_jaccard(qan, can)
                num_jac = numeric_jaccard(qnums, cnums)
                name_exact = 1.0 if (qns and qns == cns) else 0.0

                # TF-IDF
                name_tfidf = tfidf_cosine(qnn, cnn, idf_n, def_n)
                addr_tfidf = tfidf_cosine(qan, can, idf_a, def_a)

                # Levenshtein (Normalized)
                name_lev = fuzz.ratio(qnn, cnn) / 100.0
                addr_lev = fuzz.ratio(qan, can) / 100.0

                f_out.write(f"{qid},{cid},{name_cos:.4f},{addr_cos:.4f},{name_jac:.4f},{addr_jac:.4f},{num_jac:.4f},{name_exact:.1f},{name_tfidf:.4f},{addr_tfidf:.4f},{name_lev:.4f},{addr_lev:.4f}\n")

    print("Writing feature matrix to Parquet...")
    df = pl.read_csv(csv_temp)
    df.write_parquet(out_file)
    os.remove(csv_temp)
    print(f"Saved {len(df)} rows to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args.split)
