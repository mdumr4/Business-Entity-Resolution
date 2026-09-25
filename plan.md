# Amazon ML Challenge 2026 - Master Architecture & Execution Plan

## 1. The Core Objective
We are solving a massive-scale Entity Resolution problem. 
- **Source 1 (S1):** ~2.2 Million Queries
- **Source 2 & 3 (S2/S3):** ~10.3 Million Candidates
Our goal is to map S1 entities to their true matches in S2/S3 while maximizing the **F-0.5 Macro Score**. Because comparing every S1 against every S2/S3 yields **22 Trillion** combinations, a single model is impossible. We must use a funnel architecture.

---

## 2. The Multi-Stage ML Pipeline Strategy

### Phase 1: Blocking (Candidate Generation)
The goal is to reduce 10.3M candidates down to the Top-K (e.g., Top 20) most likely candidates per query. We discussed building three independent blocking flows to test which one yields the highest recall for the hackathon:
1. **Exact Match Baseline:** Fast, naive string matching using Polars to catch the obvious pairs.
2. **Lexical Retrieval (TF-IDF):** 
   - **The Thinking:** Catches word-overlap and frequency but fails on typos/synonyms.
   - **The Engineering Hurdle:** Hit a 157GB RAM explosion. Solved it mathematically by bypassing `.toarray()` and extracting pointers directly from the SciPy CSR sparse matrix.
3. **Dense Retrieval (Semantic Embeddings):** 
   - **The Thinking:** Uses Neural Networks to catch semantic meaning, missing words, and messy OCR. 
   - **The Engineering Hurdle:** Hit a 186GB VRAM PyTorch crash during cosine similarity matrix multiplication. Solved it by batching queries into chunks of 1,000 on the L4 GPU. (Currently using `all-MiniLM-L6-v2`).

### Phase 2: Pairwise Feature Engineering
Once candidates are generated, we join them with their original text and calculate strict distance metrics using `rapidfuzz`:
- `feat_name_jw` (Jaro-Winkler string similarity for business names)
- `feat_name_lev` (Levenshtein distance for business names)
- `feat_addr_jw` (Jaro-Winkler for addresses)
- `feat_addr_lev` (Levenshtein for addresses)

### Phase 3: Reranking (XGBoost)
We train an XGBoost Binary Classifier to look at those 4 features and predict `is_match` (0.0 to 1.0).
- **The Engineering Hurdle:** Extreme Class Imbalance. Our Exact Match run gave us 34k matches vs 23M negatives (99.8% false positives), causing XGBoost to lazily predict 0 for everything.
- **The Fix:** We implemented a dynamic `scale_pos_weight` to aggressively penalize false negatives and force the model to learn the true match boundaries.

### Phase 4: Inference & Thresholding
- Load the Hackathon Test Dataset.
- Run Phase 1 & 2 on the test data.
- Score all candidates with the trained XGBoost model.
- Filter out garbage matches (e.g., score < 0.5), group by the S1 Query ID, and select the Top-1 highest-scoring candidate.
- Output perfectly formatted `submission.csv`.

---

## 3. Current Execution State: "Flow 1"
We are explicitly defining our current completed, working pipeline as **Flow 1**. 

**Flow 1 Definition:**
- **Blocker:** Dense Retrieval (`all-MiniLM-L6-v2`) globally across all countries.
- **Features:** Jaro-Winkler & Levenshtein.
- **Reranker:** XGBoost (with scale_pos_weight).

### Immediate Next Steps for Flow 1:
1. Do **not** change the architecture (e.g., no country separation yet). We must establish a baseline.
2. Run `inference.py` using Flow 1 on the Test set to generate `submission.csv`.
3. Submit to the leaderboard and record our F-0.5 Macro score.
4. Rename/Archive the current weights as `models/Flow1_xgb_reranker.json`.

Only *after* we evaluate Flow 1 will we branch out to test Flow 2 (e.g., country separation, heavier BGE-m3 embeddings, or lexical ensembling) to scientifically measure if those ideas actually improve the leaderboard score.
