# Amazon ML Challenge 2026 - Master Architecture & Execution Plan

## 1. The Core Objective
We are solving a massive-scale Entity Resolution problem. 
- **Source 1 (S1):** ~2.2 Million Queries
- **Source 2 & 3 (S2/S3):** ~10.3 Million Candidates
Our goal is to map S1 entities to their true matches in S2/S3 while maximizing the **F-0.5 Macro Score**. Because comparing every S1 against every S2/S3 yields **22 Trillion** combinations, a single model is mathematically impossible. We must use a staged funnel architecture.

---

## 2. Original Strategy Brief & Architectural Blueprint
At the start of the project, we aligned on a robust architecture that merged two different conceptual plans into one unified vision:

### Where Both Plans Converged (The Core Architecture)
- **Two-stage pipeline:** Blocking (Candidate Generation) → Pairwise Matching (GBDT Classifier) → Threshold tuning.
- **Country Agnosticism:** Country logic must *never* be hardcoded. The pipeline must generalize cleanly to unseen regions (like France).
- **Threshold Tuning:** The decision cutoff cannot be a generic `0.5` probability. It must be dynamically tuned specifically for the F-0.5 metric using a grid search, acknowledging the all-or-nothing singleton scoring mechanism.
- **Missing-field Handling:** The system must be explicitly aware of nulls and singletons.

### Crucial Additions from the Merged Plans
- **Dense Retrieval (Semantic Embeddings):** Added as a crucial blocking channel (`all-MiniLM-L6-v2`) to catch cross-script pairs (e.g., a Devanagari name vs. its Latin transliteration) where N-gram/MinHash/Lexical overlap completely fails.
- **Cross-Encoder Escalation:** Demoted from a blanket pass to an *optional* last-mile reranker on a tiny shortlist (Top-2 per entity) to save GPU throughput constraints.
- **Normalization & Transliteration:** An explicit pre-processing stage to handle Unicode normalization, legal-suffix canonicalization, and script transliteration (Devanagari → Latin) *before* blocking. Without this, cross-script matches are invisible.
- **Union of Independent Blocking Signals:** Real Entity Resolution requires orthogonal blocking keys unioned together (Name LSH + Address Geo + Dense ANN) to maximize the recall ceiling, rather than picking just one method.
- **Generalization Validation Proxy (France Proxy):** Because France is hidden, we must build a zero-shot proxy (e.g., Train on US, Validate on India) to measure degradation on unseen countries.
- **Systems & Memory Realism:** Explicitly acknowledging that naive pandas loads will OOM immediately. We must use out-of-core processing (Polars/Dask) and strict memory constraints.

---

## 3. What We Actually Did (The Reality of Execution)

We translated that blueprint into code, tackling massive engineering constraints along the way. Here is exactly what is built and functioning:

### Phase 1: Blocking (Candidate Generation)
We built three independent blocking channels (to eventually be unioned):
1. **Exact Match Baseline:** Fast, naive string matching using Polars out-of-core engine to catch obvious pairs.
2. **Lexical Retrieval (TF-IDF):** Hit a 157GB RAM explosion. Solved mathematically by bypassing `.toarray()` and extracting CSR matrix pointers directly.
3. **Dense Retrieval (Semantic Embeddings):** Implemented `sentence-transformers/all-MiniLM-L6-v2`. Hit a 186GB PyTorch VRAM crash during cosine similarity dot-products. Solved by chunking into GPU batches of 1,000. 

### Phase 2: Pairwise Feature Engineering
We implemented `address_normalizer.py` for basic lowercasing and punctuation stripping, though full Devanagari transliteration is still pending. We calculate strict distance metrics using `rapidfuzz`:
- `feat_name_jw` (Jaro-Winkler) & `feat_name_lev` (Levenshtein)
- `feat_addr_jw` (Jaro-Winkler) & `feat_addr_lev` (Levenshtein)

### Phase 3: Pairwise Matching (XGBoost Reranker)
We trained a GBDT binary classifier. 
- **The Execution Hurdle:** Extreme Class Imbalance. The exact match block yielded 34k matches vs 23M negatives (99.8% false positives). XGBoost lazily predicted `0` for everything.
- **The Fix:** We implemented dynamic `scale_pos_weight` to aggressively penalize false negatives, forcing the GBDT to learn the actual match boundaries.

---

## 4. Current Execution State: "Flow 1"
We are explicitly defining our current completed, end-to-end pipeline as **Flow 1**. It represents a single, cohesive path through our planned architecture.

**Flow 1 Definition:**
- **Normalization:** Basic Polars lowercase + stripping.
- **Blocker:** Pure Dense Retrieval (`all-MiniLM-L6-v2`) globally across all countries (Country Agnostic).
- **Features:** Jaro-Winkler & Levenshtein.
- **Reranker:** XGBoost (with `scale_pos_weight`).
- **Thresholding:** Currently static filtering (`>= 0.5`) with a Top-1 selection rule.

### Immediate Next Steps for Flow 1:
1. Do **not** change the code. Establish the Flow 1 baseline.
2. Run `inference.py` using Flow 1 on the Test set to generate `submission.csv`.
3. Submit to the leaderboard and record our F-0.5 Macro score.
4. *Already completed:* Locked in Flow 1 weights as `models/Flow1_xgb_reranker.json`.

### Future Iterations (Flow 2 & Beyond):
Once Flow 1 is benchmarked, we will iterate back on the original brief:
- **Unioning Blockers:** Combine Exact Match + Dense + Lexical to hit the true recall ceiling.
- **France Validation Proxy:** Implement the "Train US, Validate India" split to test generalizability before relying on the leaderboard.
- **F-0.5 Grid Search:** Replace the static `0.5` cutoff with an empirical grid search maximizing the macro F-0.5 formula.
- **Transliteration:** Implement proper Devanagari to Latin transliteration in the normalizer.
