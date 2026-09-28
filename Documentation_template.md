# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** The A-Team  
**Submission Date:** September 2026

---

## 1. Executive Summary
Our solution uses a scalable, 3-Phase Entity Resolution pipeline designed for high recall and exact macro-$F_{0.5}$ precision. By leveraging a Dual-Lane FAISS Architecture (Name and Address) using lightweight `BGE-M3` embeddings, an L2 XGBoost Pruner for fast candidate reduction, and a final feature-rich XGBoost Ranker trained on relative intra-query features, we achieve exceptional accuracy while maintaining strict control over processing time and memory overhead.

---

## 2. Methodology

### 2.1 Problem Analysis
During EDA, we discovered that simple lexical overlaps (e.g. TF-IDF) fail on noise variations, abbreviations, and Indian transliterations. Furthermore, we identified that computing distances on raw strings versus properly normalized strings was causing systemic global recall drops (e.g., US recall jumping from 93.89% to 96.41% upon fixing a normalization mismatch). We also noticed a stark precision tradeoff: normalizing correctly allowed Devanagari text to match validly, but artificially boosted the score of true singletons (Correct-Empty) via generic shared vocabulary, pushing the L2 Pruner to misclassify zero-match entities.

### 2.2 Solution Strategy

**Approach Type:** Multi-Stage (Blocking + L2 Pruner + Feature-Rich Classifier)

**Core Innovation:** 
1. **Dual-Lane FAISS Blocking:** Searching addresses and names independently in Float16 HNSW FAISS indices allows extreme recall (99%+) without memory saturation (using `faiss.IO_FLAG_MMAP`).
2. **Strict Multi-Phase Subsampling:** Instead of overwhelming memory with expensive Jaccard/Fuzzy features on 20 million candidates, we use an L2 XGBoost Pruner on computationally cheap cosine similarities to dramatically filter the space to 4.8 million highly-probable pairs.
3. **Relative Rank Features:** We augment the final ranker with intra-query metrics (e.g., `addr_cos_gap_from_top`) to simulate list-wise learning (LambdaMART) while preserving binary logistic outputs for hard $F_{0.5}$ thresholding.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** Dual-Lane Dense Embeddings via `BAAI/bge-m3` (Name Lane, Address Lane).
- **Candidate pairs generated:** Top-100 per lane, generating ~20 million raw candidate pairs.
- **How you ensured true matches were not lost:** We used an aggressive Phase 2 "L2 Pruner". Instead of using the Top-100 raw candidates, we trained an intermediate XGBoost model using the combined Name and Address cosine similarities to learn the non-linear relationship between the two lanes. By sweeping an empirical threshold, we eliminated 75% of candidates (down to 4.8 million) while retaining 96.4% recall on US data and 90.9% on Indian data.

---

## 4. Matching Model

**Features used:**
- Name features: `name_cos` (BGE-M3), `name_jaccard` (token-level), `name_fuzzy` (Token Sort Ratio), `name_idf_overlap`, `name_cos_gap_from_top`
- Address features: `addr_cos` (BGE-M3), `addr_jaccard`, `addr_fuzzy`, `addr_idf_overlap`, `addr_cos_gap_from_top`, `addr_cos_rank`
- Other features: `num_overlap`, `num_diff`, `name_len_diff`, `addr_len_diff`

**Model type:** XGBoost Binary Logistic Ranker  
**Threshold selection method:** We implemented an exact Python translation of the competition's Macro-Averaged $F_{0.5}$ metric (which rewards "Correct-Empty" predictions). To prevent data leakage (double-dipping), the ranker is trained using a **3-way split** via `GroupShuffleSplit`: 
- 70% Train 
- 10% Tune (used for `early_stopping_rounds`)
- 20% Holdout (used exclusively for sweeping the optimal $F_{0.5}$ threshold).

---

## 5. Results & Error Analysis

- **Common false positives (wrong merges):** True singletons (Correct-Empty instances) are particularly vulnerable to false merges when generic localized vocabulary (like city names or business suffixes like "Pvt Ltd") heavily dominate the dense embeddings.
- **Common false negatives (missed matches):** Drastic spelling mutations or extreme truncations in addresses that push the true match out of the Top-100 HNSW FAISS search radius.

---

## 6. Conclusion
By treating Entity Resolution as a cascading search problem, we bypassed the $O(N^2)$ bottleneck. The combination of memory-mapped GPU FAISS indexing, an L2 Pruner for scale reduction, and a tightly stratified 3-way XGBoost Ranker yields a system that is robust, empirically verifiable, and production-ready.
