# Post drafts

**Before posting:** (1) check the challenge rules on sharing solutions and wait until the leaderboard is final if in doubt; (2) don't paste raw dataset rows or the competition's name/data without checking; (3) transfer result (⟦transfer⟧) pending `--ablate --transfer` run.

---

## LinkedIn

**I built a business entity-resolution pipeline on 22 trillion possible pairs — alone, with an AI pair-programmer. The most valuable thing I learned is how many bugs looked like success.**

The task: link millions of noisy business records across 3 sources with no shared ID. 2.2M queries × 10.3M candidates ≈ 22.8 trillion comparisons. Metric: per-entity F0.5, so a false merge hurts twice as much as a miss.

**Architecture (two-stage funnel):**
• BGE-M3 embeddings → two FAISS-HNSW indices (name, address) → top-50 each
• Cheap XGBoost pruner: 50–100 candidates → ~14 per query
• Precision-tuned XGBoost ranker (dense + TF-IDF + edit-distance + numeric features), threshold tuned on the *actual* metric
• One-owner-per-candidate step (I verified in the labels: 0 of 7.6M true records belong to two entities)

**Result on a clean holdout the model never touched:** F0.5 = **0.9082** (95% CI 0.9075–0.9089), 14.9 candidates per query, singletons handled correctly 88.5%.

**Bugs that silently corrupted results (and how they were caught):**
1. Two text normalisers (Python regex vs Rust regex) disagreed on Devanagari vowel signs — stored vs fresh embeddings had cosine **0.52** instead of ~1.0. Fixing it moved absolute recall by ~2.5 points.
2. A "99.999% precise" retrieval lane I assumed was **8.3%** when measured.
3. FAISS defaulted to L2 distance while I treated scores as cosine.
4. A validation metric that ignored every match my retrieval had already lost — flattering, wrong.
5. Early stopping and threshold selection on the same split.

**Compute reality:** one GPU. A 4B-parameter embedder would have needed ~80 GPU-hours, so I chose a 0.5B model. Naïve TF-IDF wanted 157 GB of RAM; a naïve GPU matmul wanted 186 GB of VRAM.

**What I'd do next:** transliteration for Indic scripts, a sparse retrieval lane, S2↔S3 clustering, and proper zero-shot testing for a country not in training (my proxy: train on US → test on India, F0.5 ⟦transfer⟧).

Not an expert — I used AI as a reviewer that kept asking "did you measure that?" The discipline was: build → measure → explain → then move on. Full write-up and code: ⟦link⟧

#MachineLearning #EntityResolution #FAISS #XGBoost #MLOps

---

## X / Twitter thread

1/ Built an entity-resolution pipeline: 2.2M records × 10.3M candidates = 22.8 *trillion* possible pairs. Solo, one GPU, AI as pair-programmer. Thread on what actually mattered 🧵

2/ Architecture: BGE-M3 → 2 FAISS-HNSW indices (name/address) → cheap pruner (50→~14 candidates) → precision-tuned XGBoost ranker → one-owner-per-candidate. Clean holdout F0.5: **0.9082**.

3/ Bug #1: two regex engines disagreed on Devanagari vowel signs. Same record, stored vs fresh embedding: cosine 0.52. After the fix: ~1.000. Silent, no error, just worse recall.

4/ Bug #2: I assumed a rare-token retrieval lane was ~99.999% precise "by construction." Measured: 8.3%. Measure your assumptions.

5/ Bug #3: my validation metric only counted matches that survived retrieval. Matches lost earlier weren't penalised. Score looked great. It was wrong. Now: per-entity F0.5 against the full ground truth + bootstrap CIs.

6/ Compute: a 4B embedder ≈ 80 GPU-hours on my card → picked a 0.5B one. Naïve TF-IDF wanted 157 GB RAM. Naïve GPU matmul wanted 186 GB VRAM. Chunking and choosing the right model beat "more hardware."

7/ Honest limits: recall on Indic-script records is lower; the test set has a country I never trained on (proxy test: train US → test India, F0.5 ⟦transfer⟧). Write-up + code: ⟦link⟧
