# Amazon ML Challenge 2026: Scalable Business Entity Resolution

This project implements a highly scalable, multi-stage Machine Learning pipeline for Business Entity Resolution across noisy, unlinked datasets. It compares millions of reference entities (Source 1) against a massive pool of candidate entities (Sources 2 and 3) to find true matches with a 22.8 trillion pair search space.

## Architecture

The matching pipeline is built as a highly-efficient 3-stage funnel:

1. **L1 Retrieval (FAISS + BGE-M3)**
   - Queries and candidates are safely normalized for Unicode.
   - Separate BGE-M3 (0.5B) dense embeddings are computed for the `name` and `address` lanes.
   - FAISS HNSW indices are used to retrieve the top-50 candidates per lane in roughly O(log N) time.
2. **L2 Pruner (XGBoost - Recall-Biased)**
   - A lightweight depth-3 XGBoost model runs on 8 computationally cheap features.
   - It filters the 50-100 candidates down to ~14 candidates per query using a generous 0.05 probability threshold.
3. **L3 Ranker (XGBoost - Precision-Biased)**
   - A heavier XGBoost model extracts 13 rich features (dense cosine, TF-IDF cosine, Levenshtein distance, Numeric Jaccard, and intra-query relative gap/rank).
   - A tuned threshold converts probability scores into definitive match decisions.
   - Post-processing enforces a "one-record-one-business" rule to resolve assignment conflicts.

## Directory Structure

```
├── dataset/
│   └── train/                  # Contains *_source1.tsv, *_source2.tsv, *_source3.tsv
├── src/                        # Core pipeline source code
│   ├── blocking/               # L1 Retrieval and L2 Pruner scripts
│   │   ├── build_faiss_index.py
│   │   ├── generate_candidate_pairs.py
│   │   └── train_l2_pruner.py
│   ├── ranking/                # L3 Ranker and inference scripts
│   │   ├── extract_ranker_features.py
│   │   ├── generate_submission.py
│   │   └── train_ranker.py
│   └── advanced_evaluation.py  # Advanced evaluation script
├── utils/
│   └── validate_submission.py  # Script to validate submission format
├── output/                     # Generated artifacts (embeddings, FAISS indices, models, outputs)
├── PROJECT_REPORT.md           # Detailed engineering and evaluation report
├── CHALLENGE_PROMPT.md         # Original competition instructions
└── evaluate_pipeline.py        # End-to-end evaluation pipeline with strict holdout splitting
```

*(Note: The `dataset/test` data has been removed from this pipeline. Both training and testing stages now operate strictly using subsets of the `train` dataset to allow for rigorous local evaluation).*

## Instructions

To run the pipeline from scratch on the `train` dataset:

### 1. Build L1 FAISS Indices
Encodes all target candidates into dense representations and builds the HNSW index.
```bash
python src/blocking/build_faiss_index.py
```

### 2. Generate Candidate Pairs (L1 + L2)
Queries the FAISS index to get top candidates, then applies the cheap L2 pruner.
```bash
python src/blocking/generate_candidate_pairs.py --threshold 0.05
```

### 3. Extract L3 Ranker Features
Computes the expensive TF-IDF, Levenshtein, and Jaccard features for the surviving pairs.
```bash
python src/ranking/extract_ranker_features.py
```

### 4. Evaluate the Pipeline
Evaluates the full end-to-end pipeline (including training the L3 ranker) using a strict 70/15/15 deterministic hash split (train/tune/holdout) to ensure zero data leakage. 
```bash
python evaluate_pipeline.py --ablate --transfer --save-model
```

### 5. Generate Final Submission (Inference)
Runs the trained L3 Ranker on the chosen dataset to produce the final `matching_results.tsv` file.
```bash
python src/ranking/generate_submission.py
```

### 6. Validate Output
Ensures the output format conforms to the submission guidelines.
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv
```
