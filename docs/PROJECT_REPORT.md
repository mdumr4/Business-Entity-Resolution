# Scalable Business Entity Resolution — Architecture, Evaluation & Lessons

> **How to use this file.** Numeric results in §4 were filled from
> `output/eval/eval_results.json` (produced by `evaluate_pipeline.py`). Every other number
> comes from run logs or direct checks of the data. Ablation and transfer results are pending (`--ablate --transfer`).

Solo project, built with AI assistance (Claude) as a pair-programmer/reviewer. All code was run and validated by me on a single Vertex AI Workbench GPU instance.

---

## 1. Problem

Link business records across three noisy sources with no shared identifier. Source 1 is the deduplicated reference; for each S1 entity find all matching S2/S3 records (zero, one, or many).

| | Train | Test |
|---|---|---|
| S1 entities | 2,206,821 (US 1.32M, India 0.88M) | ≈1.73M (US 0.66M, India 0.81M, **France 0.26M – never seen in training**) |
| S2 + S3 candidate pool | 10,320,219 | ≈9.97M |
| True matches per S1 | mean 3.46, median 3, max 11; **5.6 % have none** | – |

Naïve comparison = 2.2M × 10.3M ≈ **22.8 trillion pairs**. Metric: macro-averaged F₀.₅ *per S1 entity* (precision counts 2× recall; correctly predicting "no match" scores 1.0, any false merge on a singleton scores 0.0).

## 2. Architecture

```
 S1 query ─► normalise (Unicode-safe) ─► BGE-M3 embeddings (name lane, address lane)
                                                 │
 L1  RETRIEVAL   two FAISS-HNSW indices (name / address), top-50 per lane, merged
                                                 │           ~10.3M × 2 lanes, ~84–94 GB
 L2  PRUNER      depth-3 XGBoost on 8 cheap features (recall-biased, threshold 0.05)
                                                 │           50–100 → ~14 candidates / query  ==> candidate_pairs.tsv
 L3  RANKER      XGBoost on 13 features (dense + TF-IDF + Levenshtein + numeric Jaccard
                 + intra-query relative rank/gap), threshold tuned on entity-level F0.5
                                                 │
 POST            keep each S2/S3 record only for the S1 entity that scores it highest
                                                 ▼
                                        matching_results.tsv
```

**Why these choices (evidence noted where it exists)**

| Decision | Evidence |
|---|---|
| Dense retrieval, not TF-IDF/exact | Exact match: 23.5M pairs for only 29 % recall. TF-IDF: OOM on 10M rows. |
| BGE-M3 (0.5B), not Qwen3-Embedding-4B | 4B model ≈ **80 GPU-hours** to embed the pool on one L4 – rejected on measured throughput. |
| Separate name/address lanes, rank-fusion (RRF) | Lanes are complementary: at K=30, name 69.7 %, address 89.5 %, fused 93.9 % recall. |
| RRF for *ordering*, raw cosine for cutoffs | RRF scores of rank 1 vs rank 30 differ by only 32 % (0.0164 vs 0.0111) – a margin cutoff on them never removed a single candidate. |
| Token-join lane dropped | Assumed ~99.999 % precision; **measured 8.3 %** on 500k queries (street numbers and common developer names collide by chance). |
| Binary classifier, not LambdaMART | Reasoned, **not benchmarked**: the metric rewards an absolute accept/reject per pair (incl. empty output), which needs calibrated probabilities. |
| Two-stage pruner→ranker | Pruner is recall-biased and cheap; ranker is precision-biased and richer. Different jobs, different losses. |
| Pruner intentionally passes look-alikes | 20 inspected "dense-trap" pairs (same name, one-digit street-number change) score 0.86–0.99 in the pruner; the ranker's numeric/lexical features are what reject them. |

## 3. Evaluation methodology (`evaluate_pipeline.py`)

No test labels exist, so all evaluation uses the training labels with these safeguards:

1. **Universe = all 2.2M train S1 entities**, including those with zero candidates.
2. **Deterministic 70/15/15 hash split** (train / tune / holdout) by entity ID. Entities from the diagnostic samples used for design decisions (seeds 42 and 43, incl. the L2-pruner training sample) are forced into *train*, so the holdout never influenced any design choice.
3. **Ranker is trained on *train*, early-stopped and thresholded on *tune*, reported on *holdout*.** (The original script used one split for early stopping *and* threshold selection.)
4. **Leaderboard-faithful metric.** Per-entity F₀.₅ against the *full* ground truth, so matches lost by L1/L2 count against recall. (The original validation metric only counted matches that survived into the candidate set, which overstates recall.) The vectorised implementation is checked against a plain-Python reference and reproduces the README's worked example (0.714).
5. **Uncertainty:** 95 % bootstrap CI over entities.
6. **Floor and ceiling:** "predict nothing" scores the singleton rate; a *perfect* ranker on the candidate set scores the ceiling imposed by L1+L2 recall. Real performance should be read between them.
7. **Segments:** country, number of true matches (0 / 1 / 2–3 / 4–5 / 6+), candidate-list size (0 / 1–9 / 10–19 / 20+ "fat tail").
8. **Calibration** (reliability curve, ECE) – the threshold is only meaningful if probabilities are.
9. **Ablations** (`--ablate`): drop relative features / dense embeddings / TF-IDF / Levenshtein / numeric Jaccard, retrain, report holdout delta.
10. **Zero-shot-country proxy** (`--transfer`): train on US only → test on India and vice-versa, with the threshold taken from the *source* country (as it would be for France) versus an oracle threshold. **This stresses the ranker only** – the embeddings and L2 pruner saw both countries – so it is a proxy, not a France measurement.
11. **Train/serve parity:** features rounded to 4 decimals as in the inference script's CSV output.

## 4. Results (holdout, 315,196 entities never used for any decision)

| Metric | Value |
|---|---|
| **Macro F₀.₅** | **0.9082** (95 % CI 0.9075–0.9089) |
| Floor (predict nothing) / ceiling (perfect ranker) | 0.0563 / 0.9788 |
| Recall: candidate set → final | 94.3 % → 82.3 % |
| Precision (micro) | 97.1 % |
| Singletons correctly predicted empty | 88.5 % |
| Entities with a match but predicted empty | 2.3 % |
| Candidates / query (mean, median, P95, max) | 14.9, 12, 39, 100 |
| Calibration ECE | 0.003 |
| + one-owner-per-candidate step (lower bound) | 0.9087 (Δ +0.0004) |

| Segment | n entities | F₀.₅ |
|---|---|---|
| country=US | 189,191 | 0.9208 |
| country=India | 126,005 | 0.8893 |
| true_matches=0 (singleton) | 17,745 | 0.8854 |
| true_matches=1 | 16,902 | 0.7836 |
| true_matches=2–3 | 129,644 | 0.9024 |
| true_matches=4–5 | 115,084 | 0.9280 |
| true_matches≥6 | 35,821 | 0.9361 |
| candidates=0 | 71 | 0.6479 |
| candidates=1–9 | 121,928 | 0.9020 |
| candidates=10–19 | 122,879 | 0.9170 |
| candidates≥20 (fat tail) | 70,318 | 0.9040 |

Ablations and country-transfer results pending (`--ablate --transfer`).

**Blocking stage, measured on a 50k-query diagnostic sample (pruner threshold 0.05, after the normalisation fixes in §6):**
absolute recall 94.2 % overall (US 96.4 %, India 90.9 %), 14.9 candidates/query. Note this threshold was chosen on that sample, so treat it as a diagnostic, not a held-out figure.

## 5. Compute log (single node, one ~22 GB-VRAM GPU)

| Step | Measured |
|---|---|
| Embed 10.32M candidates, name lane / address lane | ≈ 55 min / ≈ 70 min (batch 2048, FP16; ≈3.2k / ≈2.5k texts/s) |
| Two HNSW indices (M=32, efConstruction=200) | ≈ 84–94 GB total |
| L1 retrieval + merge, 2.2M queries | 78.7 min |
| L2 pruning (Python), 2.2M queries | 54.9 min (≈ 670 queries/s, single process) |
| Ranker features, train | query encoding ≈ 28 min + per-pair Python feature loop |
| Rejected: Qwen3-4B embeddings | ≈ 80 h (estimated from throughput) |
| Failed attempts | TF-IDF `.toarray()` → 157 GB RAM; naive GPU matmul (50k × 2M) → 186 GB VRAM; full-corpus dict of reconstructed vectors → crash (est. ~78 GB RAM) |

## 6. Challenges and what I learned (chronological failures that silently corrupted results)

1. **Memory walls.** Fixed by using CSR internals instead of `.toarray()`, chunking GPU matmuls, and reconstructing embeddings from the FAISS index on demand instead of re-encoding or caching them.
2. **Rank-fusion scores are not confidences.** A margin cutoff on RRF scores was mathematically inert (see §2).
3. **An assumption I never checked:** token-join precision "by construction" 99.999 % → measured 8.3 %.
4. **Learned fusion lost to plain RRF** (India recall@10: 84.5 % vs 86.4 %); its negative agreement weight was a feature-encoding artifact (sentinel zeros + a derived AND-feature).
5. **FAISS metric mismatch.** `IndexHNSWFlat` defaults to L2; scores fed to a model as "cosine" were inverted. Neighbour *ranking* was unaffected, features were not.
6. **ID-prefix bugs** (`S1-…` vs bare digits) produced empty lookups and 20 identical "predictions" – caught only because identical outputs looked wrong.
7. **Label bug:** ground truth treated as one match per entity; it is a list (median 3).
8. **Wrong metric:** scikit-learn "macro" F₀.₅ is per-*class*, not per-entity; replaced with the leaderboard definition.
9. **Unicode normalisation mismatch (the big one).** The index was built with Polars' regex engine, queries were normalised with Python's `re`. Python's `\w` treats Devanagari vowel signs (combining marks) as punctuation and strips them; Rust's does not. A stored vector vs a fresh encode of the same record had cosine **0.52**; after unifying normalisation it is **≈1.000** across Devanagari, Gujarati, Odia, Telugu and accented-Latin samples. Absolute recall at the pruner rose from 88.6 → 90.9 % (India) *and* 93.9 → 96.4 % (US) – the US gain is consistent with a second bug fixed at the same time (the pruner had been trained on un-normalised query text); I did not ablate the two fixes separately.
10. **Recall accounting:** "relative recall" (of what retrieval returned) looked like 98 % while absolute recall was 88–91 %. Always gate on absolute.
11. **Validation hygiene:** early-stopping and threshold selection on the same split; design decisions made repeatedly on the same 50k sample → hence the forced-train exclusion in §3.

## 7. Scalability

**Why it scales in principle.** Retrieval is an ANN graph search (roughly O(log N) per query) instead of an O(N) scan; the pruner and ranker score a bounded list (≈14 candidates, hard cap ≈100) so their cost per query does not grow with corpus size and is embarrassingly parallel. Partitioning the index by a generic field (e.g. `country`) needs no per-country logic and gives an unseen country its own shard automatically. Ground truth confirms it is valid: **all 7,638,365 true S2/S3 IDs belong to exactly one S1 entity.**

**What would have to change at 1 billion candidates (back-of-envelope, not benchmarked):**

| Component | Today (measured) | At ~1B (estimate) | Change needed |
|---|---|---|---|
| Embedding | ≈ 2 h for 10.3M × 2 lanes | ≈ 200 GPU-hours | data-parallel over GPUs (~1 day on 8) |
| Index memory | ≈ 8 KB / candidate (two 1024-d fp32 vectors + graph) → 84–94 GB | ≈ 8 TB flat | product quantisation (e.g. 64–128 B/vector) and/or sharding across machines |
| Query serving | ≈ 470 q/s for L1 retrieval incl. query encoding on one box | needs replicas | shard + replica fan-out, merge top-K |
| Pruner/ranker | ≈ 670 q/s, single Python process | linear in #queries | vectorise / multiprocess; per-pair Python loops are the bottleneck |
| Index build | one pass, offline | incremental | HNSW supports insertion; periodic rebuild |

Inference already memory-maps the index (`IO_FLAG_MMAP`) so it need not be fully resident; the training-time scripts loaded it fully. Random-access HNSW on a cold memory-mapped file is slow, so this is a memory-saving option, not a free lunch.

## 8. Known limitations and suggested next steps

1. **Recall ceiling, especially India.** Transliteration/script normalisation was never implemented; BGE-M3 is relied on to bridge scripts. Try BGE-M3's sparse/multi-vector outputs as an extra retrieval lane (untested here), or a transliterated-text lane.
2. **France is untested.** Thresholds and features were tuned on US/India only; suffix lists were extended (SARL/SAS/…) by hand. Report the transfer proxy honestly and expect some calibration shift.
3. **Global assignment.** Because each candidate has exactly one owner, use one-owner-per-candidate resolution (implemented as post-processing) and consider full bipartite assignment.
4. **S2↔S3 consistency.** S2 and S3 records of the same business could be linked first (clusters), then matched to S1 as a group.
5. **Near-duplicate decoys.** The data contains look-alikes (same name, street number ±1, suffix swapped) that are *not* matches; a dedicated pair-level model on numeric/suffix disagreement may beat generic string similarity.
6. **Engineering:** rewrite the per-pair Python loops in Polars/NumPy or Rust; add unit tests for normalisation parity between index build and query time; log run manifests (code hash, data hash, thresholds) for every artifact.
7. **Not benchmarked:** end-to-end latency at scale, alternative embedding models, per-source (S2 vs S3) error rates.

## 9. Reproduce

```
python build_faiss_index.py --split train        # ~2 h, GPU
python generate_candidate_pairs.py --threshold 0.05 --split train
python extract_ranker_features.py --split train
python evaluate_pipeline.py --ablate --transfer --save-model      # this report's numbers
python build_faiss_index.py --split test         # separate index for the test candidate pool
python generate_submission.py
python validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
