# Amazon ML Challenge 2026 - Master Architecture & Execution Plan

## 1. The Core Objective
We are solving a massive-scale Entity Resolution problem.
- **Source 1 (S1):** ~2.2 Million Queries
- **Source 2 & 3 (S2/S3):** ~10.3 Million Candidates
Our goal is to map S1 entities to their true matches in S2/S3 while maximizing the **F-0.5 Macro Score**. Because comparing every S1 against every S2/S3 yields **22 Trillion** combinations, a single model is mathematically impossible. We must use a staged funnel architecture.

---

## 2. Operating Principle

> At every phase: **build the thing → produce a number you can explain → don't advance until that number clears a gate.**
>
> "XGBoost trained, F₀.₅ = 0.71" is **not** a checkpoint.
> "Blocking recall ceiling is 0.94, driven by a 0.81 ceiling on France proxy vs 0.97 on US/India — the gap is address-token blocking failing on French postal formats" **is** a checkpoint.

---

## 3. Original Strategy Brief & Architectural Blueprint

At the start of the project, we merged two conceptual plans into one unified vision:

### Where Both Plans Converged (The Core Architecture)
- **Two-stage pipeline:** Blocking (Candidate Generation) → Pairwise Matching (GBDT Classifier) → Threshold tuning.
- **Country Agnosticism:** Country logic must *never* be hardcoded. The pipeline must generalize cleanly to unseen regions (like France).
- **Threshold Tuning:** The decision cutoff cannot be a generic `0.5` probability. It must be dynamically tuned specifically for F-0.5 using a grid search, acknowledging the all-or-nothing singleton scoring mechanism.
- **Missing-field Handling:** The system must be explicitly aware of nulls and singletons.

### Crucial Additions from the Merged Plans
- **Dense Retrieval (Semantic Embeddings):** Added as a crucial blocking channel (`all-MiniLM-L6-v2`) to catch cross-script pairs (e.g., Devanagari name vs. its Latin transliteration) where N-gram/MinHash/Lexical overlap completely fails.
- **Cross-Encoder Escalation:** Demoted from a blanket pass to an *optional* last-mile reranker on a tiny shortlist (Top-2 per entity), not throughput-feasible at 34M pairs without significant GPU infra.
- **Normalization & Transliteration:** An explicit pre-processing stage for Unicode normalization, legal-suffix canonicalization, and Devanagari→Latin transliteration *before* blocking. Without this, cross-script matches are invisible.
- **Union of Independent Blocking Signals:** Real ER requires 2-4 orthogonal blocking keys unioned (Name LSH + Address Geo + Dense ANN + Exact) to maximize the recall ceiling, not picking just one method.
- **Generalization Validation Proxy (France Proxy):** Because France is hidden, we must build a zero-shot proxy (e.g., Train on US, Validate on India) to measure degradation on unseen countries.
- **Systems & Memory Realism:** Naive pandas OOMs on 3.9GB. We must use out-of-core processing (Polars) and strict memory/VRAM constraints throughout.
- **Concrete pair-volume estimate:** 1.7M × 20 ≈ 34M test pairs — a sanity number to carry through for sizing the feature/scoring stage.

### Four-Candidate Ensemble Architecture (A-D)
- **Candidate A:** Frozen-embedding reranker (cosine similarity as a feature).
- **Candidate B:** Cross-encoder shortlist (BERT-class, only on Top-2 per entity after GBDT).
- **Candidate C:** Feature-stack GBDT (XGBoost/LightGBM with string-distance features).
- **Candidate D:** Graph consistency / conflict resolution (greedy argmax-with-margin assignment).

---

## 4. What We Actually Built So Far (Flow 1)

We translated the blueprint into code, tackling massive engineering constraints. Here is exactly what is built and functioning:

### Blocking (Candidate Generation)
Three independent blocking channels built (to be unioned):
1. **Exact Match Baseline:** Fast Polars string matching. Produces candidates in ~5 seconds.
2. **Lexical Retrieval (TF-IDF):** Hit a 157GB RAM explosion on `.toarray()`. Solved by extracting CSR sparse matrix pointers directly — zero dense allocation.
3. **Dense Retrieval (MiniLM):** Implemented `all-MiniLM-L6-v2` in float16. Hit a 186GB VRAM crash on `torch.matmul(50k, 2M)`. Solved by chunking queries into batches of 1,000 on the L4 GPU (24GB VRAM).

### Feature Engineering
Basic lowercasing via Polars. Four pairwise distance features via `rapidfuzz`:
- `feat_name_jw` (Jaro-Winkler name similarity)
- `feat_name_lev` (Levenshtein name ratio)
- `feat_addr_jw` (Jaro-Winkler address similarity)
- `feat_addr_lev` (Levenshtein address ratio)

### Reranking (XGBoost)
Trained a GBDT binary classifier on MiniLM dense candidates.
- **Hurdle:** Extreme class imbalance — 94k matches vs 44M negatives. XGBoost predicted 0 for everything.
- **Fix:** Dynamic `scale_pos_weight = 464.23` to penalize false negatives.

### Inference
Top-1 selection with static threshold >= 0.5. Outputs `submission.csv`.

### Flow 1 Definition (Locked)
| Component | Choice |
|---|---|
| Normalization | Basic lowercase + strip |
| Blocker | Dense only (MiniLM, global, country-agnostic) |
| Features | JW + Levenshtein (name + address) |
| Reranker | XGBoost (scale_pos_weight) |
| Threshold | Static >= 0.5, Top-1 |
| Weights | `models/Flow1_xgb_reranker.json` |

**Status:** Running inference on test set. Awaiting leaderboard baseline score.

---

## 5. The Diagnostic Execution Roadmap (Post-Flow-1 Baseline)

Once we have a Flow 1 leaderboard number, we execute the following phases. Each has a **build step**, a **measurement step**, and a **gate**. We do not advance until the gate passes, and if it doesn't, the diagnostic tells us *which* assumption broke.

### Phase 0 — Ground-Truth Diagnostics (Before Touching a Model)

**Build:** Pure analysis, no new code.

**Measure:**
- Match-count and singleton-rate distribution, broken down by **country** and by **source-pair** (S1–S2 vs S1–S3). We already know the aggregate (5.6% singleton, median 3) — now split it to know if India/US behave differently, since France will follow whichever pattern is closer.
- Missing-field rates by country/source — quantify how much of the ML has to work with degraded input.
- **One-record-one-business assumption check:** Does any true S2/S3 id appear under two different S1 rows in ground truth? This determines whether Phase 4's conflict resolution is safe to apply.

**Gate:** We can state, in one sentence per country, what the "expected difficulty" is. If we can't, we don't understand the data well enough to blame later failures correctly.

### Phase 1 — Normalization + Per-Channel Blocking, Each Channel Measured Independently

**Build:** Normalization (including transliteration — this is **not optional**, it's what makes cross-script matches visible) + three blocking channels kept **separate**, not unioned yet.

**Measure, per channel, per country:**
- **Recall:** Fraction of ground-truth matches the channel alone catches.
- **Candidate volume:** Pairs produced.
- **Precision-at-blocking:** Fraction of candidates that are true matches — tells you how much work the model stage has to do.

**Then union** and re-measure combined recall/volume.

**Gate:** Union recall ceiling ≥ 0.97 overall, and no country sits more than ~3–5 points below the others. If France-proxy (India-held-out) recall lags badly, that's a blocking problem to fix *before* any model tuning — a model can't recover a match that blocking never produced.

### Phase 1.5 — Meta-Blocking (Weighted Candidate Pruning)

**Why this exists:** The ER literature (Papadakis et al.) treats meta-blocking as its own established stage, separate from blocking itself. Our current Phase 1 unions blocking channels flatly — every candidate that survives any channel gets treated equally. A pair that both exact-match and dense retrieval surfaced is a much stronger candidate than one only the dense channel found. Right now that signal is discarded at the union step. Given F₀.₅'s precision weighting, recovering this signal for free (no extra recall cost, since we're only pruning within the union, not adding new candidates) is one of the highest-leverage, lowest-risk additions available.

**Build:** Restructure the union of blocking-channel outputs into a weighted comparison graph. Each candidate pair gets a score based on *how many independent channels agreed on it, and how strongly* (e.g., channel-agreement count, per-channel rank position). Prune the weak-agreement tail before it ever reaches the GBDT.

**Measure:** Precision/recall tradeoff of pruning at different weight cutoffs — same measure-then-gate discipline as everything else. Specifically: at what pruning threshold do we lose <0.5% recall while cutting candidate volume by X%?

**Gate:** Meta-blocking achieves a measurable precision lift at blocking stage with recall loss < 0.5%. If it doesn't, skip it — the flat union is fine.

### Phase 2 — Candidate Model Exploration (A–D), Each Judged on Segments

**Build:** Candidate C (feature-stack GBDT) first — cheapest, closest to Flow 1. Recalibrate without aggressive `scale_pos_weight`. Run Candidate A (frozen-embedding rerank) and, budget-permitting, B (cross-encoder shortlist) and D (graph consistency) as **separate experiments**, not sequential replacements.

**Candidate B upgrade (Ditto-style):** Li et al. (VLDB 2020) validated cross-encoder reranking on a real 789K×412K company-matching task (96.5% F1) — same domain as ours. Three concrete techniques to adopt:
- **Domain knowledge injection (span highlighting):** Wrap important sub-fields (postal code, city tokens) in special markers so the transformer knows which spans matter, rather than feeding raw concatenated text.
- **Hard-negative data augmentation:** Attribute shuffling, span deletion — forces the model to learn the actual decision boundary rather than spurious correlates like "same country + similar length → match."
- **Compute budget gate:** Ditto's benchmark was ~an order of magnitude smaller than our post-blocking volume. Size it explicitly: pairs-per-second throughput on L4 GPU × ~3.4M shortlisted pairs. If it doesn't fit in the time budget, Candidate B stays as Top-2-only or gets cut.

**Measure, per model:**
- **Precision/recall curve** (not a single point) at the pair level.
- **Stratified F₀.₅** by country, by source-pair, and by true match-count bucket (singleton vs multi-match). A model great on singletons but bad on multi-match has a completely different fix than the reverse.
- **Calibration reliability curve:** Does P=0.8 actually mean ~80% are true matches? This is what makes the threshold search in Phase 3 trustworthy instead of guesswork.

**Gate:** For each candidate model, we can **name its specific failure segment** (e.g., "Candidate C underperforms on India multi-match entities because address tokens rarely include postal codes there"). If we can only report an aggregate number, we don't have a diagnosis, we have a score.

### Phase 3 — Threshold + Decision Rule, Grounded in Calibration

**Build:** Replace Top-1 with per-entity multi-select. Grid-search threshold centered at t* ≈ 0.8 (the F₀.₅-derived starting point), searched against the real macro formula.

**Measure:** Threshold stability across the country-held-out proxy split. If optimal threshold for US is 0.75 and for India is 0.85, that's telling you something real about differential calibration, not noise to average away.

**Gate:** Chosen threshold (or per-segment thresholds, if justified) produces stable macro-F₀.₅ across at least two different validation splits, not just one lucky holdout.

### Phase 4 — Conflict Resolution / Global Consistency (Only If Phase 0 Justified It)

**Build:** Greedy argmax-with-margin assignment for candidates claimed by multiple S1 entities (Candidate D).

**Measure:** Ablation — F₀.₅ with vs. without this layer, on the segment where conflicts actually occur. If Phase 0 showed the one-record-one-business assumption is nearly always true in ground truth, this should be a clean, measurable win.

**Gate:** Documented, positive, segment-attributed lift — not "it seems more principled."

### Phase 5 — Stacking Ensemble Across A–D

**Build:** Meta-learner (or Candidate D's graph layer) over each base model's OOF probabilities.

**Measure:** **Leave-one-model-out ablation.** For each of A/B/C/D, what does removing it cost the ensemble? This is the step most teams skip, and it's exactly what separates "we built 4 models" from "we understand what each model contributes." A model that adds nothing to the stack should be cut.

**Gate:** Every model in the final stack has a measured, non-zero marginal contribution.

### Phase 6 — Validation-vs-Leaderboard Drift Audit

**Build:** Nothing new — this is discipline, not code.

**Measure:** Once we have a public leaderboard number, compare it against our country-held-out proxy estimate. If they diverge, break the gap down by segment (is it France specifically? a particular source-pair?) rather than re-tuning the whole pipeline blind.

**Gate:** We can explain any leaderboard-vs-local gap before making further changes based on the leaderboard score alone.

### Phase 7 — Production Packaging

Final `matching_results.tsv` / `candidate_pairs.tsv` generation, `validate_submission.py` clean run, code/README/requirements matching the submission structure, methodology doc filled in with the actual diagnostics from Phases 0–7 (not just the final architecture) — the diagnostic trail *is* the differentiator the review of top teams' packages will be looking for.

---

## 6. Risk Flags

These are load-bearing assumptions and constraints that the plan doesn't fully resolve yet. Each must be addressed or explicitly accepted before the phase that depends on it.

| Risk | Depends On | Mitigation |
|---|---|---|
| **No published benchmark matches this exact setup** (3-source, cross-script, one country held out entirely at test time). The country-held-out proxy (India-as-France-stand-in) is our only calibration signal; there's no external number to sanity-check it against. | Phase 6 | Accept this — Phase 6's drift audit is carrying more weight than usual. Be extra conservative about trusting a single proxy split. |
| **One-record-one-business assumption** underlying Phase 4 is still unverified. | Phase 0 | Must be checked in Phase 0 before Phase 4 is built. If violated, Phase 4's greedy assignment is unsafe. |
| **Compute budget for Candidate B (cross-encoder).** Ditto's own benchmark is ~10x smaller than our post-blocking volume. "Top-2 shortlist only" is the right mitigation, but it needs a real throughput number, not a design intent. | Phase 2 | Size it explicitly in Phase 2: measure pairs/second on L4, multiply by shortlist volume, compare to available wall-clock time. |
| **Class imbalance and spurious correlates.** With imbalance this extreme, any model (GBDT or cross-encoder) can learn "same country + similar length → match" instead of real name/address correspondence. | Phase 2 | Ditto-style hard-negative augmentation for Candidate B. For Candidate C, calibration reliability curves will surface this if it's happening. |

---

## 7. Literature References

- **Papadakis et al.** — Blocking/filtering/meta-blocking survey. Standard reference for the two-stage ER architecture and meta-blocking as a distinct, established stage.
- **Li et al. (Ditto), VLDB 2020** — Cross-encoder entity matching as sequence-pair classification. Domain knowledge injection (span highlighting), hard-negative augmentation. 96.5% F1 on 789K×412K company matching.
- **AWS Entity Resolution** — Industry pattern: rule-based + ML split, confirming the GBDT-with-string-features approach as production-grade.
