# Amazon ML Challenge 2026 - Master Architecture & Execution Plan (v2.0)

## 1. The Core Objective & New Directives
We are solving a massive-scale Entity Resolution problem (1.7M Queries vs 10.3M Candidates) to maximize the **F-0.5 Macro Score**. 

### 🛑 The Methodology Rule (Comprehension Over Completion)
1. **Never rush to the next phase.** The goal is deep, empirical understanding of the data and models.
2. **Build → Measure → Explain:** At every phase, ensure the code is structurally perfect, measure its performance, explain the metric empirically, and clear the diagnostic gate before advancing.
3. **No "Throwing Code Over the Wall":** Respect hardware limits and memory constraints.

### 🏆 The Hackathon Scaling Directive
Amazon explicitly grades the candidate generation code:
> *"Blocking has to scale... Your blocking step must cut the search space to a small candidate set per Source 1 entity... The approach that generates a smaller candidate set per Source 1 entity will be ranked higher."*

We cannot brute-force this. A pipeline that generates 50 Million candidates to get 99% recall will be penalized. We need an elegant, precision-first funnel.

---

## 2. What We Built, Measured, and Learned (The Empirical Discoveries)

### Discovery A: The "Shopping Mall" Trap (Phase 0 XGBoost Flaw)
* **What we measured:** We generated 5,163 negative pairs (completely different businesses that happen to share an identical address).
* **The Result:** The Phase 0 XGBoost model incorrectly predicted them as a MATCH **98.61% of the time**.
* **Why it failed:** We only fed the model Jaro-Winkler and Levenshtein string distances. String distance only measures character overlap. For example, `williams college` vs `wildlife coalition` scored a `0.82` Name Similarity because they share characters (`wil`, `l`, `e`). Combined with a `1.0` Address Similarity, the model blindly assumed it was a match.
* **The Fix:** We must build Phase 2 features based on **Token Sets** (Jaccard word overlap), not just character distances.

### Discovery B: The Empirical Reality of True Matches
We hypothesized that "Same Address + Different Name = False Match". We tested this against `train_ground_truth.tsv` (7.6 Million true pairs) and proved our hypothesis wrong:
* **Fact 1:** **9.6% of True Matches** have the exact same name but a *different* address string (due to noisy formatting/parsing).
* **Fact 2:** **6.7% of True Matches** have the exact same address but a *different* name string (e.g., `Lee and Lawson LLC` vs `Lee and Lawson`).
* **Conclusion:** The model *must* learn to forgive name differences if they are just legal suffixes/abbreviations, but heavily penalize name differences if they are completely different semantic words.

---

## 3. The Great Blocking Pivot (Solving the Scaling Rule)

We initially tried to union Exact Match, TF-IDF, and Dense Retrieval. We hit massive scaling walls:

1. **Exact Match (Failed due to O(N²)):** 
   Generated 23.5 Million candidates for only 29% recall. Matching purely on "Address" caused every store in a shopping mall to link to every other store. It is completely unscalable. **Status: Deleted.**
2. **Lexical TF-IDF (Failed due to O(N×M)):** 
   Vocabulary of 750k words against 10M rows caused an 8GB RAM explosion and projected 200+ hours of matrix multiplication. **Status: Deleted.**
3. **SPLADE / Neural Tokenizers (Rejected):**
   Neural models (like BERT/SPLADE) destroy numbers (e.g., building code `A-19283` becomes `["A", "##19", "##283"]`). This causes horrific false positives for highly specific alphanumeric fingerprints.

### The Winning Blocking Architecture:
To satisfy Amazon's "Small Candidate Set" rule, we rely on exactly two orthogonal, highly-scalable channels:

* **Channel 1: Dense Retrieval (MiniLM) — The Semantic Net**
  * *Why:* Catches typos, spelling errors, and semantic equivalents.
  * *Scaling Fix:* Instead of "Top 20" (34M candidates), we clamp it to **Top 5**, trusting the encoder to put the true match near the top.
* **Channel 2: Token-Join Inverted Index — The Fingerprint Net**
  * *Why:* Polars relational hash-join on exact tokens. We filter out common words (frequency > 50) and only join S1 and S2 if they share an ultra-rare token (e.g., `A-19283`).
  * *Scaling Fix:* It naturally scales because it strictly ignores common terms (like "Store" or "Street"). It generated only 12M candidates at blistering C++ speeds.

**The Meta-Blocker:** Unions Dense (Top 5) and Token-Join, enforcing a strict cap of **15 candidates maximum per query**. This guarantees our `candidate_pairs.tsv` is incredibly small, pure, and infinitely scalable.

---

## 4. The Execution Roadmap (Where we go from here)

### Phase 2: Feature Engineering (Fixing the XGBoost Blindspots)
We must build `src/features/feature_engineering.py` to calculate features that actually understand semantic overlap:
1. **Name/Address Jaccard Similarity:** Break strings into words. What percentage of words are shared? (Catches `Subway` vs `Starbucks` = 0.0 overlap).
2. **Longest Common Subsequence (LCS):** Is Name B entirely contained within Name A? (Catches `Lee and Lawson LLC` vs `Lee and Lawson` = 1.0 contained).
3. **Length Ratios & Token Counts.**
4. **Meta-Blocking Booleans:** `found_in_dense` and `found_in_token_join`.

### Phase 3: Candidate Model Training (XGBoost v2)
* Train the model on the new feature stack.
* Evaluate Feature Importances to guarantee `feat_addr_jw` is no longer single-handedly deciding matches.

### Phase 4: Dynamic Thresholding & Conflict Resolution (The F-0.5 Tuner)
* We cannot use `prob > 0.5`. We must run a Grid Search over thresholds (0.1 to 0.9) against the validation set, explicitly calculating the F-0.5 macro score to find the mathematical peak.
* If a single Candidate ID is assigned to multiple S1 queries (which shouldn't happen for 1-to-1 matching), we implement Greedy Argmax Resolution to resolve conflicts.
