# Phase 2: Flow 1 XGBoost Model Analysis

## Feature Importance (Gain)

Gain measures the relative contribution of each feature to the model. Higher gain = more important for decision making.

* **feat_addr_jw**: 153629.5938
* **feat_name_jw**: 38208.4688
* **feat_addr_lev**: 19397.2031
* **feat_name_lev**: 6201.5386

## Feature Frequency (Weight)

Weight is the number of times a feature is used to split the data across all trees.

* **feat_addr_lev**: 1772.0
* **feat_addr_jw**: 1513.0
* **feat_name_lev**: 1467.0
* **feat_name_jw**: 1368.0


# Phase 2: Flow 1 XGBoost Internal Analysis

While we wait for your test score, I extracted the internal decision-making metrics directly from your trained `xgb_reranker.json` model to see *how* it makes its choices.

**It revealed a massive flaw in our baseline.**

### Feature Importance (Gain)

*Gain measures how much a feature contributes to the model's confidence in a match. Higher gain = more important.*

1. **Address Jaro-Winkler (`feat_addr_jw`)** : 153,629
2. **Name Jaro-Winkler (`feat_name_jw`)** : 38,208
3. **Address Levenshtein (`feat_addr_lev`)** : 19,397
4. **Name Levenshtein (`feat_name_lev`)** : 6,201

### The Diagnosis

The XGBoost model is  **drastically over-indexing on Address similarity** . It considers the address to be *4x more important* than the business name!

**Why this is dangerous:** In business entity resolution, hundreds of completely distinct businesses often share the exact same address (e.g., shops in a shopping mall, or companies sharing a co-working space). Because our current model is obsessed with addresses, if a "Starbucks" and a "Foot Locker" are in the exact same shopping mall, this model is highly likely to falsely predict them as a match!

**How to fix this in our next model iteration:** We need to engineer better Name-specific features (like TF-IDF token overlap, or longest common subsequence) to force the tree to split on the Name first, or we need to add a "Shopping Mall Penalty" feature if the address is highly generic.
