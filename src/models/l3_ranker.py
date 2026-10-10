"""L3 Precision Ranker Model.

A precision-biased XGBoost ranker that extracts 13 features:
  - Dense cosine similarities (name & address)
  - Lexical token Jaccard & RapidFuzz Levenshtein
  - Exact legal-stripped match & numeric Jaccard
  - Dictionary-based TF-IDF cosine similarities
  - Intra-query relative features (rank, gaps from top candidate)

Includes global one-record-one-business conflict resolution where each
candidate entity is assigned strictly to the query entity scoring it highest.
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl
import xgboost as xgb

BASE_RANKER_FEATURES = [
    "name_cos",
    "addr_cos",
    "name_jac",
    "addr_jac",
    "num_jac",
    "name_exact",
    "name_tfidf",
    "addr_tfidf",
    "name_lev",
    "addr_lev",
]

RELATIVE_RANKER_FEATURES = [
    "addr_cos_gap_from_top",
    "name_cos_gap_from_top",
    "addr_cos_rank",
]

ALL_RANKER_FEATURES = BASE_RANKER_FEATURES + RELATIVE_RANKER_FEATURES


class L3Ranker:
    """Precision-biased XGBoost ranker for final match classification."""

    def __init__(
        self,
        threshold: float = 0.63,
        max_depth: int = 5,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.feature_names = ALL_RANKER_FEATURES
        self.model = xgb.XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective="binary:logistic",
            tree_method="hist",
            device=device,
            eval_metric="logloss",
            random_state=42,
        )
        self.is_fitted = False

    @staticmethod
    def add_relative_features(df: pl.DataFrame) -> pl.DataFrame:
        """Compute intra-query relative ranking features over source1_entity_id.

        Simulates listwise ranking (LambdaMART) while preserving calibrated
        probabilities for macro-F0.5 thresholding.
        """
        return df.with_columns(
            [
                (
                    pl.col("addr_cos")
                    - pl.col("addr_cos").max().over("source1_entity_id")
                ).alias("addr_cos_gap_from_top"),
                (
                    pl.col("name_cos")
                    - pl.col("name_cos").max().over("source1_entity_id")
                ).alias("name_cos_gap_from_top"),
                pl.col("addr_cos")
                .rank(method="ordinal", descending=True)
                .over("source1_entity_id")
                .cast(pl.Float64)
                .alias("addr_cos_rank"),
            ]
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = True,
    ) -> "L3Ranker":
        """Fit the L3 ranker model."""
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if not self.is_fitted:
            raise RuntimeError("L3Ranker is not fitted or loaded yet.")
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def predict_matches(
        self,
        query_ids: List[str],
        candidate_ids: List[str],
        probs: np.ndarray,
        all_query_universe: List[str],
        threshold: Optional[float] = None,
        enforce_one_owner: bool = True,
    ) -> Dict[str, List[str]]:
        """Assign candidates to query entities above threshold.

        If enforce_one_owner=True:
            Enforces one-record-one-business: each candidate ID is retained
            only for the Source 1 entity that gave it the highest probability.
        """
        thr = self.threshold if threshold is None else threshold
        matches_by_query: Dict[str, List[str]] = defaultdict(list)

        if enforce_one_owner and len(candidate_ids) > 0:
            # Find max probability for each candidate
            cand_max_prob: Dict[str, float] = {}
            cand_best_query: Dict[str, str] = {}
            for qid, cid, p in zip(query_ids, candidate_ids, probs):
                if p >= thr:
                    if cid not in cand_max_prob or p > cand_max_prob[cid]:
                        cand_max_prob[cid] = float(p)
                        cand_best_query[cid] = qid

            # Add to winner
            for cid, qid in cand_best_query.items():
                matches_by_query[qid].append(cid)
        else:
            for qid, cid, p in zip(query_ids, candidate_ids, probs):
                if p >= thr:
                    matches_by_query[qid].append(cid)

        # Ensure all entities in universe exist (even singletons)
        final_dict: Dict[str, List[str]] = {}
        for qid in all_query_universe:
            final_dict[qid] = matches_by_query.get(qid, [])
        return final_dict

    def save_model(
        self,
        model_path: str,
        threshold_path: Optional[str] = None,
    ):
        """Save ranker weights and threshold configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.model.save_model(model_path)
        if threshold_path:
            os.makedirs(os.path.dirname(os.path.abspath(threshold_path)), exist_ok=True)
            with open(threshold_path, "w", encoding="utf-8") as f:
                json.dump({"optimal_threshold": float(self.threshold)}, f, indent=2)

    def load_model(
        self,
        model_path: str,
        threshold_path: Optional[str] = None,
    ):
        """Load ranker weights and optimal threshold."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_model(model_path)
        self.is_fitted = True

        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                self.threshold = float(cfg.get("optimal_threshold", self.threshold))
