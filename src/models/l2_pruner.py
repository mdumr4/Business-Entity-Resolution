"""L2 Candidate Pruner Model.

A lightweight, recall-biased XGBoost classifier trained on 8 fast lexical
and dense similarity features. Filters 50-100 L1 candidates down to ~14.9
candidates per query at a generous probability threshold (0.05).
"""

import os
from typing import List, Optional, Set
import numpy as np
import xgboost as xgb
from src.core.similarity import token_jaccard

L2_FEATURE_NAMES = [
    "name_cos",
    "addr_cos",
    "found_n",
    "found_a",
    "name_jaccard",
    "addr_jaccard",
    "addr_num_match",
    "name_exact",
]


class L2Pruner:
    """Lightweight XGBoost pruner for fast candidate space reduction."""

    def __init__(
        self,
        threshold: float = 0.05,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        device: str = "cpu",
    ):
        self.threshold = threshold
        self.feature_names = L2_FEATURE_NAMES
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
    def extract_pair_features(
        q_name_norm: str,
        q_addr_norm: str,
        c_name_norm: str,
        c_addr_norm: str,
        q_nums: Set[str],
        c_nums: Set[str],
        q_name_stripped: str,
        c_name_stripped: str,
        name_cos: float,
        addr_cos: float,
        found_in_name_lane: bool,
        found_in_addr_lane: bool,
    ) -> List[float]:
        """Compute the 8 lightweight features for a single query-candidate pair."""
        name_jac = token_jaccard(q_name_norm, c_name_norm)
        addr_jac = token_jaccard(q_addr_norm, c_addr_norm)

        if q_nums and c_nums:
            addr_num_match = 1.0 if q_nums == c_nums else 0.0
        else:
            addr_num_match = 0.5  # Neutral if either lacks numbers

        name_exact = (
            1.0 if (q_name_stripped and q_name_stripped == c_name_stripped) else 0.0
        )

        return [
            float(name_cos),
            float(addr_cos),
            1.0 if found_in_name_lane else 0.0,
            1.0 if found_in_addr_lane else 0.0,
            float(name_jac),
            float(addr_jac),
            float(addr_num_match),
            float(name_exact),
        ]

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[list] = None,
        verbose: bool = True,
    ) -> "L2Pruner":
        """Fit the L2 XGBoost model."""
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities."""
        if not self.is_fitted:
            raise RuntimeError("L2Pruner is not fitted or loaded yet.")
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def filter_candidates(
        self,
        X: np.ndarray,
        candidate_ids: List[str],
        threshold: Optional[float] = None,
    ) -> List[str]:
        """Filter candidates using the probability threshold."""
        if not candidate_ids:
            return []
        thr = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return [cid for cid, p in zip(candidate_ids, probs) if p >= thr]

    def save_model(self, model_path: str):
        """Save model to JSON format."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        self.model.save_model(model_path)

    def load_model(self, model_path: str):
        """Load model from JSON format."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model.load_model(model_path)
        self.is_fitted = True
