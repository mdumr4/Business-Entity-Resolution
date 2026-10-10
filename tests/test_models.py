"""Unit tests for L2Pruner and L3Ranker model wrappers."""

import unittest
import numpy as np
import polars as pl
from src.models.l2_pruner import L2Pruner, L2_FEATURE_NAMES
from src.models.l3_ranker import L3Ranker, ALL_RANKER_FEATURES


class TestModels(unittest.TestCase):
    def test_l2_pruner_fit_predict(self):
        pruner = L2Pruner(threshold=0.5, n_estimators=5, max_depth=2)
        X = np.random.rand(20, len(L2_FEATURE_NAMES)).astype(np.float32)
        y = np.random.choice([0, 1], size=20)
        pruner.fit(X, y, verbose=False)
        probs = pruner.predict_proba(X)
        self.assertEqual(len(probs), 20)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_l3_ranker_relative_features(self):
        df = pl.DataFrame({
            "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-3"],
            "addr_cos": [0.8, 0.9, 0.5],
            "name_cos": [0.7, 0.6, 0.4],
        })
        augmented = L3Ranker.add_relative_features(df)
        self.assertIn("addr_cos_gap_from_top", augmented.columns)
        self.assertIn("addr_cos_rank", augmented.columns)

        # For S1-1: max addr_cos is 0.9. First row gap should be 0.8 - 0.9 = -0.1
        gaps = augmented["addr_cos_gap_from_top"].to_list()
        self.assertAlmostEqual(gaps[0], -0.1, places=5)
        self.assertAlmostEqual(gaps[1], 0.0, places=5)

    def test_one_owner_constraint(self):
        ranker = L3Ranker(threshold=0.5)
        qids = ["S1-A", "S1-B"]
        cids = ["C1", "C1"]  # Both claim candidate C1
        probs = np.array([0.7, 0.9])  # S1-B gives higher probability (0.9 > 0.7)
        universe = ["S1-A", "S1-B", "S1-C"]

        matches = ranker.predict_matches(
            query_ids=qids,
            candidate_ids=cids,
            probs=probs,
            all_query_universe=universe,
            enforce_one_owner=True,
        )

        # S1-B should win C1, S1-A should not have C1, S1-C is a singleton
        self.assertEqual(matches["S1-B"], ["C1"])
        self.assertEqual(matches["S1-A"], [])
        self.assertEqual(matches["S1-C"], [])


if __name__ == "__main__":
    unittest.main()
