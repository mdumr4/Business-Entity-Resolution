"""Unit tests for similarity metrics and Macro F0.5 calculation."""

import unittest
import numpy as np
from src.core.similarity import (
    token_jaccard,
    numeric_jaccard,
    levenshtein_ratio,
    build_idf_dict,
    tfidf_cosine,
    l2_sq_to_cosine,
    calculate_f05,
    evaluate_macro_f05,
)


class TestSimilarityCore(unittest.TestCase):
    def test_token_jaccard(self):
        self.assertEqual(token_jaccard("apple inc", "apple corp"), 1.0 / 3.0)
        self.assertEqual(token_jaccard("same name", "same name"), 1.0)
        self.assertEqual(token_jaccard("", "apple"), 0.0)

    def test_numeric_jaccard(self):
        # Neutral indicator when neither has digits
        self.assertEqual(numeric_jaccard(set(), set()), 0.5)
        # Mismatch when one has digits and other does not
        self.assertEqual(numeric_jaccard({"10"}, set()), 0.0)
        # Match
        self.assertEqual(numeric_jaccard({"42", "100"}, {"42", "200"}), 1.0 / 3.0)

    def test_levenshtein_ratio(self):
        self.assertAlmostEqual(levenshtein_ratio("microsoft", "microsoft"), 1.0)
        self.assertGreater(levenshtein_ratio("walmart", "walmrt"), 0.85)

    def test_tfidf_cosine(self):
        corpus = ["walmart supercenter", "target store", "walmart neighborhood market"]
        idf_dict, default_idf = build_idf_dict(corpus)
        sim = tfidf_cosine("walmart supercenter", "walmart store", idf_dict, default_idf)
        self.assertGreater(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_l2_sq_to_cosine(self):
        l2_sq = np.array([0.0, 2.0, 4.0])
        cos = l2_sq_to_cosine(l2_sq)
        np.testing.assert_allclose(cos, [1.0, 0.0, -1.0])

    def test_calculate_f05_worked_example(self):
        # README worked example: Precision = 2/3, Recall = 1.0 -> F0.5 = 0.714
        f05 = calculate_f05(2.0 / 3.0, 1.0)
        self.assertAlmostEqual(f05, 0.7142857, places=5)

    def test_evaluate_macro_f05_singletons(self):
        # Entity 1: Match predicted, perfect (1.0)
        # Entity 2: Singleton, predicted empty (1.0)
        # Entity 3: Singleton, false merge predicted (0.0)
        preds = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": {"S2-99"},
        }
        gt = {
            "S1-1": {"S2-1"},
            "S1-2": set(),
            "S1-3": set(),
        }
        qids = ["S1-1", "S1-2", "S1-3"]
        result = evaluate_macro_f05(preds, gt, qids)
        # Scores: S1-1 = 1.0, S1-2 = 1.0, S1-3 = 0.0 -> Mean = 2/3 = 0.6667
        self.assertAlmostEqual(result["macro_f05"], 2.0 / 3.0, places=4)
        self.assertEqual(result["singleton_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
