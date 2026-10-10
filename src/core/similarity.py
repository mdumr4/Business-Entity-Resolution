"""Similarity functions and metrics for Entity Resolution.

Provides string distance, token overlap, numeric Jaccard, memory-efficient
IDF dictionary computation, and competition leaderboard metric evaluation.
"""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple
import numpy as np
from rapidfuzz import fuzz


def token_jaccard(s1: str, s2: str) -> float:
    """Compute token Jaccard similarity between two whitespace-split strings."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    union_len = len(set1 | set2)
    return len(set1 & set2) / union_len if union_len > 0 else 0.0


def numeric_jaccard(nums1: Set[str], nums2: Set[str]) -> float:
    """Compute Jaccard similarity on sets of street/entity numbers.

    Returns:
        0.5 if neither string contains digits (neutral),
        0.0 if one contains digits and the other does not,
        intersection / union if both contain digits.
    """
    if not nums1 and not nums2:
        return 0.5  # Neutral indicator
    if not nums1 or not nums2:
        return 0.0
    union_len = len(nums1 | nums2)
    return len(nums1 & nums2) / union_len if union_len > 0 else 0.0


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0] using RapidFuzz."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return fuzz.ratio(s1, s2) / 100.0


def build_idf_dict(corpus: Iterable[str]) -> Tuple[Dict[str, float], float]:
    """Build an in-memory token IDF dictionary.

    Bypasses Scipy sparse matrix overhead (which previously caused a 157GB RAM OOM).
    Returns (idf_dict, default_idf_for_oov).
    """
    df_dict: Dict[str, int] = defaultdict(int)
    n_docs = 0
    for text in corpus:
        tokens = set(text.split())
        for token in tokens:
            df_dict[token] += 1
        n_docs += 1

    idf_dict: Dict[str, float] = {}
    default_idf = math.log((n_docs + 1) / 1.0) + 1.0
    for token, df in df_dict.items():
        idf_dict[token] = math.log((n_docs + 1) / (df + 1)) + 1.0
    return idf_dict, default_idf


def tfidf_cosine(
    s1: str, s2: str, idf_dict: Dict[str, float], default_idf: float
) -> float:
    """Compute TF-IDF cosine similarity using an IDF lookup dictionary."""
    tokens1 = s1.split()
    tokens2 = s2.split()
    if not tokens1 or not tokens2:
        return 0.0

    vec1: Dict[str, float] = defaultdict(float)
    for t in tokens1:
        vec1[t] += idf_dict.get(t, default_idf)

    vec2: Dict[str, float] = defaultdict(float)
    for t in tokens2:
        vec2[t] += idf_dict.get(t, default_idf)

    common_tokens = set(vec1.keys()) & set(vec2.keys())
    if not common_tokens:
        return 0.0

    dot = sum(vec1[t] * vec2[t] for t in common_tokens)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def l2_sq_to_cosine(scores_l2_sq: np.ndarray) -> np.ndarray:
    """Convert squared L2 distance to cosine similarity for unit-normalized vectors.

    cos(theta) = 1 - (L2^2 / 2)
    """
    return 1.0 - (scores_l2_sq / 2.0)


def calculate_f05(precision: float, recall: float) -> float:
    """Calculate F0.5 score: (1.25 * P * R) / (0.25 * P + R)."""
    denominator = 0.25 * precision + recall
    if denominator <= 0.0:
        return 0.0
    return (1.25 * precision * recall) / denominator


def evaluate_macro_f05(
    preds_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    all_qids: List[str],
) -> Dict[str, float]:
    """Macro-averaged F0.5 evaluation per Source 1 entity across the full universe.

    Correctly scores singletons (true_count == 0):
      - 1.0 if predicted empty (correct negative)
      - 0.0 if any candidate is predicted (false merge)
    """
    f05_scores = []
    precisions = []
    recalls = []
    total_singletons = 0
    correct_singletons = 0

    for qid in all_qids:
        true_matches = gt_dict.get(qid, set())
        pred_matches = preds_dict.get(qid, set())

        true_count = len(true_matches)
        pred_count = len(pred_matches)

        if true_count == 0:
            total_singletons += 1
            if pred_count == 0:
                correct_singletons += 1
                f05_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
        elif pred_count == 0:
            f05_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            tp = len(true_matches & pred_matches)
            prec = tp / pred_count
            rec = tp / true_count
            precisions.append(prec)
            recalls.append(rec)
            f05_scores.append(calculate_f05(prec, rec))

    return {
        "macro_f05": float(np.mean(f05_scores)) if f05_scores else 0.0,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "singleton_accuracy": (
            correct_singletons / total_singletons if total_singletons > 0 else 0.0
        ),
    }
