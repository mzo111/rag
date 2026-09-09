import math

import pytest

from eval.metrics import dcg_at_k, mrr, ndcg_at_k, recall_at_k

# Hand-computed example used throughout:
#   grades: a=2, b=1, c=0 (judged irrelevant), x unjudged
#   ranked: [b, x, a]
#   DCG@10 = (2^1-1)/log2(2) + 0/log2(3) + (2^2-1)/log2(4) = 1 + 0 + 1.5 = 2.5
#   IDCG   = 3/log2(2) + 1/log2(3) = 3 + 0.63093 = 3.63093
#   nDCG   = 2.5 / 3.63093 = 0.68853
GRADES = {"a": 2, "b": 1, "c": 0}
RANKED = ["b", "x", "a"]


def test_recall_at_k_hand_computed():
    assert recall_at_k(RANKED, GRADES, 1) == 0.5
    assert recall_at_k(RANKED, GRADES, 2) == 0.5
    assert recall_at_k(RANKED, GRADES, 3) == 1.0
    assert recall_at_k(RANKED, GRADES, 100) == 1.0  # k beyond list length


def test_recall_edge_cases():
    assert recall_at_k([], GRADES, 5) == 0.0
    assert recall_at_k(RANKED, {}, 5) == 0.0
    assert recall_at_k(RANKED, {"c": 0}, 5) == 0.0  # only irrelevant judgments
    assert recall_at_k(RANKED, GRADES, 0) == 0.0
    assert recall_at_k(["c", "c", "a"], GRADES, 2) == 0.5  # duplicate c collapses, a is rank 2


def test_mrr_hand_computed():
    assert mrr(RANKED, GRADES) == 1.0
    assert mrr(["x", "c", "a"], GRADES) == pytest.approx(1 / 3)
    assert mrr(["x", "x", "a"], GRADES) == 0.5  # duplicate x occupies one rank
    assert mrr(["x", "c"], GRADES) == 0.0
    assert mrr([], GRADES) == 0.0


def test_dcg_at_k():
    assert dcg_at_k([3.0, 0.0, 1.0], 10) == pytest.approx(3 + 1 / math.log2(4))
    assert dcg_at_k([3.0, 0.0, 1.0], 2) == 3.0
    assert dcg_at_k([], 10) == 0.0


def test_ndcg_hand_computed():
    assert ndcg_at_k(RANKED, GRADES, 10) == pytest.approx(2.5 / (3 + 1 / math.log2(3)), abs=1e-6)
    assert ndcg_at_k(RANKED, GRADES, 10) == pytest.approx(0.68853, abs=1e-5)
    assert ndcg_at_k(["a", "b"], GRADES, 10) == 1.0  # ideal ordering
    assert ndcg_at_k(["b", "a"], GRADES, 10) == pytest.approx(
        (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    )
    assert ndcg_at_k(["a"], GRADES, 1) == 1.0  # ideal is truncated to k too
    assert ndcg_at_k(["b"], GRADES, 1) == pytest.approx(1 / 3)


def test_ndcg_edge_cases():
    assert ndcg_at_k([], GRADES, 10) == 0.0
    assert ndcg_at_k(RANKED, {}, 10) == 0.0
    assert ndcg_at_k(RANKED, {"a": 0, "b": 0}, 10) == 0.0
    assert ndcg_at_k(RANKED, GRADES, 0) == 0.0
    assert ndcg_at_k(["a", "a", "b"], GRADES, 10) == 1.0  # duplicate does not push b down
