"""Ranking metrics as pure functions.

All functions take ``ranked``, the retrieved item ids in rank order (best first), and
``grades``, a mapping item id -> relevance grade (0, 1 or 2). Items missing from ``grades``
are unjudged and count as grade 0. Duplicates in ``ranked`` count once, at their first
position. When there are no relevant items every metric is 0.0 (callers should normally
exclude such queries from averages).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _dedupe(ranked: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Fraction of relevant items (grade > 0) that appear in the top k."""
    relevant = {item for item, g in grades.items() if g > 0}
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item in _dedupe(ranked)[:k] if item in relevant)
    return hits / len(relevant)


def mrr(ranked: Sequence[str], grades: Mapping[str, int]) -> float:
    """Reciprocal rank of the first relevant item (grade > 0), 0 if none is retrieved."""
    for i, item in enumerate(_dedupe(ranked), 1):
        if grades.get(item, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], 1))


def ndcg_at_k(ranked: Sequence[str], grades: Mapping[str, int], k: int = 10) -> float:
    """Normalised DCG with gain 2^grade - 1 and log2(rank + 1) discount.

    The ideal ranking is every judged item sorted by grade (so a query with more relevant
    items than k has an IDCG computed over the top k of them).
    """
    if k <= 0:
        return 0.0
    gains = [float(2 ** grades.get(item, 0) - 1) for item in _dedupe(ranked)]
    ideal = sorted((float(2**g - 1) for g in grades.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg
