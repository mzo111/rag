"""Hybrid retrieval: reciprocal rank fusion over BM25 and dense.

BM25 and dense fail on different queries - BM25 wins when the query names an identifier,
dense when it describes a task. Fusion is the standard way to take both without training a
reranker, and reciprocal rank fusion is the standard fusion: it uses only the *ranks* each
system assigns, so it needs no score normalisation between a BM25 score (unbounded, corpus
dependent) and a cosine (bounded, [-1, 1]). Scores from those two scales cannot be added
meaningfully; ranks can.

    RRF(d) = sum over systems s of  1 / (RRF_K + rank_s(d))

**The constant is RRF_K = 60, taken from the paper that introduced the method:**

    Cormack, Clarke and Buettcher (2009), "Reciprocal Rank Fusion Outperforms Condorcet and
    Individual Rank Learning Methods", SIGIR '09, pp. 758-759.

60 is the value used there, and the paper reports the method to be insensitive to it over a
wide range. It is adopted here **as published and is not tuned against this eval set** - the
judgment set is small (40 queries), optimistically biased, and was pooled from the two systems
being fused, so any constant fitted on it would be fitting pooling artifacts and judgment
noise. The same applies to ``depth``: 100 is a conventional fusion depth fixed a priori, not
searched.

Fusion happens at the level of the **canonical page**, not the chunk. Both base retrievers
already collapse alias pages, so each returns at most one chunk per page, but they may pick
*different* chunks of the same page - and the eval scores pages, not chunks. Fusing on chunk
ids would let one page occupy two of the k slots and would split the evidence for it across
two entries instead of adding it. The best-ranked chunk of a page is returned as its
representative.

Usage:
    python -m retrieval.hybrid "how do I clip gradients"
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from corpus.store import Store

# Cormack, Clarke & Buettcher (SIGIR 2009). Published value, deliberately not tuned here.
RRF_K = 60
# How deep to read each system's ranking before fusing. Conventional value, fixed a priori.
DEFAULT_DEPTH = 100


@dataclass
class Fused:
    """One page's fusion state: its score and the best chunk seen for it."""

    score: float = 0.0
    chunk_id: str = ""
    best_rank: int = 1 << 30
    ranks: dict[str, int] = field(default_factory=dict)


def rrf_scores(rankings: Mapping[str, Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """Reciprocal rank fusion over named rankings of item ids, best first.

    Returns item id -> summed 1/(k + rank). Ranks are 1-based, as in the paper.
    """
    out: dict[str, float] = {}
    for ids in rankings.values():
        for rank, item in enumerate(ids, 1):
            out[item] = out.get(item, 0.0) + 1.0 / (k + rank)
    return out


class HybridRetriever:
    """RRF over BM25 and dense. Implements the ``search(query, k)`` protocol.

    ``systems`` maps a name to any object with ``search(query, k)``, so the fusion is
    testable without loading a model and a third system could be added without touching the
    fusion itself.
    """

    def __init__(
        self,
        store: Store,
        systems: Mapping[str, object] | None = None,
        *,
        rrf_k: int = RRF_K,
        depth: int = DEFAULT_DEPTH,
    ):
        self.store = store
        self.rrf_k = rrf_k
        self.depth = depth
        if systems is None:
            from retrieval.bm25 import Bm25Retriever
            from retrieval.dense import DenseRetriever

            systems = {"bm25": Bm25Retriever(store), "dense": DenseRetriever.load(store)}
        self.systems = dict(systems)

    def _pages(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        """chunk_id -> canonical page url, so fusion can group chunks by the page they show."""
        ids = list(dict.fromkeys(chunk_ids))
        out: dict[str, str] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            for row in self.store.conn.execute(
                "SELECT c.chunk_id, c.url, p.canonical_url FROM chunks c "
                f"JOIN pages p ON p.url = c.url WHERE c.chunk_id IN ({placeholders})",
                batch,
            ):
                out[row["chunk_id"]] = row["canonical_url"] or row["url"]
        return out

    def _fuse(self, query: str, k: int) -> list[tuple[str, Fused]]:
        """Pages ordered by fused score. Ties break on best single-system rank, then url."""
        if not query.strip() or k <= 0:
            return []
        depth = max(self.depth, k)
        rankings = {name: list(s.search(query, depth)) for name, s in self.systems.items()}
        located = self._pages([cid for ids in rankings.values() for cid in ids])

        fused: dict[str, Fused] = {}
        for name, ids in rankings.items():
            seen_here: set[str] = set()
            for rank, cid in enumerate(ids, 1):
                page = located.get(cid)
                if page is None or page in seen_here:
                    # A system ranking one page twice must not pay it twice.
                    continue
                seen_here.add(page)
                f = fused.setdefault(page, Fused())
                f.score += 1.0 / (self.rrf_k + rank)
                f.ranks[name] = rank
                if rank < f.best_rank:
                    f.best_rank, f.chunk_id = rank, cid

        order = sorted(fused.items(), key=lambda kv: (-kv[1].score, kv[1].best_rank, kv[0]))
        return order[:k]

    def search(self, query: str, k: int) -> list[str]:
        return [f.chunk_id for _, f in self._fuse(query, k)]

    def explain(self, query: str, k: int) -> list[tuple[str, float, dict[str, int]]]:
        """(page, fused score, per-system rank) for the top k - for inspection, not scoring."""
        return [(page, f.score, dict(f.ranks)) for page, f in self._fuse(query, k)]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ad-hoc RRF search over data/corpus.db")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    args = ap.parse_args(argv)

    with Store(args.db) as store:
        r = HybridRetriever(store, depth=args.depth)
        q = " ".join(args.query)
        print(f"query: {q}\nRRF k={r.rrf_k}  depth={r.depth}\n")
        for i, (page, score, ranks) in enumerate(r.explain(q, args.k), 1):
            where = "  ".join(f"{n}@{v}" for n, v in sorted(ranks.items()))
            print(f"{i:>3}. {page.replace('https://docs.pytorch.org/', '')}")
            print(f"     rrf={score:.5f}  {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
