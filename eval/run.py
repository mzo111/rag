"""Evaluation harness: run any retriever over eval/queries.yaml and print a metrics table.

A retriever is any object with ``search(query: str, k: int) -> list[str]`` returning chunk
ids, best first. Retrieved chunks are mapped to *judgment units* (a page URL, or a
URL#anchor when the judgments name a section) via the store, de-duplicated in rank order,
and scored with recall@k, MRR and nDCG@10. Queries with no judgments are skipped (and
counted) unless ``--include-unjudged`` is given.

Alias pages (see :mod:`corpus.quality`) are collapsed to their canonical URL on *both*
sides, so a judgment written against ``torch.optim.Adam`` still matches a retriever that
returns ``torch.optim.adam.Adam_class``, and the same content cannot be counted twice.

Usage:
    python -m eval.run --retriever bm25|dense|hybrid|rerank|random [--k 10]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from eval.metrics import mrr, ndcg_at_k, recall_at_k

QUERIES_PATH = Path(__file__).with_name("queries.yaml")
CATEGORIES = ("api_lookup", "conceptual", "multi_hop", "tutorial")
Locator = Callable[[Iterable[str]], dict[str, tuple[str, str]]]


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]: ...


@dataclass(frozen=True)
class Judgment:
    url: str
    grade: int
    anchor: str = ""

    @property
    def key(self) -> str:
        return f"{self.url}#{self.anchor}" if self.anchor else self.url


@dataclass
class Query:
    id: str
    query: str
    category: str
    judgments: list[Judgment] = field(default_factory=list)

    @property
    def grades(self) -> dict[str, int]:
        return {j.key: j.grade for j in self.judgments}


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    queries: list[Query] = []
    for q in data["queries"]:
        judgments = [
            Judgment(url=j["url"], grade=int(j["grade"]), anchor=j.get("anchor") or "")
            for j in (q.get("judgments") or [])
        ]
        queries.append(Query(q["id"], q["query"], q["category"], judgments))
    return queries


def canonical_key(key: str, canonical: Mapping[str, str] | None) -> str:
    """Rewrite a judgment unit (``url`` or ``url#anchor``) to its canonical page."""
    if not canonical:
        return key
    url, sep, anchor = key.partition("#")
    return canonical.get(url, url) + sep + anchor


def to_units(
    chunk_ids: Sequence[str],
    located: dict[str, tuple[str, str]],
    judged_keys: set[str],
    canonical: Mapping[str, str] | None = None,
) -> list[str]:
    """Map retrieved chunk ids to judgment units in rank order, first occurrence wins.

    A chunk maps to ``url#anchor`` when that exact section was judged, else to its page URL.
    Alias URLs are rewritten to their canonical page. Unknown chunk ids are dropped.
    """
    units: list[str] = []
    for cid in chunk_ids:
        if cid not in located:
            continue
        url, anchor = located[cid]
        if canonical:
            url = canonical.get(url, url)
        key = f"{url}#{anchor}"
        unit = key if anchor and key in judged_keys else url
        if unit not in units:
            units.append(unit)
    return units


@dataclass
class QueryResult:
    id: str
    category: str
    judged: bool
    recall_5: float
    recall_10: float
    mrr: float
    ndcg_10: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


METRIC_NAMES = ("recall_5", "recall_10", "mrr", "ndcg_10")


def evaluate(
    retriever: Retriever,
    queries: Sequence[Query],
    locate: Locator,
    *,
    k: int = 10,
    include_unjudged: bool = False,
    canonical: Mapping[str, str] | None = None,
) -> list[QueryResult]:
    results: list[QueryResult] = []
    for q in queries:
        grades = {canonical_key(key, canonical): g for key, g in q.grades.items()}
        if not grades and not include_unjudged:
            continue
        ids = list(retriever.search(q.query, max(k, 10)))
        located = locate(ids)
        ranked = to_units(ids, located, set(grades), canonical)
        results.append(
            QueryResult(
                id=q.id,
                category=q.category,
                judged=bool(grades),
                recall_5=recall_at_k(ranked, grades, 5),
                recall_10=recall_at_k(ranked, grades, 10),
                mrr=mrr(ranked, grades),
                ndcg_10=ndcg_at_k(ranked, grades, 10),
            )
        )
    return results


def summarize(results: Sequence[QueryResult]) -> dict[str, dict[str, float | int]]:
    """Mean of each metric overall and per category, plus query counts."""
    groups: dict[str, list[QueryResult]] = {"all": list(results)}
    for r in results:
        groups.setdefault(r.category, []).append(r)
    out: dict[str, dict[str, float | int]] = {}
    for name, rs in groups.items():
        row: dict[str, float | int] = {"n": len(rs)}
        for m in METRIC_NAMES:
            row[m] = statistics.fmean(getattr(r, m) for r in rs) if rs else 0.0
        out[name] = row
    return out


def format_table(
    results: Sequence[QueryResult], n_total: int, n_skipped: int, verbose: bool = False
) -> str:
    header = f"{'group':<14}{'n':>4}  {'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG@10':>9}"
    lines = [header, "-" * len(header)]
    summary = summarize(results)
    order = ["all"] + [c for c in CATEGORIES if c in summary]
    order += [c for c in summary if c not in order]
    for name in order:
        s = summary[name]
        lines.append(
            f"{name:<14}{s['n']:>4}  {s['recall_5']:>7.3f}{s['recall_10']:>7.3f}"
            f"{s['mrr']:>7.3f}{s['ndcg_10']:>9.3f}"
        )
    if verbose and results:
        lines += ["", f"{'query':<14}{'cat':<12}{'R@5':>7}{'R@10':>7}{'MRR':>7}{'nDCG@10':>9}"]
        for r in results:
            flag = "" if r.judged else "  (unjudged)"
            lines.append(
                f"{r.id:<14}{r.category:<12}{r.recall_5:>7.3f}{r.recall_10:>7.3f}"
                f"{r.mrr:>7.3f}{r.ndcg_10:>9.3f}{flag}"
            )
    lines.append("")
    lines.append(
        f"{len(results)} of {n_total} queries scored"
        + (f", {n_skipped} skipped (no judgments yet)" if n_skipped else "")
    )
    return "\n".join(lines)


class RandomBaseline:
    """Returns k random chunk ids. Exists only so the harness runs end to end."""

    def __init__(self, chunk_ids: Sequence[str], seed: int = 0):
        self.chunk_ids = list(chunk_ids)
        self.rng = random.Random(seed)

    def search(self, query: str, k: int) -> list[str]:
        return self.rng.sample(self.chunk_ids, min(k, len(self.chunk_ids)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid", "rerank", "random"],
        default="bm25",
    )
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dense-model", default=None, help="dense: model name (default: gte-modernbert-base)"
    )
    ap.add_argument("--dense-index", type=Path, default=None, help="dense: index directory")
    ap.add_argument(
        "--rerank-base",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="rerank: first-stage retriever whose top N is reordered",
    )
    ap.add_argument("--include-unjudged", action="store_true")
    ap.add_argument(
        "--no-canonical",
        action="store_true",
        help="do not collapse alias pages to their canonical URL",
    )
    ap.add_argument("--json", action="store_true", help="print per-query results as JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-query rows")
    args = ap.parse_args(argv)

    from corpus.store import Store

    queries = load_queries(args.queries)
    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        if args.retriever == "bm25":
            from retrieval.bm25 import Bm25Retriever

            retriever: Retriever = Bm25Retriever(store)
        elif args.retriever == "dense":
            from retrieval.dense import DEFAULT_INDEX_DIR, DEFAULT_MODEL, DenseRetriever

            retriever = DenseRetriever.load(
                store,
                model=args.dense_model or DEFAULT_MODEL,
                directory=args.dense_index or DEFAULT_INDEX_DIR,
            )
        elif args.retriever == "hybrid":
            from retrieval.hybrid import HybridRetriever

            retriever = HybridRetriever(store)
        elif args.retriever == "rerank":
            from retrieval.rerank import RerankRetriever, _make_base

            retriever = RerankRetriever(store, _make_base(args.rerank_base, store))
        else:
            retriever = RandomBaseline(store.chunk_ids(), seed=args.seed)
        canonical = None if args.no_canonical else store.canonical_map()
        results = evaluate(
            retriever,
            queries,
            store.locate,
            k=args.k,
            include_unjudged=args.include_unjudged,
            canonical=canonical,
        )
    n_skipped = sum(1 for q in queries if not q.judgments) if not args.include_unjudged else 0
    if args.json:
        print(
            json.dumps(
                {"summary": summarize(results), "queries": [r.as_dict() for r in results]}, indent=1
            )
        )
    else:
        print(f"retriever: {args.retriever}   corpus: {args.db}   k={args.k}\n")
        print(format_table(results, len(queries), n_skipped, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
