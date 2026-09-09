"""Paired evaluation: compare retrievers under a *shared* draw of the judgments.

The marginal intervals in the README are wide - nDCG@10 for BM25 spans [0.58, 0.72] - because
they carry the full weight of judgment error. But that error is *common to every system*: if a
page is re-graded from 1 to 0, it stops counting for BM25 and for dense and for hybrid, all at
once. In a difference between two systems it largely cancels, so a paired comparison can
resolve gaps far smaller than either marginal interval.

Method, per draw:
  1. Sample the re-grade transition rows from a Dirichlet posterior (counts + 1), as in the
     earlier bootstrap, so the n=20 behind each row is priced in.
  2. Resample every judgment ONCE under those rows, and score every system against that one
     resampled judgment set. This is what makes the comparison paired.
  3. Resample the 40 queries with replacement and take the mean, so the interval covers query
     sampling as well as judgment error.

Reported side by side: the paired interval on the difference and the marginal interval on each
system alone. The width gap between them is the point of the design.

Scoring reuses the frozen harness - :mod:`eval.metrics` and :func:`eval.run.to_units` - rather
than reimplementing any metric. Retrieval does not depend on the grades and no judgment carries
an anchor, so each system is run once and only the scoring is repeated.

recall@5 is the primary metric, raw and as a fraction of its achievable ceiling (with ~9.8
relevant pages per query, five slots cannot hold them all). MRR is reported as a secondary line
only: at this judgment density it is saturated and is not expected to discriminate.

Usage:
    python -m eval.paired [--draws 4000] [--k 10] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from eval.metrics import mrr, recall_at_k
from eval.run import CATEGORIES, RandomBaseline, canonical_key, load_queries, to_units

EVAL_DIR = Path(__file__).parent
PROBE_1 = (EVAL_DIR / "recheck.yaml", EVAL_DIR / "recheck_key.yaml")
PROBE_2 = (EVAL_DIR / "recheck_2v1.yaml", EVAL_DIR / "recheck_2v1_key.yaml")
DEFAULT_DRAWS = 4000
SEED = 20260908
RECALL_K = 5
SYSTEMS = ("random", "bm25", "dense", "hybrid", "rerank_bm25", "rerank_hybrid")
# Fusion comparisons, then the two reranking comparisons: a reranker is measured against the
# exact base whose candidates it reorders, which is the only comparison it is entitled to.
PAIRS = (
    ("hybrid", "bm25"),
    ("hybrid", "dense"),
    ("bm25", "dense"),
    ("rerank_hybrid", "hybrid"),
    ("rerank_bm25", "bm25"),
)


def probe_row(probe: Path, key: Path, original: int) -> np.ndarray:
    """Counts of re-grades 0/1/2 for candidates that originally held ``original``."""
    entries = yaml.safe_load(probe.read_text(encoding="utf-8"))["entries"]
    key_map = {x["id"]: x for x in yaml.safe_load(key.read_text(encoding="utf-8"))["key"]}
    row = np.zeros(3)
    for e in entries:
        if key_map[e["id"]]["original"] == original and e["grade"] is not None:
            row[int(e["grade"])] += 1
    return row


def transition_counts() -> np.ndarray:
    """Rows of the re-grade matrix: grade 0 from probe 1, grades 1 and 2 from probe 2.

    Probe 1 is the only measurement of grade 0; probe 2 is the only measurement of grade 2 and
    the later measurement of grade 1. The two probes differ in contrast and in whether the
    written rubric existed, so this mixes protocols - see the README limitations.
    """
    return np.vstack(
        [
            probe_row(*PROBE_1, 0),
            probe_row(*PROBE_2, 1),
            probe_row(*PROBE_2, 2),
        ]
    )


@dataclass
class QueryRun:
    """One system's retrieval for one query, plus that query's judgments."""

    query_id: str
    category: str
    ranked: list[str]  # judgment units in rank order
    keys: list[str]  # judged unit keys
    grades: np.ndarray  # original grade per key


def build_cache(store, queries, systems, k: int) -> dict[str, list[QueryRun]]:
    """Run each system over every query once. Rankings do not depend on the grades."""
    canonical = store.canonical_map()
    cache: dict[str, list[QueryRun]] = {}
    for name, retriever in systems.items():
        runs = []
        for q in queries:
            grades = {canonical_key(key, canonical): g for key, g in q.grades.items()}
            ids = list(retriever.search(q.query, k))
            ranked = to_units(ids, store.locate(ids), set(grades), canonical)
            keys = list(grades)
            runs.append(
                QueryRun(
                    q.id,
                    q.category,
                    ranked,
                    keys,
                    np.array([grades[key] for key in keys], dtype=int),
                )
            )
        cache[name] = runs
    return cache


def per_query_scores(
    runs: Sequence[QueryRun], drawn: Sequence[np.ndarray]
) -> dict[str, np.ndarray]:
    """recall@5 raw, recall@5 as a fraction of its ceiling, and MRR, per query.

    The ceiling is min(k, R)/R for R relevant pages in this draw: with more than k relevant
    pages, recall@k cannot reach 1 however good the ranking is. Queries with no relevant page
    in a draw have no defined ceiling and are marked NaN so they can be dropped from that mean
    rather than counted as zero.
    """
    n = len(runs)
    raw = np.empty(n)
    ceil = np.empty(n)
    rr = np.empty(n)
    for i, (run, g) in enumerate(zip(runs, drawn, strict=True)):
        gd = dict(zip(run.keys, g.tolist(), strict=True))
        rel = int((g > 0).sum())
        raw[i] = recall_at_k(run.ranked, gd, RECALL_K)
        rr[i] = mrr(run.ranked, gd)
        ceil[i] = raw[i] / (min(RECALL_K, rel) / rel) if rel else np.nan
    return {"recall5": raw, "recall5_ceil": ceil, "mrr": rr}


def _nanmean(v: np.ndarray) -> float:
    return float(np.nanmean(v)) if not np.all(np.isnan(v)) else 0.0


def bootstrap(
    cache: dict[str, list[QueryRun]],
    counts: np.ndarray,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = SEED,
) -> dict:
    """Paired and marginal bootstrap distributions, overall and per category."""
    rng = np.random.default_rng(seed)
    names = [n for n in SYSTEMS if n in cache]
    runs0 = cache[names[0]]
    n_q = len(runs0)
    groups = {"all": np.arange(n_q)}
    for cat in CATEGORIES:
        idx = np.array([i for i, r in enumerate(runs0) if r.category == cat])
        if len(idx):
            groups[cat] = idx

    metrics = ("recall5", "recall5_ceil", "mrr")
    pairs = [(a, b) for a, b in PAIRS if a in cache and b in cache]
    marg = {(s, g, m): np.empty(draws) for s in names for g in groups for m in metrics}
    diff = {(a, b, g, m): np.empty(draws) for a, b in pairs for g in groups for m in metrics}

    for d in range(draws):
        Md = np.vstack([rng.dirichlet(counts[i] + 1.0) for i in range(3)])
        # one shared resampling of the judgments, used by every system
        drawn = []
        for run in runs0:
            cum = Md[run.grades].cumsum(1)
            u = rng.random((len(run.grades), 1))
            drawn.append((u > cum).sum(1))
        scores = {s: per_query_scores(cache[s], drawn) for s in names}
        # one shared resampling of the queries, so paired differences stay paired
        for g, idx in groups.items():
            pick = idx[rng.integers(0, len(idx), len(idx))]
            for m in metrics:
                for s in names:
                    marg[(s, g, m)][d] = _nanmean(scores[s][m][pick])
                for a, b in pairs:
                    diff[(a, b, g, m)][d] = _nanmean(scores[a][m][pick] - scores[b][m][pick])
    return {"names": names, "groups": list(groups), "pairs": pairs, "marginal": marg, "diff": diff}


def win_loss_tie(cache, metric: str = "recall5") -> dict:
    """Per-query wins on the judgments as they actually stand, ties counted explicitly."""
    out = {}
    for a, b in PAIRS:
        if a not in cache or b not in cache:
            continue
        sa = per_query_scores(cache[a], [r.grades for r in cache[a]])[metric]
        sb = per_query_scores(cache[b], [r.grades for r in cache[b]])[metric]
        d = sa - sb
        out[(a, b)] = {
            "wins": int((d > 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
            "ties": int((np.abs(d) <= 1e-12).sum()),
            "n": len(d),
        }
    return out


def ci(v: np.ndarray) -> tuple[float, float, float]:
    lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def format_report(res: dict, wlt: dict, cache: dict) -> str:
    L: list[str] = []
    names, groups, pairs = res["names"], res["groups"], res["pairs"]
    marg, diff = res["marginal"], res["diff"]

    L.append("PRIMARY: recall@5\n")
    for label, metric in (("raw", "recall5"), ("of ceiling", "recall5_ceil")):
        L.append(f"-- marginal (unpaired) intervals, recall@5 {label} --")
        L.append(f"{'group':<12}" + "".join(f"{s:>26}" for s in names))
        for g in groups:
            row = f"{g:<12}"
            for s in names:
                lo, med, hi = ci(marg[(s, g, metric)])
                row += f"{med:>10.3f} [{lo:.3f},{hi:.3f}]"
            L.append(row)
        L.append("")
        L.append(f"-- PAIRED intervals on the difference, recall@5 {label} --")
        L.append(
            f"{'group':<12}{'comparison':<18}{'median':>9}{'2.5%':>9}"
            f"{'97.5%':>9}{'width':>8}  verdict"
        )
        for g in groups:
            for a, b in pairs:
                lo, med, hi = ci(diff[(a, b, g, metric)])
                verdict = (
                    "favours " + (a if lo > 0 else b) if (lo > 0 or hi < 0) else "no difference"
                )
                L.append(
                    f"{g:<12}{a + ' - ' + b:<18}{med:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
                    f"{hi - lo:>8.4f}  {verdict}"
                )
        L.append("")

    L.append("-- interval width: marginal vs paired (recall@5 raw, all queries) --")
    L.append(
        f"{'comparison':<18}{'marginal A':>12}{'marginal B':>12}"
        f"{'naive sum':>11}{'paired':>9}{'shrink':>9}"
    )
    for a, b in pairs:
        wa = ci(marg[(a, "all", "recall5")])[2] - ci(marg[(a, "all", "recall5")])[0]
        wb = ci(marg[(b, "all", "recall5")])[2] - ci(marg[(b, "all", "recall5")])[0]
        lo, _, hi = ci(diff[(a, b, "all", "recall5")])
        wp = hi - lo
        L.append(
            f"{a + ' - ' + b:<18}{wa:>12.4f}{wb:>12.4f}{wa + wb:>11.4f}"
            f"{wp:>9.4f}{(wa + wb) / wp:>8.1f}x"
        )
    L.append("")

    L.append("-- per-query wins on recall@5, judgments as they stand --")
    L.append(f"{'comparison':<18}{'wins':>6}{'losses':>8}{'ties':>6}{'n':>4}   note")
    for (a, b), w in wlt.items():
        L.append(
            f"{a + ' > ' + b:<18}{w['wins']:>6}{w['losses']:>8}{w['ties']:>6}{w['n']:>4}"
            f"   ties = identical recall@5, counted, not split"
        )
    L.append("")

    L.append("SECONDARY: MRR (saturated at this judgment density - not expected to discriminate)")
    L.append(f"{'group':<12}{'comparison':<18}{'median':>9}{'2.5%':>9}{'97.5%':>9}")
    for a, b in pairs:
        lo, med, hi = ci(diff[(a, b, "all", "mrr")])
        L.append(f"{'all':<12}{a + ' - ' + b:<18}{med:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--no-rerank", action="store_true", help="skip the cross-encoder systems (much faster)"
    )
    args = ap.parse_args(argv)

    from corpus.store import Store
    from retrieval.bm25 import Bm25Retriever
    from retrieval.dense import DenseRetriever
    from retrieval.hybrid import HybridRetriever
    from retrieval.rerank import CrossEncoderScorer, RerankRetriever

    queries = load_queries()
    counts = transition_counts()
    print("re-grade transition matrix (row = original, col = re-grade):")
    M = counts / counts.sum(1, keepdims=True)
    for i in range(3):
        print(f"  {i} -> {M[i, 0]:.2f} {M[i, 1]:.2f} {M[i, 2]:.2f}   (n={int(counts[i].sum())})")
    print()

    with Store(args.db) as store:
        bm25 = Bm25Retriever(store)
        dense = DenseRetriever.load(store)
        hybrid = HybridRetriever(store, {"bm25": bm25, "dense": dense})
        systems = {
            "random": RandomBaseline(store.chunk_ids(), seed=0),
            "bm25": bm25,
            "dense": dense,
            "hybrid": hybrid,
        }
        if not args.no_rerank:
            # One scorer shared by both reranking systems: the model is the same, and loading
            # it twice would double the memory for no reason.
            scorer = CrossEncoderScorer()
            systems["rerank_bm25"] = RerankRetriever(store, bm25, scorer)
            systems["rerank_hybrid"] = RerankRetriever(store, hybrid, scorer)
            print(f"reranker: {scorer.name} on {scorer.device}\n")
        cache = build_cache(store, queries, systems, args.k)

    res = bootstrap(cache, counts, draws=args.draws, seed=args.seed)
    wlt = win_loss_tie(cache)
    print(f"{len(queries)} queries, {args.draws} draws, shared judgment + query resampling\n")
    print(format_report(res, wlt, cache))

    if args.json:
        payload = {
            "draws": args.draws,
            "marginal": {f"{s}|{g}|{m}": ci(v) for (s, g, m), v in res["marginal"].items()},
            "diff": {f"{a}-{b}|{g}|{m}": ci(v) for (a, b, g, m), v in res["diff"].items()},
            "win_loss_tie": {f"{a}-{b}": w for (a, b), w in wlt.items()},
        }
        args.json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
