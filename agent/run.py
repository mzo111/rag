"""Run the retrieval agent over eval/queries.yaml and report how it behaved.

This reports *operations*, not quality: which path the graph took per query, how often a
decomposition failed to parse, and where the wall clock went. It deliberately prints no
retrieval metrics. Scoring the agent against the current judgments would compare it to a
pool built by other retrievers, and :mod:`agent.coverage` exists to measure how badly that
pool under-covers the agent before any such number is quoted.

Latency is split into **model time** and **retrieval time**, which are measured differently
and must be read differently:

* Model time is the duration of the Ollama call. On a cache hit it is the duration recorded
  when that call was originally made live, *not* the microseconds the replay took - a
  replay's own timing would understate the system by three orders of magnitude and flatter
  it for no reason.
* Retrieval time is always measured now, live, because the retrievers run for real on every
  invocation.

So a cached run reports the model latency of the original live run and the retrieval latency
of this one. ``--fresh-timings`` refuses cache hits so both halves come from the same run.

Usage:
    python -m agent.run [--retriever bm25|dense|hybrid] [--k 10] [--offline] [-v]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from agent.graph import AgentRetriever
from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient, OllamaError, ResponseCache
from eval.run import CATEGORIES, QUERIES_PATH, load_queries

BASE_RETRIEVERS = ("bm25", "dense", "hybrid")
DEFAULT_BASE = "hybrid"


def make_base_retriever(name: str, store):
    """Any of the project's retrievers, by name. The agent is pluggable over all three."""
    if name == "bm25":
        from retrieval.bm25 import Bm25Retriever

        return Bm25Retriever(store)
    if name == "dense":
        from retrieval.dense import DenseRetriever

        return DenseRetriever.load(store)
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever(store)
    raise SystemExit(f"unknown retriever: {name!r} (choose from {', '.join(BASE_RETRIEVERS)})")


def make_agent(
    store,
    base: str = DEFAULT_BASE,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    cache_path: Path | None = None,
    offline: bool = False,
    check: bool = True,
) -> AgentRetriever:
    """An :class:`AgentRetriever` over the named base retriever.

    ``check`` probes Ollama up front so a missing server or unpulled model fails with a
    fixable message before 40 queries' worth of work, rather than midway through. It is
    skipped when ``offline``, which is the whole point of the committed cache.
    """
    cache = ResponseCache(cache_path) if cache_path is not None else ResponseCache()
    client = OllamaClient(model=model, base_url=base_url, cache=cache, offline=offline)
    if check and not offline:
        client.check_available()
    return AgentRetriever(store, make_base_retriever(base, store), client)


def summarize_traces(traces: Sequence[dict]) -> dict:
    """Route mix, fallback rates and latency split over a run's traces."""
    n = len(traces)
    if not n:
        return {"n": 0}
    decomposed = [t for t in traces if t["route"] == "decompose"]
    dec_fallbacks = [t for t in traces if t["decompose_fallback"]]

    def stat(field: str) -> dict[str, float]:
        xs = [t[field] for t in traces]
        return {
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "max": max(xs),
            "total": sum(xs),
        }

    model_total = sum(t["llm_seconds"] for t in traces)
    retrieval_total = sum(t["retrieval_seconds"] for t in traces)
    measured = model_total + retrieval_total
    return {
        "n": n,
        "route_decompose": len(decomposed),
        "route_direct": n - len(decomposed),
        "route_fallbacks": sum(1 for t in traces if t["route_fallback"]),
        # Denominator is decompose-path queries: a direct query never attempts a
        # decomposition, so counting it as a success would dilute the rate.
        "decompose_fallbacks": len(dec_fallbacks),
        "decompose_attempts": len(decomposed),
        "decompose_fallback_rate": (len(dec_fallbacks) / len(decomposed)) if decomposed else 0.0,
        "sub_queries_total": sum(t["n_sub_queries"] for t in traces),
        "sub_queries_mean": statistics.fmean(t["n_sub_queries"] for t in traces),
        "llm_calls": sum(t["llm_calls"] for t in traces),
        "cache_hits": sum(t["cache_hits"] for t in traces),
        "prompt_tokens": sum(t["prompt_tokens"] for t in traces),
        "completion_tokens": sum(t["completion_tokens"] for t in traces),
        "llm_seconds": stat("llm_seconds"),
        "retrieval_seconds": stat("retrieval_seconds"),
        "total_seconds": stat("total_seconds"),
        "model_share": (model_total / measured) if measured else 0.0,
    }


def format_report(traces: Sequence[dict], summary: dict, *, verbose: bool = False) -> str:
    if not traces:
        return "no queries run"
    lines: list[str] = []
    if verbose:
        header = (
            f"{'query':<14}{'route':<11}{'sub':>4}{'fb':>4}"
            f"{'model s':>9}{'retr s':>9}{'total s':>9}"
        )
        lines += [header, "-" * len(header)]
        for t in traces:
            fb = "!" if t["decompose_fallback"] or t["route_fallback"] else ""
            lines.append(
                f"{t.get('id', t['query'][:13]):<14}{t['route']:<11}{t['n_sub_queries']:>4}"
                f"{fb:>4}{t['llm_seconds']:>9.3f}{t['retrieval_seconds']:>9.3f}"
                f"{t['total_seconds']:>9.3f}"
            )
        lines.append("")

    s = summary
    lines += [
        f"queries              {s['n']}",
        f"route                {s['route_decompose']} decompose, {s['route_direct']} direct"
        + (f"  ({s['route_fallbacks']} unparseable -> direct)" if s["route_fallbacks"] else ""),
        f"sub-queries          {s['sub_queries_total']} total, "
        f"{s['sub_queries_mean']:.2f} per query",
        f"decomposition        {s['decompose_fallbacks']}/{s['decompose_attempts']} fell back "
        f"to the original query ({s['decompose_fallback_rate']:.1%} of decompose-path queries, "
        f"{s['decompose_fallbacks'] / s['n']:.1%} of all queries)",
        f"model calls          {s['llm_calls']} ({s['cache_hits']} from cache), "
        f"{s['prompt_tokens']} prompt + {s['completion_tokens']} completion tokens",
        "",
    ]
    head = f"{'latency (s)':<14}{'mean':>9}{'median':>9}{'max':>9}{'total':>9}"
    lines += [head, "-" * len(head)]
    for label, key in (
        ("model", "llm_seconds"),
        ("retrieval", "retrieval_seconds"),
        ("end to end", "total_seconds"),
    ):
        st = s[key]
        lines.append(
            f"{label:<14}{st['mean']:>9.3f}{st['median']:>9.3f}{st['max']:>9.3f}{st['total']:>9.2f}"
        )
    lines += ["", f"model time is {s['model_share']:.1%} of measured work"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=list(BASE_RETRIEVERS), default=DEFAULT_BASE)
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cache", type=Path, default=None, help="response cache (default: committed)")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="serve every call from the cache; a miss is an error, not a live call",
    )
    ap.add_argument(
        "--fresh-timings",
        action="store_true",
        help="ignore the cache so model and retrieval latency come from the same run",
    )
    ap.add_argument("--category", choices=list(CATEGORIES), default=None)
    ap.add_argument("--traces", type=Path, default=None, help="write per-query traces as JSON")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-query rows")
    args = ap.parse_args(argv)

    if args.offline and args.fresh_timings:
        print("--offline and --fresh-timings contradict: one replays, the other refuses to.")
        return 2

    from corpus.store import Store

    queries = load_queries(args.queries)
    if args.category:
        queries = [q for q in queries if q.category == args.category]
    if not queries:
        print("no queries to run")
        return 1

    # --fresh-timings gets a scratch cache rather than the committed one: it must neither
    # read hits (that is the point) nor write this run's timings into the artifact.
    cache_path = Path(tempfile.mkdtemp()) / "fresh.json" if args.fresh_timings else args.cache

    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        try:
            agent = make_agent(
                store,
                args.retriever,
                model=args.model,
                base_url=args.base_url,
                cache_path=cache_path,
                offline=args.offline,
            )
            for q in queries:
                agent.search(q.query, args.k)
                agent.traces[-1]["id"] = q.id
                agent.traces[-1]["category"] = q.category
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # Live calls extend the committed cache so the next run replays them for free.
        if agent.client.calls and not args.fresh_timings:
            agent.client.cache.save()

    summary = summarize_traces(agent.traces)
    if args.traces:
        args.traces.write_text(json.dumps(agent.traces, indent=1), encoding="utf-8")
    if args.json:
        print(json.dumps({"summary": summary, "traces": agent.traces}, indent=1))
    else:
        mode = (
            "offline replay"
            if args.offline
            else ("live, no cache" if args.fresh_timings else "live + cache")
        )
        print(f"agent over {args.retriever}   model: {args.model}   k={args.k}   [{mode}]\n")
        print(format_report(agent.traces, summary, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    sys.exit(main())
