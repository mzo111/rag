"""How much of what the agent retrieves has never been judged.

The judgments in ``eval/queries.yaml`` were pooled from BM25 and dense. The agent retrieves
against a *different* distribution - it fuses up to three sub-query result lists, so pages
that no single-query retriever ranked in its top 10 can arrive near the top. Every such page
is unjudged, and :mod:`eval.run` scores anything unjudged as grade 0. An agent that surfaced
a perfect page the pool never saw would be punished for it.

So this module measures the hole before anyone quotes a score through it:

* **unjudged rate** - of the (query, page) pairs the agent returns, the share carrying no
  judgment at any grade. Grade 0 counts as judged: the judge saw that page and ruled it out.
* **per category** - the same rate split by query category, because the exposure is not
  uniform; multi_hop is where decomposition changes the ranking most.
* **distinct new pages** - how many different documents a re-pool would put in front of a
  human. The pair count is the number of grading decisions; the distinct-page count is the
  number of pages actually read, and one page judged for three queries is three decisions
  but one read.

An unjudged rate near zero means the existing pool already covers the agent and an ablation
against it is interpretable. A high rate means the ablation is measuring pool coverage
rather than retrieval quality, and the delta pool this module can write is the fix.

Usage:
    python -m agent.coverage [--retriever hybrid] [--k 10] [--offline] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from agent.graph import AgentRetriever
from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaError
from agent.run import BASE_RETRIEVERS, DEFAULT_BASE, make_base_retriever, make_client
from eval.pool import build_pool, delta_pool, judged_urls, write_pool
from eval.run import CATEGORIES, QUERIES_PATH, Query, load_queries


def coverage(delta: Mapping, queries: Sequence[Query]) -> dict:
    """Roll a delta pool up into unjudged rates, overall, per category and per query.

    ``delta`` is the output of :func:`eval.pool.delta_pool`, which has already canonicalized
    both the candidates and the judgments, so alias pages are not double counted here.
    """
    category = {q.id: q.category for q in queries}
    per_query: list[dict] = []
    groups: dict[str, dict[str, set | int]] = {}
    new_pages: set[str] = set()

    for qid, entry in delta["queries"].items():
        n_new = entry["n_new"]
        pooled = n_new + entry["n_already_judged"]
        urls = [c["url"] for c in entry["candidates"]]
        new_pages.update(urls)
        cat = category.get(qid, entry.get("category", "?"))
        per_query.append(
            {
                "id": qid,
                "category": cat,
                "retrieved": pooled,
                "unjudged": n_new,
                "rate": (n_new / pooled) if pooled else 0.0,
                "new_urls": urls,
            }
        )
        for name in ("all", cat):
            g = groups.setdefault(
                name, {"n_queries": 0, "retrieved": 0, "unjudged": 0, "pages": set()}
            )
            g["n_queries"] += 1
            g["retrieved"] += pooled
            g["unjudged"] += n_new
            g["pages"].update(urls)

    summary = {}
    for name, g in groups.items():
        summary[name] = {
            "n_queries": g["n_queries"],
            "retrieved": g["retrieved"],
            "unjudged": g["unjudged"],
            "rate": (g["unjudged"] / g["retrieved"]) if g["retrieved"] else 0.0,
            "distinct_new_pages": len(g["pages"]),
        }
    return {
        "summary": summary,
        "per_query": per_query,
        # Distinct across the whole run: the union, which is smaller than the sum of the
        # per-category unions whenever a page is new for queries in different categories.
        "distinct_new_pages_total": len(new_pages),
        "queries_fully_covered": sum(1 for r in per_query if r["unjudged"] == 0),
    }


AGENT_PREFIX = "agent/"
RERANK_PREFIX = "rerank/"
#: Every system this module can pool: a plain retriever, an agent over one, or a
#: cross-encoder rerank of one. The rerankers were never pooled, so their unjudged rate is
#: the one number here that is a finding rather than a check.
SYSTEMS = (
    tuple(BASE_RETRIEVERS)
    + tuple(AGENT_PREFIX + b for b in BASE_RETRIEVERS)
    + tuple(RERANK_PREFIX + b for b in ("bm25", "hybrid"))
    + ("random",)
)

RE_POOL_HEADER = """\
# Re-pool for the agent layer - ONLY pages that carry no judgment in eval/queries.yaml.
#
#   2 = directly answers the query (the page you would send someone to)
#   1 = partially answers, or is useful context / answers a sub-part
#   0 = not relevant
#
# Every candidate here was returned by one of the systems in `built_by` and has never been
# graded at any grade. Pages already judged - including at grade 0, which is a verdict and
# not a gap - were subtracted, so nothing you have already ruled on appears again.
#
# `found_by` names which systems surfaced the page and at what rank, so a candidate that
# only one system found is identifiable. `n_new` and `n_already_judged` per query record
# what was kept and what was subtracted.
#
# This file exists because the existing judgments were pooled from BM25 and dense alone.
# Those two score 0%% unjudged by construction - the pool IS their top-k - while the agent
# and the hybrid return pages the pool never saw, which eval.run scores as grade 0. Until
# these are graded, an agent-vs-hybrid comparison measures pool coverage as much as
# retrieval quality, and the bias runs against the agent.
#
# Fill in every `grade:`, then APPEND (never merge - merge replaces a query's whole
# judgments block and would destroy the existing 611):
#
#     python -m eval.pool append --pool %(path)s
"""


def build_systems(names: Sequence[str], store, client, *, scorer=None) -> dict[str, object]:
    """``label -> retriever`` for names like ``hybrid`` or ``agent/hybrid``.

    Base retrievers are built once and shared, so ``hybrid`` and ``agent/hybrid`` are the
    same object and the dense model is loaded once rather than per system. The agents share
    one model client for the same reason.

    ``scorer`` is passed straight to :class:`~retrieval.rerank.RerankRetriever`, which builds
    a real cross-encoder when given none. It exists so the composition can be tested without
    sentence-transformers installed, the same way the dense tests inject an encoder: CI
    installs `requirements.txt` only, so a test that constructs a real scorer can pass
    locally and fail there.
    """
    bases: dict[str, object] = {}

    def base_for(name: str):
        if name not in bases:
            bases[name] = make_base_retriever(name, store)
        return bases[name]

    out: dict[str, object] = {}
    for name in names:
        if name.startswith(AGENT_PREFIX):
            out[name] = AgentRetriever(store, base_for(name[len(AGENT_PREFIX) :]), client)
        elif name == "random":
            from eval.run import RandomBaseline

            # The control: what an unpooled system's coverage looks like.
            out[name] = RandomBaseline(store.chunk_ids(), seed=0)
        elif name.startswith(RERANK_PREFIX):
            from retrieval.rerank import RerankRetriever

            # Built on the shared base, so `hybrid` and `rerank/hybrid` read the same
            # first stage and the difference in coverage is the reranking alone.
            out[name] = RerankRetriever(store, base_for(name[len(RERANK_PREFIX) :]), scorer)
        else:
            out[name] = base_for(name)
    return out


def format_systems(per_system: Mapping[str, Mapping], union: Mapping) -> str:
    """Each system's own unjudged rate, and what the union of them costs to grade.

    The union is smaller than the sum: systems overlap heavily, so a page unjudged for four
    of them is one candidate, not four.
    """
    header = f"{'system':<16}{'retrieved':>11}{'unjudged':>10}{'rate':>8}{'pages':>8}"
    lines = [header, "-" * len(header)]
    for name, rep in per_system.items():
        r = rep["summary"]["all"]
        lines.append(
            f"{name:<16}{r['retrieved']:>11}{r['unjudged']:>10}{r['rate']:>7.1%}"
            f"{rep['distinct_new_pages_total']:>8}"
        )
    u = union["summary"]["all"]
    lines += [
        "-" * len(header),
        f"{'union':<16}{u['retrieved']:>11}{u['unjudged']:>10}{u['rate']:>7.1%}"
        f"{union['distinct_new_pages_total']:>8}",
        "",
        f"{sum(r['summary']['all']['unjudged'] for r in per_system.values())} unjudged pairs "
        f"summed over systems collapse to {u['unjudged']} distinct candidates once pooled",
    ]
    return "\n".join(lines)


def analyze(retriever, name: str, queries: Sequence[Query], store, judged, k: int) -> dict:
    """Unjudged-coverage report for one ``search(query, k)`` retriever."""
    delta = delta_pool(build_pool({name: retriever}, queries, store, k=k), judged)
    report = coverage(delta, queries)
    report["retriever"] = name
    report["k"] = k
    report["_delta"] = delta
    return report


def format_comparison(agent_report: Mapping, base_report: Mapping) -> str:
    """The agent's unjudged rate next to its base retriever's.

    Without this control the agent's rate is unreadable. BM25 and dense score 0% by
    construction - the pool *is* their top-k, so they cannot surface anything unjudged - and
    any retriever that was not pooled starts with a hole of its own. Only the difference
    between the two rows is attributable to the agent.
    """
    a, b = agent_report["summary"], base_report["summary"]
    header = f"{'retriever':<16}{'retrieved':>11}{'unjudged':>10}{'rate':>8}{'pages':>8}"
    lines = [header, "-" * len(header)]
    for label, r, rep in (
        (base_report["retriever"], b["all"], base_report),
        (agent_report["retriever"], a["all"], agent_report),
    ):
        lines.append(
            f"{label:<16}{r['retrieved']:>11}{r['unjudged']:>10}{r['rate']:>7.1%}"
            f"{rep['distinct_new_pages_total']:>8}"
        )
    delta_pp = (a["all"]["rate"] - b["all"]["rate"]) * 100
    lines.append(
        f"{'attributable':<16}{'':>11}{a['all']['unjudged'] - b['all']['unjudged']:>10}"
        f"{delta_pp:>+6.1f}pp"
    )
    return "\n".join(lines)


def format_coverage(report: Mapping, *, verbose: bool = False) -> str:
    s = report["summary"]
    header = f"{'group':<14}{'q':>4}{'retrieved':>11}{'unjudged':>10}{'rate':>8}{'pages':>8}"
    lines = [header, "-" * len(header)]
    order = (
        ["all"]
        + [c for c in CATEGORIES if c in s]
        + [c for c in s if c not in CATEGORIES and c != "all"]
    )
    for name in order:
        r = s[name]
        lines.append(
            f"{name:<14}{r['n_queries']:>4}{r['retrieved']:>11}{r['unjudged']:>10}"
            f"{r['rate']:>7.1%}{r['distinct_new_pages']:>8}"
        )
    if verbose:
        lines += ["", f"{'query':<14}{'category':<12}{'retr':>6}{'new':>6}{'rate':>8}"]
        for r in sorted(report["per_query"], key=lambda r: -r["rate"]):
            lines.append(
                f"{r['id']:<14}{r['category']:<12}{r['retrieved']:>6}{r['unjudged']:>6}"
                f"{r['rate']:>7.1%}"
            )
    total = s.get("all", {})
    lines += [
        "",
        f"{total.get('unjudged', 0)} unjudged (query, page) pairs = grading decisions to make",
        f"{report['distinct_new_pages_total']} distinct pages to read across all queries",
        f"{report['queries_fully_covered']}/{total.get('n_queries', 0)} queries are already "
        "fully covered by the existing judgments",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=list(BASE_RETRIEVERS), default=DEFAULT_BASE)
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--offline", action="store_true", help="replay from the cache only")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the unjudged pages as a gradeable pool (writes nothing by default; "
        "refuses to overwrite an existing file)",
    )
    ap.add_argument(
        "--system",
        action="append",
        choices=list(SYSTEMS),
        default=None,
        help="repeatable; pool the union of these systems (default: agent/<--retriever>)",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also measure the base retriever alone, so the agent's rate has a control",
    )
    ap.add_argument("--force", action="store_true", help="allow --out to overwrite a file")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-query rows")
    args = ap.parse_args(argv)

    from corpus.store import Store

    names = args.system or [f"{AGENT_PREFIX}{args.retriever}"]
    if args.baseline and args.retriever not in names:
        names = [args.retriever, *names]
    names = list(dict.fromkeys(names))

    # The re-pool is a file a human then grades; clobbering one would destroy judgments that
    # cannot be recovered. Refuse rather than overwrite.
    if args.out and args.out.exists() and not args.force:
        print(f"error: {args.out} exists; refusing to overwrite it (--force to override)")
        return 2

    queries = load_queries(args.queries)
    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        judged = judged_urls(queries, store.canonical_map())
        client = None
        try:
            if any(n.startswith(AGENT_PREFIX) for n in names):
                client = make_client(
                    model=args.model,
                    base_url=args.base_url,
                    cache_path=args.cache,
                    offline=args.offline,
                )
            systems = build_systems(names, store, client)
            # One pool over every system at once: build_pool unions their top-k per query and
            # records `found_by`, so a page four systems found is one candidate, not four.
            union_delta = delta_pool(build_pool(systems, queries, store, k=args.k), judged)
            per_system = {
                name: analyze(r, name, queries, store, judged, args.k)
                for name, r in systems.items()
            }
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if client is not None and client.calls:
            client.cache.save()

    report = coverage(union_delta, queries)
    report["systems"] = names
    report["k"] = args.k
    report["per_system"] = {
        n: {k: v for k, v in r.items() if k != "_delta"} for n, r in per_system.items()
    }

    if args.out:
        header = RE_POOL_HEADER % {"path": args.out}
        n = write_pool({**union_delta, "built_by": names}, args.out, header=header)
        print(f"wrote {args.out}: {n} candidates to grade\n")
    if args.json:
        print(json.dumps(report, indent=1, default=list))
    else:
        print(f"systems: {', '.join(names)}   k={args.k}   judgments: {args.queries}\n")
        print(format_coverage(report, verbose=args.verbose))
        if len(per_system) > 1:
            print("\nper system:\n")
            print(format_systems(report["per_system"], report))
        # Exactly a base and its agent: the difference between them is the interesting
        # number, and it is not the agent's raw rate.
        pair = {args.retriever, f"{AGENT_PREFIX}{args.retriever}"}
        if set(names) == pair:
            print()
            print(
                format_comparison(
                    report["per_system"][f"{AGENT_PREFIX}{args.retriever}"],
                    report["per_system"][args.retriever],
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
