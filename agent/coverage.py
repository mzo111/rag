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

from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaError
from agent.run import BASE_RETRIEVERS, DEFAULT_BASE, make_agent
from eval.pool import DELTA_HEADER, build_pool, delta_pool, judged_urls, write_pool
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
        help="also write a gradeable delta pool of the unjudged pages (writes nothing "
        "by default; never point this at eval/pool_delta.yaml)",
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="also measure the base retriever alone, so the agent's rate has a control",
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true", help="per-query rows")
    args = ap.parse_args(argv)

    from corpus.store import Store

    queries = load_queries(args.queries)
    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        judged = judged_urls(queries, store.canonical_map())
        try:
            agent = make_agent(
                store,
                args.retriever,
                model=args.model,
                base_url=args.base_url,
                cache_path=args.cache,
                offline=args.offline,
            )
            report = analyze(agent, f"agent/{args.retriever}", queries, store, judged, args.k)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if agent.client.calls:
            agent.client.cache.save()
        base = (
            analyze(agent.retriever, args.retriever, queries, store, judged, args.k)
            if args.baseline
            else None
        )

    delta = report.pop("_delta")
    if base is not None:
        base.pop("_delta")
        report["baseline"] = base

    if args.out:
        n = write_pool(delta, args.out, header=DELTA_HEADER)
        print(f"wrote {args.out}: {n} candidates to grade\n")
    if args.json:
        print(json.dumps(report, indent=1, default=list))
    else:
        print(f"agent over {args.retriever}   k={args.k}   judgments: {args.queries}\n")
        print(format_coverage(report, verbose=args.verbose))
        if base is not None:
            print("\nagainst the base retriever alone:\n")
            print(format_comparison(report, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
