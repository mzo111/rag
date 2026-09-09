"""Build a judging pool: the candidate pages a human needs to grade, and nothing else.

Judging 11,832 chunks by hand is not possible; judging the union of what a few retrievers
rank highly is. This is standard TREC-style pooling. ``build`` runs each configured
retriever over every query, unions their top-k pages, and writes ``eval/pool.yaml`` with a
blank ``grade`` on each candidate plus enough context (title, section, snippet) to grade it
without opening a browser. ``merge`` folds the graded pool back into ``eval/queries.yaml``.

The pool is only as good as the retrievers that built it: a page no retriever surfaced is
never judged and will look like a miss forever. Re-run ``build`` after adding a retriever of
a different kind (a semantic one, say) and grade only the candidates that are new.

``delta`` is the second half of that loop. It pools a *new* retriever the same way, then
subtracts every page the judge has already seen, and writes only the remainder to
``eval/pool_delta.yaml``. Grading that file is the cheapest honest way to un-bias a pool that
one retriever built by itself: it asks for judgments only where the new retriever disagrees
with what is already judged.

Usage:
    python -m eval.pool build [--k 10] [--retriever bm25 ...]
    python -m eval.pool delta [--k 10] [--retriever dense ...]
    python -m eval.pool merge [--force]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from eval.run import QUERIES_PATH, Query, load_queries

POOL_PATH = Path(__file__).with_name("pool.yaml")
DELTA_PATH = Path(__file__).with_name("pool_delta.yaml")
SNIPPET_CHARS = 220

HEADER = """\
# Judging pool - fill in every `grade:` then run `python -m eval.pool merge`.
#
#   2 = directly answers the query (the page you would send someone to)
#   1 = partially answers, or is useful context / answers a sub-part
#   0 = not relevant
#
# Leave a grade blank to skip that candidate; blanks are NOT written to queries.yaml, and a
# page you never grade counts as 0 when the metrics run. `title`, `section` and `snippet`
# are context for you, and are ignored by the merge. To narrow a judgment to one section,
# add the judgment by hand in queries.yaml with an `anchor:` key.
#
# Candidates come from the retrievers named in `built_by` below. Anything no retriever
# surfaced is absent, so add pages you know are relevant by hand rather than trusting the
# pool to be complete.
"""

DELTA_HEADER = """\
# Delta judging pool - only candidates that are NOT already judged in queries.yaml.
#
#   2 = directly answers the query (the page you would send someone to)
#   1 = partially answers, or is useful context / answers a sub-part
#   0 = not relevant
#
# Every page here was surfaced by the retrievers in `built_by` and has never been graded, so
# there is no need to re-read anything you have already judged. Pages already carrying a
# judgment of ANY grade (including 0) were subtracted - a 0 means the judge saw it and ruled
# it out, which is just as much a judgment as a 2.
#
# `n_new` per query and `n_already_judged` record what was kept and what was subtracted.
#
# NOTE: `python -m eval.pool merge` REPLACES a query's judgments rather than appending to
# them, so it must not be pointed at this file while queries.yaml holds the original pool's
# grades. Merging this delta needs an append-mode merge that does not exist yet.
"""


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """One-line preview of a chunk for the judge."""
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("..." if len(flat) > limit else "")


def build_pool(
    retrievers: Mapping[str, object],
    queries: Sequence[Query],
    store,
    *,
    k: int = 10,
) -> dict:
    """Union each retriever's top-k pages per query into a gradeable structure."""
    canonical = store.canonical_map()
    out: dict = {"built_by": sorted(retrievers), "k": k, "queries": {}}
    for q in queries:
        seen: dict[str, dict] = {}
        for name, r in retrievers.items():
            for rank, cid in enumerate(r.search(q.query, k), 1):
                chunk = store.get(cid)
                if chunk is None:
                    continue
                url = canonical.get(chunk.url, chunk.url)
                if url in seen:
                    seen[url]["found_by"].append(f"{name}@{rank}")
                    continue
                seen[url] = {
                    "grade": None,
                    "url": url,
                    "title": chunk.title,
                    "section": " > ".join(chunk.heading_path[-2:]),
                    "snippet": snippet(chunk.text),
                    "found_by": [f"{name}@{rank}"],
                }
        for cand in seen.values():
            cand["found_by"] = ", ".join(cand["found_by"])
        out["queries"][q.id] = {
            "query": q.query,
            "category": q.category,
            "candidates": list(seen.values()),
        }
    return out


def write_pool(pool: dict, path: Path = POOL_PATH, header: str = HEADER) -> int:
    body = yaml.safe_dump(pool, sort_keys=False, width=100, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return sum(len(q["candidates"]) for q in pool["queries"].values())


def judged_urls(
    queries: Sequence[Query], canonical: Mapping[str, str] | None = None
) -> dict[str, set[str]]:
    """query id -> canonical page urls that already carry a judgment, at any grade.

    Grade 0 counts as judged: the judge looked at that page and ruled it out, which is a
    verdict, not a gap. Re-listing it would ask for the same decision twice.

    Judgment urls are canonicalized on the way in, the same rewrite :mod:`eval.run` applies,
    so a judgment written against an alias still matches a candidate pooled under the
    canonical page.
    """
    out: dict[str, set[str]] = {}
    for q in queries:
        urls = set()
        for j in q.judgments:
            urls.add(canonical.get(j.url, j.url) if canonical else j.url)
        out[q.id] = urls
    return out


def delta_pool(pool: dict, judged: Mapping[str, set[str]]) -> dict:
    """Strip candidates whose page is already judged, keeping the pool's shape and order.

    Queries that gain nothing are kept with an empty ``candidates`` list: "this retriever
    surfaced nothing new here" is a result worth recording, not an omission.
    """
    out: dict = {k: v for k, v in pool.items() if k != "queries"}
    out["queries"] = {}
    for qid, entry in pool["queries"].items():
        seen = judged.get(qid, set())
        new = [c for c in entry["candidates"] if c["url"] not in seen]
        out["queries"][qid] = {
            **{k: v for k, v in entry.items() if k != "candidates"},
            "n_new": len(new),
            "n_already_judged": len(entry["candidates"]) - len(new),
            "candidates": new,
        }
    return out


def format_delta_report(delta: dict) -> str:
    """Per-query and total counts of what the delta pool adds."""
    header = f"{'query':<14}{'category':<12}{'new':>5}{'judged':>8}{'pooled':>8}"
    lines = [header, "-" * len(header)]
    n_new = n_seen = 0
    for qid, entry in delta["queries"].items():
        new, seen = entry["n_new"], entry["n_already_judged"]
        n_new += new
        n_seen += seen
        lines.append(f"{qid:<14}{entry['category']:<12}{new:>5}{seen:>8}{new + seen:>8}")
    n_q = len(delta["queries"]) or 1
    lines += [
        "-" * len(header),
        f"{'total':<14}{'':<12}{n_new:>5}{n_seen:>8}{n_new + n_seen:>8}",
        "",
        f"{n_new} new candidates over {len(delta['queries'])} queries "
        f"({n_new / n_q:.1f} per query); {n_seen} already judged and skipped",
    ]
    empty = [qid for qid, e in delta["queries"].items() if e["n_new"] == 0]
    if empty:
        lines.append(f"queries with nothing new ({len(empty)}): {', '.join(empty)}")
    return "\n".join(lines)


def graded_judgments(pool: dict) -> dict[str, list[tuple[str, int]]]:
    """query id -> [(url, grade)] for candidates that were actually graded."""
    out: dict[str, list[tuple[str, int]]] = {}
    for qid, entry in pool["queries"].items():
        rows = [
            (c["url"], int(c["grade"]))
            for c in entry["candidates"]
            if c.get("grade") is not None and str(c["grade"]).strip() != ""
        ]
        if rows:
            out[qid] = rows
    return out


def merge_into_queries(
    judgments: Mapping[str, Sequence[tuple[str, int]]],
    queries_path: Path = QUERIES_PATH,
    *,
    force: bool = False,
) -> tuple[int, list[str]]:
    """Write judgments into queries.yaml, preserving its comments and layout.

    Queries that already have judgments are left alone unless ``force``; their ids are
    returned so the caller can report them.
    """
    src = queries_path.read_text(encoding="utf-8")
    lines = src.split("\n")
    out: list[str] = []
    skipped: list[str] = []
    written = 0
    cur: str | None = None
    dropping = False
    for line in lines:
        m = re.match(r"  - id: (\S+)", line)
        if m:
            cur = m.group(1)
        if dropping:
            if line.strip().startswith("- {") or line.strip().startswith("- url:"):
                continue
            dropping = False
        stripped = line.strip()
        if cur in judgments and stripped in ("judgments: []", "judgments:"):
            if stripped == "judgments:" and not force:
                skipped.append(cur)
                out.append(line)
                continue
            dropping = stripped == "judgments:"
            out.append("    judgments:")
            for url, grade in judgments[cur]:
                out.append(f"      - {{url: {url}, grade: {grade}}}")
            written += 1
            continue
        out.append(line)
    queries_path.write_text("\n".join(out), encoding="utf-8")
    return written, skipped


JUDGMENT_LINE = re.compile(r"^      - \{url: ")
QUERY_ID_LINE = re.compile(r"  - id: (\S+)")


def append_into_queries(
    judgments: Mapping[str, Sequence[tuple[str, int]]],
    queries_path: Path = QUERIES_PATH,
    *,
    backup: bool = True,
) -> tuple[list[str], int, list[tuple[str, str]]]:
    """Add judgments to the END of each query's existing block, replacing nothing.

    Deliberately separate from :func:`merge_into_queries`, which rewrites a whole
    ``judgments:`` block and would destroy earlier rounds. This one only ever inserts lines,
    so every judgment already in the file survives byte for byte.

    A url already judged for that query is a conflict: it is skipped and returned rather
    than written twice or silently overwritten, because the same page cannot hold two grades
    and picking one for you would be a judgment call, not a merge.

    Returns ``(query ids touched, rows added, conflicts)``.
    """
    src = queries_path.read_text(encoding="utf-8")
    if backup:
        queries_path.with_suffix(queries_path.suffix + ".bak").write_text(src, encoding="utf-8")

    existing = {q.id: {j.url for j in q.judgments} for q in load_queries(queries_path)}
    conflicts: list[tuple[str, str]] = []
    touched: list[str] = []
    added = 0

    def rows_for(qid: str | None) -> list[str]:
        rows = []
        for url, grade in judgments.get(qid or "", []):
            if url in existing.get(qid or "", set()):
                conflicts.append((qid or "", url))
                continue
            rows.append(f"      - {{url: {url}, grade: {grade}}}")
        return rows

    out: list[str] = []
    cur: str | None = None
    in_judgments = False
    for line in src.split("\n"):
        is_id = QUERY_ID_LINE.match(line)
        # The block ends at the first line that is not a judgment row (blank line, next key,
        # next query). Insert there, before that line, so the new rows land inside the block.
        if in_judgments and (is_id or not JUDGMENT_LINE.match(line)):
            rows = rows_for(cur)
            if rows:
                out.extend(rows)
                added += len(rows)
                touched.append(cur or "")
            in_judgments = False
        if is_id:
            cur = is_id.group(1)
        stripped = line.strip()
        if cur and stripped == "judgments:":
            in_judgments = True
        elif cur and stripped == "judgments: []":
            rows = rows_for(cur)
            if rows:
                out.append("    judgments:")
                out.extend(rows)
                added += len(rows)
                touched.append(cur)
                continue
        out.append(line)
    if in_judgments:  # file ended inside a judgments block
        rows = rows_for(cur)
        if rows:
            out.extend(rows)
            added += len(rows)
            touched.append(cur or "")

    queries_path.write_text("\n".join(out), encoding="utf-8")
    return touched, added, conflicts


def _make_retrievers(names: Sequence[str], store, seed: int = 0) -> dict[str, object]:
    retrievers: dict[str, object] = {}
    for name in names:
        if name == "bm25":
            from retrieval.bm25 import Bm25Retriever

            retrievers[name] = Bm25Retriever(store)
        elif name == "dense":
            from retrieval.dense import DenseRetriever

            retrievers[name] = DenseRetriever.load(store)
        elif name == "random":
            from eval.run import RandomBaseline

            retrievers[name] = RandomBaseline(store.chunk_ids(), seed=seed)
        else:
            raise SystemExit(f"unknown retriever: {name}")
    return retrievers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write eval/pool.yaml for hand grading")
    b.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    b.add_argument("--queries", type=Path, default=QUERIES_PATH)
    b.add_argument("--out", type=Path, default=POOL_PATH)
    b.add_argument(
        "--retriever", action="append", default=None, choices=["bm25", "dense", "random"]
    )
    b.add_argument("-k", type=int, default=10)

    d = sub.add_parser("delta", help="write eval/pool_delta.yaml: only not-yet-judged pages")
    d.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    d.add_argument("--queries", type=Path, default=QUERIES_PATH)
    d.add_argument("--out", type=Path, default=DELTA_PATH)
    d.add_argument(
        "--retriever", action="append", default=None, choices=["bm25", "dense", "random"]
    )
    d.add_argument("-k", type=int, default=10)

    ap_ = sub.add_parser("append", help="ADD a graded pool's judgments, replacing nothing")
    ap_.add_argument("--pool", type=Path, default=DELTA_PATH)
    ap_.add_argument("--queries", type=Path, default=QUERIES_PATH)
    ap_.add_argument("--no-backup", action="store_true", help="skip writing queries.yaml.bak")

    m = sub.add_parser("merge", help="fold graded eval/pool.yaml into eval/queries.yaml")
    m.add_argument("--pool", type=Path, default=POOL_PATH)
    m.add_argument("--queries", type=Path, default=QUERIES_PATH)
    m.add_argument("--force", action="store_true", help="overwrite existing judgments")

    args = ap.parse_args(argv)

    if args.cmd == "build":
        from corpus.store import Store

        names = args.retriever or ["bm25"]
        with Store(args.db) as store:
            if store.count() == 0:
                print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store")
                return 1
            queries = load_queries(args.queries)
            pool = build_pool(_make_retrievers(names, store), queries, store, k=args.k)
        n = write_pool(pool, args.out)
        per = n / len(pool["queries"]) if pool["queries"] else 0
        print(
            f"wrote {args.out}: {n} candidates over {len(pool['queries'])} queries "
            f"({per:.1f} per query, from {', '.join(names)} top-{args.k})"
        )
        print("grade every `grade:` field, then: python -m eval.pool merge")
        return 0

    if args.cmd == "delta":
        from corpus.store import Store

        names = args.retriever or ["dense"]
        with Store(args.db) as store:
            if store.count() == 0:
                print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store")
                return 1
            queries = load_queries(args.queries)
            canonical = store.canonical_map()
            pool = build_pool(_make_retrievers(names, store), queries, store, k=args.k)
        delta = delta_pool(pool, judged_urls(queries, canonical))
        n = write_pool(delta, args.out, header=DELTA_HEADER)
        print(format_delta_report(delta))
        print(f"\nwrote {args.out}: {n} candidates to grade (from {', '.join(names)} top-{args.k})")
        print(f"{args.queries} was not modified.")
        return 0

    if args.cmd == "append":
        pool = yaml.safe_load(args.pool.read_text(encoding="utf-8"))
        judgments = graded_judgments(pool)
        if not judgments:
            print(f"no grades filled in {args.pool}; nothing to append")
            return 1
        before = {q.id: len(q.judgments) for q in load_queries(args.queries)}
        touched, added, conflicts = append_into_queries(
            judgments, args.queries, backup=not args.no_backup
        )
        after = {q.id: len(q.judgments) for q in load_queries(args.queries)}
        print(f"appended {added} judgments across {len(touched)} queries in {args.queries}")
        print(f"total judgments: {sum(before.values())} -> {sum(after.values())}")
        if not args.no_backup:
            print(f"backup: {args.queries}.bak")
        if conflicts:
            print(f"skipped {len(conflicts)} already-judged urls (a page cannot hold two grades):")
            for qid, url in conflicts[:10]:
                print(f"  {qid}  {url}")
        return 0

    pool = yaml.safe_load(args.pool.read_text(encoding="utf-8"))
    judgments = graded_judgments(pool)
    if not judgments:
        print(f"no grades filled in {args.pool}; nothing to merge")
        return 1
    written, skipped = merge_into_queries(judgments, args.queries, force=args.force)
    total = sum(len(v) for v in judgments.values())
    print(f"merged {total} judgments into {written} queries in {args.queries}")
    if skipped:
        print(f"left alone (already judged, use --force): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
