"""BM25 retrieval over the FTS5 index built by :mod:`corpus.store`.

This is the lexical baseline: whatever comes later has to beat it. It uses SQLite's own
``bm25()`` ranking with per-column weights, so the whole retriever is one SQL query.

Query handling. FTS5's MATCH grammar treats bare punctuation, hyphens and its own keywords
(``AND``, ``OR``, ``NEAR``) as syntax, so a natural-language question is not a valid query
string. :func:`to_match_query` extracts identifier-like terms, quotes each one and joins
them with ``OR``: documents matching any term become candidates and BM25 does the ranking.
Dotted names are additionally split, so ``torch.nn.functional.cross_entropy`` also matches
pages that only mention ``cross_entropy``.

Two corpus-quality behaviours from :mod:`corpus.quality` are on by default and can be
turned off to measure their effect:

- deprecated stub pages are excluded, since they are redirect notices that answer nothing;
- alias pages are collapsed, so the same content published at several URLs takes one slot
  in the top k rather than several.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from corpus.store import Store

# Terms worth searching for: identifiers, dotted names, numbers. Everything else is syntax.
TERM = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|\d+")
# Very common English words carry no signal and blow up the candidate set.
STOPWORDS = frozenset(
    """a an and are as at be by do does for from how i if in into is it its of on or that
    the to what when where which who why with you your me my we our will can should would
    could does doing done get got make made use used using want need""".split()
)
DEFAULT_WEIGHTS = (3.0, 2.0, 1.0)  # title, heading_path, text


def to_match_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression, or '' if nothing is searchable."""
    terms: list[str] = []
    for raw in TERM.findall(query):
        term = raw.strip(".").lower()
        if not term or term in STOPWORDS:
            continue
        terms.append(term)
        if "." in term:
            # Also match the bare parts of a dotted name (cross_entropy from the full path).
            terms.extend(p for p in term.split(".") if p and p not in STOPWORDS)
    seen: dict[str, None] = {}
    for t in terms:
        seen.setdefault(t, None)
    return " OR ".join(f'"{t}"' for t in seen)


class Bm25Retriever:
    """Rank chunks with SQLite FTS5 BM25. Implements the ``search(query, k)`` protocol."""

    def __init__(
        self,
        store: Store,
        *,
        weights: Sequence[float] = DEFAULT_WEIGHTS,
        exclude_stubs: bool = True,
        collapse_aliases: bool = True,
        candidate_pool: int = 500,
    ):
        self.store = store
        self.weights = tuple(weights)
        self.exclude_stubs = exclude_stubs
        self.collapse_aliases = collapse_aliases
        self.candidate_pool = candidate_pool

    def search(self, query: str, k: int) -> list[str]:
        match = to_match_query(query)
        if not match or k <= 0:
            return []
        sql = f"""
            WITH hits AS (
                SELECT rowid, bm25(chunks_fts, {", ".join(str(w) for w in self.weights)}) AS score
                FROM chunks_fts WHERE chunks_fts MATCH ?
                ORDER BY score LIMIT ?
            )
            SELECT c.chunk_id, c.url, p.canonical_url, p.status, h.score
            FROM hits h JOIN chunks c ON c.rowid = h.rowid JOIN pages p ON p.url = c.url
            ORDER BY h.score
        """
        try:
            rows = self.store.conn.execute(sql, (match, self.candidate_pool)).fetchall()
        except sqlite3.OperationalError:  # malformed MATCH despite quoting
            return []

        out: list[str] = []
        seen_pages: set[str] = set()
        for row in rows:
            if self.exclude_stubs and row["status"] != "ok":
                continue
            if self.collapse_aliases:
                page = row["canonical_url"] or row["url"]
                if page in seen_pages:
                    continue
                seen_pages.add(page)
            out.append(row["chunk_id"])
            if len(out) >= k:
                break
        return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ad-hoc BM25 search over data/corpus.db")
    ap.add_argument("query", nargs="+")
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--keep-stubs", action="store_true")
    ap.add_argument("--keep-aliases", action="store_true")
    args = ap.parse_args(argv)

    with Store(args.db) as store:
        r = Bm25Retriever(
            store, exclude_stubs=not args.keep_stubs, collapse_aliases=not args.keep_aliases
        )
        q = " ".join(args.query)
        print(f"query: {q}\nmatch: {to_match_query(q)}\n")
        for i, cid in enumerate(r.search(q, args.k), 1):
            c = store.get(cid)
            path = " > ".join(c.heading_path[-2:])
            print(f"{i:>3}. {c.url.replace('https://docs.pytorch.org/', '')}")
            print(f"     {path}  [{c.n_tokens} tok]")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
