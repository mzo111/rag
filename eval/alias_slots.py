"""How many of a retriever's top-k slots are pages it has already returned.

The corpus-quality sections of the README quote three counts about alias pages: how many
result slots are duplicates, how many slots the stub filter is worth, and how many queries
have their top-k change when aliases are collapsed. All three describe retriever *output*
rather than the corpus, so none of them can be read off `corpus/quality.py`; this is what
produces them.

A slot is a *duplicate* when the chunk filling it maps to a canonical page that an
earlier slot in the same query's ranking already covered. Counting is over the whole query
set, so the denominator is `queries x k`.

The two arms differ only in `exclude_stubs`, because the README quotes both: the
alias-ablation figure runs with stubs excluded (the default policy, so aliases are the only
thing switched off), and the corpus-quality figure runs fully unfiltered.

Usage:
    python -m eval.alias_slots [--retriever bm25] [--k 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

QUERIES = None  # resolved in main, so importing this module costs nothing


def duplicate_slots(retriever, queries, store, canonical, k: int) -> tuple[int, int]:
    """``(duplicate slots, total slots)`` over every query's top-k."""
    duplicate = total = 0
    for q in queries:
        ids = list(retriever.search(q.query, k))[:k]
        located = store.locate(ids)
        seen: set[str] = set()
        for cid in ids:
            total += 1
            url, _ = located.get(cid, ("", ""))
            page = canonical.get(url, url)
            if page in seen:
                duplicate += 1
            seen.add(page)
    return duplicate, total


def pages_returned(retriever, query: str, store, k: int) -> list[str]:
    """The raw page URLs filling the top-k, in rank order and not canonicalized."""
    ids = list(retriever.search(query, k))[:k]
    located = store.locate(ids)
    return [located.get(cid, ("", ""))[0] for cid in ids]


def main(argv: list[str] | None = None) -> int:
    from corpus.store import Store
    from eval.run import load_queries

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "dense"])
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args(argv)

    queries = load_queries()
    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        canonical = store.canonical_map()

        def build(*, collapse: bool, stubs: bool):
            if args.retriever == "bm25":
                from retrieval.bm25 import Bm25Retriever

                return Bm25Retriever(store, collapse_aliases=collapse, exclude_stubs=stubs)
            from retrieval.dense import DenseRetriever

            return DenseRetriever.load(store, collapse_aliases=collapse, exclude_stubs=stubs)

        print(f"retriever: {args.retriever}   k={args.k}   queries: {len(queries)}\n")
        header = f"{'aliases':<10}{'stubs':<10}{'duplicate slots':>18}{'rate':>8}"
        print(header)
        print("-" * len(header))
        counts = {}
        for collapse in (True, False):
            for stubs in (True, False):
                dup, total = duplicate_slots(
                    build(collapse=collapse, stubs=stubs), queries, store, canonical, args.k
                )
                counts[(collapse, stubs)] = (dup, total)
                print(
                    f"{'collapsed' if collapse else 'kept':<10}"
                    f"{'excluded' if stubs else 'kept':<10}"
                    f"{f'{dup} of {total}':>18}{dup / total:>8.1%}"
                )

        # How many queries' top-k changes when alias collapsing is switched on. Compared on
        # the raw page URLs filling the slots, not canonicalized: canonicalizing first would
        # map an alias onto the page it duplicates and hide the very change being counted.
        kept = build(collapse=False, stubs=True)
        collapsed = build(collapse=True, stubs=True)
        changed = sum(
            pages_returned(kept, q.query, store, args.k)
            != pages_returned(collapsed, q.query, store, args.k)
            for q in queries
        )
        # Slots a deprecated stub occupies when the stub filter is off. This is a different
        # question from the duplicate count above, so it is measured directly rather than
        # inferred from the difference between the two arms.
        stubs = store.stub_urls()
        unfiltered = build(collapse=False, stubs=False)
        stub_slots = sum(
            url in stubs
            for q in queries
            for url in pages_returned(unfiltered, q.query, store, args.k)
        )

        dup_stub_excluded = counts[(False, True)][0]
        dup_unfiltered = counts[(False, False)][0]
        print()
        print(f"top-{args.k} changes when aliases are collapsed: {changed} of {len(queries)}")
        print(f"duplicate slots, stubs excluded (the ablation arm):  {dup_stub_excluded}")
        print(f"duplicate slots, fully unfiltered (corpus quality):  {dup_unfiltered}")
        print(f"slots held by a deprecated stub, unfiltered:         {stub_slots}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
