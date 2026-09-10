"""Faithfulness: is every claim in an answer stated by the passages it was given?

**What this measures.** Groundedness in retrieved text. A claim counts as supported when a
retrieved passage says it. Nothing here checks whether the passage is *right*, whether it
answers the question, or whether the answer is useful. An answer that faithfully reproduces
a wrong or irrelevant chunk scores 1.0, and that is not a defect in the metric - it is the
boundary of what the metric is. Retrieval quality is measured separately, against the hand
judgments, by :mod:`eval.run`; the two numbers are about different failures and neither
substitutes for the other.

**No new human judgments.** Two LLM passes replace them, and both are the same model family
as the generator, which is the central weakness of the whole measurement: the checker
shares a tokenizer, a pretraining corpus and a set of blind spots with the thing it is
judging. Where the generator misreads a passage, the checker is disposed to misread it the
same way and call the claim supported. This biases the support rate **up** by an unmeasured
amount, and nothing in this repo bounds it. Treat the rates as comparable *between
retrievers*, which share the checker, and not as absolute levels.

The two passes are kept apart on purpose. The decomposer never sees the passages, so it
cannot shape claims to check out; the checker never sees the question, so it cannot drift
from "is this stated" to "does this answer the question".

**The refusal probe.** Measuring "refuses when the context lacks the answer" needs cases
where the context provably lacks it. Rather than label any, each query is re-asked over the
passages retrieved for a *different* query - a fixed rotation by
:data:`MISMATCH_OFFSET` over the 40 queries, chosen a priori and never searched. A pair is
dropped when the donor passages contain a page the *existing* judgments already mark
relevant (grade >= 1) for the recipient query, since the context would then not be missing
the answer. That check reads ``eval/queries.yaml``; it writes nothing.

Usage:
    python -m generate.faithfulness --retriever bm25|dense|hybrid|rerank|agent [--offline]
    python -m generate.faithfulness --all --offline --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient, ResponseCache
from generate.prompts import CLAIMS_PROMPT, SUPPORT_PROMPT
from generate.run import DEFAULT_CACHE, NUM_CTX, NUM_PREDICT, RETRIEVERS, Answer, Generator
from generate.run import format_context as _format_context

# --- fixed a priori; asserted in tests/test_generate.py ---------------------------------
MAX_CLAIMS = 8  # cap on atomic claims per answer; answers are capped at 120 words
MISMATCH_OFFSET = 20  # query i is re-asked over the passages retrieved for query i+20


@dataclass
class Verdict:
    claim: str
    supported: bool
    passage: int | None = None


@dataclass
class Checked:
    """One answer, its claims, and the checker's verdict on each."""

    answer: Answer
    claims: list[str] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    malformed: str = ""
    check_seconds: float = 0.0
    check_prompt_tokens: int = 0
    check_completion_tokens: int = 0

    @property
    def supported(self) -> int:
        return sum(v.supported for v in self.verdicts)

    @property
    def n_claims(self) -> int:
        return len(self.verdicts)


def parse_claims(raw: str, max_claims: int) -> tuple[list[str], str]:
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"invalid JSON: {exc.msg}"
    if not isinstance(body, dict) or not isinstance(body.get("claims"), list):
        return [], "missing 'claims' list"
    claims = [c.strip() for c in body["claims"] if isinstance(c, str) and c.strip()]
    if not claims:
        return [], "no usable claims"
    # Over-long lists are truncated rather than rejected: the cap is a bound on work, and a
    # model that ignores it has still produced usable claims up to that point.
    return claims[:max_claims], ""


def parse_verdicts(raw: str, claims: list[str], k: int) -> tuple[list[Verdict], str]:
    """Verdicts aligned to ``claims`` by the model's 1-based claim numbers.

    A claim the checker skipped is counted as unsupported rather than dropped: silently
    dropping it would raise the support rate by removing the cases the checker found hard.
    """
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [Verdict(c, False) for c in claims], f"invalid JSON: {exc.msg}"
    rows = body.get("verdicts") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return [Verdict(c, False) for c in claims], "missing 'verdicts' list"

    seen: dict[int, Verdict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            n = int(row.get("claim"))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= len(claims) or n in seen:
            continue
        passage = row.get("passage")
        passage = passage if isinstance(passage, int) and 1 <= passage <= k else None
        seen[n] = Verdict(claims[n - 1], bool(row.get("supported")), passage)

    verdicts = [seen.get(i + 1, Verdict(c, False)) for i, c in enumerate(claims)]
    missing = len(claims) - len(seen)
    return verdicts, f"{missing} claim(s) unjudged, counted unsupported" if missing else ""


class Checker:
    """Decomposes an answer into claims, then judges each against the passages."""

    def __init__(self, client: OllamaClient, *, max_claims: int = MAX_CLAIMS):
        self.client = client
        self.max_claims = max_claims

    def _call(self, prompt: str, *, query: str, tag: str):
        return self.client.generate(
            prompt,
            query=query,
            retriever=tag,
            options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        )

    def check(self, answer: Answer, chunks: list) -> Checked:
        out = Checked(answer=answer)
        if answer.refused or answer.malformed or not answer.text:
            return out  # nothing to decompose; refusals are scored by the probe instead

        r1 = self._call(
            CLAIMS_PROMPT.format(answer=answer.text, max_claims=self.max_claims),
            query=answer.query,
            tag=f"{answer.retriever}:claims",
        )
        claims, bad = parse_claims(r1.text, self.max_claims)
        out.check_seconds += r1.seconds
        out.check_prompt_tokens += r1.prompt_tokens
        out.check_completion_tokens += r1.completion_tokens
        if bad:
            out.malformed = f"claims: {bad}"
            return out
        out.claims = claims

        numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
        r2 = self._call(
            SUPPORT_PROMPT.format(context=_format_context(chunks), claims=numbered),
            query=answer.query,
            tag=f"{answer.retriever}:support",
        )
        verdicts, bad = parse_verdicts(r2.text, claims, len(chunks))
        out.check_seconds += r2.seconds
        out.check_prompt_tokens += r2.prompt_tokens
        out.check_completion_tokens += r2.completion_tokens
        out.verdicts = verdicts
        if bad:
            out.malformed = f"support: {bad}"
        return out


def relevant_pages(query, canonical) -> set[str]:
    """Canonical pages the existing judgments mark relevant (grade >= 1) for a query."""
    out = set()
    for j in query.judgments:
        if j.grade >= 1:
            out.add(canonical.get(j.url, j.url))
    return out


def mismatch_pairs(queries, offset: int = MISMATCH_OFFSET) -> list[tuple[int, int]]:
    """``(recipient, donor)`` index pairs under the fixed rotation."""
    n = len(queries)
    return [(i, (i + offset) % n) for i in range(n)]


def run_refusal_probe(gen: Generator, queries, store, canonical, *, verbose: bool = False):
    """Ask each query over another query's passages; count how often the system refuses."""
    contexts = {}
    for q in queries:
        ids, chunks, _ = gen.context_for(q.query)
        contexts[q.id] = (ids, chunks)

    rows = []
    for i, donor_i in mismatch_pairs(queries):
        q, donor = queries[i], queries[donor_i]
        ids, chunks = contexts[donor.id]
        located = store.locate(ids)
        pages = {canonical.get(url, url) for url, _ in located.values()}
        overlap = pages & relevant_pages(q, canonical)
        if overlap:
            rows.append({"query_id": q.id, "donor": donor.id, "skipped": True})
            continue
        a = gen.answer_from(q.id, q.query, ids, chunks, 0.0, tag="mismatch")
        rows.append(
            {
                "query_id": q.id,
                "donor": donor.id,
                "skipped": False,
                "refused": a.refused,
                "malformed": a.malformed,
                "model_seconds": a.model_seconds,
                "prompt_tokens": a.prompt_tokens,
                "completion_tokens": a.completion_tokens,
            }
        )
        if verbose:
            print(f"  probe {q.id} <- {donor.id}: {'REFUSED' if a.refused else 'answered'}")
    return rows


def summarize(retriever: str, checks: list[Checked], probe: list[dict]) -> dict:
    answered = [c for c in checks if not c.answer.refused and not c.answer.malformed]
    claims = sum(c.n_claims for c in answered)
    supported = sum(c.supported for c in answered)
    scored = [r for r in probe if not r["skipped"]]
    lat = [c.answer.model_seconds for c in checks]
    ret = [c.answer.retrieval_seconds for c in checks]
    return {
        "retriever": retriever,
        "queries": len(checks),
        "refused": sum(c.answer.refused for c in checks),
        "malformed": sum(bool(c.answer.malformed) for c in checks),
        "dangling_citations": sum(len(c.answer.dangling) for c in checks),
        "answers_scored": len(answered),
        "claims": claims,
        "claims_per_answer": round(claims / len(answered), 2) if answered else 0.0,
        "supported": supported,
        "support_rate": round(supported / claims, 3) if claims else 0.0,
        "answers_fully_supported": sum(
            1 for c in answered if c.n_claims and c.supported == c.n_claims
        ),
        "probe_pairs": len(probe),
        "probe_scored": len(scored),
        "probe_skipped": len(probe) - len(scored),
        "probe_refused": sum(bool(r["refused"]) for r in scored),
        "refusal_rate": (
            round(sum(bool(r["refused"]) for r in scored) / len(scored), 3) if scored else 0.0
        ),
        "median_model_seconds": round(statistics.median(lat), 2) if lat else 0.0,
        "median_retrieval_seconds": round(statistics.median(ret), 4) if ret else 0.0,
        "mean_prompt_tokens": round(statistics.fmean(c.answer.prompt_tokens for c in checks))
        if checks
        else 0,
        "mean_completion_tokens": round(
            statistics.fmean(c.answer.completion_tokens for c in checks)
        )
        if checks
        else 0,
        "check_seconds": round(sum(c.check_seconds for c in checks), 1),
        "check_tokens": sum(c.check_prompt_tokens + c.check_completion_tokens for c in checks),
    }


def format_table(rows: list[dict]) -> str:
    head = (
        f"{'retriever':<10}{'n':>4}{'refuse':>8}{'claims':>8}{'/ans':>6}"
        f"{'support':>9}{'full':>6}{'probe':>7}{'refusal':>9}{'model s':>9}{'p_tok':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['retriever']:<10}{r['queries']:>4}{r['refused']:>8}{r['claims']:>8}"
            f"{r['claims_per_answer']:>6.1f}{r['support_rate']:>9.3f}"
            f"{r['answers_fully_supported']:>6}{r['probe_scored']:>7}"
            f"{r['refusal_rate']:>9.3f}{r['median_model_seconds']:>9.1f}"
            f"{r['mean_prompt_tokens']:>7}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from corpus.store import Store
    from eval.run import load_queries

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    ap.add_argument("--all", action="store_true", help="every retriever in turn")
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true", help="skip the refusal probe")
    ap.add_argument("--json", type=Path, default=None, help="write the full record here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    names = list(RETRIEVERS) if args.all else [args.retriever]

    cache = ResponseCache(args.cache)
    client = OllamaClient(
        model=args.model, base_url=args.base_url, cache=cache, offline=args.offline
    )
    if not args.offline:
        client.check_available()

    rows, record = [], {}
    with Store(args.db) as store:
        canonical = store.canonical_map()
        checker = Checker(client)
        for name in names:
            from generate.run import make_retriever

            gen = Generator(store, make_retriever(name, store, client=client), name, client)
            checks = []
            for i, q in enumerate(queries, start=1):
                # retrieve once and reuse: the agent retriever costs model calls, and the
                # checker must see exactly the passages the answer was written from
                ids, chunks, retrieval_seconds = gen.context_for(q.query)
                a = gen.answer_from(q.id, q.query, ids, chunks, retrieval_seconds)
                checks.append(checker.check(a, chunks))
                if args.verbose:
                    c = checks[-1]
                    print(
                        f"  {name} {i:>3}/{len(queries)} {q.id:<6} "
                        f"{'REFUSED' if a.refused else f'{c.supported}/{c.n_claims} supported'}",
                        flush=True,
                    )
                if i % 5 == 0:
                    cache.save()
            probe = (
                []
                if args.no_probe
                else run_refusal_probe(gen, queries, store, canonical, verbose=args.verbose)
            )
            cache.save()
            rows.append(summarize(name, checks, probe))
            record[name] = {
                "summary": rows[-1],
                "answers": [
                    {
                        **c.answer.as_dict(),
                        "claims": c.claims,
                        "verdicts": [v.__dict__ for v in c.verdicts],
                        "check_malformed": c.malformed,
                    }
                    for c in checks
                ],
                "probe": probe,
            }
    cache.save()

    print()
    print(format_table(rows))
    if args.json:
        args.json.write_text(json.dumps(record, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
