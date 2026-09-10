"""Render the generation section's tables from a faithfulness record.

`generate.faithfulness` prints a one-line-per-retriever summary; the README quotes more than
that — support rates with bootstrap intervals, a breakdown by query category, citation
counts and a latency spread. This produces exactly those tables, so every figure in that
section has a command behind it instead of a claim.

**The bootstrap resamples queries, not claims.** Claims inside one answer stand or fall
together: they come from one generation over one context, and a context that is off-topic
makes all of them unsupported at once. Resampling claims would treat them as independent and
report an interval several times too narrow. 4,000 draws, seed fixed.

**Two support columns, and neither is "the" rate.** The checker returns no verdict for a
share of claims (`Verdict.judged` is False for those). The left column counts them as
unsupported, the right drops them. The gap between the columns is the size of the checker's
silence, and it differs by retriever, which is why the section refuses to rank on either.

Usage:
    python -m generate.faithfulness --all --offline --json record.json
    python -m generate.report record.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval.run import CATEGORIES, load_queries

DRAWS = 4000
SEED = 0


def bootstrap(per_query: list[tuple[int, int]], draws: int = DRAWS, seed: int = SEED):
    """95% interval for a ratio of sums, resampling the (supported, total) pairs."""
    if not per_query:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(per_query)
    out = []
    for _ in range(draws):
        sample = [per_query[rng.randrange(n)] for _ in range(n)]
        total = sum(b for _, b in sample)
        if total:
            out.append(sum(a for a, _ in sample) / total)
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


def scored_answers(blob: dict) -> list[dict]:
    """Answers that produced claims: not refused, not malformed, with verdicts."""
    return [
        a for a in blob["answers"] if not a["refused"] and not a["malformed"] and a.get("verdicts")
    ]


def pairs(answers: list[dict], judged_only: bool) -> list[tuple[int, int]]:
    out = []
    for a in answers:
        verdicts = (
            [v for v in a["verdicts"] if v.get("judged", True)] if judged_only else a["verdicts"]
        )
        if verdicts:
            out.append((sum(v["supported"] for v in verdicts), len(verdicts)))
    return out


def support_table(record: dict) -> str:
    lines = [
        "| retriever | claims | judged | unjudged = unsupported | judged only |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, blob in record.items():
        answers = scored_answers(blob)
        allp, judp = pairs(answers, False), pairs(answers, True)
        claims = sum(t for _, t in allp)
        judged = sum(t for _, t in judp)
        r_all = sum(s for s, _ in allp) / claims if claims else 0.0
        r_jud = sum(s for s, _ in judp) / judged if judged else 0.0
        lo1, hi1 = bootstrap(allp)
        lo2, hi2 = bootstrap(judp)
        lines.append(
            f"| {name} | {claims} | {judged} | **{r_all:.3f}** [{lo1:.2f}, {hi1:.2f}] "
            f"| **{r_jud:.3f}** [{lo2:.2f}, {hi2:.2f}] |"
        )
    return "\n".join(lines)


def category_tables(record: dict, category: dict[str, str]) -> str:
    cell: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for name, blob in record.items():
        for a in scored_answers(blob):
            c = category[a["query_id"]]
            cell[(name, c)][0] += sum(v["supported"] for v in a["verdicts"])
            cell[(name, c)][1] += len(a["verdicts"])
            cell[(name, c)][2] += sum(v.get("judged", True) for v in a["verdicts"])

    out = ["| retriever | " + " | ".join(CATEGORIES) + " |", "|---" * (len(CATEGORIES) + 1) + "|"]
    for name in record:
        row = [name]
        for c in CATEGORIES:
            s, t, j = cell[(name, c)]
            row.append(f"{s / t:.2f} / {s / j:.2f} (n={t})" if t and j else "—")
        out.append("| " + " | ".join(row) + " |")

    out += [
        "",
        "| category | claims | judged | unjudged = unsupported | judged only | refusals |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CATEGORIES:
        s = sum(cell[(n, c)][0] for n in record)
        t = sum(cell[(n, c)][1] for n in record)
        j = sum(cell[(n, c)][2] for n in record)
        answers = [a for b in record.values() for a in b["answers"] if category[a["query_id"]] == c]
        refused = sum(a["refused"] for a in answers)
        out.append(f"| {c} | {t} | {j} | {s / t:.3f} | {s / j:.3f} | {refused}/{len(answers)} |")
    return "\n".join(out)


def citation_table(record: dict) -> str:
    out = [
        "| retriever | scored answers | citations | dangling | rate |",
        "|---|---:|---:|---:|---:|",
    ]
    tot_c = tot_d = 0
    for name, blob in record.items():
        answers = [a for a in blob["answers"] if not a["refused"] and not a["malformed"]]
        cits = sum(len(a["cited"]) for a in answers)
        dang = sum(len(a["dangling"]) for a in answers)
        tot_c += cits
        tot_d += dang
        rate = dang / (cits + dang) if cits + dang else 0.0
        out.append(f"| {name} | {len(answers)} | {cits} | {dang} | {rate:.3f} |")
    uncited = sum(
        1
        for b in record.values()
        for a in b["answers"]
        if not a["refused"] and not a["malformed"] and not a["cited"]
    )
    out.append("")
    out.append(f"totals: {tot_c} citations, {tot_d} dangling, {uncited} answers with none")
    return "\n".join(out)


def latency_table(record: dict) -> str:
    def pct(values: list[float], q: float) -> float:
        return sorted(values)[min(int(q * len(values)), len(values) - 1)]

    out = [
        "| retriever | answer min / p50 / p90 / max | retrieval p50 | end-to-end p50 "
        "| prompt tok | completion tok | tok/query |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, blob in record.items():
        a = blob["answers"]
        lat = sorted(x["model_seconds"] for x in a)
        ret = sorted(x["retrieval_seconds"] for x in a)
        e2e = sorted(x["model_seconds"] + x["retrieval_seconds"] for x in a)
        pt = statistics.fmean(x["prompt_tokens"] for x in a)
        ct = statistics.fmean(x["completion_tokens"] for x in a)
        out.append(
            f"| {name} | {lat[0]:.1f} / **{pct(lat, 0.5):.1f}** / {pct(lat, 0.9):.1f} / "
            f"{lat[-1]:.1f} s | {1000 * pct(ret, 0.5):.0f} ms | {pct(e2e, 0.5):.1f} s "
            f"| {pt:,.0f} | {ct:,.0f} | {pt + ct:,.0f} |"
        )
    return "\n".join(out)


def refusal_table(record: dict) -> str:
    out = [
        "| retriever | refusals on real context | probe pairs | dropped | correct refusal |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, blob in record.items():
        probe = blob["probe"]
        scored = [r for r in probe if not r["skipped"]]
        refused = sum(bool(r["refused"]) for r in scored)
        rate = refused / len(scored) if scored else 0.0
        out.append(
            f"| {name} | {sum(a['refused'] for a in blob['answers'])}/{len(blob['answers'])} "
            f"| {len(scored)} | {len(probe) - len(scored)} "
            f"| **{refused}/{len(scored)} = {rate:.3f}** |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record", type=Path, help="JSON written by `generate.faithfulness --json`")
    ap.add_argument("--queries", type=Path, default=None)
    args = ap.parse_args(argv)

    record = json.loads(args.record.read_text(encoding="utf-8"))
    queries = load_queries(args.queries) if args.queries else load_queries()
    category = {q.id: q.category for q in queries}

    total_claims = sum(len(a["verdicts"]) for b in record.values() for a in b["answers"])
    total_answers = sum(len(b["answers"]) for b in record.values())
    print(
        f"{total_answers} answers, {total_claims} claims, "
        f"{DRAWS}-draw bootstrap over queries, seed {SEED}"
    )
    answers = [a for b in record.values() for a in b["answers"]]
    replayed = sum(a.get("cached", False) for a in answers)
    if replayed == len(answers):
        print(
            "\nNOTE: every answer here came from the cache, so this record is an --offline "
            "replay.\nModel timings are the original live calls, stored with the response. "
            "Retrieval\ntimings are measured fresh on every run and so are the replay's own -- "
            "for the agent\nespecially, they are not comparable to a live run, because the agent "
            "retrieves by\ncalling the model and those calls are served from cache here."
        )
    elif replayed:
        print(f"\nNOTE: {replayed} of {len(answers)} answers were cache hits; timings are mixed.")
    print()
    for title, table in (
        ("Support", support_table(record)),
        ("By category", category_tables(record, category)),
        ("Refusal", refusal_table(record)),
        ("Citations", citation_table(record)),
        ("Latency and tokens", latency_table(record)),
    ):
        print(f"### {title}\n")
        print(table)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
