"""Re-grade reliability: reproduction rates, Cohen's kappa, and the two probes compared.

Two blind re-grade probes are committed: `recheck.yaml` (grade 1 shown against grade 0,
graded before `eval/GRADING.md` was written) and `recheck_2v1.yaml` (grade 1 against grade 2,
graded after). Each has a key naming the original grade, withheld while re-grading. This
reads both and reports how repeatable the grading is.

**Kappa is reported per probe, not only pooled.** The two probes differ in which contrast the
grader saw *and* in whether the rubric existed, so pooling them averages over a difference
that is itself the thing in question. The README quotes probe 2's kappa; probe 1's is roughly
four times larger, and the pooled figure sits between. All three are printed so no one has to
guess which is which.

**The Fisher test asks one narrow question**: is grade 1 more stable in one probe than the
other? It compares how often a grade-1 candidate came back as grade 1, probe against probe.
It cannot attribute any difference to the rubric or to the contrast, because the probes
changed both at once - that confound is limitation 2 in the README, and no test here removes
it. Implemented directly from the hypergeometric distribution so the repo keeps its
numpy-only dependency.

Usage:
    python -m eval.reliability
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import numpy as np
import yaml

EVAL_DIR = Path(__file__).parent
PROBE_1 = (EVAL_DIR / "recheck.yaml", EVAL_DIR / "recheck_key.yaml")
PROBE_2 = (EVAL_DIR / "recheck_2v1.yaml", EVAL_DIR / "recheck_2v1_key.yaml")
GRADES = (0, 1, 2)


def load_pairs(probe: Path, key: Path) -> list[tuple[int, int]]:
    """``(original grade, re-grade)`` for every candidate the probe actually graded."""
    entries = yaml.safe_load(probe.read_text(encoding="utf-8"))["entries"]
    key_map = {x["id"]: x for x in yaml.safe_load(key.read_text(encoding="utf-8"))["key"]}
    return [
        (int(key_map[e["id"]]["original"]), int(e["grade"]))
        for e in entries
        if e.get("grade") is not None
    ]


def confusion(pairs: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((len(GRADES), len(GRADES)))
    for original, regrade in pairs:
        m[original, regrade] += 1
    return m


def cohens_kappa(pairs: list[tuple[int, int]]) -> tuple[float, float, float]:
    """``(kappa, observed agreement, chance agreement)`` over the three-grade scale."""
    m = confusion(pairs)
    total = m.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    observed = float(np.trace(m)) / total
    chance = float(sum(m[i, :].sum() * m[:, i].sum() for i in range(len(GRADES)))) / total**2
    if chance == 1.0:
        return 0.0, observed, chance
    return (observed - chance) / (1 - chance), observed, chance


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table ``[[a, b], [c, d]]``.

    Sums the hypergeometric probability of every table at least as extreme as the observed
    one, "as extreme" meaning no more probable - the conventional two-sided definition.
    """
    n, row1, col1 = a + b + c + d, a + b, a + c
    if n == 0 or row1 in (0, n) or col1 in (0, n):
        return 1.0

    def prob(x: int) -> float:
        return comb(row1, x) * comb(n - row1, col1 - x) / comb(n, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - (n - row1)), min(row1, col1)
    # The tolerance keeps a table that ties the observed probability from being excluded by
    # floating-point noise; dropping it would understate p.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-12)))


def reproduction(pairs: list[tuple[int, int]], grade: int) -> tuple[int, int]:
    """``(came back as the same grade, total shown)`` for one original grade."""
    shown = [p for p in pairs if p[0] == grade]
    return sum(1 for _, regrade in shown if regrade == grade), len(shown)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe1", nargs=2, type=Path, default=PROBE_1, metavar=("PROBE", "KEY"))
    ap.add_argument("--probe2", nargs=2, type=Path, default=PROBE_2, metavar=("PROBE", "KEY"))
    args = ap.parse_args(argv)

    probes = {
        "probe 1 (1s against 0s, before the rubric)": load_pairs(*args.probe1),
        "probe 2 (1s against 2s, after the rubric)": load_pairs(*args.probe2),
    }

    for label, pairs in probes.items():
        print(f"{label}: {len(pairs)} graded")
        m = confusion(pairs)
        print("  re-grade matrix (row = original, col = re-grade)")
        print(f"           {'0':>6}{'1':>6}{'2':>6}   reproduced")
        for g in GRADES:
            shown = int(m[g, :].sum())
            if not shown:
                continue
            same, total = reproduction(pairs, g)
            print(
                f"     {g} ->{int(m[g, 0]):>6}{int(m[g, 1]):>6}{int(m[g, 2]):>6}"
                f"   {same}/{total} = {same / total:.0%}"
            )
        k, observed, chance = cohens_kappa(pairs)
        print(f"  Cohen's kappa: {k:.4f}   (observed {observed:.3f}, chance {chance:.3f})")
        print()

    pooled = [p for pairs in probes.values() for p in pairs]
    k, observed, chance = cohens_kappa(pooled)
    print(
        f"pooled over both probes: n={len(pooled)}  kappa {k:.4f} "
        f"(observed {observed:.3f}, chance {chance:.3f})"
    )
    print("  Pooling averages over the contrast and the rubric at once; the per-probe rows")
    print("  above are the ones to quote.\n")

    (s1, n1), (s2, n2) = (reproduction(pairs, 1) for pairs in probes.values())
    p = fisher_exact_two_sided(s1, n1 - s1, s2, n2 - s2)
    print("is grade 1 more stable in one probe than the other?")
    print(f"  probe 1: {s1}/{n1} reproduced      probe 2: {s2}/{n2} reproduced")
    print(f"  two-sided Fisher exact p = {p:.4f}")
    print("  The probes changed contrast and rubric together, so this cannot attribute a")
    print("  difference to either one; it only says whether there is one to attribute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
