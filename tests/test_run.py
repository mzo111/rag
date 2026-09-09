import pytest

from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import (
    Judgment,
    Query,
    RandomBaseline,
    Retriever,
    canonical_key,
    evaluate,
    format_table,
    load_queries,
    summarize,
    to_units,
)

P1 = "https://docs.pytorch.org/docs/2.14/p1.html"
P2 = "https://docs.pytorch.org/docs/2.14/p2.html"
T1 = "https://docs.pytorch.org/tutorials/t1.html"

LOCATED = {
    "p1a": (P1, "a"),
    "p1b": (P1, "b"),
    "p2": (P2, "s"),
    "t1": (T1, "intro"),
}


def locate(ids):
    return {i: LOCATED[i] for i in ids if i in LOCATED}


class FakeRetriever:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.answers[query][:k]


def test_to_units_maps_chunks_to_pages_or_judged_sections():
    judged = {f"{P1}#b", P2}
    assert to_units(["p1a", "p1b", "p1b", "p2", "t1", "zzz"], LOCATED, judged) == [
        P1,  # section a not judged -> page unit
        f"{P1}#b",  # judged section -> section unit
        P2,
        T1,  # unjudged page still yields a unit (grade 0)
    ]


def test_evaluate_hand_computed():
    queries = [
        Query("q1", "first", "api_lookup", [Judgment(P1, 2), Judgment(P2, 1)]),
        Query("q2", "second", "tutorial", [Judgment(T1, 2, anchor="intro")]),
        Query("q3", "third", "conceptual", []),  # unjudged -> skipped
    ]
    r = FakeRetriever({"first": ["p2", "t1", "p1a", "p1b"], "second": ["p1a", "p2"], "third": []})
    results = evaluate(r, queries, locate, k=10)
    assert [x.id for x in results] == ["q1", "q2"]
    q1, q2 = results
    # q1 units: [P2(1), T1(0), P1(2)] -> R@5 = 1.0, MRR = 1.0,
    # nDCG = (1 + 0 + 3/log2(4)) / (3 + 1/log2(3)) = 2.5 / 3.63093 = 0.68853
    assert q1.recall_5 == q1.recall_10 == 1.0 and q1.mrr == 1.0
    assert q1.ndcg_10 == pytest.approx(0.68853, abs=1e-5)
    assert (q2.recall_5, q2.recall_10, q2.mrr, q2.ndcg_10) == (0.0, 0.0, 0.0, 0.0)
    assert r.calls == [("first", 10), ("second", 10)]
    assert isinstance(r, Retriever)


def test_evaluate_include_unjudged_scores_zero():
    queries = [Query("q3", "third", "conceptual", [])]
    r = FakeRetriever({"third": ["p1a"]})
    (res,) = evaluate(r, queries, locate, k=10, include_unjudged=True)
    assert res.judged is False and res.ndcg_10 == 0.0


def test_summarize_and_table():
    queries = [
        Query("q1", "first", "api_lookup", [Judgment(P1, 2)]),
        Query("q2", "second", "api_lookup", [Judgment(P2, 2)]),
        Query("q3", "third", "tutorial", [Judgment(T1, 1)]),
    ]
    r = FakeRetriever({"first": ["p1a"], "second": ["t1", "p2"], "third": ["p2"]})
    results = evaluate(r, queries, locate)
    s = summarize(results)
    assert s["all"]["n"] == 3 and s["api_lookup"]["n"] == 2 and s["tutorial"]["n"] == 1
    assert s["api_lookup"]["mrr"] == pytest.approx(0.75)
    assert s["all"]["recall_10"] == pytest.approx(2 / 3)
    table = format_table(results, n_total=4, n_skipped=1, verbose=True)
    # nDCG: q1 = 1.0, q2 = (3/log2 3)/3 = 0.631, q3 = 0 -> mean 0.544
    assert "all              3    0.667  0.667  0.500    0.544" in table
    assert "3 of 4 queries scored, 1 skipped (no judgments yet)" in table
    assert "q2            api_lookup    1.000  1.000  0.500    0.631" in table


def test_random_baseline_returns_valid_ids_from_store():
    with Store(":memory:") as s:
        s.load([Chunk(f"c{i}", P1, "", "P1", ["P1"], i, f"text {i}", 2, "h") for i in range(20)])
        rb = RandomBaseline(s.chunk_ids(), seed=1)
        ids = rb.search("anything", 5)
        assert len(ids) == 5 and len(set(ids)) == 5
        assert set(ids) <= set(s.chunk_ids())
        assert set(s.locate(ids)) == set(ids)
        assert RandomBaseline(s.chunk_ids(), seed=1).search("anything", 5) == ids  # seeded
        assert len(rb.search("x", 100)) == 20  # capped at corpus size


def test_load_queries_file_roundtrip(tmp_path):
    p = tmp_path / "q.yaml"
    p.write_text(
        "queries:\n"
        "  - id: q01\n    category: api_lookup\n    query: hello\n    judgments: []\n"
        "  - id: q02\n    category: tutorial\n    query: world\n"
        "    judgments:\n      - {url: https://x/y.html, anchor: sec, grade: 2}\n"
        "      - {url: https://x/z.html, grade: 0}\n"
    )
    qs = load_queries(p)
    assert [q.id for q in qs] == ["q01", "q02"]
    assert qs[0].judgments == [] and qs[0].grades == {}
    assert qs[1].grades == {"https://x/y.html#sec": 2, "https://x/z.html": 0}


# --- alias collapsing -----------------------------------------------------------

P1_ALIAS = "https://docs.pytorch.org/docs/2.14/p1_alias.html"
CANONICAL = {P1_ALIAS: P1}


def locate_with_alias(ids):
    table = {**LOCATED, "p1alias": (P1_ALIAS, "a")}
    return {i: table[i] for i in ids if i in table}


def test_to_units_rewrites_alias_urls_to_canonical():
    assert to_units(["p1alias"], {"p1alias": (P1_ALIAS, "")}, set(), CANONICAL) == [P1]
    # without the map the alias stays distinct
    assert to_units(["p1alias"], {"p1alias": (P1_ALIAS, "")}, set()) == [P1_ALIAS]


def test_to_units_dedupes_alias_and_canonical_into_one_unit():
    located = {"p1a": (P1, "a"), "p1alias": (P1_ALIAS, "a")}
    assert to_units(["p1alias", "p1a"], located, set(), CANONICAL) == [P1]


def test_canonical_key_preserves_anchor():
    assert canonical_key(f"{P1_ALIAS}#sec", CANONICAL) == f"{P1}#sec"
    assert canonical_key(P1_ALIAS, CANONICAL) == P1
    assert canonical_key(P1_ALIAS, None) == P1_ALIAS


def test_evaluate_credits_a_retriever_that_returns_the_alias():
    """A judgment written against the canonical page matches a retrieved alias."""
    queries = [Query("q1", "first", "api_lookup", [Judgment(P1, 2)])]
    r = FakeRetriever({"first": ["p1alias"]})
    (no_map,) = evaluate(r, queries, locate_with_alias, k=10)
    assert no_map.ndcg_10 == 0.0  # alias looks like a different page
    (with_map,) = evaluate(r, queries, locate_with_alias, k=10, canonical=CANONICAL)
    assert (with_map.recall_5, with_map.mrr, with_map.ndcg_10) == (1.0, 1.0, 1.0)
