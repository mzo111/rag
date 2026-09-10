"""Tests for the two reporting producers: eval.alias_slots and generate.report."""

import pytest

from corpus.chunk import Chunk
from eval.alias_slots import duplicate_slots, pages_returned
from eval.run import Query
from generate.report import bootstrap, pairs, scored_answers

P1 = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P1_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P2 = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
CANONICAL = {P1_ALIAS: P1}


class FakeStore:
    def __init__(self, table):
        self.table = table  # chunk_id -> url

    def locate(self, ids):
        return {i: (self.table[i], "") for i in ids if i in self.table}


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k):
        return self.ids[:k]


def chunk(cid, url):
    return Chunk(cid, url, "", "T", ["T"], 0, "text", 1, cid)


# --- eval.alias_slots ---------------------------------------------------------------------


def test_an_alias_of_an_earlier_slot_counts_as_a_duplicate():
    store = FakeStore({"a": P1, "b": P1_ALIAS, "c": P2})
    q = [Query("q1", "q", "api_lookup", [])]
    dup, total = duplicate_slots(FakeRetriever(["a", "b", "c"]), q, store, CANONICAL, 10)
    assert (dup, total) == (1, 3)  # b is a's alias


def test_distinct_pages_are_never_duplicates():
    store = FakeStore({"a": P1, "c": P2})
    q = [Query("q1", "q", "api_lookup", [])]
    assert duplicate_slots(FakeRetriever(["a", "c"]), q, store, CANONICAL, 10) == (0, 2)


def test_the_same_page_twice_is_a_duplicate_without_any_alias_map():
    store = FakeStore({"a": P1, "b": P1})
    q = [Query("q1", "q", "api_lookup", [])]
    assert duplicate_slots(FakeRetriever(["a", "b"]), q, store, {}, 10) == (1, 2)


def test_slots_are_counted_over_every_query():
    store = FakeStore({"a": P1, "b": P1_ALIAS})
    qs = [Query(f"q{i}", "q", "api_lookup", []) for i in range(3)]
    assert duplicate_slots(FakeRetriever(["a", "b"]), qs, store, CANONICAL, 10) == (3, 6)


def test_pages_returned_is_not_canonicalized():
    """The top-k change count depends on seeing the alias as itself, not as its canonical."""
    store = FakeStore({"b": P1_ALIAS})
    assert pages_returned(FakeRetriever(["b"]), "q", store, 10) == [P1_ALIAS]


def test_duplicate_slots_respects_k():
    store = FakeStore({"a": P1, "b": P1_ALIAS, "c": P2})
    q = [Query("q1", "q", "api_lookup", [])]
    assert duplicate_slots(FakeRetriever(["a", "b", "c"]), q, store, CANONICAL, 2) == (1, 2)


# --- generate.report ----------------------------------------------------------------------


def answer(verdicts, refused=False, malformed=""):
    return {
        "query_id": "q1",
        "refused": refused,
        "malformed": malformed,
        "verdicts": verdicts,
        "cited": [],
        "dangling": [],
    }


def test_pairs_counts_every_verdict_by_default():
    a = answer([{"supported": True, "judged": True}, {"supported": False, "judged": False}])
    assert pairs([a], judged_only=False) == [(1, 2)]


def test_pairs_drops_unjudged_claims_when_asked():
    a = answer([{"supported": True, "judged": True}, {"supported": False, "judged": False}])
    assert pairs([a], judged_only=True) == [(1, 1)]


def test_pairs_skips_an_answer_with_nothing_judged():
    a = answer([{"supported": False, "judged": False}])
    assert pairs([a], judged_only=True) == []
    assert pairs([a], judged_only=False) == [(0, 1)]


def test_a_verdict_without_the_judged_field_is_treated_as_judged():
    """Records written before Verdict.judged existed must still report."""
    a = answer([{"supported": True}])
    assert pairs([a], judged_only=True) == [(1, 1)]


def test_scored_answers_excludes_refusals_and_malformed_and_empty():
    blob = {
        "answers": [
            answer([{"supported": True, "judged": True}]),
            answer([], refused=True),
            answer([{"supported": True, "judged": True}], malformed="bad JSON"),
            answer([]),
        ]
    }
    assert len(scored_answers(blob)) == 1


def test_bootstrap_is_seeded_and_brackets_the_point_estimate():
    data = [(1, 2), (2, 2), (0, 2), (2, 2), (1, 2)]
    lo, hi = bootstrap(data, draws=500)
    assert (lo, hi) == bootstrap(data, draws=500)  # same seed, same interval
    point = sum(a for a, _ in data) / sum(b for _, b in data)
    assert lo <= point <= hi


def test_bootstrap_on_no_variation_is_a_point():
    lo, hi = bootstrap([(1, 1), (1, 1), (1, 1)], draws=200)
    assert lo == hi == 1.0


def test_bootstrap_of_nothing_is_zero_rather_than_an_error():
    assert bootstrap([], draws=10) == (0.0, 0.0)


@pytest.mark.parametrize("judged_only", [True, False])
def test_pairs_never_reports_a_total_of_zero(judged_only):
    """A zero total would divide by zero in the caller."""
    a = answer([{"supported": False, "judged": False}])
    assert all(total > 0 for _, total in pairs([a], judged_only=judged_only))
