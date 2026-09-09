import pytest

from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import Retriever
from retrieval.hybrid import RRF_K, HybridRetriever, rrf_scores

P_CANON = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P_OTHER = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
P_THIRD = "https://docs.pytorch.org/docs/2.14/nn.html"


def chunk(cid, url, text, ordinal=0, title="T"):
    return Chunk(cid, url, "", title, [title], ordinal, text, len(text.split()), cid)


@pytest.fixture
def store():
    s = Store(":memory:")
    s.load(
        [
            chunk("a1", P_CANON, "class torch.optim.Adam(params, eps=1e-08)", title="Adam"),
            chunk("a2", P_ALIAS, "class torch.optim.adam.Adam(params, eps=1e-08)", title="Adam"),
            chunk("t1", P_OTHER, "torch.topk returns k largest", title="torch.topk"),
            chunk("t2", P_OTHER, "second chunk of the topk page", ordinal=1, title="torch.topk"),
            chunk("n1", P_THIRD, "the nn module index", title="nn"),
        ]
    )
    s.apply_quality()
    yield s
    s.close()


class FakeSystem:
    def __init__(self, ids):
        self.ids = ids
        self.last_k = None

    def search(self, query, k):
        self.last_k = k
        return self.ids[:k]


def hybrid(store, a, b, **kw):
    return HybridRetriever(store, {"a": FakeSystem(a), "b": FakeSystem(b)}, **kw)


# --- the fusion formula ---------------------------------------------------------


def test_rrf_scores_uses_one_over_k_plus_rank():
    s = rrf_scores({"a": ["x", "y"]}, k=60)
    assert s["x"] == pytest.approx(1 / 61)
    assert s["y"] == pytest.approx(1 / 62)


def test_rrf_scores_sum_across_systems():
    s = rrf_scores({"a": ["x"], "b": ["x"]}, k=60)
    assert s["x"] == pytest.approx(2 / 61)


def test_rrf_constant_is_the_published_sixty():
    # Cormack, Clarke & Buettcher (SIGIR 2009). Guards against silent tuning.
    assert RRF_K == 60


def test_rrf_default_k_matches_the_module_constant():
    assert rrf_scores({"a": ["x"]})["x"] == pytest.approx(1 / (RRF_K + 1))


# --- fusion behaviour -----------------------------------------------------------


def test_agreement_beats_a_single_system_first_place(store):
    # 'a' ranks t1 first, but n1 is ranked by both, so consensus outweighs one first place:
    # n1 = 1/62 + 1/61 = 0.0325, t1 = 1/61 = 0.0164
    r = hybrid(store, ["t1", "n1"], ["n1"])
    assert r.search("q", 2)[0] == "n1"


def test_a_symmetric_tie_is_broken_deterministically(store):
    # mirror-image rankings give both pages exactly 1/61 + 1/62, so the ordering rests
    # entirely on the tie-break (best single-system rank, then url) and must be stable
    r = hybrid(store, ["t1", "n1"], ["n1", "t1"])
    scores = [s for _, s, _ in r.explain("q", 2)]
    assert scores[0] == pytest.approx(scores[1])
    assert r.search("q", 2) == r.search("q", 2)


def test_a_page_ranked_by_both_outranks_a_page_ranked_once_higher(store):
    r = hybrid(store, ["a1", "t1"], ["t1"])
    # t1: 1/62 + 1/61 ; a1: 1/61 -> t1 wins on the sum
    assert r.search("q", 2) == ["t1", "a1"]


def test_alias_pages_fuse_into_one_entry(store):
    # a1 and a2 are the same canonical page seen by two systems
    r = hybrid(store, ["a1"], ["a2"])
    hits = r.search("q", 5)
    assert len(hits) == 1 and hits[0] in {"a1", "a2"}


def test_two_chunks_of_one_page_do_not_take_two_slots(store):
    r = hybrid(store, ["t1", "t2", "a1"], ["t2", "t1"])
    hits = r.search("q", 5)
    assert len([h for h in hits if h in {"t1", "t2"}]) == 1


def test_a_system_ranking_one_page_twice_pays_only_once(store):
    # 'a' lists both chunks of P_OTHER; that must not double its contribution
    once = hybrid(store, ["t1"], ["n1"]).explain("q", 5)
    twice = hybrid(store, ["t1", "t2"], ["n1"]).explain("q", 5)
    score_once = dict((p, s) for p, s, _ in once)[P_OTHER]
    score_twice = dict((p, s) for p, s, _ in twice)[P_OTHER]
    assert score_once == pytest.approx(score_twice)


def test_best_ranked_chunk_represents_its_page(store):
    r = hybrid(store, ["t2", "t1"], [])
    assert r.search("q", 1) == ["t2"]


def test_search_respects_k(store):
    r = hybrid(store, ["a1", "t1", "n1"], ["n1", "t1", "a1"])
    assert len(r.search("q", 1)) == 1
    assert len(r.search("q", 2)) == 2


def test_empty_and_degenerate_queries(store):
    r = hybrid(store, ["a1"], ["t1"])
    assert r.search("", 5) == []
    assert r.search("   ", 5) == []
    assert r.search("q", 0) == []
    assert r.search("q", -1) == []


def test_handles_one_system_returning_nothing(store):
    r = hybrid(store, ["a1", "t1"], [])
    assert r.search("q", 5) == ["a1", "t1"]


def test_unknown_chunk_ids_are_ignored(store):
    r = hybrid(store, ["nope", "a1"], ["a1"])
    assert r.search("q", 5) == ["a1"]


def test_conforms_to_the_retriever_protocol(store):
    assert isinstance(hybrid(store, ["a1"], ["t1"]), Retriever)


def test_reads_each_system_to_the_configured_depth(store):
    r = HybridRetriever(store, {"a": FakeSystem(["a1"]), "b": FakeSystem(["t1"])}, depth=37)
    r.search("q", 5)
    assert all(s.last_k == 37 for s in r.systems.values())


def test_depth_is_never_below_k(store):
    r = HybridRetriever(store, {"a": FakeSystem(["a1"]), "b": FakeSystem(["t1"])}, depth=2)
    r.search("q", 9)
    assert all(s.last_k == 9 for s in r.systems.values())


def test_explain_reports_per_system_ranks(store):
    r = hybrid(store, ["t1", "a1"], ["a1"])
    got = {page: ranks for page, _, ranks in r.explain("q", 5)}
    assert got[P_OTHER] == {"a": 1}
    assert got[P_CANON] == {"a": 2, "b": 1}


def test_ordering_is_deterministic_across_calls(store):
    r = hybrid(store, ["a1", "t1", "n1"], ["n1", "a1", "t1"])
    assert r.search("q", 5) == r.search("q", 5)


def test_fusion_is_symmetric_in_system_order(store):
    a, b = ["a1", "t1", "n1"], ["n1", "t1", "a1"]
    one = HybridRetriever(store, {"a": FakeSystem(a), "b": FakeSystem(b)}).search("q", 5)
    two = HybridRetriever(store, {"b": FakeSystem(b), "a": FakeSystem(a)}).search("q", 5)
    assert [s for s in one] == [s for s in two]
