import pytest

from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import Retriever
from retrieval.dense import embedding_text
from retrieval.rerank import DEFAULT_TOP_N, RerankRetriever

P1 = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P2 = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
P3 = "https://docs.pytorch.org/docs/2.14/nn.html"


def chunk(cid, url, text, title="T"):
    return Chunk(cid, url, "", title, [title, "sec"], 0, text, len(text.split()), cid)


@pytest.fixture
def store():
    s = Store(":memory:")
    s.load(
        [
            chunk("a1", P1, "class torch.optim.Adam(params, eps=1e-08)", title="Adam"),
            chunk("t1", P2, "torch.topk returns k largest", title="torch.topk"),
            chunk("n1", P3, "the nn module index", title="nn"),
        ]
    )
    s.apply_quality()
    yield s
    s.close()


class FakeBase:
    def __init__(self, ids):
        self.ids = ids
        self.last_k = None

    def search(self, query, k):
        self.last_k = k
        return self.ids[:k]


class FakeScorer:
    """Scores by a fixed table keyed on a substring of the passage."""

    def __init__(self, table):
        self.table = table
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        return [next((v for key, v in self.table.items() if key in p), 0.0) for p in passages]


def rr(store, ids, table, **kw):
    return RerankRetriever(store, FakeBase(ids), FakeScorer(table), **kw)


# --- the constant ---------------------------------------------------------------


def test_top_n_is_fixed_at_one_hundred():
    # Guards against silent tuning; 100 is the BEIR reranking depth, fixed a priori.
    assert DEFAULT_TOP_N == 100


# --- reordering behaviour -------------------------------------------------------


def test_reranker_reorders_the_base_ranking(store):
    # base puts a1 first; the scorer prefers topk
    r = rr(store, ["a1", "t1", "n1"], {"topk": 9.0, "Adam": 1.0, "nn": 0.5})
    assert r.search("q", 3) == ["t1", "a1", "n1"]


def test_reranker_can_promote_the_last_candidate_to_first(store):
    r = rr(store, ["a1", "t1", "n1"], {"nn": 9.0})
    assert r.search("q", 1) == ["n1"]


def test_reranker_only_reorders_what_the_base_returned(store):
    r = rr(store, ["a1", "t1"], {"nn": 9.0, "Adam": 1.0, "topk": 2.0})
    assert set(r.search("q", 5)) == {"a1", "t1"}


def test_ties_keep_the_base_order(store):
    r = rr(store, ["t1", "a1", "n1"], {})  # every score 0.0
    assert r.search("q", 3) == ["t1", "a1", "n1"]


def test_search_respects_k(store):
    r = rr(store, ["a1", "t1", "n1"], {"topk": 9.0})
    assert len(r.search("q", 1)) == 1
    assert len(r.search("q", 2)) == 2


def test_empty_and_degenerate_queries(store):
    r = rr(store, ["a1"], {"Adam": 1.0})
    assert r.search("", 5) == []
    assert r.search("   ", 5) == []
    assert r.search("q", 0) == []
    assert r.search("q", -1) == []


def test_empty_base_result_is_handled(store):
    assert rr(store, [], {"Adam": 1.0}).search("q", 5) == []


def test_unknown_chunk_ids_are_dropped_before_scoring(store):
    r = rr(store, ["ghost", "a1"], {"Adam": 1.0})
    assert r.search("q", 5) == ["a1"]
    assert len(r.scorer.calls[0][1]) == 1


def test_conforms_to_the_retriever_protocol(store):
    assert isinstance(rr(store, ["a1"], {}), Retriever)


# --- composition and depth ------------------------------------------------------


def test_base_is_asked_for_top_n_not_k(store):
    r = rr(store, ["a1", "t1", "n1"], {}, top_n=50)
    r.search("q", 3)
    assert r.base.last_k == 50


def test_depth_is_never_below_k(store):
    r = rr(store, ["a1", "t1", "n1"], {}, top_n=2)
    r.search("q", 7)
    assert r.base.last_k == 7


def test_scorer_sees_the_same_text_the_embedder_would(store):
    r = rr(store, ["a1"], {"Adam": 1.0})
    r.search("my query", 1)
    query, passages = r.scorer.calls[0]
    assert query == "my query"
    assert passages == [embedding_text(store.get("a1"))]
    assert "Adam" in passages[0] and "sec" in passages[0]


def test_latency_is_recorded_per_search(store):
    r = rr(store, ["a1", "t1"], {"Adam": 1.0})
    assert r.last_latency is None
    r.search("q", 2)
    assert r.last_latency is not None and r.last_latency >= 0.0


def test_composes_over_an_arbitrary_base(store):
    # any object with search(query, k) works, which is what lets it wrap hybrid
    class Reversed:
        def search(self, query, k):
            return ["n1", "t1", "a1"][:k]

    r = RerankRetriever(store, Reversed(), FakeScorer({"Adam": 5.0}), top_n=10)
    assert r.search("q", 1) == ["a1"]


def test_scorer_is_called_once_per_search(store):
    r = rr(store, ["a1", "t1", "n1"], {"topk": 1.0})
    r.search("q", 3)
    assert len(r.scorer.calls) == 1
