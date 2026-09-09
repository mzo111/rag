import pytest

from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import Retriever
from retrieval.bm25 import Bm25Retriever, to_match_query

P_CANON = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P_OTHER = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
P_STUB = "https://docs.pytorch.org/tutorials/beginner/old_adam_tutorial.html"


def chunk(cid, url, text, ordinal=0, title="T"):
    return Chunk(cid, url, "", title, [title], ordinal, text, len(text.split()), cid)


@pytest.fixture
def store():
    s = Store(":memory:")
    s.load(
        [
            chunk(
                "a1", P_CANON, "class torch.optim.Adam(params, lr=0.001, eps=1e-08)", title="Adam"
            ),
            chunk(
                "a2",
                P_ALIAS,
                "class torch.optim.adam.Adam(params, lr=0.001, eps=1e-08)",
                title="Adam",
            ),
            chunk("t1", P_OTHER, "torch.topk returns the k largest elements", title="torch.topk"),
            chunk(
                "s1",
                P_STUB,
                "This tutorial was deprecated. Redirecting in 3 seconds Adam",
                title="Old",
            ),
        ]
    )
    s.apply_quality()
    yield s
    s.close()


# --- query translation ----------------------------------------------------------


def test_to_match_query_quotes_terms_and_drops_stopwords():
    assert (
        to_match_query("what does torch.topk return")
        == '"torch.topk" OR "torch" OR "topk" OR "return"'
    )
    assert to_match_query("the and of") == ""
    assert to_match_query("") == ""


def test_to_match_query_survives_fts_syntax():
    # bare OR/AND/NEAR and punctuation would be FTS5 syntax errors unquoted
    for q in [
        "reduce-overhead mode",
        "a OR b AND c",
        'quote " inside',
        "padding='same'",
        "*",
        "NEAR(x)",
    ]:
        m = to_match_query(q)
        assert '"' in m or m == "", q
        assert not m.startswith("OR"), q


def test_to_match_query_splits_dotted_names_without_duplicates():
    m = to_match_query("torch.nn.functional.cross_entropy")
    assert m.startswith('"torch.nn.functional.cross_entropy"')
    assert '"cross_entropy"' in m and '"functional"' in m
    assert m.count('"torch"') == 1


# --- search ---------------------------------------------------------------------


def test_search_ranks_matching_chunk_first(store):
    r = Bm25Retriever(store)
    assert isinstance(r, Retriever)
    assert r.search("torch.topk k largest", 5)[0] == "t1"
    assert r.search("Adam eps default", 5)[0] in {"a1", "a2"}


def test_search_handles_empty_and_nonmatching_queries(store):
    r = Bm25Retriever(store)
    assert r.search("the of and", 5) == []
    assert r.search("torch.topk", 0) == []
    assert r.search("zzzznotpresent", 5) == []


def test_search_excludes_stub_pages_by_default(store):
    assert "s1" not in Bm25Retriever(store).search("Adam", 10)
    assert "s1" in Bm25Retriever(store, exclude_stubs=False).search("Adam", 10)


def test_search_collapses_alias_pages(store):
    ids = Bm25Retriever(store).search("Adam params lr eps", 10)
    assert len([i for i in ids if i in {"a1", "a2"}]) == 1
    both = Bm25Retriever(store, collapse_aliases=False).search("Adam params lr eps", 10)
    assert len([i for i in both if i in {"a1", "a2"}]) == 2


def test_search_respects_k(store):
    assert len(Bm25Retriever(store, collapse_aliases=False).search("Adam torch topk", 2)) == 2


def test_weights_favour_title_matches(store):
    """A term in the title should outrank the same term buried in body text."""
    s = Store(":memory:")
    s.load(
        [
            chunk(
                "title_hit",
                "https://docs.pytorch.org/docs/2.14/a.html",
                "unrelated body text",
                title="einsum",
            ),
            chunk(
                "body_hit",
                "https://docs.pytorch.org/docs/2.14/b.html",
                "einsum appears here in body",
                title="Other",
            ),
        ]
    )
    s.apply_quality()
    assert Bm25Retriever(s, weights=(50.0, 1.0, 1.0)).search("einsum", 2)[0] == "title_hit"
    s.close()
