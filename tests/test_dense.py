import hashlib
import re

import numpy as np
import pytest

from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import Retriever
from retrieval.dense import (
    DenseIndex,
    DenseRetriever,
    corpus_fingerprint,
    embedding_text,
    index_paths,
    model_slug,
    normalize,
)

P_CANON = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P_OTHER = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
P_STUB = "https://docs.pytorch.org/tutorials/beginner/old_adam_tutorial.html"

TOKEN = re.compile(r"[a-z0-9_]+")


class HashingEncoder:
    """Deterministic bag-of-words encoder: no torch, no network, real cosine geometry.

    Hashes each token into one of ``dim`` buckets and counts it. Two texts sharing tokens
    get a high cosine, so ranking assertions test the retriever rather than a mock.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.name = "hashing-test-encoder"
        self.device = "cpu"
        self.calls: list[tuple[int, bool]] = []

    def encode(self, texts, *, is_query: bool = False) -> np.ndarray:
        self.calls.append((len(texts), is_query))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in TOKEN.findall(text.lower()):
                bucket = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % self.dim
                out[i, bucket] += 1.0
        return normalize(out)


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


@pytest.fixture
def encoder():
    return HashingEncoder()


@pytest.fixture
def index(store, encoder):
    return DenseIndex.build(store, encoder)


@pytest.fixture
def retriever(store, index, encoder):
    return DenseRetriever(store, index, encoder)


# --- text composition -----------------------------------------------------------


def test_embedding_text_includes_title_heading_path_and_body():
    c = Chunk("c", "u", "", "Adam", ["Optim", "Adam"], 0, "lr defaults to 0.001", 5, "h")
    text = embedding_text(c)
    assert "Adam" in text
    assert "Optim > Adam" in text
    assert "lr defaults to 0.001" in text


def test_embedding_text_skips_empty_parts():
    c = Chunk("c", "u", "", "", [], 0, "body only", 2, "h")
    assert embedding_text(c) == "body only"


# --- normalization --------------------------------------------------------------


def test_normalize_gives_unit_rows():
    v = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_normalize_leaves_zero_rows_alone_without_nan():
    v = normalize(np.zeros((2, 3)))
    assert not np.isnan(v).any()
    assert (v == 0).all()


def test_normalize_promotes_a_single_vector_to_one_row():
    assert normalize(np.array([3.0, 4.0])).shape == (1, 2)


# --- index building -------------------------------------------------------------


def test_build_produces_one_unit_row_per_chunk(store, index, encoder):
    assert len(index) == store.count() == 4
    assert index.vectors.shape == (4, encoder.dim)
    assert index.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)


def test_build_records_stats_and_fingerprint(store, index):
    assert index.meta["n_chunks"] == 4
    assert index.meta["fingerprint"] == corpus_fingerprint(store)
    assert index.meta["embed_seconds"] >= 0
    assert index.meta["device"] == "cpu"


def test_build_rows_line_up_with_chunk_ids(store, index, encoder):
    # row i must be the embedding of chunk_ids[i], not merely the right multiset of rows
    for i, cid in enumerate(index.chunk_ids):
        expected = encoder.encode([embedding_text(store.get(cid))])[0]
        assert np.allclose(index.vectors[i], expected)


def test_index_rejects_mismatched_ids_and_vectors():
    with pytest.raises(ValueError, match="inconsistent"):
        DenseIndex(np.zeros((2, 4), dtype=np.float32), ["only-one"], "m", {})


# --- persistence ----------------------------------------------------------------


def test_save_load_round_trip(index, tmp_path):
    meta = index.save(tmp_path)
    loaded = DenseIndex.load(index.model, tmp_path)
    assert loaded.chunk_ids == index.chunk_ids
    assert loaded.model == index.model
    assert np.allclose(np.asarray(loaded.vectors), index.vectors)
    assert meta["vectors_bytes"] > 0
    assert meta["total_bytes"] == meta["vectors_bytes"] + meta["meta_bytes"]


def test_saved_files_are_named_after_the_model(index, tmp_path):
    index.save(tmp_path)
    vec_path, meta_path = index_paths(index.model, tmp_path)
    assert vec_path.exists() and meta_path.exists()
    assert "/" not in vec_path.name


def test_model_slug_is_filesystem_safe():
    assert model_slug("Alibaba-NLP/gte-modernbert-base") == "Alibaba-NLP__gte-modernbert-base"


def test_load_missing_index_names_the_build_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m retrieval.dense build"):
        DenseIndex.load("no-such-model", tmp_path)


def test_load_warns_when_the_corpus_changed(store, index, tmp_path, capsys):
    index.save(tmp_path)
    store.load([chunk("n1", P_OTHER, "a newly added chunk", ordinal=1, title="torch.topk")])
    DenseIndex.load(index.model, tmp_path, store=store)
    assert "built from a different corpus" in capsys.readouterr().err


def test_load_is_quiet_when_the_corpus_matches(store, index, tmp_path, capsys):
    index.save(tmp_path)
    loaded = DenseIndex.load(index.model, tmp_path, store=store)
    assert capsys.readouterr().err == ""
    assert loaded.check_fingerprint(store) is True


# --- search ---------------------------------------------------------------------


def test_conforms_to_the_retriever_protocol(retriever):
    assert isinstance(retriever, Retriever)


def test_search_ranks_the_semantically_closest_chunk_first(retriever):
    assert retriever.search("topk largest elements", 3)[0] == "t1"
    assert retriever.search("Adam optimizer params lr eps", 3)[0] in {"a1", "a2"}


def test_search_returns_chunk_ids_that_exist_in_the_store(store, retriever):
    for cid in retriever.search("torch", 5):
        assert store.get(cid) is not None


def test_search_respects_k(retriever):
    assert len(retriever.search("torch", 1)) == 1
    assert len(retriever.search("torch", 2)) == 2


def test_search_handles_empty_and_degenerate_queries(retriever):
    assert retriever.search("", 5) == []
    assert retriever.search("   ", 5) == []
    assert retriever.search("torch", 0) == []
    assert retriever.search("torch", -1) == []


def test_search_marks_the_query_as_a_query(retriever, encoder):
    retriever.search("torch", 3)
    assert (1, True) in encoder.calls


def test_index_search_returns_descending_scores(index, encoder):
    q = encoder.encode(["torch optim Adam"])
    scored = index.search(q, 4)
    assert [s for _, s in scored] == sorted((s for _, s in scored), reverse=True)


def test_index_search_caps_n_at_the_index_size(index, encoder):
    assert len(index.search(encoder.encode(["torch"]), 999)) == 4


def test_index_search_of_an_empty_index_is_empty(encoder):
    empty = DenseIndex(np.zeros((0, encoder.dim), dtype=np.float32), [], "m", {})
    assert empty.search(encoder.encode(["torch"]), 5) == []


# --- corpus quality behaviour (mirrors retrieval.bm25) --------------------------


def test_search_excludes_stub_pages_by_default(retriever):
    assert "s1" not in retriever.search("deprecated redirecting Adam tutorial", 10)


def test_search_keeps_stubs_when_asked(store, index, encoder):
    r = DenseRetriever(store, index, encoder, exclude_stubs=False)
    assert "s1" in r.search("deprecated redirecting Adam tutorial", 10)


def test_search_collapses_alias_pages(retriever):
    hits = retriever.search("Adam params lr eps", 10)
    assert len({"a1", "a2"} & set(hits)) == 1


def test_search_keeps_aliases_when_asked(store, index, encoder):
    r = DenseRetriever(store, index, encoder, collapse_aliases=False)
    hits = r.search("Adam params lr eps", 10)
    assert {"a1", "a2"} <= set(hits)


def test_collapsing_uses_the_canonical_page_not_the_raw_url(store, retriever):
    # a1 and a2 live at different urls; only their shared canonical_url makes them one page
    assert store.canonical_map()[P_ALIAS] == P_CANON
    assert len({"a1", "a2"} & set(retriever.search("Adam params lr eps", 10))) == 1


# --- loading a retriever end to end ---------------------------------------------


def test_retriever_load_reads_the_index_from_disk(store, index, encoder, tmp_path):
    index.save(tmp_path)
    r = DenseRetriever.load(store, model=index.model, directory=tmp_path, encoder=encoder)
    assert r.search("topk largest elements", 3)[0] == "t1"
