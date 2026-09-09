"""Dense retrieval: sentence-transformer embeddings over the chunk store, cosine similarity.

This is the semantic counterpart to :mod:`retrieval.bm25`. BM25 matches tokens; this matches
meaning, so the two fail on different queries. That difference is the point: the judging pool
in ``eval/pool.yaml`` was built from BM25 alone, and a pool built by one retriever cannot
measure another. Adding a retriever of a different kind is what makes re-pooling worthwhile.

Model: ``Alibaba-NLP/gte-modernbert-base`` (149M params, 768 dim, 8192 token context,
Apache-2.0, no ``trust_remote_code``). Chosen on size and retrieval benchmarks:

- 55.33 BEIR nDCG@10 at 149M params, above ``bge-base-en-v1.5`` (53.2) and level with
  ``snowflake-arctic-embed-m-v2.0`` (~55.5) at roughly half that model's size.
- 79.31 nDCG@10 on CoIR, a 20-dataset *code* retrieval benchmark. This corpus is API
  reference full of signatures and code blocks, so code retrieval is the benchmark that
  resembles the workload; the general BEIR average understates it.
- The 8192-token context removes a truncation question rather than answering it. Chunks are
  capped at 512 ``cl100k_base`` tokens (:mod:`corpus.chunk`), but this model tokenizes with
  its own vocabulary and code-dense text expands under subword tokenization, so a 512-token
  model could silently truncate the tail of the distribution.

The obvious default, ``all-MiniLM-L6-v2``, is rejected on this corpus specifically: it was
trained at 128 tokens and truncates at 256 word pieces, while 61.7% of these chunks are
longer than 128 tokens (mean 191.6). It is the most downloaded model and the wrong one here.

Encoding is symmetric - this model takes no query prefix, unlike E5 (``query:``) or BGE
(``Represent this sentence...``). :class:`Encoder` still passes ``is_query`` so a model that
needs a prefix can be dropped in without changing the retriever.

Index format. ``<dir>/<model-slug>.npy`` holds float32 rows, L2-normalized, where row *i*
belongs to ``chunk_ids[i]``; ``<dir>/<model-slug>.json`` holds the model name, dimension, the
ordered chunk ids, build stats and a fingerprint of the corpus it was built from. Because the
rows are normalized, cosine similarity is a plain dot product, so search is one matrix-vector
multiply over ~35 MiB. An ANN index would add a dependency and an approximation to a search
that is already exact and takes milliseconds.

The two corpus-quality behaviours of :mod:`retrieval.bm25` are mirrored here with the same
defaults, so the two retrievers are comparable: deprecated stubs are excluded and alias pages
are collapsed. Turning them off for one retriever and not the other would compare the corpus
cleanup, not the retrievers.

Usage:
    python -m retrieval.dense build [--db data/corpus.db] [--model NAME] [--batch-size N]
    python -m retrieval.dense "how do I clip gradients"
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from corpus.chunk import Chunk
from corpus.store import Store

DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"
DEFAULT_INDEX_DIR = Path("data/dense")
DEFAULT_BATCH_SIZE = 64
# Over-fetch before stub/alias filtering, matching retrieval.bm25.
CANDIDATE_POOL = 500


def _relax_triton_requirement() -> bool:
    """Fall back to aten kernels when triton cannot build, returning whether it did.

    On CUDA, torch routes some ops (``bmm_outer_product``, reached from ModernBERT's rotary
    embedding) to triton kernels, and triton JIT-compiles its driver with a *system C
    compiler*. Machines without a toolchain therefore fail at the first forward pass with
    "Failed to find C compiler" - after the model has loaded, which makes it look like a
    model problem rather than an environment one.

    Dropping the triton overrides falls back to aten. The kernels are mathematically
    equivalent, so embeddings are unchanged; only throughput is affected. When a compiler is
    present nothing is touched and the fast path stands.
    """
    if shutil.which("cc") or shutil.which("gcc"):
        return False
    try:  # private torch API - never break encoding over a missing optimisation
        from torch._native import registry

        registry.deregister_op_overrides(disable_dsl_names="triton")
    except Exception:  # pragma: no cover - depends on the torch build
        return False
    return True


def model_slug(model: str) -> str:
    """Filesystem-safe stem for a model name: Alibaba-NLP/gte-modernbert-base -> ..."""
    return model.replace("/", "__").replace(":", "_")


def index_paths(model: str, directory: Path = DEFAULT_INDEX_DIR) -> tuple[Path, Path]:
    """(vectors .npy, metadata .json) for a model in ``directory``."""
    stem = model_slug(model)
    return directory / f"{stem}.npy", directory / f"{stem}.json"


def embedding_text(chunk: Chunk) -> str:
    """Text handed to the encoder: title, heading path, then body.

    The dense analogue of the BM25 column weighting in :mod:`retrieval.bm25`, which scores
    title and heading_path above text. A chunk from the middle of a page is often
    uninterpretable alone ("It defaults to 1e-8." belongs to whichever API owns it), so the
    heading path is prepended rather than left to the body text to imply.
    """
    parts = [chunk.title, " > ".join(chunk.heading_path), chunk.text]
    return "\n".join(p for p in parts if p)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so that a dot product is cosine similarity.

    Zero rows would divide by zero; they are left as zeros, which scores 0 against every
    query rather than producing NaNs that poison an argsort.
    """
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0.0, 1.0, norms)


def corpus_fingerprint(store: Store) -> str:
    """Digest of (chunk_id, content_hash) over the whole store, in index order.

    Lets :meth:`DenseIndex.load` notice that the corpus was rebuilt after the index was, so a
    stale index is reported instead of silently scoring against the wrong vectors.
    """
    h = hashlib.sha256()
    for row in store.conn.execute("SELECT chunk_id, content_hash FROM chunks ORDER BY rowid"):
        h.update(row["chunk_id"].encode())
        h.update(row["content_hash"].encode())
    return h.hexdigest()


@runtime_checkable
class Encoder(Protocol):
    """Anything that turns text into unit-length row vectors."""

    dim: int

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """:mod:`sentence_transformers` wrapper. The model is loaded lazily and once.

    ``sentence_transformers`` is imported inside ``__init__`` rather than at module scope so
    that importing :mod:`retrieval.dense` - and therefore running the test suite or
    ``eval.run --help`` - does not require torch. The same lazy-import convention is used by
    :func:`corpus.chunk.token_counter`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "dense retrieval needs sentence-transformers, which is not in "
                "requirements.txt because it pulls in torch. Install it with:\n"
                "    pip install -r requirements-embed.txt"
            ) from exc
        self.name = model
        self.batch_size = batch_size
        self.eager_fallback = _relax_triton_requirement()
        self.model = SentenceTransformer(model, device=device)
        # renamed in sentence-transformers 5.x; support both spellings
        dim_of = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(dim_of())

    @property
    def device(self) -> str:
        return str(self.model.device)

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        """Embed ``texts`` into unit-length float32 rows.

        ``is_query`` is unused: gte-modernbert-base encodes queries and passages the same
        way. It stays in the signature so a prefixed model can be substituted here alone.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalized below, so every path agrees
            show_progress_bar=len(texts) > 1000,
        )
        return normalize(vectors)


@dataclass
class DenseIndex:
    """Unit-length chunk vectors plus the chunk ids they belong to, row for row."""

    vectors: np.ndarray  # (n, dim) float32, L2-normalized
    chunk_ids: list[str]
    model: str
    meta: dict

    def __post_init__(self) -> None:
        if len(self.chunk_ids) != self.vectors.shape[0]:
            raise ValueError(
                f"index is inconsistent: {self.vectors.shape[0]} vectors but "
                f"{len(self.chunk_ids)} chunk ids"
            )

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(
        cls,
        store: Store,
        encoder: Encoder,
        *,
        model_name: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> DenseIndex:
        """Embed every chunk in ``store``. Returns an index carrying its own build stats."""
        chunks = list(store.iter_chunks())
        texts = [embedding_text(c) for c in chunks]
        started = time.perf_counter()
        vectors = _encode_batched(encoder, texts, batch_size)
        elapsed = time.perf_counter() - started
        name = model_name or getattr(encoder, "name", encoder.__class__.__name__)
        return cls(
            vectors=vectors,
            chunk_ids=[c.chunk_id for c in chunks],
            model=name,
            meta={
                "model": name,
                "dim": int(vectors.shape[1]) if len(chunks) else getattr(encoder, "dim", 0),
                "n_chunks": len(chunks),
                "n_tokens": sum(c.n_tokens for c in chunks),
                "batch_size": batch_size,
                "embed_seconds": round(elapsed, 3),
                "chunks_per_second": round(len(chunks) / elapsed, 1) if elapsed > 0 else 0.0,
                "device": getattr(encoder, "device", "unknown"),
                "fingerprint": corpus_fingerprint(store),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    def save(self, directory: Path = DEFAULT_INDEX_DIR) -> dict:
        """Write vectors and metadata. Returns the metadata, with on-disk sizes added."""
        vec_path, meta_path = index_paths(self.model, directory)
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vec_path, self.vectors.astype(np.float32, copy=False))
        meta = dict(self.meta)
        meta["chunk_ids"] = self.chunk_ids
        meta["vectors_bytes"] = vec_path.stat().st_size
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        meta["meta_bytes"] = meta_path.stat().st_size
        meta["total_bytes"] = meta["vectors_bytes"] + meta["meta_bytes"]
        # Rewrite so the recorded sizes are in the file too (the size of the size field
        # cannot change the vector size, and meta_bytes is reported from the first write).
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        self.meta = {k: v for k, v in meta.items() if k != "chunk_ids"}
        return meta

    @classmethod
    def load(
        cls,
        model: str = DEFAULT_MODEL,
        directory: Path = DEFAULT_INDEX_DIR,
        *,
        store: Store | None = None,
        mmap: bool = True,
    ) -> DenseIndex:
        """Load an index from disk, warning if it was built from a different corpus."""
        vec_path, meta_path = index_paths(model, directory)
        if not vec_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"no dense index for {model!r} at {vec_path}. Build one with:\n"
                f"    python -m retrieval.dense build --model {model}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path, mmap_mode="r" if mmap else None)
        chunk_ids = list(meta.pop("chunk_ids", []))
        index = cls(vectors=vectors, chunk_ids=chunk_ids, model=meta.get("model", model), meta=meta)
        if store is not None:
            index.check_fingerprint(store)
        return index

    def check_fingerprint(self, store: Store) -> bool:
        """Warn on stderr if ``store`` has changed since this index was built."""
        expected = self.meta.get("fingerprint")
        if not expected:
            return True
        if expected == corpus_fingerprint(store):
            return True
        print(
            f"warning: dense index for {self.model!r} was built from a different corpus "
            f"({self.meta.get('n_chunks')} chunks, {self.meta.get('built_at')}); "
            "rebuild it with `python -m retrieval.dense build`",
            file=sys.stderr,
        )
        return False

    def search(self, query_vector: np.ndarray, n: int) -> list[tuple[str, float]]:
        """Top ``n`` (chunk_id, cosine) pairs, best first.

        ``argpartition`` finds the top n in O(len(index)) and only the n survivors are
        sorted, which matters because n here is the candidate pool, not k.
        """
        if n <= 0 or len(self) == 0:
            return []
        q = normalize(query_vector)[0]
        scores = np.asarray(self.vectors, dtype=np.float32) @ q
        n = min(n, scores.shape[0])
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]


def _encode_batched(encoder: Encoder, texts: Sequence[str], batch_size: int) -> np.ndarray:
    """Encode in batches, so a fake encoder in tests need not know about batching.

    SentenceTransformerEncoder batches internally too; this keeps peak memory bounded for
    encoders that do not.
    """
    if not texts:
        return np.zeros((0, getattr(encoder, "dim", 0)), dtype=np.float32)
    chunks_out = [
        encoder.encode(texts[i : i + batch_size]) for i in range(0, len(texts), max(batch_size, 1))
    ]
    return np.vstack(chunks_out).astype(np.float32, copy=False)


class DenseRetriever:
    """Cosine search over a :class:`DenseIndex`. Implements ``search(query, k)``.

    Stub exclusion and alias collapsing default to on, exactly as in
    :class:`retrieval.bm25.Bm25Retriever`, so the two retrievers can be compared to each
    other rather than to each other's corpus handling.
    """

    def __init__(
        self,
        store: Store,
        index: DenseIndex,
        encoder: Encoder,
        *,
        exclude_stubs: bool = True,
        collapse_aliases: bool = True,
        candidate_pool: int = CANDIDATE_POOL,
    ):
        self.store = store
        self.index = index
        self.encoder = encoder
        self.exclude_stubs = exclude_stubs
        self.collapse_aliases = collapse_aliases
        self.candidate_pool = candidate_pool

    @classmethod
    def load(
        cls,
        store: Store,
        *,
        model: str = DEFAULT_MODEL,
        directory: Path = DEFAULT_INDEX_DIR,
        encoder: Encoder | None = None,
        **kwargs,
    ) -> DenseRetriever:
        """Load the on-disk index and the model that produced it."""
        index = DenseIndex.load(model, directory, store=store)
        return cls(store, index, encoder or SentenceTransformerEncoder(index.model), **kwargs)

    def _pages(self, chunk_ids: Iterable[str]) -> dict[str, tuple[str, str]]:
        """chunk_id -> (canonical page url, page status), for filtering and collapsing."""
        ids = list(chunk_ids)
        out: dict[str, tuple[str, str]] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            rows = self.store.conn.execute(
                "SELECT c.chunk_id, c.url, p.canonical_url, p.status FROM chunks c "
                f"JOIN pages p ON p.url = c.url WHERE c.chunk_id IN ({placeholders})",
                batch,
            )
            for row in rows:
                out[row["chunk_id"]] = (row["canonical_url"] or row["url"], row["status"])
        return out

    def search(self, query: str, k: int) -> list[str]:
        if not query.strip() or k <= 0:
            return []
        qvec = self.encoder.encode([query], is_query=True)
        pool = max(self.candidate_pool, k)
        scored = self.index.search(qvec, pool)
        if not scored:
            return []

        pages = self._pages(cid for cid, _ in scored)
        out: list[str] = []
        seen_pages: set[str] = set()
        for cid, _score in scored:
            page, status = pages.get(cid, ("", "ok"))
            if self.exclude_stubs and status != "ok":
                continue
            if self.collapse_aliases:
                if page in seen_pages:
                    continue
                seen_pages.add(page)
            out.append(cid)
            if len(out) >= k:
                break
        return out


def _human_bytes(n: int) -> str:
    mib = n / (1024 * 1024)
    return f"{mib:.1f} MiB" if mib >= 1 else f"{n / 1024:.1f} KiB"


def _build(args) -> int:
    with Store(args.db) as store:
        if store.count() == 0:
            print(f"{args.db} has no chunks; run corpus.fetch, corpus.chunk, corpus.store first")
            return 1
        print(f"loading {args.model} ...")
        encoder = SentenceTransformerEncoder(args.model, device=args.device)
        print(f"device: {encoder.device}   dim: {encoder.dim}   chunks: {store.count()}")
        index = DenseIndex.build(store, encoder, model_name=args.model, batch_size=args.batch_size)
        meta = index.save(args.out)

    vec_path, meta_path = index_paths(args.model, args.out)
    secs = meta["embed_seconds"]
    print(
        f"\nembedded {meta['n_chunks']} chunks ({meta['n_tokens']} tokens) in {secs:.1f}s "
        f"= {meta['chunks_per_second']} chunks/s, {meta['n_tokens'] / secs:,.0f} tok/s"
    )
    print(
        f"index: {meta['n_chunks']} x {meta['dim']} float32 -> "
        f"{_human_bytes(meta['vectors_bytes'])} {vec_path}"
    )
    print(f"       metadata {_human_bytes(meta['meta_bytes'])} {meta_path}")
    print(f"total on disk: {_human_bytes(meta['total_bytes'])}")
    return 0


def _search(args) -> int:
    with Store(args.db) as store:
        retriever = DenseRetriever.load(store, model=args.model, directory=args.out)
        q = " ".join(args.query)
        print(f"query: {q}\nmodel: {retriever.index.model}\n")
        for i, cid in enumerate(retriever.search(q, args.k), 1):
            c = store.get(cid)
            path = " > ".join(c.heading_path[-2:])
            print(f"{i:>3}. {c.url.replace('https://docs.pytorch.org/', '')}")
            print(f"     {path}  [{c.n_tokens} tok]")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Dense retrieval index over data/corpus.db")
    ap.add_argument("query", nargs="*", help="search the index (omit when using `build`)")
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args(argv)

    if args.query and args.query[0] == "build":
        args.query = args.query[1:]
        if args.query:
            ap.error("`build` takes no query")
        return _build(args)
    if not args.query:
        ap.error("give a query, or `build` to construct the index")
    return _search(args)


if __name__ == "__main__":
    sys.exit(main())
