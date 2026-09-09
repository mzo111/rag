"""Cross-encoder reranking over the top-N candidates of any base retriever.

A bi-encoder scores a query and a page independently and compares two vectors, so it never
sees them together. A cross-encoder reads the pair jointly and can attend from the query onto
the passage, which is why it ranks better and why it cannot be used to search a corpus - it
costs one forward pass per candidate. The standard arrangement, used here, is two stage: a
cheap retriever proposes N candidates, the cross-encoder reorders them.

This wraps *any* object with ``search(query, k)``, so it composes with BM25, dense and the RRF
hybrid without any of them changing.

Model: ``Alibaba-NLP/gte-reranker-modernbert-base`` (149M params, 8192 token context,
Apache-2.0). Chosen on size and published reranking benchmarks, weighted toward code:

| model                        | params | ctx  | licence      | BEIR  | code retrieval       |
|------------------------------|--------|------|--------------|-------|----------------------|
| ms-marco-MiniLM-L6-v2        |  22.7M |  512 | Apache-2.0   |   -   | none published       |
| jina-reranker-v2-base-multi  |   278M | 1024 | CC-BY-NC-4.0 | 53.17 | CSN MRR@10 71.36 (3) |
| gte-reranker-modernbert-base |   149M | 8192 | Apache-2.0   | 56.73 | CoIR 79.99 (20 task) |

- **Best code retrieval of the three.** CoIR 79.99 over 20 code tasks, against jina's
  CodeSearchNet MRR@10 71.36 over 3. This corpus is PyTorch API reference - signatures, code
  blocks, dotted identifiers - so code reranking is the benchmark that resembles the workload.
- **Best BEIR too** (56.73 vs 53.17) at roughly half jina's size.
- **Apache-2.0.** jina-reranker-v2 is CC-BY-NC-4.0, i.e. non-commercial, which disqualifies it
  for anything but a research prototype regardless of its scores.
- ``ms-marco-MiniLM-L6-v2`` is the popular default and the wrong tool here: trained on MS MARCO
  web passages, no published code-retrieval numbers at all, and a 512-token limit.
- Same ModernBERT backbone as the embedder in :mod:`retrieval.dense`, so both stages tokenize
  code-dense text the same way.

**N is fixed at 100 a priori and is not searched.** 100 is the standard reranking depth in the
BEIR protocol, and it is already the fusion depth of :mod:`retrieval.hybrid`, so both stages
read the same candidate horizon. Searching N against this eval set would fit pooling artifacts
and judgment noise - the same argument that keeps the RRF constant at its published value.

Reranking inherits whatever corpus-quality policy the base retriever applies: BM25, dense and
hybrid all collapse alias pages and drop stubs before this sees a candidate, so the reranker
reorders one chunk per page and cannot reintroduce a duplicate.

Usage:
    python -m retrieval.rerank "how do I clip gradients" [--base hybrid] [--top-n 100]
    python -m retrieval.rerank --benchmark          # per-query latency and model size
"""

from __future__ import annotations

import sys
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from corpus.store import Store
from retrieval.dense import embedding_text

DEFAULT_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
# Standard BEIR reranking depth, and the fusion depth of retrieval.hybrid. Fixed, not searched.
DEFAULT_TOP_N = 100
DEFAULT_BATCH_SIZE = 32


@runtime_checkable
class Scorer(Protocol):
    """Anything that scores (query, passage) pairs. Higher is more relevant."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderScorer:
    """:mod:`sentence_transformers` CrossEncoder, loaded lazily and once.

    The import sits inside ``__init__`` so that importing this module - and therefore running
    the test suite or ``eval.run --help`` - does not require torch, matching the convention in
    :mod:`retrieval.dense` and :func:`corpus.chunk.token_counter`.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_length: int | None = None,
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "reranking needs sentence-transformers, which is not in requirements.txt "
                "because it pulls in torch. Install it with:\n"
                "    pip install -r requirements-embed.txt"
            ) from exc
        from retrieval.dense import _relax_triton_requirement

        # Same ModernBERT triton/rotary issue as the embedder; see retrieval.dense.
        self.eager_fallback = _relax_triton_requirement()
        self.name = model
        self.batch_size = batch_size
        kwargs = {"max_length": max_length} if max_length else {}
        self.model = CrossEncoder(model, device=device, **kwargs)

    @property
    def device(self) -> str:
        return str(self.model.model.device)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return [float(s) for s in scores]


class RerankRetriever:
    """Reorders a base retriever's top N with a cross-encoder. Implements ``search(query, k)``.

    ``scorer`` is injectable so the composition can be tested without loading a model.
    """

    def __init__(
        self,
        store: Store,
        base,
        scorer: Scorer | None = None,
        *,
        top_n: int = DEFAULT_TOP_N,
        model: str = DEFAULT_MODEL,
    ):
        self.store = store
        self.base = base
        self.top_n = top_n
        self.scorer = scorer if scorer is not None else CrossEncoderScorer(model)
        self.last_latency: float | None = None

    def _texts(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        """chunk_id -> the text shown to the cross-encoder.

        Reuses :func:`retrieval.dense.embedding_text` so first and second stage read a chunk
        the same way: title, heading path, body. A chunk from mid-page is often
        uninterpretable without its heading path, and that is as true for a cross-encoder as
        for an embedder.
        """
        out: dict[str, str] = {}
        for cid in dict.fromkeys(chunk_ids):
            chunk = self.store.get(cid)
            if chunk is not None:
                out[cid] = embedding_text(chunk)
        return out

    def search(self, query: str, k: int) -> list[str]:
        if not query.strip() or k <= 0:
            return []
        candidates = list(self.base.search(query, max(self.top_n, k)))
        if not candidates:
            return []
        texts = self._texts(candidates)
        scored_ids = [cid for cid in candidates if cid in texts]
        if not scored_ids:
            return []
        started = time.perf_counter()
        scores = self.scorer.score(query, [texts[cid] for cid in scored_ids])
        self.last_latency = time.perf_counter() - started
        # Stable sort on -score keeps the base ordering as the tie-break.
        order = sorted(range(len(scored_ids)), key=lambda i: -scores[i])
        return [scored_ids[i] for i in order[:k]]


def _make_base(name: str, store: Store):
    if name == "bm25":
        from retrieval.bm25 import Bm25Retriever

        return Bm25Retriever(store)
    if name == "dense":
        from retrieval.dense import DenseRetriever

        return DenseRetriever.load(store)
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever(store)
    raise SystemExit(f"unknown base retriever: {name}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import statistics

    ap = argparse.ArgumentParser(description="Cross-encoder reranking over data/corpus.db")
    ap.add_argument("query", nargs="*")
    ap.add_argument("--db", default="data/corpus.db")
    ap.add_argument("--base", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--benchmark", action="store_true", help="latency over eval/queries.yaml")
    args = ap.parse_args(argv)

    with Store(args.db) as store:
        scorer = CrossEncoderScorer(args.model)
        r = RerankRetriever(store, _make_base(args.base, store), scorer, top_n=args.top_n)

        if args.benchmark:
            from eval.run import load_queries

            queries = load_queries()
            params = scorer.n_parameters()
            print(f"model: {scorer.name}")
            print(f"  parameters: {params / 1e6:.1f}M   fp32 size: {params * 4 / 1024**2:.0f} MiB")
            print(f"  device: {scorer.device}   top_n: {r.top_n}   batch: {scorer.batch_size}")
            r.search(queries[0].query, args.k)  # warm up kernels, excluded from timings
            total, rerank_only, n_cand = [], [], []
            for q in queries:
                t0 = time.perf_counter()
                hits = r.search(q.query, args.k)
                total.append(time.perf_counter() - t0)
                rerank_only.append(r.last_latency or 0.0)
                n_cand.append(len(hits))
            print(f"\n{len(queries)} queries, k={args.k}")
            print(
                f"  end-to-end per query: mean {statistics.fmean(total) * 1000:.0f} ms   "
                f"median {statistics.median(total) * 1000:.0f} ms   "
                f"max {max(total) * 1000:.0f} ms"
            )
            print(
                f"  cross-encoder only  : mean {statistics.fmean(rerank_only) * 1000:.0f} ms   "
                f"median {statistics.median(rerank_only) * 1000:.0f} ms"
            )
            print(
                f"  throughput: {r.top_n / statistics.fmean(rerank_only):.0f} pairs/s "
                f"at N={r.top_n}"
            )
            return 0

        if not args.query:
            ap.error("give a query, or --benchmark")
        q = " ".join(args.query)
        print(f"query: {q}\nbase: {args.base}  N={r.top_n}  model={scorer.name}\n")
        for i, cid in enumerate(r.search(q, args.k), 1):
            c = store.get(cid)
            path = " > ".join(c.heading_path[-2:])
            print(f"{i:>3}. {c.url.replace('https://docs.pytorch.org/', '')}")
            print(f"     {path}  [{c.n_tokens} tok]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
