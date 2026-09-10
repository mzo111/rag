"""Generation stage: answer a query from a retriever's top-k chunks, with citations.

The retriever proposes, the model writes. Nothing here re-ranks or re-retrieves - this
stage is deliberately thin, so that a change in answer quality is attributable to the
retriever that fed it rather than to anything the generator did on its own.

**Everything that could be tuned is fixed a priori and asserted in tests**: the number of
chunks (:data:`K`), the prompt text (:mod:`generate.prompts`), temperature 0, the context
window and output cap, and the exact spelling of a refusal. None of them were chosen by
looking at a score. See :mod:`generate.faithfulness` for what is measured and what that
measurement does not cover.

Citations. The model is asked to mark every sentence with the passage it came from, as
``[C1]``. :func:`parse_answer` extracts those markers and maps them back to chunk ids, and
records any marker that points outside ``1..k`` as a *dangling* citation rather than
silently dropping it - a model that cites [C14] out of ten passages is telling you
something, and it should show up in the numbers instead of being cleaned away.

Refusal. When the passages do not answer the question the model must emit
:data:`~generate.prompts.REFUSAL_MARKER` and set ``refused``. Both are checked; a
disagreement between them is recorded as a malformed response, not resolved by guessing.

Usage:
    python -m generate.run --retriever bm25|dense|hybrid|rerank|agent [--offline]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.llm import DEFAULT_BASE_URL, DEFAULT_MODEL, OllamaClient, ResponseCache
from generate.prompts import ANSWER_PROMPT, REFUSAL_MARKER

# --- fixed a priori; asserted in tests/test_generate.py ---------------------------------
K = 10  # chunks of context, matching the eval harness's k so retrieval and generation agree
NUM_CTX = 8192  # ~5k tokens of passages plus the prompt, with headroom; never searched
NUM_PREDICT = 400  # output cap; 120-word answers plus JSON overhead fit well inside it
MAX_ANSWER_WORDS = 120  # stated in the prompt, measured in the results, never enforced
RETRIEVERS = ("bm25", "dense", "hybrid", "rerank", "agent")
DEFAULT_CACHE = Path(__file__).parent / "cache" / "generation.json"

CITATION = re.compile(r"\[C(\d+)\]")


@dataclass
class Answer:
    """One generated answer and everything measured about producing it."""

    query_id: str
    query: str
    retriever: str
    chunk_ids: list[str]
    refused: bool
    text: str
    cited: list[int] = field(default_factory=list)  # 1-based passage numbers, in order
    dangling: list[int] = field(default_factory=list)  # cited but outside 1..k
    malformed: str = ""  # non-empty when the response did not fit the schema
    model_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False

    @property
    def cited_chunk_ids(self) -> list[str]:
        """Chunk ids the answer cites, de-duplicated in citation order."""
        out: list[str] = []
        for n in self.cited:
            if 1 <= n <= len(self.chunk_ids):
                cid = self.chunk_ids[n - 1]
                if cid not in out:
                    out.append(cid)
        return out

    @property
    def words(self) -> int:
        return len(self.text.split())

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["cited_chunk_ids"] = self.cited_chunk_ids
        d["words"] = self.words
        return d


def format_context(chunks) -> str:
    """The passage block the model reads: one numbered entry per chunk, in rank order."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        heading = " > ".join(c.heading_path) if c.heading_path else ""
        head = f"[C{i}] {c.title}" + (f" — {heading}" if heading else "")
        parts.append(f"{head}\n{c.text}")
    return "\n\n".join(parts)


def parse_answer(raw: str, k: int) -> tuple[bool, str, list[int], list[int], str]:
    """``(refused, text, cited, dangling, malformed)`` from the model's JSON.

    Refusal must agree with itself: ``refused`` true and the marker present, or neither.
    A response that says one and not the other is reported as malformed instead of being
    coerced, because guessing which half to believe would hide exactly the failure the
    refusal rate is meant to count.
    """
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, "", [], [], f"invalid JSON: {exc.msg}"
    if not isinstance(body, dict):
        return False, "", [], [], f"expected a JSON object, got {type(body).__name__}"

    text = body.get("answer")
    if not isinstance(text, str):
        return False, "", [], [], "missing or non-string 'answer'"
    text = text.strip()

    flag = body.get("refused")
    if not isinstance(flag, bool):
        return False, text, [], [], "missing or non-boolean 'refused'"
    marker = REFUSAL_MARKER in text
    if flag != marker:
        return flag, text, [], [], f"refused={flag} but marker present={marker}"
    if flag:
        return True, text, [], [], ""

    nums = [int(n) for n in CITATION.findall(text)]
    cited = [n for n in nums if 1 <= n <= k]
    dangling = [n for n in nums if not 1 <= n <= k]
    return False, text, cited, dangling, ""


class Generator:
    """Answers a query from one retriever's top-k chunks."""

    def __init__(self, store, retriever, name: str, client: OllamaClient, *, k: int = K):
        self.store = store
        self.retriever = retriever
        self.name = name
        self.client = client
        self.k = k

    def context_for(self, query: str) -> tuple[list[str], list, float]:
        """Retrieve, then fetch the chunk rows. Retrieval time is always measured live."""
        started = time.perf_counter()
        ids = list(self.retriever.search(query, self.k))[: self.k]
        elapsed = time.perf_counter() - started
        chunks = [c for c in (self.store.get(i) for i in ids) if c is not None]
        return ids[: len(chunks)], chunks, elapsed

    def answer(self, query_id: str, query: str) -> Answer:
        ids, chunks, retrieval_seconds = self.context_for(query)
        return self.answer_from(query_id, query, ids, chunks, retrieval_seconds)

    def answer_from(
        self,
        query_id: str,
        query: str,
        ids: list[str],
        chunks: list,
        retrieval_seconds: float = 0.0,
        *,
        tag: str = "",
    ) -> Answer:
        """Answer from passages supplied by the caller.

        ``tag`` widens the cache key so a run over deliberately mismatched context (see
        :mod:`generate.faithfulness`) cannot collide with the ordinary run for the same
        query and retriever.
        """
        prompt = ANSWER_PROMPT.format(
            context=format_context(chunks), query=query, refusal=REFUSAL_MARKER
        )
        response = self.client.generate(
            prompt,
            query=query,
            retriever=f"{self.name}:{tag}" if tag else self.name,
            options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
        )
        refused, text, cited, dangling, malformed = parse_answer(response.text, len(chunks))
        return Answer(
            query_id=query_id,
            query=query,
            retriever=self.name,
            chunk_ids=list(ids),
            refused=refused,
            text=text,
            cited=cited,
            dangling=dangling,
            malformed=malformed,
            model_seconds=response.seconds,
            retrieval_seconds=retrieval_seconds,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cached=response.cached,
        )


def make_retriever(name: str, store, *, client: OllamaClient | None = None):
    """The five systems the generator can sit on top of, built with their defaults."""
    if name == "bm25":
        from retrieval.bm25 import Bm25Retriever

        return Bm25Retriever(store)
    if name == "dense":
        from retrieval.dense import DenseRetriever

        return DenseRetriever.load(store)
    if name == "hybrid":
        from retrieval.hybrid import HybridRetriever

        return HybridRetriever(store)
    if name == "rerank":
        from retrieval.rerank import RerankRetriever, _make_base

        return RerankRetriever(store, _make_base("hybrid", store))
    if name == "agent":
        from agent.graph import AgentRetriever
        from agent.run import make_base_retriever

        if client is None:
            raise SystemExit("the agent retriever needs an Ollama client")
        return AgentRetriever(store, make_base_retriever("hybrid", store), client)
    raise SystemExit(f"unknown retriever: {name!r} (choose from {', '.join(RETRIEVERS)})")


def main(argv: list[str] | None = None) -> int:
    from corpus.store import Store
    from eval.run import load_queries

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", choices=RETRIEVERS, default="hybrid")
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--queries", type=Path, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--offline", action="store_true", help="replay the committed cache only")
    ap.add_argument("--limit", type=int, default=0, help="first N queries (0 = all)")
    ap.add_argument("--out", type=Path, default=None, help="write answers as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    queries = load_queries(args.queries) if args.queries else load_queries()
    if args.limit:
        queries = queries[: args.limit]

    cache = ResponseCache(args.cache)
    client = OllamaClient(
        model=args.model, base_url=args.base_url, cache=cache, offline=args.offline
    )
    if not args.offline:
        client.check_available()

    with Store(args.db) as store:
        retriever = make_retriever(args.retriever, store, client=client)
        gen = Generator(store, retriever, args.retriever, client)
        answers = []
        for i, q in enumerate(queries, start=1):
            a = gen.answer(q.id, q.query)
            answers.append(a)
            if args.verbose:
                flag = "REFUSED" if a.refused else (a.malformed or f"{len(a.cited)} citations")
                print(f"  {i:>3}/{len(queries)} {q.id:<6} {flag}", flush=True)
            if i % 10 == 0:
                cache.save()
    cache.save()

    if args.out:
        args.out.write_text(json.dumps([a.as_dict() for a in answers], indent=1), encoding="utf-8")
        print(f"wrote {len(answers)} answers to {args.out}")
    print(summarize(answers))
    return 0


def summarize(answers) -> str:
    n = len(answers)
    if not n:
        return "no answers"
    refused = sum(a.refused for a in answers)
    malformed = sum(bool(a.malformed) for a in answers)
    dangling = sum(bool(a.dangling) for a in answers)
    model_s = sum(a.model_seconds for a in answers) / n
    return (
        f"{n} answers  refused {refused}  malformed {malformed}  dangling-citation {dangling}  "
        f"mean model {model_s:.1f}s"
    )


if __name__ == "__main__":
    sys.exit(main())
