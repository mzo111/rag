"""Retrieval agent: query decomposition and tool routing over an explicit state graph.

The graph is written out here rather than imported from a framework, because it is small
enough that a framework would hide more than it saves:

    route ──(decompose)──> decompose ──┐
      │                                ├──> retrieve ──> fuse ──> END
      └──(direct)────────> direct ─────┘

``route`` and ``decompose`` are one model call each; ``direct`` is free. Every node takes the
state and returns it, and every edge is either a node name or a function of the state, so the
whole control flow is inspectable as data (:data:`GRAPH`) and each node is testable alone.

Constants are **fixed a priori and asserted in tests**, so tuning them against eval scores
fails CI rather than passing quietly:

    MAX_SUB_QUERIES        3    at most three sub-queries per query
    DECOMPOSITION_DEPTH    1    sub-queries are never themselves decomposed
    RESULTS_PER_SUB_QUERY  10   retrieved per sub-query before fusion
    RRF_K                  60   reciprocal rank fusion constant, from retrieval.hybrid
    TEMPERATURE            0.0  greedy decoding, in agent.llm

Fusion reuses :func:`retrieval.hybrid.rrf_scores` - the same published constant and the same
formula that fuses BM25 with dense - and, like that module, fuses at the level of the
**canonical page** rather than the chunk, so two sub-queries that surface the same page from
different chunks add their evidence instead of splitting it and taking two slots.

Defensive parsing is a first-class path, not error handling. A 7B local model is markedly
less reliable at schema-following than a hosted one, so a response that does not parse falls
back to the original query unchanged and is counted; :attr:`AgentRetriever.traces` carries the
route taken, the fallback flag and the split timings for every query.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.llm import OllamaClient, Response
from agent.prompts import DECOMPOSE_PROMPT, ROUTE_PROMPT
from corpus.store import Store
from retrieval.hybrid import RRF_K, rrf_scores

# --- fixed a priori; tests assert these ----------------------------------------
MAX_SUB_QUERIES = 3
DECOMPOSITION_DEPTH = 1
RESULTS_PER_SUB_QUERY = 10

END = "__end__"
ROUTE_DECOMPOSE = "decompose"
ROUTE_DIRECT = "direct"
# First JSON object in a response; local models like to wrap it in prose or fences.
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AgentState:
    """Everything the graph reads or writes. One instance per query."""

    query: str
    k: int
    route: str = ""
    route_reason: str = ""
    sub_queries: list[str] = field(default_factory=list)
    per_sub_query: dict[str, list[str]] = field(default_factory=dict)
    results: list[str] = field(default_factory=list)
    # observability
    path: list[str] = field(default_factory=list)
    route_fallback: bool = False
    decompose_fallback: bool = False
    llm_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0

    def record(self, response: Response) -> None:
        self.llm_seconds += response.seconds
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.llm_calls += 1
        self.cache_hits += int(response.cached)


class Graph:
    """A tiny explicit state graph: named nodes, edges that are names or predicates."""

    def __init__(self) -> None:
        self.nodes: dict[str, Callable[[AgentState], AgentState]] = {}
        self.edges: dict[str, str | Callable[[AgentState], str]] = {}
        self.entry: str = ""

    def add_node(self, name: str, fn: Callable[[AgentState], AgentState]) -> Graph:
        self.nodes[name] = fn
        return self

    def add_edge(self, name: str, target: str | Callable[[AgentState], str]) -> Graph:
        self.edges[name] = target
        return self

    def set_entry(self, name: str) -> Graph:
        self.entry = name
        return self

    def run(self, state: AgentState, *, max_steps: int = 16) -> AgentState:
        node = self.entry
        for _ in range(max_steps):
            if node == END:
                return state
            if node not in self.nodes:
                raise KeyError(f"no such node: {node!r}")
            state.path.append(node)
            state = self.nodes[node](state)
            target = self.edges.get(node, END)
            node = target(state) if callable(target) else target
        raise RuntimeError(f"graph did not reach {END} within {max_steps} steps")


# --- parsing --------------------------------------------------------------------


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a response, tolerating fences and prose."""
    if not text:
        return None
    match = JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_route(text: str) -> tuple[str, str] | None:
    """``(route, reason)``, or None if the response is unusable."""
    obj = _first_json_object(text)
    if obj is None:
        return None
    route = obj.get("route")
    if not isinstance(route, str) or route.strip().lower() not in {
        ROUTE_DECOMPOSE,
        ROUTE_DIRECT,
    }:
        return None
    reason = obj.get("reason")
    return route.strip().lower(), reason.strip() if isinstance(reason, str) else ""


def parse_sub_queries(text: str, max_n: int = MAX_SUB_QUERIES) -> list[str] | None:
    """Clean list of sub-queries, or None if the response is unusable.

    Valid JSON is not the same as the requested schema, so the list is validated element by
    element: strings only, stripped, empties dropped, duplicates removed, capped at ``max_n``.
    """
    obj = _first_json_object(text)
    if obj is None:
        return None
    raw = obj.get("sub_queries")
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out[:max_n] or None


# --- nodes ----------------------------------------------------------------------


def make_graph(client: OllamaClient, retriever, store: Store) -> Graph:
    """Build the graph over a model client and any ``search(query, k)`` retriever."""

    def route_node(state: AgentState) -> AgentState:
        response = client.generate(ROUTE_PROMPT.format(query=state.query), query=state.query)
        state.record(response)
        parsed = parse_route(response.text)
        if parsed is None:
            # Unparseable verdict: take the cheap path rather than guessing.
            state.route, state.route_reason = ROUTE_DIRECT, "unparseable route verdict"
            state.route_fallback = True
        else:
            state.route, state.route_reason = parsed
        return state

    def decompose_node(state: AgentState) -> AgentState:
        prompt = DECOMPOSE_PROMPT.format(query=state.query, max_sub_queries=MAX_SUB_QUERIES)
        response = client.generate(prompt, query=state.query)
        state.record(response)
        parsed = parse_sub_queries(response.text, MAX_SUB_QUERIES)
        if parsed is None:
            state.sub_queries = [state.query]
            state.decompose_fallback = True
        else:
            state.sub_queries = parsed
        return state

    def direct_node(state: AgentState) -> AgentState:
        state.sub_queries = [state.query]
        return state

    def retrieve_node(state: AgentState) -> AgentState:
        started = time.perf_counter()
        for sub_query in state.sub_queries:
            # DECOMPOSITION_DEPTH == 1: a sub-query goes straight to the retriever and is
            # never routed or decomposed again.
            state.per_sub_query[sub_query] = list(
                retriever.search(sub_query, RESULTS_PER_SUB_QUERY)
            )
        state.retrieval_seconds += time.perf_counter() - started
        return state

    def fuse_node(state: AgentState) -> AgentState:
        state.results = fuse_sub_query_results(state.per_sub_query, store, state.k)
        return state

    return (
        Graph()
        .add_node("route", route_node)
        .add_node("decompose", decompose_node)
        .add_node("direct", direct_node)
        .add_node("retrieve", retrieve_node)
        .add_node("fuse", fuse_node)
        .add_edge("route", lambda s: "decompose" if s.route == ROUTE_DECOMPOSE else "direct")
        .add_edge("decompose", "retrieve")
        .add_edge("direct", "retrieve")
        .add_edge("retrieve", "fuse")
        .add_edge("fuse", END)
        .set_entry("route")
    )


#: The control flow as data, for documentation and tests.
GRAPH = {
    "entry": "route",
    "edges": {
        "route": {"decompose": "decompose", "direct": "direct"},
        "decompose": "retrieve",
        "direct": "retrieve",
        "retrieve": "fuse",
        "fuse": END,
    },
}


def canonical_pages(store: Store, chunk_ids: Sequence[str]) -> dict[str, str]:
    """chunk_id -> canonical page url, for page-level fusion."""
    ids = list(dict.fromkeys(chunk_ids))
    out: dict[str, str] = {}
    for i in range(0, len(ids), 500):
        batch = ids[i : i + 500]
        placeholders = ",".join("?" * len(batch))
        for row in store.conn.execute(
            "SELECT c.chunk_id, c.url, p.canonical_url FROM chunks c "
            f"JOIN pages p ON p.url = c.url WHERE c.chunk_id IN ({placeholders})",
            batch,
        ):
            out[row["chunk_id"]] = row["canonical_url"] or row["url"]
    return out


def fuse_sub_query_results(per_sub_query: dict[str, list[str]], store: Store, k: int) -> list[str]:
    """RRF over the sub-query result lists, fused per page, truncated to k.

    Reuses :func:`retrieval.hybrid.rrf_scores` with the same published constant, so the two
    places this project fuses rankings do it identically.
    """
    if k <= 0 or not per_sub_query:
        return []
    pages = canonical_pages(store, [c for ids in per_sub_query.values() for c in ids])

    rankings: dict[str, list[str]] = {}
    best_chunk: dict[str, str] = {}
    best_rank: dict[str, int] = {}
    for sub_query, chunk_ids in per_sub_query.items():
        seen: list[str] = []
        for rank, cid in enumerate(chunk_ids, 1):
            page = pages.get(cid)
            if page is None or page in seen:
                continue  # one sub-query must not pay twice for the same page
            seen.append(page)
            if rank < best_rank.get(page, 1 << 30):
                best_rank[page], best_chunk[page] = rank, cid
        rankings[sub_query] = seen

    scores = rrf_scores(rankings, k=RRF_K)
    order = sorted(scores, key=lambda p: (-scores[p], best_rank[p], p))
    return [best_chunk[p] for p in order[:k]]


class AgentRetriever:
    """The agent as a retriever: implements ``search(query, k)`` like every other system.

    ``retriever`` is any object with ``search(query, k)``, so the agent runs over BM25, dense
    or the hybrid without either side changing.
    """

    def __init__(self, store: Store, retriever, client: OllamaClient):
        self.store = store
        self.retriever = retriever
        self.client = client
        self.graph = make_graph(client, retriever, store)
        self.traces: list[dict] = []

    def search(self, query: str, k: int) -> list[str]:
        if not query.strip() or k <= 0:
            return []
        started = time.perf_counter()
        state = self.graph.run(AgentState(query=query, k=k))
        self.traces.append(
            {
                "query": query,
                "route": state.route,
                "route_reason": state.route_reason,
                "path": list(state.path),
                "n_sub_queries": len(state.sub_queries),
                "sub_queries": list(state.sub_queries),
                "route_fallback": state.route_fallback,
                "decompose_fallback": state.decompose_fallback,
                "llm_seconds": state.llm_seconds,
                "retrieval_seconds": state.retrieval_seconds,
                # total = model + retrieval, NOT the measured wall clock. On a cache hit the
                # model half is the latency recorded when that call ran live, so the wall
                # clock of a replay omits it entirely and would report a system three orders
                # of magnitude faster than the one that exists. `wall_seconds` keeps the
                # measured number for anyone who wants the replay cost itself.
                "total_seconds": state.llm_seconds + state.retrieval_seconds,
                "wall_seconds": time.perf_counter() - started,
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "llm_calls": state.llm_calls,
                "cache_hits": state.cache_hits,
                "n_results": len(state.results),
            }
        )
        return state.results
