import json

import pytest
import requests

from agent import run as retrieval_factory
from agent.coverage import (
    SYSTEMS,
    analyze,
    build_systems,
    format_comparison,
    format_coverage,
    format_systems,
)
from agent.coverage import main as coverage_main
from agent.graph import (
    DECOMPOSITION_DEPTH,
    END,
    GRAPH,
    MAX_SUB_QUERIES,
    RESULTS_PER_SUB_QUERY,
    RRF_K,
    AgentRetriever,
    AgentState,
    Graph,
    fuse_sub_query_results,
    make_graph,
    parse_route,
    parse_sub_queries,
)
from agent.llm import TEMPERATURE, OllamaClient, OllamaError, Response, ResponseCache
from agent.prompts import DECOMPOSE_PROMPT, ROUTE_PROMPT
from agent.run import format_report, make_agent, make_base_retriever, summarize_traces
from corpus.chunk import Chunk
from corpus.store import Store
from eval.run import Retriever

P1 = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P1_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P2 = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
P3 = "https://docs.pytorch.org/docs/2.14/nn.html"


def chunk(cid, url, text, ordinal=0, title="T"):
    return Chunk(cid, url, "", title, [title], ordinal, text, len(text.split()), cid)


@pytest.fixture
def store():
    s = Store(":memory:")
    s.load(
        [
            chunk("a1", P1, "class torch.optim.Adam(params, eps=1e-08)", title="Adam"),
            chunk("a2", P1_ALIAS, "class torch.optim.adam.Adam(params, eps=1e-08)", title="Adam"),
            chunk("t1", P2, "torch.topk returns k largest", title="torch.topk"),
            chunk("t2", P2, "second chunk of topk", ordinal=1, title="torch.topk"),
            chunk("n1", P3, "the nn module index", title="nn"),
        ]
    )
    s.apply_quality()
    yield s
    s.close()


class FakeClient:
    """Stands in for Ollama. Returns canned text per prompt kind; records what it saw."""

    def __init__(self, route="direct", sub_queries=None, route_text=None, decompose_text=None):
        self.route_text = (
            route_text if route_text is not None else json.dumps({"route": route, "reason": "test"})
        )
        self.decompose_text = (
            decompose_text
            if decompose_text is not None
            else json.dumps({"sub_queries": sub_queries or ["a", "b"]})
        )
        self.prompts = []

    def generate(self, prompt, *, query, json_mode=True):
        self.prompts.append(prompt)
        text = self.route_text if "routing a search query" in prompt else self.decompose_text
        return Response(text=text, prompt_tokens=10, completion_tokens=5, seconds=0.01)


class FakeRetriever:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.table.get(query, [])[:k]


# --- constants fixed a priori ---------------------------------------------------


def test_constants_are_fixed():
    # Guards against silent tuning: changing any of these must fail CI, not pass quietly.
    assert MAX_SUB_QUERIES == 3
    assert DECOMPOSITION_DEPTH == 1
    assert RESULTS_PER_SUB_QUERY == 10
    assert RRF_K == 60
    assert TEMPERATURE == 0.0


def test_rrf_constant_is_shared_with_the_hybrid_retriever():
    from retrieval.hybrid import RRF_K as HYBRID_K

    assert RRF_K is HYBRID_K


def test_prompts_are_not_empty_and_take_the_documented_placeholders():
    assert "{query}" in ROUTE_PROMPT
    assert "{query}" in DECOMPOSE_PROMPT and "{max_sub_queries}" in DECOMPOSE_PROMPT
    ROUTE_PROMPT.format(query="q")
    DECOMPOSE_PROMPT.format(query="q", max_sub_queries=MAX_SUB_QUERIES)


# --- defensive parsing ----------------------------------------------------------


def test_parse_route_accepts_clean_json():
    assert parse_route('{"route": "decompose", "reason": "two APIs"}') == (
        "decompose",
        "two APIs",
    )


def test_parse_route_tolerates_prose_and_fences():
    assert parse_route('Sure!\n```json\n{"route": "direct", "reason": "one API"}\n```')[0] == (
        "direct"
    )


def test_parse_route_normalises_case_and_whitespace():
    assert parse_route('{"route": " Decompose "}')[0] == "decompose"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        "{malformed",
        '{"route": "maybe"}',  # not one of the two verdicts
        '{"route": 3}',
        '["decompose"]',  # a list, not an object
    ],
)
def test_parse_route_rejects_unusable_responses(text):
    assert parse_route(text) is None


def test_parse_sub_queries_accepts_clean_json():
    assert parse_sub_queries('{"sub_queries": ["a", "b"]}') == ["a", "b"]


def test_parse_sub_queries_caps_at_the_maximum():
    got = parse_sub_queries('{"sub_queries": ["a", "b", "c", "d", "e"]}')
    assert got == ["a", "b", "c"] and len(got) == MAX_SUB_QUERIES


def test_parse_sub_queries_drops_blanks_dupes_and_non_strings():
    assert parse_sub_queries('{"sub_queries": ["a", "  ", "a", 7, null, " b "]}') == ["a", "b"]


@pytest.mark.parametrize(
    "text",
    ["", "nope", '{"sub_queries": "a"}', '{"sub_queries": []}', '{"other": ["a"]}'],
)
def test_parse_sub_queries_rejects_unusable_responses(text):
    assert parse_sub_queries(text) is None


# --- the graph ------------------------------------------------------------------


def test_graph_shape_is_declared_as_data():
    assert GRAPH["entry"] == "route"
    assert GRAPH["edges"]["route"] == {"decompose": "decompose", "direct": "direct"}
    assert GRAPH["edges"]["fuse"] == END


def test_graph_runs_nodes_in_order_and_stops_at_end():
    g = Graph()
    g.add_node("a", lambda s: s).add_edge("a", "b")
    g.add_node("b", lambda s: s).add_edge("b", END)
    state = g.set_entry("a").run(AgentState(query="q", k=5))
    assert state.path == ["a", "b"]


def test_graph_raises_on_a_missing_node():
    g = Graph().add_node("a", lambda s: s).add_edge("a", "ghost").set_entry("a")
    with pytest.raises(KeyError, match="ghost"):
        g.run(AgentState(query="q", k=5))


def test_graph_raises_rather_than_looping_forever():
    g = Graph().add_node("a", lambda s: s).add_edge("a", "a").set_entry("a")
    with pytest.raises(RuntimeError, match="did not reach"):
        g.run(AgentState(query="q", k=5), max_steps=5)


def test_direct_route_takes_the_direct_path_and_makes_one_llm_call(store):
    client = FakeClient(route="direct")
    r = FakeRetriever({"q": ["a1"]})
    state = make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert state.path == ["route", "direct", "retrieve", "fuse"]
    assert state.sub_queries == ["q"]
    assert state.llm_calls == 1


def test_decompose_route_takes_the_decompose_path_and_makes_two_llm_calls(store):
    client = FakeClient(route="decompose", sub_queries=["x", "y"])
    r = FakeRetriever({"x": ["a1"], "y": ["t1"]})
    state = make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert state.path == ["route", "decompose", "retrieve", "fuse"]
    assert state.sub_queries == ["x", "y"]
    assert state.llm_calls == 2


def test_each_sub_query_is_retrieved_at_the_fixed_depth(store):
    client = FakeClient(route="decompose", sub_queries=["x", "y"])
    r = FakeRetriever({"x": ["a1"], "y": ["t1"]})
    make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert r.calls == [("x", RESULTS_PER_SUB_QUERY), ("y", RESULTS_PER_SUB_QUERY)]


def test_sub_queries_are_never_themselves_decomposed(store):
    """DECOMPOSITION_DEPTH == 1: exactly two model calls however many sub-queries there are."""
    client = FakeClient(route="decompose", sub_queries=["x", "y", "z"])
    r = FakeRetriever({"x": ["a1"], "y": ["t1"], "z": ["n1"]})
    state = make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert state.llm_calls == 2
    assert len(client.prompts) == 2


# --- fallback -------------------------------------------------------------------


def test_unparseable_decomposition_falls_back_to_the_original_query(store):
    client = FakeClient(route="decompose", decompose_text="I cannot answer that.")
    r = FakeRetriever({"q": ["a1"]})
    state = make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert state.sub_queries == ["q"]
    assert state.decompose_fallback is True


def test_unparseable_route_falls_back_to_direct(store):
    client = FakeClient(route_text="???")
    r = FakeRetriever({"q": ["a1"]})
    state = make_graph(client, r, store).run(AgentState(query="q", k=5))
    assert state.route == "direct"
    assert state.route_fallback is True
    assert state.path == ["route", "direct", "retrieve", "fuse"]


def test_a_successful_run_records_no_fallback(store):
    client = FakeClient(route="decompose", sub_queries=["x"])
    state = make_graph(client, FakeRetriever({"x": ["a1"]}), store).run(AgentState(query="q", k=5))
    assert not state.decompose_fallback and not state.route_fallback


# --- fusion ---------------------------------------------------------------------


def test_fusion_ranks_a_page_found_by_two_sub_queries_first(store):
    fused = fuse_sub_query_results({"x": ["n1", "a1"], "y": ["a1"]}, store, 5)
    assert fused[0] == "a1"


def test_fusion_collapses_alias_pages_across_sub_queries(store):
    fused = fuse_sub_query_results({"x": ["a1"], "y": ["a2"]}, store, 5)
    assert len(fused) == 1 and fused[0] in {"a1", "a2"}


def test_fusion_gives_one_page_one_slot_per_sub_query(store):
    fused = fuse_sub_query_results({"x": ["t1", "t2"], "y": ["n1"]}, store, 5)
    assert len([c for c in fused if c in {"t1", "t2"}]) == 1


def test_fusion_truncates_to_k(store):
    fused = fuse_sub_query_results({"x": ["a1", "t1", "n1"]}, store, 2)
    assert len(fused) == 2


def test_fusion_handles_empty_input(store):
    assert fuse_sub_query_results({}, store, 5) == []
    assert fuse_sub_query_results({"x": []}, store, 5) == []
    assert fuse_sub_query_results({"x": ["a1"]}, store, 0) == []


# --- the retriever facade -------------------------------------------------------


def test_agent_conforms_to_the_retriever_protocol(store):
    agent = AgentRetriever(store, FakeRetriever({}), FakeClient())
    assert isinstance(agent, Retriever)


def test_agent_search_truncates_to_the_requested_k(store):
    client = FakeClient(route="decompose", sub_queries=["x", "y"])
    r = FakeRetriever({"x": ["a1", "t1"], "y": ["n1"]})
    assert len(AgentRetriever(store, r, client).search("q", 2)) == 2


def test_agent_handles_degenerate_queries(store):
    agent = AgentRetriever(store, FakeRetriever({}), FakeClient())
    assert agent.search("", 5) == []
    assert agent.search("   ", 5) == []
    assert agent.search("q", 0) == []


def test_agent_logs_the_path_taken_for_every_query(store):
    client = FakeClient(route="decompose", sub_queries=["x"])
    agent = AgentRetriever(store, FakeRetriever({"x": ["a1"]}), client)
    agent.search("q", 5)
    trace = agent.traces[0]
    assert trace["route"] == "decompose"
    assert trace["path"] == ["route", "decompose", "retrieve", "fuse"]
    assert trace["n_sub_queries"] == 1
    assert trace["llm_calls"] == 2
    assert trace["prompt_tokens"] == 20 and trace["completion_tokens"] == 10
    assert trace["llm_seconds"] > 0 and trace["retrieval_seconds"] >= 0


def test_agent_is_pluggable_over_any_base_retriever(store):
    class Constant:
        def search(self, query, k):
            return ["n1"][:k]

    agent = AgentRetriever(store, Constant(), FakeClient(route="direct"))
    assert agent.search("anything", 5) == ["n1"]


# --- cache ----------------------------------------------------------------------


def test_cache_key_covers_model_prompt_and_query():
    k = ResponseCache.key
    base = k("m", "p", "q")
    assert base != k("m2", "p", "q")
    assert base != k("m", "p2", "q")
    assert base != k("m", "p", "q2")
    assert base == k("m", "p", "q")


def test_cache_round_trips_through_disk(tmp_path):
    c = ResponseCache(tmp_path / "c.json")
    c.put("k1", Response(text="hi", prompt_tokens=3, completion_tokens=4, seconds=1.5))
    assert c.save() == 1
    again = ResponseCache(tmp_path / "c.json")
    got = again.get("k1")
    assert got.text == "hi" and got.prompt_tokens == 3 and got.completion_tokens == 4
    assert got.cached is True


def test_cache_miss_returns_none(tmp_path):
    assert ResponseCache(tmp_path / "c.json").get("nope") is None


def test_corrupt_cache_is_ignored_rather_than_fatal(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{ this is not json")
    assert len(ResponseCache(p)) == 0


def test_client_serves_from_cache_without_touching_the_network(tmp_path):
    cache = ResponseCache(tmp_path / "c.json")
    client = OllamaClient(cache=cache, base_url="http://127.0.0.1:1")  # unroutable
    key = cache.key(client.model, "PROMPT", "q")
    cache.put(key, Response(text="cached!", prompt_tokens=1, completion_tokens=1))
    got = client.generate("PROMPT", query="q")
    assert got.text == "cached!" and got.cached is True
    assert client.calls == 0 and client.cache_hits == 1


def test_offline_client_raises_on_a_cache_miss_instead_of_calling_out(tmp_path):
    client = OllamaClient(cache=ResponseCache(tmp_path / "c.json"), offline=True)
    with pytest.raises(OllamaError, match="cache miss"):
        client.generate("PROMPT", query="q")


# --- clear failure when Ollama is absent ----------------------------------------


def test_check_available_explains_an_unreachable_server(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", boom)
    client = OllamaClient(cache=ResponseCache(tmp_path / "c.json"))
    with pytest.raises(OllamaError, match="cannot reach Ollama"):
        client.check_available()


def test_check_available_explains_a_missing_model(tmp_path, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "llama3:8b"}]}

    monkeypatch.setattr(requests, "get", lambda *a, **kw: Resp())
    client = OllamaClient(cache=ResponseCache(tmp_path / "c.json"), model="qwen2.5:7b")
    with pytest.raises(OllamaError, match="ollama pull qwen2.5:7b"):
        client.check_available()


def test_generate_wraps_a_transport_failure_in_ollama_error(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    client = OllamaClient(cache=ResponseCache(tmp_path / "c.json"))
    with pytest.raises(OllamaError, match="Ollama call failed"):
        client.generate("PROMPT", query="q")


def test_generate_sends_temperature_zero_and_json_mode(tmp_path, monkeypatch):
    sent = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "{}", "prompt_eval_count": 2, "eval_count": 1}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(cache=ResponseCache(tmp_path / "c.json"))
    client.generate("PROMPT", query="q")
    assert sent["options"]["temperature"] == 0.0
    assert sent["format"] == "json"
    assert sent["stream"] is False


def test_a_live_call_is_written_to_the_cache(tmp_path, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "hello", "prompt_eval_count": 7, "eval_count": 3}

    monkeypatch.setattr(requests, "post", lambda *a, **kw: Resp())
    cache = ResponseCache(tmp_path / "c.json")
    client = OllamaClient(cache=cache)
    first = client.generate("PROMPT", query="q")
    assert first.cached is False and first.total_tokens == 10
    second = client.generate("PROMPT", query="q")
    assert second.cached is True and client.calls == 1


# --- latency accounting ---------------------------------------------------------


def test_total_latency_is_model_plus_retrieval_not_the_wall_clock(store):
    """A cache hit's model time is the original live latency, which no wall clock sees."""
    client = FakeClient(route="direct")
    agent = AgentRetriever(store, FakeRetriever({"q": ["a1"]}), client)
    agent.search("q", 5)
    t = agent.traces[0]
    assert t["total_seconds"] == pytest.approx(t["llm_seconds"] + t["retrieval_seconds"])
    # The replay's own wall clock is kept, and is far smaller than the recorded model time.
    assert t["wall_seconds"] < t["total_seconds"]


# --- the run harness ------------------------------------------------------------


def _trace(route="direct", *, dec_fb=False, llm=1.0, retr=0.1, n_sub=1):
    return {
        "route": route,
        "route_fallback": False,
        "decompose_fallback": dec_fb,
        "n_sub_queries": n_sub,
        "llm_seconds": llm,
        "retrieval_seconds": retr,
        "total_seconds": llm + retr,
        "wall_seconds": retr,
        "llm_calls": 2 if route == "decompose" else 1,
        "cache_hits": 0,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }


def test_summarize_counts_the_route_mix():
    s = summarize_traces([_trace("direct"), _trace("decompose"), _trace("decompose")])
    assert s["n"] == 3
    assert s["route_direct"] == 1 and s["route_decompose"] == 2


def test_fallback_rate_is_over_decompose_attempts_not_all_queries():
    """A direct query never attempts a decomposition; counting it would dilute the rate."""
    traces = [_trace("direct")] * 6 + [_trace("decompose"), _trace("decompose", dec_fb=True)]
    s = summarize_traces(traces)
    assert s["decompose_attempts"] == 2
    assert s["decompose_fallbacks"] == 1
    assert s["decompose_fallback_rate"] == 0.5


def test_fallback_rate_is_zero_when_nothing_was_decomposed():
    assert summarize_traces([_trace("direct")])["decompose_fallback_rate"] == 0.0


def test_summarize_splits_latency_into_model_and_retrieval():
    s = summarize_traces([_trace(llm=1.0, retr=0.25), _trace(llm=3.0, retr=0.75)])
    assert s["llm_seconds"]["total"] == 4.0 and s["llm_seconds"]["mean"] == 2.0
    assert s["retrieval_seconds"]["total"] == 1.0
    assert s["retrieval_seconds"]["max"] == 0.75
    assert s["model_share"] == pytest.approx(4.0 / 5.0)


def test_summarize_handles_an_empty_run():
    assert summarize_traces([])["n"] == 0


def test_report_states_the_fallback_rate_and_both_latency_halves():
    traces = [_trace("decompose", dec_fb=True), _trace("direct")]
    text = format_report(traces, summarize_traces(traces))
    assert "1/1" in text and "model" in text and "retrieval" in text


def test_make_base_retriever_rejects_an_unknown_name(store):
    with pytest.raises(SystemExit, match="unknown retriever"):
        make_base_retriever("nope", store)


def test_offline_agent_never_probes_the_network(store, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("offline must not touch the network")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr(retrieval_factory, "make_base_retriever", lambda name, s: FakeRetriever({}))
    agent = make_agent(store, "bm25", offline=True)
    assert agent.client.offline is True


# --- coverage -------------------------------------------------------------------


def _queries():
    from eval.run import Judgment, Query

    return [
        Query("q1", "first", "api_lookup", [Judgment(P1, 2), Judgment(P2, 0)]),
        Query("q2", "second", "multi_hop", [Judgment(P1, 1)]),
    ]


def _judged(queries, store):
    from eval.pool import judged_urls

    return judged_urls(queries, store.canonical_map())


def test_coverage_rate_counts_pages_with_no_judgment_at_any_grade(store):
    queries = _queries()
    # q1: P1 and P2 judged, P3 not -> 1 of 3 unjudged. q2: only P1 judged -> 1 of 2.
    r = FakeRetriever({"first": ["a1", "t1", "n1"], "second": ["a1", "n1"]})
    report = analyze(r, "fake", queries, store, _judged(queries, store), k=10)
    all_ = report["summary"]["all"]
    assert all_["retrieved"] == 5 and all_["unjudged"] == 2
    assert all_["rate"] == pytest.approx(2 / 5)


def test_a_grade_zero_judgment_counts_as_judged(store):
    """P2 is graded 0: the judge saw it and ruled it out. That is a verdict, not a gap."""
    queries = _queries()
    r = FakeRetriever({"first": ["t1"], "second": []})
    report = analyze(r, "fake", queries, store, _judged(queries, store), k=10)
    assert report["summary"]["all"]["unjudged"] == 0


def test_coverage_splits_by_category(store):
    queries = _queries()
    r = FakeRetriever({"first": ["a1"], "second": ["n1"]})
    s = analyze(r, "fake", queries, store, _judged(queries, store), k=10)["summary"]
    assert s["api_lookup"]["unjudged"] == 0
    assert s["multi_hop"]["unjudged"] == 1 and s["multi_hop"]["rate"] == 1.0


def test_distinct_new_pages_is_a_union_not_a_sum(store):
    """The same page unjudged for two queries is two decisions but one page to read."""
    queries = _queries()
    r = FakeRetriever({"first": ["n1"], "second": ["n1"]})
    report = analyze(r, "fake", queries, store, _judged(queries, store), k=10)
    assert report["summary"]["all"]["unjudged"] == 2
    assert report["distinct_new_pages_total"] == 1


def test_alias_pages_are_not_counted_as_new(store):
    """A judgment on the canonical page covers a retrieval of its alias."""
    from eval.run import Judgment, Query

    queries = [Query("q1", "first", "api_lookup", [Judgment(P1, 2)])]
    r = FakeRetriever({"first": ["a2"]})  # a2 lives on the alias url
    report = analyze(r, "fake", queries, store, _judged(queries, store), k=10)
    assert report["summary"]["all"]["unjudged"] == 0


def test_fully_covered_queries_are_counted(store):
    queries = _queries()
    r = FakeRetriever({"first": ["a1"], "second": ["n1"]})
    assert (
        analyze(r, "fake", queries, store, _judged(queries, store), k=10)["queries_fully_covered"]
        == 1
    )


def test_comparison_reports_the_difference_attributable_to_the_agent(store):
    queries = _queries()
    judged = _judged(queries, store)
    base = analyze(
        FakeRetriever({"first": ["a1"], "second": ["a1"]}), "base", queries, store, judged, k=10
    )
    agent = analyze(
        FakeRetriever({"first": ["a1", "n1"], "second": ["a1"]}),
        "agent/base",
        queries,
        store,
        judged,
        k=10,
    )
    text = format_comparison(agent, base)
    assert "base" in text and "agent/base" in text and "attributable" in text
    assert base["summary"]["all"]["unjudged"] == 0
    assert agent["summary"]["all"]["unjudged"] == 1


def test_coverage_table_renders(store):
    queries = _queries()
    r = FakeRetriever({"first": ["n1"], "second": []})
    text = format_coverage(
        analyze(r, "fake", queries, store, _judged(queries, store), k=10), verbose=True
    )
    assert "api_lookup" in text and "distinct pages" in text


# --- pooling the union of several systems ---------------------------------------


def test_systems_lists_every_poolable_system():
    """Base retrievers, their agent wrappers, both rerankers, and the random control.

    The rerankers matter here: they are the systems the README calls least trustworthy on
    coverage grounds, and leaving them out of SYSTEMS is what made that claim unmeasurable.
    """
    assert set(SYSTEMS) == {
        "bm25",
        "dense",
        "hybrid",
        "agent/bm25",
        "agent/dense",
        "agent/hybrid",
        "rerank/bm25",
        "rerank/hybrid",
        "random",
    }


class FakeScorer:
    """Stands in for the cross-encoder. Never loaded, so no torch is needed.

    Injected rather than skipped: CI installs `requirements.txt` only, and a test that builds
    a real CrossEncoderScorer would import sentence-transformers and pass locally while
    failing there. Same guard the dense tests use when they inject an encoder.
    """

    name = "fake-scorer"
    device = "cpu"
    batch_size = 1

    def score(self, query, texts):
        return [0.0] * len(texts)


def test_build_systems_gives_a_reranker_the_shared_base(store):
    """rerank/bm25 must reorder the same first stage `bm25` uses, or the coverage
    difference between them would confound reranking with a different base."""
    systems = build_systems(["bm25", "rerank/bm25"], store, FakeClient(), scorer=FakeScorer())
    assert not isinstance(systems["bm25"], AgentRetriever)
    assert systems["rerank/bm25"].base is systems["bm25"]


def test_build_systems_does_not_load_a_cross_encoder_when_given_a_scorer(store):
    """The guard itself: the injected scorer is the one that ends up on the retriever.

    Over a bm25 base rather than hybrid: hybrid builds a dense retriever, which needs
    sentence-transformers, so testing the scorer over it would reintroduce the very import
    this test exists to avoid.
    """
    scorer = FakeScorer()
    systems = build_systems(["rerank/bm25"], store, FakeClient(), scorer=scorer)
    assert systems["rerank/bm25"].scorer is scorer


def test_build_systems_makes_random_a_seeded_baseline(store):
    systems = build_systems(["random"], store, FakeClient())
    first = systems["random"].search("anything", 5)
    assert build_systems(["random"], store, FakeClient())["random"].search("anything", 5) == first


def test_build_systems_wraps_only_the_agent_prefixed_names(store):
    systems = build_systems(["bm25", "agent/bm25"], store, FakeClient())
    assert not isinstance(systems["bm25"], AgentRetriever)
    assert isinstance(systems["agent/bm25"], AgentRetriever)


def test_build_systems_shares_one_base_retriever_with_its_agent(store):
    """Otherwise `hybrid` and `agent/hybrid` would each load the dense model separately."""
    systems = build_systems(["bm25", "agent/bm25"], store, FakeClient())
    assert systems["agent/bm25"].retriever is systems["bm25"]


def test_build_systems_shares_one_client_across_agents(store):
    client = FakeClient()
    systems = build_systems(["agent/bm25", "agent/bm25"], store, client)
    assert all(s.client is client for s in systems.values())


def test_union_pool_counts_a_page_found_by_several_systems_once(store):
    """The re-pool is what a human grades, so overlap must collapse, not multiply."""
    from eval.pool import build_pool, delta_pool
    from eval.run import Query

    queries = [Query("q1", "first", "api_lookup", [])]
    systems = {
        "a": FakeRetriever({"first": ["a1", "t1"]}),
        "b": FakeRetriever({"first": ["a1", "n1"]}),
    }
    delta = delta_pool(build_pool(systems, queries, store, k=10), {"q1": set()})
    entry = delta["queries"]["q1"]
    # P1, P2 and P3: three distinct pages from four retrieved slots.
    assert entry["n_new"] == 3
    found = {c["url"]: c["found_by"] for c in entry["candidates"]}
    assert "a@1" in found[P1] and "b@1" in found[P1]


def test_union_pool_records_which_system_found_a_page_alone(store):
    from eval.pool import build_pool, delta_pool
    from eval.run import Query

    queries = [Query("q1", "first", "api_lookup", [])]
    systems = {"a": FakeRetriever({"first": ["a1"]}), "b": FakeRetriever({"first": ["n1"]})}
    delta = delta_pool(build_pool(systems, queries, store, k=10), {"q1": set()})
    found = {c["url"]: c["found_by"] for c in delta["queries"]["q1"]["candidates"]}
    assert found[P3] == "b@1" and found[P1] == "a@1"


def test_format_systems_shows_the_union_and_the_collapse(store):
    queries = _queries()
    judged = _judged(queries, store)
    per_system = {
        "a": analyze(
            FakeRetriever({"first": ["n1"], "second": []}), "a", queries, store, judged, k=10
        ),
        "b": analyze(
            FakeRetriever({"first": ["n1"], "second": []}), "b", queries, store, judged, k=10
        ),
    }
    union = analyze(
        FakeRetriever({"first": ["n1"], "second": []}), "u", queries, store, judged, k=10
    )
    text = format_systems(per_system, union)
    assert "union" in text
    # 1 + 1 unjudged pairs across the two systems, but the same page: 1 candidate pooled.
    assert "2 unjudged pairs summed over systems collapse to 1 distinct candidates" in text


def test_out_refuses_to_overwrite_an_existing_file(tmp_path, capsys):
    """A graded pool is unrecoverable; clobbering one must not be a typo away."""
    existing = tmp_path / "pool.yaml"
    existing.write_text("precious: judgments")
    assert coverage_main(["--system", "bm25", "--out", str(existing)]) == 2
    assert "refusing to overwrite" in capsys.readouterr().out
    assert existing.read_text() == "precious: judgments"
