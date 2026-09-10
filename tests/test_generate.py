import hashlib
import json

import pytest

from agent.llm import TEMPERATURE, Response, ResponseCache
from corpus.chunk import Chunk
from eval.run import Judgment, Query
from generate import prompts
from generate.faithfulness import (
    MAX_CLAIMS,
    MISMATCH_OFFSET,
    Checked,
    Checker,
    Verdict,
    mismatch_pairs,
    parse_claims,
    parse_verdicts,
    relevant_pages,
    summarize,
)
from generate.prompts import ANSWER_PROMPT, CLAIMS_PROMPT, REFUSAL_MARKER, SUPPORT_PROMPT
from generate.run import (
    NUM_CTX,
    NUM_PREDICT,
    RETRIEVERS,
    Answer,
    Generator,
    K,
    format_context,
    parse_answer,
)

P1 = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P2 = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"


def chunk(cid, url=P1, text="body text", title="T", headings=("T", "Args")):
    return Chunk(cid, url, "", title, list(headings), 0, text, len(text.split()), cid)


class FakeClient:
    """Stands in for Ollama. Canned text per prompt kind; records the calls it saw."""

    def __init__(self, answer=None, claims=None, verdicts=None):
        self.answer_text = (
            answer
            if answer is not None
            else json.dumps({"refused": False, "answer": "Adam takes lr [C1]."})
        )
        self.claims_text = (
            claims if claims is not None else json.dumps({"claims": ["Adam takes lr"]})
        )
        self.verdicts_text = (
            verdicts
            if verdicts is not None
            else json.dumps({"verdicts": [{"claim": 1, "supported": True, "passage": 1}]})
        )
        self.calls = []

    def generate(self, prompt, *, query, json_mode=True, retriever="", options=None):
        self.calls.append(
            {"prompt": prompt, "query": query, "retriever": retriever, "options": options or {}}
        )
        if "split an answer into atomic claims" in prompt:
            text = self.claims_text
        elif "check whether numbered passages support" in prompt:
            text = self.verdicts_text
        else:
            text = self.answer_text
        return Response(text=text, prompt_tokens=100, completion_tokens=20, seconds=0.5)


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k):
        return self.ids[:k]


class FakeStore:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def get(self, cid):
        return self.chunks.get(cid)

    def locate(self, ids):
        return {i: (self.chunks[i].url, "") for i in ids if i in self.chunks}


# --- constants fixed a priori ------------------------------------------------------------
# These are the whole point of the stage: every one was chosen before any number was
# produced, and none may be moved in response to a result. A failure here means someone
# tuned the system against a metric.


def test_k_is_ten_and_matches_the_eval_harness():
    from eval.run import main as eval_main  # noqa: F401  (import proves the module pairs up)

    assert K == 10


def test_decoding_is_greedy_and_the_windows_are_fixed():
    assert TEMPERATURE == 0.0
    assert NUM_CTX == 8192
    assert NUM_PREDICT == 400
    assert MAX_CLAIMS == 8
    assert MISMATCH_OFFSET == 20


def test_options_cannot_smuggle_in_a_non_zero_temperature(tmp_path, monkeypatch):
    """The client sets temperature last, so no caller can raise it off greedy decoding."""
    import requests

    from agent.llm import OllamaClient, ResponseCache

    sent = {}

    class Resp:
        status_code = 200

        def raise_for_status(self): ...

        def json(self):
            return {"response": "{}", "prompt_eval_count": 1, "eval_count": 1}

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    c = OllamaClient(cache=ResponseCache(tmp_path / "c.json"))
    c.generate("p", query="q", retriever="bm25", options={"temperature": 0.9, "num_ctx": 42})
    assert sent["options"] == {"num_ctx": 42, "temperature": 0.0}


def test_the_five_retrievers_are_the_ones_the_repo_evaluates():
    assert RETRIEVERS == ("bm25", "dense", "hybrid", "rerank", "agent")


def test_refusal_marker_has_exactly_one_spelling():
    assert REFUSAL_MARKER == "INSUFFICIENT_CONTEXT"


@pytest.mark.parametrize(
    "name,digest",
    [
        ("ANSWER_PROMPT", "aab298fe5df1e25757c82396896a699dbee7d999c05de04645b4d6f9f7075f00"),
        ("CLAIMS_PROMPT", "5fdafa31e11f63a5c26245e868be77ff65f3dfe3b90f5136c420f40af30caf0f"),
        ("SUPPORT_PROMPT", "e90dbecbb5e143be7b2391892133e712e3abf63ba8a0a8b578fba9abccb32b7d"),
    ],
)
def test_prompts_are_frozen(name, digest):
    """The prompts are pinned by hash. Editing one is a deliberate act that invalidates the
    committed cache and every number in the README, so it has to break a test first."""
    text = getattr(prompts, name)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest, (
        f"{name} changed. If that is intended, re-measure and update the README; "
        "do not edit a prompt to improve a score."
    )


def test_the_answer_prompt_forbids_outside_knowledge_and_demands_citations():
    assert "Do not use anything you know about PyTorch that is not in them" in ANSWER_PROMPT
    assert "must cite the passage it came from" in ANSWER_PROMPT
    assert "{refusal}" in ANSWER_PROMPT


def test_the_checker_prompts_are_kept_apart():
    """The decomposer must not see passages; the checker must not see the question."""
    assert "{answer}" in CLAIMS_PROMPT and "{context}" not in CLAIMS_PROMPT
    assert "{context}" in SUPPORT_PROMPT and "{query}" not in SUPPORT_PROMPT
    # support, not truth: the distinction the whole metric rests on
    assert "a claim that is true but that no passage states is NOT supported" in SUPPORT_PROMPT


# --- refusal behaviour -------------------------------------------------------------------


def test_a_refusal_needs_both_the_flag_and_the_marker():
    raw = json.dumps({"refused": True, "answer": REFUSAL_MARKER})
    refused, text, cited, dangling, bad = parse_answer(raw, K)
    assert refused and not bad and text == REFUSAL_MARKER
    assert cited == [] and dangling == []


@pytest.mark.parametrize(
    "body",
    [
        {"refused": True, "answer": "Adam takes lr [C1]."},  # flag without marker
        {"refused": False, "answer": f"{REFUSAL_MARKER}"},  # marker without flag
    ],
)
def test_a_half_refusal_is_malformed_rather_than_guessed(body):
    _, _, _, _, bad = parse_answer(json.dumps(body), K)
    assert "marker present" in bad


def test_a_normal_answer_yields_its_citations():
    raw = json.dumps({"refused": False, "answer": "A [C1]. B [C3][C1]."})
    refused, _, cited, dangling, bad = parse_answer(raw, K)
    assert not refused and not bad
    assert cited == [1, 3, 1] and dangling == []


def test_citations_outside_the_context_are_recorded_not_dropped():
    raw = json.dumps({"refused": False, "answer": "A [C1]. B [C14]."})
    _, _, cited, dangling, bad = parse_answer(raw, K)
    assert cited == [1] and dangling == [14] and not bad


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("not json at all", "invalid JSON"),
        ('["a list"]', "expected a JSON object"),
        (json.dumps({"refused": False}), "missing or non-string 'answer'"),
        (json.dumps({"answer": "x"}), "missing or non-boolean 'refused'"),
    ],
)
def test_malformed_responses_are_reported(raw, fragment):
    _, _, _, _, bad = parse_answer(raw, K)
    assert fragment in bad


def test_cited_chunk_ids_map_back_in_citation_order_without_duplicates():
    a = Answer(
        "q1", "q", "bm25", ["c1", "c2", "c3"], False, "A [C3]. B [C1]. C [C3].", cited=[3, 1, 3]
    )
    assert a.cited_chunk_ids == ["c3", "c1"]


# --- context and generation --------------------------------------------------------------


def test_context_is_numbered_from_one_in_rank_order():
    ctx = format_context([chunk("a", text="first"), chunk("b", text="second")])
    assert ctx.startswith("[C1] T — T > Args\nfirst")
    assert "[C2] T — T > Args\nsecond" in ctx


def test_generator_sends_k_chunks_the_frozen_options_and_the_retriever_key():
    store = FakeStore([chunk(f"c{i}") for i in range(12)])
    client = FakeClient()
    gen = Generator(store, FakeRetriever([f"c{i}" for i in range(12)]), "bm25", client)
    a = gen.answer("q1", "how do I use Adam?")
    assert len(a.chunk_ids) == K
    call = client.calls[0]
    assert call["retriever"] == "bm25"
    assert call["options"] == {"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT}
    assert "[C10]" in call["prompt"] and "[C11]" not in call["prompt"]


def test_a_mismatch_run_cannot_collide_with_the_ordinary_run_in_the_cache():
    store = FakeStore([chunk("c0")])
    client = FakeClient()
    gen = Generator(store, FakeRetriever(["c0"]), "bm25", client)
    gen.answer_from("q1", "q", ["c0"], [store.get("c0")])
    gen.answer_from("q1", "q", ["c0"], [store.get("c0")], tag="mismatch")
    assert [c["retriever"] for c in client.calls] == ["bm25", "bm25:mismatch"]


# --- cache keying ------------------------------------------------------------------------


def test_the_retriever_widens_the_cache_key():
    a = ResponseCache.key("m", "p", "q", "bm25")
    b = ResponseCache.key("m", "p", "q", "dense")
    assert a != b


def test_an_absent_retriever_keeps_the_pre_existing_key():
    """The agent's committed cache was written without this field and must still resolve."""
    assert ResponseCache.key("m", "p", "q") == ResponseCache.key("m", "p", "q", "")


# --- claim decomposition and checking ----------------------------------------------------


def test_claims_are_parsed_and_capped():
    claims, bad = parse_claims(json.dumps({"claims": [f"c{i}" for i in range(20)]}), MAX_CLAIMS)
    assert not bad and len(claims) == MAX_CLAIMS


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("junk", "invalid JSON"),
        (json.dumps({}), "missing 'claims' list"),
        (json.dumps({"claims": ["", "  "]}), "no usable claims"),
    ],
)
def test_bad_claim_responses_are_reported(raw, fragment):
    claims, bad = parse_claims(raw, MAX_CLAIMS)
    assert claims == [] and fragment in bad


def test_verdicts_align_to_claims_by_number():
    claims = ["a", "b"]
    raw = json.dumps(
        {
            "verdicts": [
                {"claim": 2, "supported": True, "passage": 3},
                {"claim": 1, "supported": False, "passage": None},
            ]
        }
    )
    verdicts, bad = parse_verdicts(raw, claims, K)
    assert not bad
    assert [(v.claim, v.supported, v.passage) for v in verdicts] == [
        ("a", False, None),
        ("b", True, 3),
    ]


def test_an_unjudged_claim_counts_as_unsupported_but_is_marked_unjudged():
    """Dropping it instead would raise the support rate by discarding the hard cases, and
    conflating it with a real 'no' would hide how much of the rate is checker silence."""
    raw = json.dumps({"verdicts": [{"claim": 1, "supported": True, "passage": 1}]})
    verdicts, bad = parse_verdicts(raw, ["a", "b"], K)
    assert len(verdicts) == 2 and verdicts[1].supported is False
    assert verdicts[0].judged is True and verdicts[1].judged is False
    assert "1 claim(s) unjudged" in bad


def test_a_passage_outside_the_context_is_not_recorded_as_the_source():
    raw = json.dumps({"verdicts": [{"claim": 1, "supported": True, "passage": 99}]})
    verdicts, _ = parse_verdicts(raw, ["a"], K)
    assert verdicts[0].supported is True and verdicts[0].passage is None


def test_a_refusal_is_not_decomposed():
    client = FakeClient()
    a = Answer("q1", "q", "bm25", ["c1"], True, REFUSAL_MARKER)
    out = Checker(client).check(a, [chunk("c1")])
    assert out.claims == [] and out.verdicts == [] and client.calls == []


def test_checker_runs_two_calls_and_tags_them_separately():
    client = FakeClient()
    a = Answer("q1", "q", "bm25", ["c1"], False, "Adam takes lr [C1].")
    out = Checker(client).check(a, [chunk("c1")])
    assert [c["retriever"] for c in client.calls] == ["bm25:claims", "bm25:support"]
    assert out.n_claims == 1 and out.supported == 1


# --- the refusal probe -------------------------------------------------------------------


def test_the_rotation_never_pairs_a_query_with_itself():
    pairs = mismatch_pairs([object()] * 40)
    assert len(pairs) == 40
    assert all(i != d for i, d in pairs)
    assert len({d for _, d in pairs}) == 40  # every query donates exactly once


def test_relevant_pages_reads_grade_one_and_two_only():
    q = Query(
        "q1", "q", "api_lookup", [Judgment(P1, 2), Judgment(P2, 1), Judgment("https://x/z.html", 0)]
    )
    assert relevant_pages(q, {}) == {P1, P2}


def test_relevant_pages_canonicalizes_before_comparing():
    q = Query("q1", "q", "api_lookup", [Judgment(P2, 2)])
    assert relevant_pages(q, {P2: P1}) == {P1}


# --- reporting ---------------------------------------------------------------------------


def test_summarize_counts_support_over_answers_only_and_refusals_over_the_probe():
    answered = Checked(
        answer=Answer("q1", "q", "bm25", ["c1"], False, "A [C1]."),
        claims=["a", "b"],
        verdicts=[Verdict("a", True, 1), Verdict("b", False)],
    )
    refused = Checked(answer=Answer("q2", "q", "bm25", ["c1"], True, REFUSAL_MARKER))
    probe = [
        {"query_id": "q1", "skipped": False, "refused": True},
        {"query_id": "q2", "skipped": False, "refused": False},
        {"query_id": "q3", "skipped": True},
    ]
    s = summarize("bm25", [answered, refused], probe)
    assert s["queries"] == 2 and s["refused"] == 1
    assert s["answers_scored"] == 1 and s["claims"] == 2
    assert s["supported"] == 1 and s["support_rate"] == 0.5
    assert s["answers_fully_supported"] == 0
    assert s["probe_scored"] == 2 and s["probe_skipped"] == 1
    assert s["refusal_rate"] == 0.5
