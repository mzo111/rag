import yaml

from corpus.chunk import Chunk
from corpus.store import Store
from eval.pool import (
    DELTA_HEADER,
    append_into_queries,
    build_pool,
    delta_pool,
    format_delta_report,
    graded_judgments,
    judged_urls,
    merge_into_queries,
    snippet,
    write_pool,
)
from eval.run import Judgment, Query, load_queries

P1 = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
P1_ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
P2 = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"

QUERIES_SRC = """\
# a comment that must survive
corpus_version: "2.14.0"

queries:
  - id: q01
    category: api_lookup
    query: "adam eps"
    judgments: []

  - id: q02
    category: api_lookup
    query: "topk"
    judgments: []
"""


def mk(cid, url, text, title="T"):
    return Chunk(cid, url, "", title, [title, "sec"], 0, text, len(text.split()), cid)


def make_store():
    s = Store(":memory:")
    s.load(
        [
            mk("a1", P1, "class torch.optim.Adam(params, eps=1e-08)", title="Adam"),
            mk("a2", P1_ALIAS, "class torch.optim.adam.Adam(params, eps=1e-08)", title="Adam"),
            mk("t1", P2, "torch.topk returns k largest", title="torch.topk"),
        ]
    )
    s.apply_quality()
    return s


class FakeRetriever:
    def __init__(self, answers):
        self.answers = answers

    def search(self, query, k):
        return self.answers.get(query, [])[:k]


def test_snippet_flattens_and_truncates():
    assert snippet("a\n  b\tc") == "a b c"
    assert snippet("x" * 300).endswith("...")
    assert len(snippet("x" * 300)) == 223


def test_build_pool_unions_retrievers_and_collapses_aliases():
    s = make_store()
    qs = [Query("q01", "adam eps", "api_lookup"), Query("q02", "topk", "api_lookup")]
    rs = {
        "bm25": FakeRetriever({"adam eps": ["a1", "a2"], "topk": ["t1"]}),
        "other": FakeRetriever({"adam eps": ["a2", "t1"]}),
    }
    pool = build_pool(rs, qs, s, k=10)
    assert pool["built_by"] == ["bm25", "other"]
    cands = pool["queries"]["q01"]["candidates"]
    # a1 and a2 are aliases: one candidate, credited to both retrievers
    assert [c["url"] for c in cands] == [P1, P2]
    assert all(c["grade"] is None for c in cands)
    assert "bm25@1" in cands[0]["found_by"] and "other@1" in cands[0]["found_by"]
    assert cands[0]["title"] == "Adam" and "eps" in cands[0]["snippet"]
    assert pool["queries"]["q02"]["candidates"][0]["url"] == P2
    s.close()


def test_write_pool_is_valid_yaml_with_header(tmp_path):
    s = make_store()
    qs = [Query("q01", "adam eps", "api_lookup")]
    pool = build_pool({"bm25": FakeRetriever({"adam eps": ["a1"]})}, qs, s, k=5)
    p = tmp_path / "pool.yaml"
    assert write_pool(pool, p) == 1
    text = p.read_text()
    assert text.startswith("# Judging pool")
    assert yaml.safe_load(text)["queries"]["q01"]["candidates"][0]["url"] == P1
    s.close()


def test_graded_judgments_keeps_only_filled_grades():
    pool = {
        "queries": {
            "q01": {
                "candidates": [
                    {"url": P1, "grade": 2},
                    {"url": P2, "grade": 0},
                    {"url": "https://x/", "grade": None},
                ]
            },
            "q02": {"candidates": [{"url": P2, "grade": None}]},
        }
    }
    assert graded_judgments(pool) == {"q01": [(P1, 2), (P2, 0)]}


def test_merge_writes_judgments_and_preserves_comments(tmp_path):
    qp = tmp_path / "queries.yaml"
    qp.write_text(QUERIES_SRC)
    written, skipped = merge_into_queries({"q01": [(P1, 2), (P2, 0)]}, qp)
    assert (written, skipped) == (1, [])
    text = qp.read_text()
    assert "# a comment that must survive" in text
    data = yaml.safe_load(text)
    assert data["queries"][0]["judgments"] == [{"url": P1, "grade": 2}, {"url": P2, "grade": 0}]
    assert data["queries"][1]["judgments"] == []
    assert [q["id"] for q in data["queries"]] == ["q01", "q02"]


def test_merge_does_not_clobber_existing_judgments_without_force(tmp_path):
    qp = tmp_path / "queries.yaml"
    qp.write_text(QUERIES_SRC)
    merge_into_queries({"q01": [(P1, 2)]}, qp)
    written, skipped = merge_into_queries({"q01": [(P2, 1)]}, qp)
    assert (written, skipped) == (0, ["q01"])
    assert yaml.safe_load(qp.read_text())["queries"][0]["judgments"] == [{"url": P1, "grade": 2}]

    written, skipped = merge_into_queries({"q01": [(P2, 1)]}, qp, force=True)
    assert (written, skipped) == (1, [])
    assert yaml.safe_load(qp.read_text())["queries"][0]["judgments"] == [{"url": P2, "grade": 1}]


def test_merge_round_trips_through_load_queries(tmp_path):
    qp = tmp_path / "queries.yaml"
    qp.write_text(QUERIES_SRC)
    merge_into_queries({"q01": [(P1, 2)], "q02": [(P2, 1)]}, qp)
    qs = load_queries(qp)
    assert [q.id for q in qs] == ["q01", "q02"]
    assert qs[0].judgments == [Judgment(url=P1, grade=2)]
    assert qs[1].grades == {P2: 1}


# --- delta pool -----------------------------------------------------------------


def _pool_of(*urls_per_query):
    """A minimal pool shaped like build_pool's output."""
    queries = {}
    for qid, urls in urls_per_query:
        queries[qid] = {
            "query": qid,
            "category": "api_lookup",
            "candidates": [
                {
                    "grade": None,
                    "url": u,
                    "title": "T",
                    "section": "s",
                    "snippet": "x",
                    "found_by": "dense@1",
                }
                for u in urls
            ],
        }
    return {"built_by": ["dense"], "k": 10, "queries": queries}


def test_judged_urls_collects_every_grade_including_zero():
    qs = [
        Query("q01", "a", "api_lookup", [Judgment(P1, 2), Judgment(P2, 0)]),
        Query("q02", "b", "api_lookup", []),
    ]
    assert judged_urls(qs) == {"q01": {P1, P2}, "q02": set()}


def test_judged_urls_canonicalizes_alias_judgments():
    qs = [Query("q01", "a", "api_lookup", [Judgment(P1_ALIAS, 2)])]
    assert judged_urls(qs, {P1_ALIAS: P1}) == {"q01": {P1}}


def test_delta_pool_drops_already_judged_candidates():
    pool = _pool_of(("q01", [P1, P2]))
    delta = delta_pool(pool, {"q01": {P1}})
    assert [c["url"] for c in delta["queries"]["q01"]["candidates"]] == [P2]
    assert delta["queries"]["q01"]["n_new"] == 1
    assert delta["queries"]["q01"]["n_already_judged"] == 1


def test_delta_pool_keeps_queries_that_gain_nothing():
    delta = delta_pool(_pool_of(("q01", [P1])), {"q01": {P1}})
    assert delta["queries"]["q01"]["candidates"] == []
    assert delta["queries"]["q01"]["n_new"] == 0


def test_delta_pool_keeps_everything_when_nothing_is_judged():
    delta = delta_pool(_pool_of(("q01", [P1, P2])), {})
    assert delta["queries"]["q01"]["n_new"] == 2
    assert delta["queries"]["q01"]["n_already_judged"] == 0


def test_delta_pool_preserves_candidate_fields_and_order():
    delta = delta_pool(_pool_of(("q01", [P1, P2])), {"q01": set()})
    cand = delta["queries"]["q01"]["candidates"][0]
    assert sorted(cand) == ["found_by", "grade", "section", "snippet", "title", "url"]
    assert [c["url"] for c in delta["queries"]["q01"]["candidates"]] == [P1, P2]
    assert delta["built_by"] == ["dense"] and delta["k"] == 10


def test_delta_pool_does_not_mutate_the_source_pool():
    pool = _pool_of(("q01", [P1, P2]))
    delta_pool(pool, {"q01": {P1}})
    assert len(pool["queries"]["q01"]["candidates"]) == 2
    assert "n_new" not in pool["queries"]["q01"]


def test_delta_pool_writes_its_own_header(tmp_path):
    delta = delta_pool(_pool_of(("q01", [P1, P2])), {"q01": {P1}})
    p = tmp_path / "pool_delta.yaml"
    assert write_pool(delta, p, header=DELTA_HEADER) == 1
    text = p.read_text()
    assert text.startswith("# Delta judging pool")
    assert yaml.safe_load(text)["queries"]["q01"]["candidates"][0]["url"] == P2


def test_delta_report_counts_new_and_skipped():
    delta = delta_pool(_pool_of(("q01", [P1, P2]), ("q02", [P1])), {"q01": {P1}, "q02": {P1}})
    report = format_delta_report(delta)
    assert "1 new candidates over 2 queries" in report
    assert "2 already judged and skipped" in report
    assert "queries with nothing new (1): q02" in report


def test_delta_pool_end_to_end_over_a_store():
    s = make_store()
    qs = [Query("q01", "adam eps", "api_lookup", [Judgment(P1, 2)])]
    pool = build_pool({"dense": FakeRetriever({"adam eps": ["a1", "t1"]})}, qs, s, k=10)
    delta = delta_pool(pool, judged_urls(qs, s.canonical_map()))
    # P1 was judged; only the topk page is new
    assert [c["url"] for c in delta["queries"]["q01"]["candidates"]] == [P2]
    s.close()


def test_delta_pool_subtracts_an_alias_of_a_judged_page():
    s = make_store()
    # judged under the alias url; the pool surfaces it under the canonical url
    qs = [Query("q01", "adam eps", "api_lookup", [Judgment(P1_ALIAS, 2)])]
    pool = build_pool({"dense": FakeRetriever({"adam eps": ["a2"]})}, qs, s, k=10)
    delta = delta_pool(pool, judged_urls(qs, s.canonical_map()))
    assert delta["queries"]["q01"]["candidates"] == []
    s.close()


# --- append-mode merge ----------------------------------------------------------

JUDGED_SRC = (
    """\
# a comment that must survive
corpus_version: "2.14.0"

queries:
  - id: q01
    category: api_lookup
    query: "adam eps"
    judgments:
      - {url: URL_A, grade: 2}
      - {url: URL_B, grade: 0}

  - id: q02
    category: api_lookup
    query: "topk"
    judgments:
      - {url: URL_C, grade: 1}
""".replace("URL_A", P1)
    .replace("URL_B", P2)
    .replace("URL_C", P2)
)


def _write(tmp_path, src=JUDGED_SRC):
    p = tmp_path / "queries.yaml"
    p.write_text(src)
    return p


def test_append_adds_without_replacing_existing(tmp_path):
    p = _write(tmp_path)
    touched, added, conflicts = append_into_queries({"q01": [(P1_ALIAS, 1)]}, p, backup=False)
    qs = {q.id: q for q in load_queries(p)}
    assert (touched, added, conflicts) == (["q01"], 1, [])
    # the two originals survive, unchanged, and the new row is appended after them
    assert [(j.url, j.grade) for j in qs["q01"].judgments] == [
        (P1, 2),
        (P2, 0),
        (P1_ALIAS, 1),
    ]
    assert [(j.url, j.grade) for j in qs["q02"].judgments] == [(P2, 1)]


def test_append_preserves_comments_and_other_keys(tmp_path):
    p = _write(tmp_path)
    append_into_queries({"q01": [(P1_ALIAS, 1)]}, p, backup=False)
    text = p.read_text()
    assert "# a comment that must survive" in text
    assert 'corpus_version: "2.14.0"' in text
    assert "category: api_lookup" in text


def test_append_touches_several_queries(tmp_path):
    p = _write(tmp_path)
    _, added, _ = append_into_queries(
        {"q01": [(P1_ALIAS, 1)], "q02": [(P1, 2), (P1_ALIAS, 0)]}, p, backup=False
    )
    qs = {q.id: q for q in load_queries(p)}
    assert added == 3
    assert len(qs["q01"].judgments) == 3 and len(qs["q02"].judgments) == 3


def test_append_skips_and_reports_already_judged_urls(tmp_path):
    p = _write(tmp_path)
    touched, added, conflicts = append_into_queries({"q01": [(P1, 0)]}, p, backup=False)
    qs = {q.id: q for q in load_queries(p)}
    assert (added, conflicts) == (0, [("q01", P1)])
    # the original grade is untouched, not overwritten with the incoming 0
    assert [(j.url, j.grade) for j in qs["q01"].judgments] == [(P1, 2), (P2, 0)]


def test_append_writes_a_backup_of_the_original(tmp_path):
    p = _write(tmp_path)
    original = p.read_text()
    append_into_queries({"q01": [(P1_ALIAS, 1)]}, p, backup=True)
    assert (tmp_path / "queries.yaml.bak").read_text() == original
    assert p.read_text() != original


def test_append_fills_an_empty_judgments_list(tmp_path):
    p = _write(tmp_path, QUERIES_SRC)
    _, added, _ = append_into_queries({"q01": [(P1, 2)]}, p, backup=False)
    qs = {q.id: q for q in load_queries(p)}
    assert added == 1
    assert [(j.url, j.grade) for j in qs["q01"].judgments] == [(P1, 2)]


def test_append_is_idempotent_on_a_second_run(tmp_path):
    p = _write(tmp_path)
    append_into_queries({"q01": [(P1_ALIAS, 1)]}, p, backup=False)
    after_first = p.read_text()
    _, added, conflicts = append_into_queries({"q01": [(P1_ALIAS, 1)]}, p, backup=False)
    assert added == 0 and len(conflicts) == 1
    assert p.read_text() == after_first


def test_append_leaves_file_loadable_and_grades_intact(tmp_path):
    p = _write(tmp_path)
    append_into_queries({"q02": [(P1, 2)]}, p, backup=False)
    qs = {q.id: q for q in load_queries(p)}
    assert sum(len(q.judgments) for q in qs.values()) == 4
    assert all(isinstance(j.grade, int) for q in qs.values() for j in q.judgments)
