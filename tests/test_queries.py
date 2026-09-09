"""Schema checks for eval/queries.yaml (the judgments themselves are human-written)."""

from collections import Counter

import yaml

from eval.run import CATEGORIES, QUERIES_PATH, load_queries


def raw():
    return yaml.safe_load(QUERIES_PATH.read_text(encoding="utf-8"))


def test_forty_queries_with_unique_ids():
    qs = raw()["queries"]
    assert len(qs) == 40
    ids = [q["id"] for q in qs]
    assert len(set(ids)) == 40
    assert ids == sorted(ids)
    texts = [q["query"].strip().lower() for q in qs]
    assert len(set(texts)) == 40


def test_every_query_has_required_fields():
    for q in raw()["queries"]:
        assert set(q) >= {"id", "category", "query", "judgments"}, q["id"]
        assert q["category"] in CATEGORIES, q["id"]
        assert isinstance(q["query"], str) and len(q["query"].split()) >= 3, q["id"]
        assert isinstance(q["judgments"], list), q["id"]


def test_categories_cover_all_difficulties():
    counts = Counter(q["category"] for q in raw()["queries"])
    assert set(counts) == set(CATEGORIES)
    assert min(counts.values()) >= 6


def test_judgments_are_well_formed():
    for q in raw()["queries"]:
        seen = set()
        for j in q["judgments"]:
            assert set(j) <= {"url", "anchor", "grade"}, (q["id"], j)
            assert j["url"].startswith("https://docs.pytorch.org/"), (q["id"], j)
            assert j["grade"] in (0, 1, 2), (q["id"], j)
            key = (j["url"], j.get("anchor") or "")
            assert key not in seen, (q["id"], "duplicate judgment", key)
            seen.add(key)


def test_load_queries_parses_the_real_file():
    qs = load_queries()
    assert len(qs) == 40
    assert all(q.grades == {j.key: j.grade for j in q.judgments} for q in qs)
