import sqlite3
from dataclasses import replace

from corpus.chunk import Chunk
from corpus.store import Store, source_for_url

URL = "https://docs.pytorch.org/docs/2.14/generated/torch.foo.html"
TUT = "https://docs.pytorch.org/tutorials/beginner/x.html"


def mk(cid, url=URL, ordinal=0, text="hello world", anchor="a", path=("Foo",)):
    return Chunk(cid, url, anchor, "Foo", list(path), ordinal, text, len(text.split()), cid[::-1])


def rows(store):
    return [(c.chunk_id, c.url, c.ordinal, c.text, c.heading_path) for c in store.iter_chunks()]


def test_schema_and_roundtrip():
    with Store(":memory:") as s:
        s.load([mk("c1"), mk("c2", ordinal=1, text="cross_entropy loss", path=("Foo", "Bar"))])
        assert s.count() == 2 and s.page_count() == 1 and s.fts_count() == 2
        c = s.get("c2")
        assert c.heading_path == ["Foo", "Bar"] and c.text == "cross_entropy loss"
        assert s.get("nope") is None
        assert s.chunk_ids() == ["c1", "c2"]


def test_load_is_idempotent():
    chunks = [mk("c1"), mk("c2", ordinal=1), mk("t1", url=TUT)]
    with Store(":memory:") as s:
        r1 = s.load(chunks, version="2.14.0", fetched_at="2026-09-07")
        before = rows(s)
        r2 = s.load(chunks, version="2.14.0", fetched_at="2026-09-07")
        assert r1 == r2 == {"pages": 2, "chunks": 3, "deleted": 0}
        assert rows(s) == before
        assert s.count() == 3 and s.page_count() == 2 and s.fts_count() == 3
        page = s.conn.execute("SELECT * FROM pages WHERE url = ?", (TUT,)).fetchone()
        assert (page["source"], page["version"], page["n_chunks"]) == ("tutorials", "2.14.0", 1)


def test_reload_updates_changed_and_removes_stale():
    with Store(":memory:") as s:
        s.load([mk("c1"), mk("c2", ordinal=1), mk("t1", url=TUT)])
        # page re-chunked: c2 changed text, c3 new, c1 gone; tutorial page untouched
        r = s.load(
            [replace(mk("c2", ordinal=0), text="changed", content_hash="new"), mk("c3", ordinal=1)]
        )
        assert r == {"pages": 1, "chunks": 2, "deleted": 1}
        assert s.chunk_ids() == ["c2", "t1", "c3"]  # c1 removed, c2 updated in place (rowid kept)
        assert s.get("c2").text == "changed" and s.get("c2").ordinal == 0
        assert s.fts_count() == s.count() == 3
        assert s.conn.execute("SELECT n_chunks FROM pages WHERE url = ?", (URL,)).fetchone()[0] == 2


def test_fts_index_stays_in_sync_and_tokenizes_identifiers():
    with Store(":memory:") as s:
        s.load([mk("c1", text="use torch.nn.functional.cross_entropy for classification")])
        hit = s.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("cross_entropy",)
        ).fetchall()
        assert len(hit) == 1
        hit = s.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("functional",)
        ).fetchall()
        assert len(hit) == 1
        s.load([mk("c1", text="nothing here")])
        assert (
            s.conn.execute(
                "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?", ("cross_entropy",)
            ).fetchone()[0]
            == 0
        )


def test_locate():
    with Store(":memory:") as s:
        s.load([mk("c1", anchor="x"), mk("t1", url=TUT, anchor="")])
        assert s.locate(["t1", "c1", "missing", "c1"]) == {"c1": (URL, "x"), "t1": (TUT, "")}
        assert s.locate([]) == {}


def test_source_for_url():
    assert source_for_url(URL) == "docs"
    assert source_for_url(TUT) == "tutorials"


# --- quality analysis -----------------------------------------------------------

ALIAS = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.adam.Adam_class.html"
CANON = "https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html"
STUB = "https://docs.pytorch.org/tutorials/beginner/old.html"


def quality_store():
    s = Store(":memory:")
    s.load(
        [
            mk("c1", CANON, text="class torch.optim.Adam(p) implements Adam"),
            mk("c2", ALIAS, text="class torch.optim.adam.Adam(p) implements Adam"),
            mk("c3", URL, text="something entirely different"),
            mk("c4", STUB, text="This tutorial was deprecated. Redirecting in 3 seconds"),
        ]
    )
    return s


def test_apply_quality_records_canonical_and_status():
    with quality_store() as s:
        report = s.apply_quality()
        assert report.alias_groups == [[CANON, ALIAS]]
        assert s.canonical_map() == {ALIAS: CANON}
        assert s.stub_urls() == {STUB}
        assert s.quality_counts() == {"stubs": 1, "redundant": 1, "unanalyzed": 0}
        # non-alias pages are their own canonical
        row = s.conn.execute("SELECT canonical_url FROM pages WHERE url = ?", (URL,)).fetchone()
        assert row["canonical_url"] == URL


def test_apply_quality_is_idempotent():
    with quality_store() as s:
        first = s.apply_quality()
        before = dict(s.canonical_map())
        second = s.apply_quality()
        assert first.alias_groups == second.alias_groups
        assert s.canonical_map() == before
        assert s.quality_counts()["stubs"] == 1


def test_quality_columns_are_added_to_an_existing_database(tmp_path):
    """A database created before these columns existed is migrated in place."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE pages (url TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '', "
        "source TEXT NOT NULL DEFAULT '', version TEXT NOT NULL DEFAULT '', "
        "fetched_at TEXT NOT NULL DEFAULT '', n_chunks INTEGER NOT NULL DEFAULT 0);"
    )
    con.execute("INSERT INTO pages(url) VALUES (?)", (URL,))
    con.commit()
    con.close()

    with Store(path) as s:
        cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(pages)")}
        assert {"canonical_url", "status"} <= cols
        assert s.quality_counts()["unanalyzed"] == 1  # existing row, not yet analyzed
        s.load([mk("c1", URL)])
        s.apply_quality()
        assert s.quality_counts()["unanalyzed"] == 0
