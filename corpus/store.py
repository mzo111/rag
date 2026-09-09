"""SQLite storage for chunks with an FTS5 index.

Schema
------
pages(url PK, title, source, version, fetched_at, n_chunks, canonical_url, status)
chunks(chunk_id PK, url -> pages, anchor, title, heading_path JSON, ordinal, text,
       n_tokens, content_hash)
chunks_fts: external-content FTS5 table over (title, heading_path, text), kept in sync with
``chunks`` by triggers. Tokenizer: porter stemming + unicode61 with ``_`` as a token
character, so ``cross_entropy`` stays one token while ``torch.nn.functional`` splits on dots.

``canonical_url`` and ``status`` are filled by :meth:`Store.apply_quality`, which runs the
alias/stub analysis in :mod:`corpus.quality`. They are advisory: no page is deleted, so the
decision to merge or drop aliases stays open. Retrievers and the eval harness use them to
collapse alias pages to one judgment unit and to skip deprecated stubs.

The loader is idempotent: rows are upserted by ``chunk_id`` and, for every page present in
the batch, chunks that no longer exist are deleted. Loading the same chunks twice leaves the
database unchanged.

Usage:
    python -m corpus.store [--db data/corpus.db] [--chunks data/chunks.jsonl]
"""

from __future__ import annotations

import json
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path

from corpus.chunk import Chunk
from corpus.quality import STATUS_OK, QualityReport, analyze

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    url        TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT '',
    source     TEXT NOT NULL DEFAULT '',
    version    TEXT NOT NULL DEFAULT '',
    fetched_at TEXT NOT NULL DEFAULT '',
    n_chunks   INTEGER NOT NULL DEFAULT 0,
    canonical_url TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'ok'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    url          TEXT NOT NULL REFERENCES pages(url) ON DELETE CASCADE,
    anchor       TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '[]',
    ordinal      INTEGER NOT NULL,
    text         TEXT NOT NULL,
    n_tokens     INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_url ON chunks(url, ordinal);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title, heading_path, text,
    content='chunks', content_rowid='rowid',
    tokenize="porter unicode61 tokenchars '_'"
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, title, heading_path, text)
    VALUES (new.rowid, new.title, new.heading_path, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, heading_path, text)
    VALUES ('delete', old.rowid, old.title, old.heading_path, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, title, heading_path, text)
    VALUES ('delete', old.rowid, old.title, old.heading_path, old.text);
    INSERT INTO chunks_fts(rowid, title, heading_path, text)
    VALUES (new.rowid, new.title, new.heading_path, new.text);
END;
"""


def source_for_url(url: str) -> str:
    return "tutorials" if "/tutorials/" in url else "docs"


class Store:
    def __init__(self, path: str | Path = "data/corpus.db"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        have = {r["name"] for r in self.conn.execute("PRAGMA table_info(pages)")}
        if "canonical_url" not in have:
            self.conn.execute("ALTER TABLE pages ADD COLUMN canonical_url TEXT NOT NULL DEFAULT ''")
        if "status" not in have:
            self.conn.execute("ALTER TABLE pages ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- loading -------------------------------------------------------------
    def load(
        self, chunks: Iterable[Chunk | dict], *, version: str = "", fetched_at: str = ""
    ) -> dict[str, int]:
        """Upsert chunks; remove stale chunks of any page present in the batch.

        Returns counts: {"pages": n, "chunks": n, "deleted": n}.
        """
        rows = [asdict(c) if isinstance(c, Chunk) else dict(c) for c in chunks]
        by_url: dict[str, list[dict]] = {}
        for r in rows:
            by_url.setdefault(r["url"], []).append(r)

        deleted = 0
        with self.conn:  # one transaction
            for url, page_rows in by_url.items():
                title = page_rows[0]["title"]
                self.conn.execute(
                    "INSERT INTO pages(url, title, source, version, fetched_at, n_chunks) "
                    "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET "
                    "title=excluded.title, source=excluded.source, version=excluded.version, "
                    "fetched_at=excluded.fetched_at, n_chunks=excluded.n_chunks",
                    (url, title, source_for_url(url), version, fetched_at, len(page_rows)),
                )
                keep = [r["chunk_id"] for r in page_rows]
                placeholders = ",".join("?" * len(keep))
                cur = self.conn.execute(
                    f"DELETE FROM chunks WHERE url = ? AND chunk_id NOT IN ({placeholders})",
                    [url, *keep],
                )
                deleted += cur.rowcount
                self.conn.executemany(
                    "INSERT INTO chunks(chunk_id, url, anchor, title, heading_path, ordinal, "
                    "text, n_tokens, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET url=excluded.url, "
                    "anchor=excluded.anchor, title=excluded.title, "
                    "heading_path=excluded.heading_path, ordinal=excluded.ordinal, "
                    "text=excluded.text, n_tokens=excluded.n_tokens, "
                    "content_hash=excluded.content_hash "
                    "WHERE chunks.text != excluded.text OR chunks.url != excluded.url "
                    "OR chunks.ordinal != excluded.ordinal OR chunks.anchor != excluded.anchor "
                    "OR chunks.heading_path != excluded.heading_path "
                    "OR chunks.title != excluded.title",
                    [
                        (
                            r["chunk_id"],
                            r["url"],
                            r["anchor"],
                            r["title"],
                            json.dumps(r["heading_path"], ensure_ascii=False),
                            r["ordinal"],
                            r["text"],
                            r["n_tokens"],
                            r["content_hash"],
                        )
                        for r in page_rows
                    ],
                )
        return {"pages": len(by_url), "chunks": len(rows), "deleted": deleted}

    # -- quality ------------------------------------------------------------
    def apply_quality(self) -> QualityReport:
        """Run the alias/stub analysis over stored chunks and record it on ``pages``.

        Non-destructive: pages keep their rows, and re-running is safe. A page that is not
        an alias gets its own URL as ``canonical_url``.
        """
        pages: dict[str, list[str]] = {}
        tokens: dict[str, int] = {}
        for row in self.conn.execute(
            "SELECT url, text, n_tokens FROM chunks ORDER BY url, ordinal"
        ):
            pages.setdefault(row["url"], []).append(row["text"])
            tokens[row["url"]] = tokens.get(row["url"], 0) + row["n_tokens"]
        report = analyze(pages, tokens)
        with self.conn:
            self.conn.executemany(
                "UPDATE pages SET canonical_url = ?, status = ? WHERE url = ?",
                [(report.canonical.get(url, url), report.status_of(url), url) for url in pages],
            )
        return report

    def canonical_map(self) -> dict[str, str]:
        """url -> canonical url, for every page whose canonical differs from itself."""
        return {
            r["url"]: r["canonical_url"]
            for r in self.conn.execute(
                "SELECT url, canonical_url FROM pages "
                "WHERE canonical_url != '' AND canonical_url != url"
            )
        }

    def stub_urls(self) -> set[str]:
        return {
            r[0] for r in self.conn.execute("SELECT url FROM pages WHERE status != ?", (STATUS_OK,))
        }

    def quality_counts(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT sum(status != 'ok') AS stubs, "
            "sum(canonical_url != '' AND canonical_url != url) AS redundant, "
            "sum(canonical_url = '') AS unanalyzed FROM pages"
        ).fetchone()
        return {k: row[k] or 0 for k in ("stubs", "redundant", "unanalyzed")}

    # -- reading -------------------------------------------------------------
    @staticmethod
    def _to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            url=row["url"],
            anchor=row["anchor"],
            title=row["title"],
            heading_path=json.loads(row["heading_path"]),
            ordinal=row["ordinal"],
            text=row["text"],
            n_tokens=row["n_tokens"],
            content_hash=row["content_hash"],
        )

    def get(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._to_chunk(row) if row else None

    def count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks").fetchone()[0]

    def page_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM pages").fetchone()[0]

    def fts_count(self) -> int:
        return self.conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]

    def iter_chunks(self) -> Iterator[Chunk]:
        for row in self.conn.execute("SELECT * FROM chunks ORDER BY url, ordinal"):
            yield self._to_chunk(row)

    def chunk_ids(self) -> list[str]:
        return [r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks ORDER BY rowid")]

    def locate(self, chunk_ids: Iterable[str]) -> dict[str, tuple[str, str]]:
        """chunk_id -> (url, anchor) for the given ids (unknown ids are omitted)."""
        ids = list(dict.fromkeys(chunk_ids))
        out: dict[str, tuple[str, str]] = {}
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            placeholders = ",".join("?" * len(batch))
            for row in self.conn.execute(
                f"SELECT chunk_id, url, anchor FROM chunks WHERE chunk_id IN ({placeholders})",
                batch,
            ):
                out[row["chunk_id"]] = (row["url"], row["anchor"])
        return out


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Load data/chunks.jsonl into SQLite + FTS5")
    ap.add_argument("--db", type=Path, default=Path("data/corpus.db"))
    ap.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    args = ap.parse_args(argv)

    version = fetched_at = ""
    if args.manifest.exists():
        m = json.loads(args.manifest.read_text())
        version = m.get("docs_version") or m.get("docs_version_dir") or ""
        fetched_at = m.get("fetched_at", "")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    with Store(args.db) as store:
        result = store.load(read_jsonl(args.chunks), version=version, fetched_at=fetched_at)
        report = store.apply_quality()
        print(
            f"loaded {result['chunks']} chunks over {result['pages']} pages "
            f"({result['deleted']} stale removed); db now has {store.count()} chunks / "
            f"{store.page_count()} pages / {store.fts_count()} fts rows -> {args.db}"
        )
        print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
