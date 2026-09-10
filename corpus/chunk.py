"""Parse PyTorch documentation HTML into text and chunk it.

Strategy: section-aware block packing, no overlap.

1. Locate the content root (Sphinx article) and drop navigation, header-links, download
   boxes and similar chrome. KaTeX math is replaced by its TeX source.
2. Walk the DOM in document order and turn it into a flat list of *blocks* (paragraph,
   list item, code, table, signature, ...), each tagged with the heading path in force at
   that point. Headings h1-h6 push/pop a stack by level. Autodoc entries
   (``<dt class="sig" id="torch.nn.Linear">``) are treated as headings one level below
   their enclosing heading, so every documented API object becomes its own section.
3. Pack consecutive blocks *within one section* greedily up to ``target_tokens`` and never
   beyond ``max_tokens``. Code blocks and table rows are atomic: a code block is only split
   at line boundaries when it alone exceeds ``max_tokens`` (fence repeated); a table is only
   split by rows, and the header row is repeated on every piece. Tiny trailing chunks are
   merged into their predecessor. Chunks never cross section boundaries otherwise.
4. Every chunk carries a stable ``chunk_id`` derived from (url, heading path, index within
   section) rather than from its text, so ids survive edits elsewhere on the page and a
   refresh of the corpus does not invalidate relevance judgments. ``content_hash`` is
   there to detect changed text.

The module is pure: ``chunk_html`` does no IO. ``python -m corpus.chunk`` is the CLI that
walks ``data/raw`` and writes ``data/chunks.jsonl`` plus corpus statistics.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

TokenCounter = Callable[[str], int]

TARGET_TOKENS = 350
MAX_TOKENS = 512
MIN_TOKENS = 40

# Chrome to remove before extracting text.
DROP_SELECTORS = [
    "nav",
    "script",
    "style",
    "a.headerlink",
    "a.reference.internal.viewcode-link",  # [source] links
    "span.viewcode-link",
    "div.sphx-glr-download",
    "div.sphx-glr-download-link-note",
    "div.sphx-glr-footer",
    "div.sphx-glr-signature",
    "p.sphx-glr-timing",
    "p.sphx-glr-example-title",  # "Learn the Basics || Quickstart || ..." breadcrumb line
    "p.date-info-last-verified",  # "Created On: ... | Last Updated: ..." line
    "div.sphx-glr-thumbnails",
    "div.sphx-glr-thumbcontainer",
    "div.toctree-wrapper",
    "img",
    "span.katex-html",  # visual rendering; the MathML annotation carries the TeX source
]
CONTENT_ROOT_SELECTORS = ["article.bd-article", "#pytorch-article", "article", "main", "body"]
HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
SKIP_TAGS = {"nav", "script", "style"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    url: str
    anchor: str
    title: str
    heading_path: list[str]
    ordinal: int
    text: str
    n_tokens: int
    content_hash: str


@dataclass
class Block:
    """An atomic unit of text with the section it belongs to."""

    kind: str  # paragraph | code | table | signature | list
    text: str
    heading_path: tuple[str, ...]
    anchor: str
    rows: list[str] = field(default_factory=list)  # table rows (for row-wise splitting)
    header: str = ""  # table header line, repeated on split
    lang: str = ""  # code language / "output"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def default_token_counter() -> TokenCounter:
    """tiktoken cl100k_base; loaded lazily so unit tests can inject a cheap counter."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s, disallowed_special=()))


def word_count(text: str) -> int:
    """Cheap, deterministic stand-in counter (used in tests)."""
    return len(text.split())


# ---------------------------------------------------------------------------
# HTML -> blocks
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split())


def _content_root(soup: BeautifulSoup) -> Tag | None:
    for sel in CONTENT_ROOT_SELECTORS:
        node = soup.select_one(sel)
        if node is not None:
            return node
    return None


def _clean(root: Tag) -> None:
    # Replace KaTeX with its TeX source before dropping the rendered spans.
    for katex in root.select("span.katex"):
        ann = katex.select_one("annotation")
        tex = _norm(ann.get_text()) if ann else _norm(katex.get_text())
        katex.replace_with(NavigableString(f"${tex}$"))
    for sel in DROP_SELECTORS:
        for node in root.select(sel):
            node.decompose()


def _classes(el: Tag) -> set[str]:
    return set(el.get("class") or [])


def _table_rows(table: Tag) -> tuple[str, list[str]]:
    header = ""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [_norm(c.get_text()) for c in tr.find_all(["th", "td"], recursive=False)]
        if not cells:
            continue
        line = "| " + " | ".join(cells) + " |"
        is_header = tr.find("th", recursive=False) is not None and not rows and not header
        if is_header:
            header = line
        else:
            rows.append(line)
    return header, rows


def _list_lines(lst: Tag, depth: int = 0) -> list[str]:
    lines: list[str] = []
    ordered = lst.name == "ol"
    for i, li in enumerate(lst.find_all("li", recursive=False), 1):
        own: list[str] = []
        sublists: list[Tag] = []
        for child in li.children:
            if isinstance(child, Tag) and child.name in ("ul", "ol"):
                sublists.append(child)
            elif isinstance(child, Tag) and child.name == "pre":
                own.append(child.get_text().strip())
            elif isinstance(child, Tag):
                nested = child.find_all(["ul", "ol"])
                for n in nested:
                    sublists.append(n)
                    n.extract()
                own.append(child.get_text())
            else:
                own.append(str(child))
        bullet = f"{i}." if ordered else "-"
        lines.append("  " * depth + f"{bullet} {_norm(' '.join(own))}".rstrip())
        for sub in sublists:
            lines.extend(_list_lines(sub, depth + 1))
    return lines


_PROGRESS_LINE = re.compile(r"^\s*\d+%\|")
MAX_OUTPUT_LINES = 40


def _trim_output(code: str) -> str:
    """Drop progress-bar lines from sphinx-gallery output and cap its length: output is far
    less useful than the code that produced it."""
    lines = [ln for ln in code.split("\n") if not _PROGRESS_LINE.match(ln)]
    lines = [ln for i, ln in enumerate(lines) if ln.strip() or (i and lines[i - 1].strip())]
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES] + ["..."]
    return "\n".join(lines).strip("\n")


class _Walker:
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.stack: list[tuple[int, str]] = []  # (level, heading text)
        self.anchor = ""
        self.title = ""
        self.prefix = ""  # pending admonition prefix for the next paragraph

    # -- section state -----------------------------------------------------
    @property
    def path(self) -> tuple[str, ...]:
        return tuple(t for _, t in self.stack)

    def _set_heading(self, level: int, text: str, anchor: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, text))
        self.anchor = anchor or self.anchor

    def _emit(self, kind: str, text: str, **kw) -> None:
        text = text.strip("\n")
        if not text.strip():
            return
        if self.prefix and kind == "paragraph":
            text = f"{self.prefix} {text}"
            self.prefix = ""
        self.blocks.append(Block(kind, text, self.path, self.anchor, **kw))

    # -- traversal ---------------------------------------------------------
    def walk(self, node: Tag) -> None:
        for child in list(node.children):
            if isinstance(child, NavigableString):
                text = _norm(str(child))
                if text and node.name not in ("section", "article", "div", "main", "body"):
                    self._emit("paragraph", text)
                elif text:
                    self._emit("paragraph", text)
                continue
            if not isinstance(child, Tag) or child.name in SKIP_TAGS:
                continue
            self.visit(child)

    def visit(self, el: Tag) -> None:  # noqa: C901 - dispatch table
        name = el.name
        cls = _classes(el)

        if name in HEADING_TAGS:
            text = _norm(el.get_text())
            parent = el.parent
            anchor = ""
            if isinstance(parent, Tag) and parent.name == "section":
                anchor = parent.get("id", "") or ""
            anchor = anchor or el.get("id", "") or ""
            if name == "h1" and not self.title:
                self.title = text
            self._set_heading(HEADING_TAGS[name], text, anchor)
            return

        if name == "dl" and cls & {"py", "c", "cpp", "std"} and el.find("dt", recursive=False):
            self._visit_autodoc(el)
            return

        if name == "dl" and "field-list" in cls:
            self._visit_field_list(el)
            return

        if name == "dl":
            for child in el.find_all(["dt", "dd"], recursive=False):
                if child.name == "dt":
                    self._emit("paragraph", _norm(child.get_text()))
                else:
                    self.walk(child)
            return

        if name == "pre":
            self._visit_pre(el)
            return

        if name == "table":
            header, rows = _table_rows(el)
            if rows or header:
                text = "\n".join(([header] if header else []) + rows)
                self._emit("table", text, rows=rows, header=header)
            return

        if name in ("ul", "ol"):
            for line in _list_lines(el):
                self._emit("list", line)
            return

        if name == "p":
            self._emit("paragraph", _norm(el.get_text()))
            return

        if name == "div" and "admonition" in cls:
            title_el = el.find("p", class_="admonition-title")
            title = _norm(title_el.get_text()) if title_el else "Note"
            if title_el:
                title_el.decompose()
            self.prefix = f"{title}:"
            self.walk(el)
            self.prefix = ""
            return

        if name == "div" and cls & {"math", "line-block"}:
            self._emit("paragraph", _norm(el.get_text()))
            return

        if name == "figcaption":
            self._emit("paragraph", _norm(el.get_text()))
            return

        # Inline / container elements: recurse (or take text if purely inline).
        if name in ("section", "div", "article", "main", "body", "dd", "blockquote", "details",
                    "figure", "aside", "summary", "span", "em", "strong", "cite", "code", "a",
                    "li", "td", "th", "tr", "tbody", "thead", "small", "b", "i", "sub", "sup",
                    "label", "abbr", "kbd"):  # fmt: skip
            if name in ("section", "div", "article", "main", "body", "dd", "blockquote",
                        "details", "figure", "aside", "tbody", "thead", "tr"):  # fmt: skip
                self.walk(el)
            else:
                self._emit("paragraph", _norm(el.get_text()))
            return
        # Unknown tag: fall back to recursing so no text is lost.
        self.walk(el)

    def _visit_pre(self, el: Tag) -> None:
        lang = ""
        output = False
        for anc in el.parents:
            if not isinstance(anc, Tag):
                break
            acls = _classes(anc)
            if "sphx-glr-script-out" in acls:
                output = True
            for c in acls:
                if c.startswith("highlight-") and c != "highlight-none" and not lang:
                    lang = c.removeprefix("highlight-").lower()
            if anc.name in ("section", "article"):
                break
        code = el.get_text().rstrip()
        if output:
            code = _trim_output(code)
            if not code.strip():
                return
            self._emit("code", "Out:\n```\n" + code + "\n```", lang="output")
        else:
            fence = f"```{lang}" if lang and lang != "default" else "```"
            self._emit("code", f"{fence}\n{code}\n```", lang=lang)

    def _visit_autodoc(self, dl: Tag) -> None:
        base_level = self.stack[-1][0] if self.stack else 0
        saved = list(self.stack)
        saved_anchor = self.anchor
        for child in dl.find_all(["dt", "dd"], recursive=False):
            if child.name == "dt":
                ident = child.get("id", "") or ""
                sig = _norm(child.get_text())
                self._set_heading(base_level + 1, ident or sig, ident)
                self._emit("signature", sig)
            else:
                self.walk(child)
        self.stack = saved
        self.anchor = saved_anchor

    def _visit_field_list(self, dl: Tag) -> None:
        for child in dl.find_all(["dt", "dd"], recursive=False):
            if child.name == "dt":
                self._emit("paragraph", _norm(child.get_text()).rstrip(":") + ":")
            else:
                self.walk(child)


def html_to_blocks(html: str) -> tuple[str, list[Block]]:
    """Parse HTML into (page title, ordered blocks with heading paths)."""
    soup = BeautifulSoup(html, "lxml")
    root = _content_root(soup)
    if root is None:
        return "", []
    _clean(root)
    w = _Walker()
    w.walk(root)
    title = w.title or _norm(soup.title.get_text()) if soup.title else w.title
    return title, w.blocks


# ---------------------------------------------------------------------------
# Blocks -> chunks
# ---------------------------------------------------------------------------


def _split_lines(text: str, max_tokens: int, count: TokenCounter) -> list[str]:
    """Split text at line boundaries into pieces <= max_tokens (a single huge line is
    split at whitespace)."""
    pieces: list[str] = []
    cur: list[str] = []
    for line in text.split("\n"):
        if count(line) > max_tokens:
            if cur:
                pieces.append("\n".join(cur))
                cur = []
            words = line.split(" ")
            buf: list[str] = []
            for wd in words:
                if buf and count(" ".join(buf + [wd])) > max_tokens:
                    pieces.append(" ".join(buf))
                    buf = []
                buf.append(wd)
            if buf:
                pieces.append(" ".join(buf))
            continue
        if cur and count("\n".join(cur + [line])) > max_tokens:
            pieces.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _split_block(block: Block, max_tokens: int, count: TokenCounter) -> list[Block]:
    if count(block.text) <= max_tokens:
        return [block]
    out: list[Block] = []
    if block.kind == "code":
        lines = block.text.split("\n")
        prefix = ""
        if lines[0] == "Out:":
            prefix, lines = "Out:\n", lines[1:]
        fence_open, body, fence_close = lines[0], lines[1:-1], lines[-1]
        overhead = count(f"{prefix}{fence_open}\n\n{fence_close}") + 1
        for piece in _split_lines("\n".join(body), max_tokens - overhead, count):
            out.append(
                Block(
                    "code",
                    f"{prefix}{fence_open}\n{piece}\n{fence_close}",
                    block.heading_path,
                    block.anchor,
                    lang=block.lang,
                )  # fmt: skip
            )
        return out
    if block.kind == "table":
        overhead = count(block.header) + 1 if block.header else 0
        cur: list[str] = []
        for row in block.rows:
            if cur and count("\n".join(cur + [row])) + overhead > max_tokens:
                out.append(_table_piece(block, cur))
                cur = []
            cur.append(row)
        if cur:
            out.append(_table_piece(block, cur))
        return out
    # paragraph / list / signature: split at sentence-ish boundaries, then whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", block.text)
    for piece in _split_lines("\n".join(sentences), max_tokens, count):
        out.append(Block(block.kind, piece.replace("\n", " "), block.heading_path, block.anchor))
    return out


def _table_piece(block: Block, rows: list[str]) -> Block:
    text = "\n".join(([block.header] if block.header else []) + rows)
    return Block("table", text, block.heading_path, block.anchor, rows=rows, header=block.header)


def _join(texts: list[str]) -> str:
    return "\n\n".join(texts)


def pack_blocks(
    blocks: list[Block],
    *,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    count: TokenCounter,
) -> list[tuple[tuple[str, ...], str, str]]:
    """Greedy packing of blocks into (heading_path, anchor, text) groups.

    Blocks are packed within their section only. A group is closed when adding the next
    block would push it past ``target_tokens`` (hard cap ``max_tokens``). Undersized
    groups are merged into the previous group of the same page when the result fits.
    """
    groups: list[tuple[tuple[str, ...], str, list[str]]] = []
    cur_path: tuple[str, ...] | None = None
    cur_anchor = ""
    cur: list[str] = []

    def flush() -> None:
        nonlocal cur
        if cur and cur_path is not None:
            groups.append((cur_path, cur_anchor, cur))
        cur = []

    for raw in blocks:
        for block in _split_block(raw, max_tokens, count):
            if block.heading_path != cur_path:
                flush()
                cur_path, cur_anchor = block.heading_path, block.anchor
            if cur:
                joined = count(_join(cur + [block.text]))
                if joined > target_tokens:
                    flush()
            cur.append(block.text)
    flush()

    # Merge undersized groups: into the previous group when the result fits, otherwise
    # prepend to the next group (keeping the next group's path/anchor, e.g. a one-line
    # page intro before the first heading). A tiny group that fits nowhere stays as is.
    merged: list[tuple[tuple[str, ...], str, list[str]]] = []
    pending: tuple[tuple[str, ...], str, list[str]] | None = None  # tiny group awaiting a home
    for path, anchor, texts in groups:
        if pending is not None:
            if count(_join(pending[2] + texts)) <= max_tokens:
                texts = pending[2] + texts
            else:
                merged.append(pending)
            pending = None
        if merged and count(_join(texts)) < min_tokens:
            ppath, panchor, ptexts = merged[-1]
            if count(_join(ptexts + texts)) <= max_tokens:
                merged[-1] = (ppath, panchor, ptexts + texts)
                continue
        if count(_join(texts)) < min_tokens:
            pending = (path, anchor, texts)
            continue
        merged.append((path, anchor, texts))
    if pending is not None:
        if merged and count(_join(merged[-1][2] + pending[2])) <= max_tokens:
            ppath, panchor, ptexts = merged[-1]
            merged[-1] = (ppath, panchor, ptexts + pending[2])
        else:
            merged.append(pending)
    return [(p, a, _join(t)) for p, a, t in merged]


def make_chunk_id(url: str, heading_path: tuple[str, ...] | list[str], index: int) -> str:
    key = "\x00".join([url, "/".join(heading_path), str(index)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def chunk_html(
    html: str,
    url: str,
    *,
    target_tokens: int = TARGET_TOKENS,
    max_tokens: int = MAX_TOKENS,
    min_tokens: int = MIN_TOKENS,
    count_tokens: TokenCounter | None = None,
) -> list[Chunk]:
    """Pure function: page HTML + URL -> list of chunks (empty for pages with no content)."""
    if target_tokens > max_tokens:
        raise ValueError("target_tokens must be <= max_tokens")
    count = count_tokens or default_token_counter()
    title, blocks = html_to_blocks(html)
    packed = pack_blocks(
        blocks, target_tokens=target_tokens, max_tokens=max_tokens, min_tokens=min_tokens,
        count=count,
    )  # fmt: skip
    per_section: dict[tuple[str, ...], int] = {}
    chunks: list[Chunk] = []
    for ordinal, (path, anchor, text) in enumerate(packed):
        idx = per_section.get(path, 0)
        per_section[path] = idx + 1
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(url, path, idx),
                url=url,
                anchor=anchor,
                title=title,
                heading_path=list(path),
                ordinal=ordinal,
                text=text,
                n_tokens=count(text),
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# CLI: data/raw -> data/chunks.jsonl + stats
# ---------------------------------------------------------------------------


def url_for_path(path: Path, raw_dir: Path, host: str = "https://docs.pytorch.org") -> str:
    return f"{host}/{path.relative_to(raw_dir).as_posix()}"


def iter_raw_pages(raw_dir: Path) -> Iterator[tuple[str, str]]:
    for path in sorted(raw_dir.rglob("*.html")):
        yield url_for_path(path, raw_dir), path.read_text(encoding="utf-8", errors="replace")


def percentile(sorted_vals: list[int], q: float) -> int:
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def summarize(lengths: list[int], n_pages: int, n_empty: int) -> dict:
    s = sorted(lengths)
    edges = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 512, 10**9]
    hist = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        n = sum(1 for x in s if lo <= x < hi)
        hist.append({"range": f"{lo}-{hi - 1}" if hi < 10**9 else f"{lo}+", "count": n})
    return {
        "pages": n_pages,
        "pages_without_chunks": n_empty,
        "chunks": len(s),
        "tokens": sum(s),
        "tokens_per_chunk": {
            "mean": round(sum(s) / len(s), 1) if s else 0,
            "p5": percentile(s, 0.05),
            "p25": percentile(s, 0.25),
            "p50": percentile(s, 0.50),
            "p75": percentile(s, 0.75),
            "p95": percentile(s, 0.95),
            "max": s[-1] if s else 0,
        },
        "histogram": hist,
    }


def source_breakdown(rows: list[tuple[str, int, int]]) -> dict:
    """Pages, chunks and tokens per source, from ``(url, n_chunks, n_tokens)`` per page.

    The README quotes docs and tutorials separately; without this they were only derivable
    from `data/chunks.jsonl`, which is gitignored, so a clone could not check them.
    """
    from corpus.store import source_for_url

    out: dict[str, dict[str, int]] = {}
    for url, n_chunks, n_tokens in rows:
        s = out.setdefault(
            source_for_url(url), {"pages": 0, "pages_with_content": 0, "chunks": 0, "tokens": 0}
        )
        s["pages"] += 1
        s["pages_with_content"] += 1 if n_chunks else 0
        s["chunks"] += n_chunks
        s["tokens"] += n_tokens
    return out


def short_chunk_breakdown(
    rows: list[tuple[str, int, int]], short_by_page: dict[str, int], min_tokens: int
) -> dict:
    """How many chunks fall under ``min_tokens``, and how many of those are a whole page.

    A one-signature API stub is a page whose entire content is one short chunk. Separating
    those from short *fragments* is the point: the first is the corpus being what it is, the
    second would be the packer misbehaving.
    """
    per_page = {url: n for url, n, _ in rows}
    short = sum(short_by_page.values())
    whole_page = sum(n for url, n in short_by_page.items() if per_page.get(url) == 1)
    return {
        "under_min_tokens": short,
        "whole_page": whole_page,
        "fragments": short - whole_page,
        "min_tokens": min_tokens,
    }


def format_stats(stats: dict) -> str:
    t = stats["tokens_per_chunk"]
    lines = [
        f"pages: {stats['pages']} ({stats['pages_without_chunks']} without content)",
        f"chunks: {stats['chunks']}",
        f"tokens: {stats['tokens']}",
        "tokens/chunk: mean {mean}  p5 {p5}  p25 {p25}  p50 {p50}  p75 {p75}  p95 {p95}  "
        "max {max}".format(**t),
        "",
        "chunk length distribution (tokens):",
    ]
    top = max((h["count"] for h in stats["histogram"]), default=1) or 1
    for h in stats["histogram"]:
        bar = "#" * round(40 * h["count"] / top)
        lines.append(f"  {h['range']:>8} {h['count']:>7} {bar}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Chunk data/raw HTML into data/chunks.jsonl")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/chunks.jsonl"))
    ap.add_argument("--stats", type=Path, default=Path("data/stats.json"))
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    args = ap.parse_args(argv)

    count = default_token_counter()
    lengths: list[int] = []
    per_page: list[tuple[str, int, int]] = []
    short_by_page: dict[str, int] = {}
    n_pages = n_empty = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for url, html in iter_raw_pages(args.raw_dir):
            n_pages += 1
            chunks = chunk_html(
                html, url, target_tokens=args.target_tokens, max_tokens=args.max_tokens,
                min_tokens=args.min_tokens, count_tokens=count,
            )  # fmt: skip
            if not chunks:
                n_empty += 1
            per_page.append((url, len(chunks), sum(c.n_tokens for c in chunks)))
            short = sum(1 for c in chunks if c.n_tokens < args.min_tokens)
            if short:
                short_by_page[url] = short
            for c in chunks:
                lengths.append(c.n_tokens)
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            if n_pages % 500 == 0:
                print(f"[{n_pages} pages, {len(lengths)} chunks]", file=sys.stderr, flush=True)
    stats = summarize(lengths, n_pages, n_empty)
    stats["by_source"] = source_breakdown(per_page)
    stats["short_chunks"] = short_chunk_breakdown(per_page, short_by_page, args.min_tokens)
    stats["params"] = {
        "target_tokens": args.target_tokens,
        "max_tokens": args.max_tokens,
        "min_tokens": args.min_tokens,
        "tokenizer": "tiktoken cl100k_base",
    }
    args.stats.write_text(json.dumps(stats, indent=1) + "\n")
    print(format_stats(stats))
    print(f"\nwrote {args.out} and {args.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
