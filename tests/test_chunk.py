from pathlib import Path

import pytest

from corpus.chunk import (
    Block,
    chunk_html,
    html_to_blocks,
    make_chunk_id,
    pack_blocks,
    percentile,
    short_chunk_breakdown,
    source_breakdown,
    summarize,
    word_count,
)

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://example.org/page.html"


def wrap(body: str) -> str:
    return f'<html><body><article class="bd-article">{body}</article></body></html>'


def chunk(body: str, **kw) -> list:
    kw.setdefault("count_tokens", word_count)
    return chunk_html(wrap(body), URL, **kw)


# --- HTML -> blocks -------------------------------------------------------------


def test_empty_page_gives_no_chunks():
    assert chunk_html("<html><body></body></html>", URL, count_tokens=word_count) == []
    assert chunk("") == []


def test_heading_path_tracks_levels():
    body = """
    <section id="top"><h1>Top</h1><p>intro</p>
      <section id="a"><h2>A</h2><p>in a</p>
        <section id="a1"><h3>A1</h3><p>in a1</p></section>
      </section>
      <section id="b"><h2>B</h2><p>in b</p></section>
    </section>"""
    title, blocks = html_to_blocks(wrap(body))
    assert title == "Top"
    got = [(b.heading_path, b.anchor, b.text) for b in blocks]
    assert got == [
        (("Top",), "top", "intro"),
        (("Top", "A"), "a", "in a"),
        (("Top", "A", "A1"), "a1", "in a1"),
        (("Top", "B"), "b", "in b"),
    ]


def test_autodoc_entries_become_sections_with_signature():
    body = """
    <section id="linear"><h1>Linear</h1>
    <dl class="py class">
      <dt class="sig" id="torch.nn.Linear">class torch.nn.Linear(in_features)
        <a class="headerlink" href="#x">#</a></dt>
      <dd><p>Applies a linear map.</p>
        <dl class="py method">
          <dt class="sig" id="torch.nn.Linear.forward">forward(input)</dt>
          <dd><p>Runs forward.</p></dd>
        </dl>
      </dd>
    </dl>
    <p>after</p>
    </section>"""
    _, blocks = html_to_blocks(wrap(body))
    got = [(b.kind, b.heading_path, b.anchor, b.text) for b in blocks]
    assert got == [
        ("signature", ("Linear", "torch.nn.Linear"), "torch.nn.Linear",
         "class torch.nn.Linear(in_features)"),
        ("paragraph", ("Linear", "torch.nn.Linear"), "torch.nn.Linear", "Applies a linear map."),
        ("signature", ("Linear", "torch.nn.Linear", "torch.nn.Linear.forward"),
         "torch.nn.Linear.forward", "forward(input)"),
        ("paragraph", ("Linear", "torch.nn.Linear", "torch.nn.Linear.forward"),
         "torch.nn.Linear.forward", "Runs forward."),
        ("paragraph", ("Linear",), "linear", "after"),
    ]  # fmt: skip


def test_code_block_is_fenced_with_language_and_kept_verbatim():
    body = """<section id="s"><h1>S</h1>
    <div class="highlight-python notranslate"><div class="highlight"><pre>x = 1
if x:
    y = 2</pre></div></div></section>"""
    _, blocks = html_to_blocks(wrap(body))
    assert blocks[0].kind == "code"
    assert blocks[0].text == "```python\nx = 1\nif x:\n    y = 2\n```"


def test_gallery_output_is_labelled_and_progress_bars_dropped():
    body = """<section id="s"><h1>S</h1>
    <div class="sphx-glr-script-out highlight-none notranslate"><div class="highlight"><pre>
 10%|#         | 1/10
100%|##########| 10/10
tensor([1., 2.])</pre></div></div></section>"""
    _, blocks = html_to_blocks(wrap(body))
    assert blocks[0].text == "Out:\n```\ntensor([1., 2.])\n```"


def test_table_rows_with_header():
    body = """<section id="s"><h1>S</h1>
    <table><thead><tr><th>Name</th><th>Desc</th></tr></thead>
    <tbody><tr><td>relu</td><td>Rectifier</td></tr><tr><td>tanh</td><td>Hyperbolic</td></tr></tbody>
    </table></section>"""
    _, blocks = html_to_blocks(wrap(body))
    b = blocks[0]
    assert b.kind == "table"
    assert b.header == "| Name | Desc |"
    assert b.rows == ["| relu | Rectifier |", "| tanh | Hyperbolic |"]
    assert b.text == "| Name | Desc |\n| relu | Rectifier |\n| tanh | Hyperbolic |"


def test_lists_admonitions_field_lists_and_math():
    body = """<section id="s"><h1>S</h1>
    <ul><li><p>one</p><ul><li><p>nested</p></li></ul></li><li><p>two</p></li></ul>
    <div class="admonition note"><p class="admonition-title">Note</p><p>be careful</p></div>
    <dl class="field-list simple"><dt>Parameters</dt>
      <dd><ul><li><p>x (int) – the x</p></li></ul></dd></dl>
    <p>eq <span class="math"><span class="katex"><span class="katex-mathml">
    <math><semantics><mrow><mi>y</mi></mrow>
    <annotation encoding="application/x-tex">y = xA^T</annotation></semantics></math></span>
    <span class="katex-html">y=xAT</span></span></span> here</p>
    </section>"""
    _, blocks = html_to_blocks(wrap(body))
    assert [b.text for b in blocks] == [
        "- one",
        "  - nested",
        "- two",
        "Note: be careful",
        "Parameters:",
        "- x (int) – the x",
        "eq $y = xA^T$ here",
    ]


def test_chrome_is_dropped():
    body = """<nav>skip me</nav><section id="s"><h1>S<a class="headerlink" href="#s">#</a></h1>
    <p class="date-info-last-verified">Created On: 2020</p>
    <div class="sphx-glr-download"><p>Download python</p></div>
    <script>alert(1)</script><p>keep</p></section>"""
    _, blocks = html_to_blocks(wrap(body))
    assert [b.text for b in blocks] == ["keep"]
    assert blocks[0].heading_path == ("S",)


# --- packing ---------------------------------------------------------------------


def blk(text, path=("S",), kind="paragraph", **kw):
    return Block(kind, text, path, "s", **kw)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_packing_respects_target_and_section_boundaries():
    blocks = [blk(words(6)), blk(words(6)), blk(words(6)), blk(words(6), path=("S", "T"))]
    groups = pack_blocks(blocks, target_tokens=12, max_tokens=20, min_tokens=1, count=word_count)
    assert [(p, word_count(t)) for p, _, t in groups] == [
        (("S",), 12),
        (("S",), 6),
        (("S", "T"), 6),
    ]


def test_packing_never_exceeds_max_tokens():
    blocks = [blk(words(7, f"b{i}")) for i in range(30)]
    groups = pack_blocks(blocks, target_tokens=20, max_tokens=24, min_tokens=1, count=word_count)
    assert all(word_count(t) <= 24 for _, _, t in groups)
    assert sum(word_count(t) for _, _, t in groups) == 210


def test_oversized_code_block_is_split_at_lines_and_fence_repeated():
    code = "```python\n" + "\n".join(f"line{i} a b" for i in range(10)) + "\n```"
    groups = pack_blocks(
        [blk(code, kind="code")], target_tokens=12, max_tokens=12, min_tokens=1, count=word_count
    )
    assert len(groups) > 1
    for _, _, t in groups:
        assert t.startswith("```python\n") and t.endswith("\n```")
        assert word_count(t) <= 12
    body = [ln for _, _, t in groups for ln in t.split("\n")[1:-1]]
    assert body == [f"line{i} a b" for i in range(10)]


def test_small_code_block_is_never_split():
    code = "```\n" + "\n".join(f"l{i}" for i in range(5)) + "\n```"
    blocks = [blk(words(9)), blk(code, kind="code"), blk(words(9))]
    groups = pack_blocks(blocks, target_tokens=10, max_tokens=12, min_tokens=1, count=word_count)
    assert [t for _, _, t in groups][1] == code


def test_oversized_table_split_by_rows_with_header_repeated():
    header = "| n | d |"
    rows = [f"| r{i} | desc{i} |" for i in range(8)]
    table = blk("\n".join([header] + rows), kind="table", rows=rows, header=header)
    groups = pack_blocks([table], target_tokens=20, max_tokens=20, min_tokens=1, count=word_count)
    assert len(groups) > 1
    seen = []
    for _, _, t in groups:
        lines = t.split("\n")
        assert lines[0] == header
        assert word_count(t) <= 20
        seen.extend(lines[1:])
    assert seen == rows  # no row split, none lost


def test_tiny_trailing_section_merges_into_previous():
    blocks = [blk(words(30)), blk("tiny one", path=("S", "T"))]
    groups = pack_blocks(blocks, target_tokens=40, max_tokens=60, min_tokens=5, count=word_count)
    assert len(groups) == 1
    assert groups[0][0] == ("S",)
    assert groups[0][2].endswith("tiny one")


def test_tiny_leftover_after_full_chunk_merges_into_next_section():
    # previous chunk is full (30 of max 31), so "tiny one" cannot join it; it is prepended
    # to the following section's chunk instead of becoming a 2-word chunk.
    blocks = [blk(words(30)), blk("tiny one", path=("S", "T")), blk(words(20), path=("S", "U"))]
    groups = pack_blocks(blocks, target_tokens=30, max_tokens=31, min_tokens=5, count=word_count)
    assert [(p, word_count(t)) for p, _, t in groups] == [(("S",), 30), (("S", "U"), 22)]
    assert groups[1][2].startswith("tiny one")


def test_tiny_group_that_fits_nowhere_is_kept():
    blocks = [blk(words(30)), blk("tiny one", path=("S", "T")), blk(words(31), path=("S", "U"))]
    groups = pack_blocks(blocks, target_tokens=31, max_tokens=31, min_tokens=5, count=word_count)
    assert [(p, word_count(t)) for p, _, t in groups] == [
        (("S",), 30),
        (("S", "T"), 2),
        (("S", "U"), 31),
    ]


def test_tiny_leading_group_merges_into_next_and_keeps_next_path():
    blocks = [blk("nav line", path=()), blk(words(30), path=("S",))]
    groups = pack_blocks(blocks, target_tokens=40, max_tokens=60, min_tokens=5, count=word_count)
    assert len(groups) == 1
    assert groups[0][0] == ("S",)
    assert groups[0][2].startswith("nav line")


# --- chunk_html end to end ------------------------------------------------------


def test_chunk_fields_and_ordinals():
    body = (
        '<section id="s"><h1>S</h1><p>%s</p><section id="t"><h2>T</h2><p>%s</p></section></section>'
    )
    chunks = chunk(body % (words(50), words(50)), target_tokens=60, max_tokens=80, min_tokens=5)
    assert [c.ordinal for c in chunks] == [0, 1]
    assert [c.heading_path for c in chunks] == [["S"], ["S", "T"]]
    assert [c.anchor for c in chunks] == ["s", "t"]
    assert all(c.url == URL and c.title == "S" for c in chunks)
    assert all(c.n_tokens == word_count(c.text) for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 2
    assert all(len(c.content_hash) == 16 for c in chunks)


def test_chunk_ids_are_deterministic_and_position_based():
    body = (
        '<section id="s"><h1>S</h1><p>%s</p><section id="t"><h2>T</h2><p>%s</p></section></section>'
    )
    a = chunk(body % (words(50), words(50)), target_tokens=60, max_tokens=80, min_tokens=5)
    b = chunk(body % (words(50), words(50)), target_tokens=60, max_tokens=80, min_tokens=5)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    # editing section T leaves S's id unchanged but changes T's content hash, not its id
    c = chunk(
        body % (words(50), words(50, "edited")), target_tokens=60, max_tokens=80, min_tokens=5
    )
    assert c[0].chunk_id == a[0].chunk_id
    assert c[1].chunk_id == a[1].chunk_id
    assert c[1].content_hash != a[1].content_hash
    assert make_chunk_id(URL, ["S"], 0) != make_chunk_id(URL, ["S"], 1)
    assert make_chunk_id(URL, ["S"], 0) != make_chunk_id("https://other/", ["S"], 0)


def test_target_above_max_is_rejected():
    with pytest.raises(ValueError):
        chunk("<p>x</p>", target_tokens=10, max_tokens=5)


def test_real_api_page_fixture():
    html = (FIXTURES / "torch.nn.Linear.html").read_text()
    chunks = chunk_html(html, URL, count_tokens=word_count)
    assert chunks, "expected chunks from the Linear page"
    assert chunks[0].title == "Linear"
    assert chunks[0].heading_path == ["Linear", "torch.nn.Linear"]
    assert chunks[0].anchor == "torch.nn.Linear"
    assert chunks[0].text.startswith("class torch.nn.Linear(in_features, out_features")
    joined = "\n".join(c.text for c in chunks)
    assert "$y = xA^T + b$" in joined  # KaTeX replaced by TeX source
    assert "- in_features (int) – size of each input sample" in joined
    assert ">>> m = nn.Linear(20, 30)" in joined
    assert "forward(input)" in joined
    assert "#" not in joined.replace("#L", "")  # no headerlink pilcrows / [source] links
    assert "[source]" not in joined


def test_real_tutorial_page_fixture():
    html = (FIXTURES / "data_tutorial.html").read_text()
    chunks = chunk_html(html, URL, count_tokens=word_count)
    paths = [c.heading_path for c in chunks]
    assert chunks[0].title == "Datasets & DataLoaders"
    assert ["Datasets & DataLoaders", "Loading a Dataset"] in paths
    assert [
        "Datasets & DataLoaders",
        "Creating a Custom Dataset for your files",
        "__getitem__",
    ] in paths
    joined = "\n".join(c.text for c in chunks)
    assert "Created On" not in joined
    assert "Learn the Basics ||" not in joined
    assert "Download Python source code" not in joined
    assert "```python\nimport torch\nfrom torch.utils.data import Dataset" in joined
    assert "%|" not in joined  # progress bars stripped from outputs
    assert all(c.n_tokens <= 512 for c in chunks)


# --- stats helpers ---------------------------------------------------------------


def test_percentile_and_summary():
    assert percentile([], 0.5) == 0
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([10, 20], 0.5) == 15
    s = summarize([10, 60, 120, 600], n_pages=3, n_empty=1)
    assert s["chunks"] == 4 and s["tokens"] == 790 and s["pages"] == 3
    assert s["tokens_per_chunk"]["max"] == 600
    assert {h["range"]: h["count"] for h in s["histogram"]}["512+"] == 1


# --- stats breakdowns -----------------------------------------------------------------------

DOC = "https://docs.pytorch.org/docs/2.14/generated/torch.topk.html"
TUT = "https://docs.pytorch.org/tutorials/beginner/basics/intro.html"


def test_source_breakdown_splits_docs_from_tutorials():
    rows = [(DOC, 3, 300), (DOC + "?x", 1, 40), (TUT, 2, 500)]
    out = source_breakdown(rows)
    assert out["docs"] == {"pages": 2, "pages_with_content": 2, "chunks": 4, "tokens": 340}
    assert out["tutorials"] == {"pages": 1, "pages_with_content": 1, "chunks": 2, "tokens": 500}


def test_source_breakdown_counts_an_empty_page_but_not_as_content():
    out = source_breakdown([(DOC, 0, 0), (DOC + "?x", 1, 10)])
    assert out["docs"]["pages"] == 2 and out["docs"]["pages_with_content"] == 1


def test_short_chunks_separate_whole_pages_from_fragments():
    """A page whose only chunk is short is an API stub; a short chunk beside others is not."""
    rows = [(DOC, 1, 12), (TUT, 4, 900)]
    out = short_chunk_breakdown(rows, {DOC: 1, TUT: 2}, 40)
    assert out["under_min_tokens"] == 3
    assert out["whole_page"] == 1  # DOC's single chunk
    assert out["fragments"] == 2  # TUT's two short chunks among four


def test_short_chunks_are_zero_when_nothing_is_short():
    out = short_chunk_breakdown([(DOC, 2, 400)], {}, 40)
    assert out == {"under_min_tokens": 0, "whole_page": 0, "fragments": 0, "min_tokens": 40}
