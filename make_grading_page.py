#!/usr/bin/env python3
"""Turn a judging pool YAML into one self-contained HTML grading page.

Everything needed to grade is inside the output file: the rubric, every
candidate's title, section path and snippet, and a link to the live page.
No other file has to be open.

Usage, from the repo root:
    python3 make_grading_page.py eval/pool_agent.yaml grading.html

Then open grading.html in a browser. Click 0/1/2 per candidate. Progress
autosaves in the browser. When done, press "Copy grades" and paste the result
back into the chat, or "Download" for a .txt copy.
"""
import html
import json
import sys

import yaml

RUBRIC = """
<h2>How to grade</h2>
<p class="lead">For each page, against the query <strong>as written</strong>:
if this were the only page a user got back, how much closer are they to their answer?</p>

<div class="grades">
  <div class="g g2"><span class="pill">2</span>
    <strong>The page carries the answer.</strong>
    The specific thing asked for is on this page and the user stops looking.
    More than one page can be a 2.</div>
  <div class="g g1"><span class="pill">1</span>
    <strong>The page carries a pointer, or one necessary piece.</strong>
    They don't finish here, but they leave with something load-bearing: the name of the
    function they actually need, an index entry listing it, a prerequisite concept.</div>
  <div class="g g0"><span class="pill">0</span>
    <strong>The page does not move the user.</strong>
    Same library, same vocabulary, no help. They'd restart their search.</div>
</div>

<h3>The test that decides 1 vs 0</h3>
<p><strong>Name the next step.</strong> If you can say "they'd read this, then go to X"
&mdash; that's a <strong>1</strong>, and X is what makes it one. If the honest sentence is
"they'd read this, then go back to the search box" &mdash; that's a <strong>0</strong>.
Shared vocabulary is not containment.</p>

<h3>The test that decides 1 vs 2</h3>
<p>A <strong>2</strong> needs the query's specific ask answered <em>on the page</em>. If the query
names an argument, a parameter or a return value, that thing must be documented here.
A page about the right function that doesn't cover the asked-about detail is a <strong>1</strong>.</p>

<h3>Rules</h3>
<ul>
  <li><strong>Torn? Take the lower grade.</strong> More than ~15 seconds means torn.</li>
  <li><strong>Grade against the original query</strong>, not the sub-query that surfaced the page.
      These candidates came from agent sub-queries; that's what's being measured.</li>
  <li><strong>Ignore which system found it.</strong> Provenance is not evidence.</li>
  <li><strong>No quotas.</strong> Some queries have many relevant pages, some have few.</li>
  <li><strong>Grade everything.</strong> A blank counts as 0 in the metrics, so a skip is an assertion.</li>
</ul>
"""

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relevance grading &mdash; @@N@@ candidates</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --panel2:#1d212a; --line:#2a2f3a;
    --fg:#e6e9ef; --dim:#9aa3b2; --accent:#7aa2f7;
    --c0:#f7768e; --c1:#e0af68; --c2:#9ece6a;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  .wrap { max-width:1000px; margin:0 auto; padding:24px 20px 120px; }
  h1 { font-size:22px; margin:0 0 4px; }
  h2 { font-size:17px; margin:22px 0 8px; }
  h3 { font-size:14px; margin:16px 0 6px; color:var(--accent); }
  .sub { color:var(--dim); margin:0 0 20px; }
  .card { background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:18px 20px; margin-bottom:20px; }
  .lead { margin-top:0; }
  .grades { display:grid; gap:8px; margin:12px 0; }
  .g { background:var(--panel2); border-radius:8px; padding:10px 12px;
    border-left:3px solid var(--line); font-size:14px; }
  .g0 { border-left-color:var(--c0); } .g1 { border-left-color:var(--c1); }
  .g2 { border-left-color:var(--c2); }
  .pill { display:inline-block; min-width:22px; text-align:center;
    font-weight:700; margin-right:8px; }
  .g2 .pill { color:var(--c2); } .g1 .pill { color:var(--c1); } .g0 .pill { color:var(--c0); }
  ul { margin:6px 0; padding-left:20px; } li { margin:3px 0; }
  .qhead { position:sticky; top:0; z-index:5; background:var(--bg);
    padding:14px 0 10px; border-bottom:1px solid var(--line); margin-bottom:14px; }
  .qid { color:var(--dim); font-size:12px; letter-spacing:.08em; text-transform:uppercase; }
  .qtext { font-size:18px; font-weight:650; margin:2px 0 4px; }
  .qmeta { color:var(--dim); font-size:13px; }
  .cand { background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:14px 16px; margin-bottom:12px; transition:border-color .12s; }
  .cand.done { border-color:#313a2c; }
  .cand.done[data-g="0"] { border-left:3px solid var(--c0); }
  .cand.done[data-g="1"] { border-left:3px solid var(--c1); }
  .cand.done[data-g="2"] { border-left:3px solid var(--c2); }
  .ctitle { font-weight:650; font-size:15px; margin-bottom:2px; }
  .csec { color:var(--dim); font-size:12.5px; margin-bottom:8px; word-break:break-word; }
  .csnip { background:var(--panel2); border-radius:6px; padding:10px 12px;
    font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
    color:#c9d1e3; margin-bottom:10px; white-space:pre-wrap; word-break:break-word; }
  .row { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  .btn { background:var(--panel2); color:var(--fg); border:1px solid var(--line);
    border-radius:7px; padding:7px 16px; font-size:14px; font-weight:600;
    cursor:pointer; font-family:inherit; }
  .btn:hover { border-color:var(--accent); }
  .btn.sel[data-v="0"] { background:var(--c0); color:#1a1013; border-color:var(--c0); }
  .btn.sel[data-v="1"] { background:var(--c1); color:#1c1608; border-color:var(--c1); }
  .btn.sel[data-v="2"] { background:var(--c2); color:#131c0c; border-color:var(--c2); }
  a.link { color:var(--accent); font-size:13px; text-decoration:none; margin-left:auto; }
  a.link:hover { text-decoration:underline; }
  .bar { position:fixed; left:0; right:0; bottom:0; background:var(--panel);
    border-top:1px solid var(--line); padding:12px 20px; z-index:10; }
  .barin { max-width:1000px; margin:0 auto; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }
  .prog { flex:1; min-width:180px; height:7px; background:var(--panel2);
    border-radius:4px; overflow:hidden; }
  .fill { height:100%; width:0; background:var(--accent); transition:width .2s; }
  .count { font-variant-numeric:tabular-nums; font-size:14px; color:var(--dim); white-space:nowrap; }
  .out { width:100%; margin-top:10px; height:110px; background:var(--panel2);
    color:var(--fg); border:1px solid var(--line); border-radius:7px; padding:10px;
    font:12px/1.4 ui-monospace,monospace; display:none; }
</style></head><body><div class="wrap">

<h1>Relevance grading</h1>
<p class="sub">@@N@@ candidates across @@NQ@@ queries &middot; from @@SRC@@</p>

<div class="card">@@RUBRIC@@</div>

@@BODY@@

</div>
<div class="bar"><div class="barin">
  <div class="prog"><div class="fill" id="fill"></div></div>
  <div class="count" id="count">0 / @@N@@</div>
  <button class="btn" onclick="jumpNext()">Next ungraded</button>
  <button class="btn" onclick="showOut()">Copy grades</button>
  <button class="btn" onclick="dl()">Download</button>
  <textarea class="out" id="out" readonly></textarea>
</div></div>

<script>
const IDS = @@IDS@@;
const TOTAL = IDS.length;
const KEY = "grading:" + @@SRCJSON@@;
let G = {};
try { G = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { G = {}; }

function paint() {
  let n = 0;
  for (const id of IDS) {
    const card = document.getElementById("c-" + id);
    const v = G[id];
    for (const b of card.querySelectorAll(".btn")) {
      b.classList.toggle("sel", String(v) === b.dataset.v);
    }
    if (v === undefined) { card.classList.remove("done"); card.removeAttribute("data-g"); }
    else { card.classList.add("done"); card.setAttribute("data-g", String(v)); n++; }
  }
  document.getElementById("count").textContent = n + " / " + TOTAL;
  document.getElementById("fill").style.width = (100 * n / TOTAL) + "%";
}

function setG(id, v) {
  G[id] = v;
  try { localStorage.setItem(KEY, JSON.stringify(G)); } catch (e) {}
  paint();
  const i = IDS.indexOf(id);
  for (let j = i + 1; j < IDS.length; j++) {
    if (G[IDS[j]] === undefined) {
      document.getElementById("c-" + IDS[j]).scrollIntoView({behavior:"smooth", block:"center"});
      return;
    }
  }
}

function jumpNext() {
  for (const id of IDS) {
    if (G[id] === undefined) {
      document.getElementById("c-" + id).scrollIntoView({behavior:"smooth", block:"center"});
      return;
    }
  }
  alert("All graded.");
}

function text() {
  return IDS.map(id => id + ": " + (G[id] === undefined ? "?" : G[id])).join("\\n");
}

function showOut() {
  const missing = IDS.filter(id => G[id] === undefined).length;
  const t = document.getElementById("out");
  t.style.display = "block";
  t.value = text();
  t.select();
  try { document.execCommand("copy"); } catch (e) {}
  if (missing) alert(missing + " still ungraded - they show as '?'.");
}

function dl() {
  const b = new Blob([text()], {type:"text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(b); a.download = "grades.txt"; a.click();
}

paint();
</script></body></html>
"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "eval/pool_agent.yaml"
    dst = sys.argv[2] if len(sys.argv) > 2 else "grading.html"

    with open(src) as f:
        doc = yaml.safe_load(f)

    queries = doc["queries"] if isinstance(doc, dict) and "queries" in doc else doc

    if isinstance(queries, dict):
        items = sorted(queries.items())
    else:
        items = [(q.get("id", "q%02d" % (i + 1)), q) for i, q in enumerate(queries)]

    esc = html.escape
    ids, parts = [], []

    for qid, q in items:
        cands = q.get("candidates", [])
        if not cands:
            continue
        parts.append(
            '<div class="qhead"><div class="qid">%s &middot; %s &middot; %d candidates</div>'
            '<div class="qtext">%s</div></div>'
            % (esc(qid), esc(str(q.get("category", ""))), len(cands), esc(str(q.get("query", ""))))
        )
        for i, c in enumerate(cands, 1):
            cid = "%s-%02d" % (qid, i)
            ids.append(cid)
            url = str(c.get("url", ""))
            btns = "".join(
                '<button class="btn" data-v="%d" onclick="setG(\'%s\',%d)">%d</button>'
                % (v, cid, v, v)
                for v in (0, 1, 2)
            )
            link = (
                '<a class="link" href="%s" target="_blank" rel="noopener">open page &rarr;</a>'
                % esc(url)
                if url
                else ""
            )
            parts.append(
                '<div class="cand" id="c-%s">'
                '<div class="ctitle">%s</div>'
                '<div class="csec">%s</div>'
                '<div class="csnip">%s</div>'
                '<div class="row">%s%s</div></div>'
                % (
                    cid,
                    esc(str(c.get("title", "(no title)"))),
                    esc(str(c.get("section", ""))),
                    esc(str(c.get("snippet", ""))),
                    btns,
                    link,
                )
            )

    out = PAGE
    for token, value in [
        ("@@RUBRIC@@", RUBRIC),
        ("@@BODY@@", "\n".join(parts)),
        ("@@IDS@@", json.dumps(ids)),
        ("@@SRCJSON@@", json.dumps(src)),
        ("@@SRC@@", esc(src)),
        ("@@NQ@@", str(len(items))),
        ("@@N@@", str(len(ids))),
    ]:
        out = out.replace(token, value)

    with open(dst, "w") as f:
        f.write(out)

    print("Wrote %s - %d candidates across %d queries." % (dst, len(ids), len(items)))
    print("Open it in a browser, grade, then press 'Copy grades'.")


if __name__ == "__main__":
    main()
