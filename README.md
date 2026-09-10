# rag — PyTorch docs retrieval: corpus + evaluation harness

Retrieval over PyTorch's documentation, built in order: corpus, then measurement, then
retrievers. The evaluation set was written before any retriever existed so nothing could be
tuned against it. All 40 queries are now judged by hand: **814 relevance judgments**,
pooled in three rounds (BM25, then dense, then the agent re-pool).

```
corpus/fetch.py     download docs (stable = 2.14.0) + tutorials HTML -> data/raw/ (gitignored)
corpus/chunk.py     HTML -> text -> chunks (pure, unit-tested)        -> data/chunks.jsonl
corpus/quality.py   alias-page and deprecated-stub detection (pure)
corpus/store.py     SQLite + FTS5 schema and idempotent loader         -> data/corpus.db
retrieval/bm25.py   BM25 baseline over the FTS5 index
retrieval/dense.py  dense retrieval: gte-modernbert-base embeddings + flat index
retrieval/hybrid.py reciprocal rank fusion over BM25 + dense (k = 60, untuned)
retrieval/rerank.py cross-encoder reranker over any first-stage retriever's top N
eval/queries.yaml   40 queries + 814 hand-written relevance judgments
eval/metrics.py     recall@k, MRR, nDCG@10 (pure functions, hand-computed tests)
eval/run.py         harness: any object with search(query, k) -> metrics table
eval/pool.py        build a judging pool from retriever output, merge/append grades back
eval/paired.py      paired bootstrap: every system scored on one shared judgment draw
eval/alias_slots.py how many top-k slots are a page the retriever already returned
eval/GRADING.md     the written grading rubric
agent/graph.py      decompose/route state graph, RRF fusion over sub-query results
agent/llm.py        Ollama HTTP client + committed on-disk response cache
agent/prompts.py    the two prompts, first draft, frozen
agent/run.py        run the agent over the query set: routes, fallbacks, latency split
agent/coverage.py   how much of what the agent retrieves has never been judged
generate/prompts.py three prompts (answer, decompose, check), first draft, frozen
generate/run.py     answer a query from a retriever's top-k chunks, with citations
generate/faithfulness.py  claim-level groundedness, plus the refusal probe
generate/report.py  the generation section's tables, with bootstrap intervals
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m corpus.fetch      # ~800 MB into data/raw, resumable, ~3 min at 8 workers
.venv/bin/python -m corpus.chunk      # writes data/chunks.jsonl + data/stats.json, prints stats
.venv/bin/python -m corpus.store      # loads into data/corpus.db + quality pass (safe to re-run)
.venv/bin/python -m retrieval.bm25 "how do I write a custom Dataset"   # ad-hoc search
.venv/bin/python -m eval.pool build   # -> eval/pool.yaml, the pages to grade by hand
.venv/bin/python -m eval.pool merge   # folds your grades into eval/queries.yaml
.venv/bin/python -m eval.run --retriever bm25 -v
```

The agent layer is optional and installs separately, like the embedder:

```bash
.venv/bin/pip install -r requirements-agent.txt
ollama serve && ollama pull qwen2.5:7b        # only needed for live calls
.venv/bin/python -m agent.run --offline                    # replays the committed cache
.venv/bin/python -m agent.coverage --offline --baseline    # unjudged-page audit
```

Generation shares those dependencies and that cache machinery; it needs no others:

```bash
.venv/bin/python -m generate.run --retriever hybrid --offline -v   # answers, from cache
.venv/bin/python -m generate.faithfulness --all --offline          # the whole table
```

CI (`.github/workflows/ci.yml`): `ruff check`, `ruff format --check`, `pytest -q`. The test
suite never contacts Ollama — every model call is mocked — so it passes with nothing served.

## Corpus

Source: `https://docs.pytorch.org/docs/stable/sitemap.xml` (stable resolves to **2.14**, source
links pin **v2.14.0**) plus `https://docs.pytorch.org/tutorials/sitemap.xml` minus `unstable/`
and index/timing pages. Chunks store the versioned URL (`/docs/2.14/...`) since `stable` moves.
`data/manifest.json` (committed) records the version, fetch date and the full URL list.

Numbers from `data/stats.json` (fetched 2026-09-07, tiktoken `cl100k_base`):

| | pages | chunks | tokens |
|---|---:|---:|---:|
| docs 2.14.0 | 3,436 (3,433 with content) | 9,058 | 1,596,670 |
| tutorials | 248 (245 with content) | 2,774 | 669,818 |
| **total** | **3,684** | **11,832** | **2,266,488** |

Six pages have no chunkable content (pure toctree index pages). Chunk length in tokens:

| mean | p5 | p25 | p50 | p75 | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 191.6 | 18 | 80 | 178 | 297 | 410 | 512 |

```
      0-49    1645 ################################
     50-99    2033 ########################################
   100-149    1435 ############################
   150-199    1450 #############################
   200-249    1227 ########################
   250-299    1180 #######################
   300-349    1948 ######################################
   350-399     296 ######
   400-449     121 ##
   450-511     488 ##########
      512+       9
```

The bump at 300-349 is the packer closing chunks at the 350-token target; the 450-511 tail is
mostly single code blocks or tables that are atomic. Of the 1,257 chunks under 40 tokens,
1,248 are whole pages (one-signature API stubs such as `torch.distributed.run.main`), which
have nothing to merge with.

## Chunking strategy

Section-aware block packing, no overlap (`corpus/chunk.py`).

- **Content root**: the Sphinx `article` element. Navigation, header-link pilcrows, `[source]`
  links, sphinx-gallery download boxes, the "Created On / Last Updated" line and the tutorial
  breadcrumb line are dropped. KaTeX math is replaced by its TeX source (`$y = xA^T + b$`).
- **Headings**: `h1..h6` maintain a heading stack by level. Every chunk records its
  `heading_path` (e.g. `["Autograd mechanics", "How autograd encodes the history"]`) and the
  `anchor` (Sphinx section id) of its innermost section, so `url#anchor` is a deep link.
  Autodoc entries (`<dt id="torch.nn.Linear.forward">`) are treated as headings one level
  below their enclosing heading, so each documented API object is its own section and an
  exact-API query maps to a chunk whose first line is the signature.
- **Blocks**: the DOM is flattened in document order into paragraphs, list items (nested
  lists indented), definition/field lists (`Parameters:` + bullets), admonitions (`Note: ...`),
  code blocks (fenced with the language; gallery outputs fenced under `Out:` with progress bars
  removed and capped at 40 lines), and tables (one `| a | b |` line per row, header first).
- **Packing**: consecutive blocks of the same section are packed greedily up to
  `target_tokens=350`, hard cap `max_tokens=512` (tiktoken `cl100k_base`). Chunks never cross
  section boundaries. **Code blocks are atomic** and only split at line boundaries when a
  single block exceeds the cap (fence repeated). **Table rows are atomic**; an oversize table is
  split by rows with the header repeated on every piece. Chunks under `min_tokens=40` are
  merged into the previous chunk, else prepended to the next, when the result fits.
- **Why no overlap**: overlap duplicates text in the FTS index and muddies judgments. The
  heading path is stored (and FTS-indexed) as its own column, which restores the context an
  overlap would have provided at zero duplication.
- **Stable ids**: `chunk_id = sha256(url, heading_path, index-within-section)[:16]`. Ids are
  positional, not content-based, so re-fetching a slightly edited page keeps ids (and therefore
  relevance judgments) valid; `content_hash` flags changed text.

## Store

`data/corpus.db`: `pages(url, title, source, version, fetched_at, n_chunks)`,
`chunks(chunk_id, url, anchor, title, heading_path, ordinal, text, n_tokens, content_hash)` and an
external-content FTS5 table `chunks_fts(title, heading_path, text)` kept in sync by triggers.
Tokenizer `porter unicode61 tokenchars '_'` keeps `cross_entropy` as one token. The loader
upserts by `chunk_id` and deletes stale chunks of any page in the batch, in one transaction;
running it twice leaves the database unchanged. `pages` also carries `canonical_url` and
`status`, filled by the quality pass below; a database made before those columns existed is
migrated in place.

## Corpus quality

Two properties of the PyTorch docs distort evaluation. Both are **recorded, not fixed by
deletion**, so the decision to merge or drop pages stays open (`corpus/quality.py`).

**Alias pages: 270 groups covering 540 pages, 270 of them redundant.** Sphinx autodoc
publishes the same object at several URLs, e.g. `torch.optim.Adam` and
`torch.optim.adam.Adam_class`. The text is identical apart from the dotted path in
signatures. Detection needs all three of: identical text after collapsing dotted identifiers
to their last segment; one URL's dotted path being a subsequence of the other's; and the same
device namespace. The last two conditions are not optional. 18 groups have identical
normalized text but document genuinely different APIs, `torch.cuda.current_device` versus
`torch.xpu.current_device` among them, and text alone merges them wrongly.

**Deprecated stubs: 36 pages.** Redirect notices such as "This tutorial was deprecated" with
no content. They match queries lexically and answer nothing. Detection requires a single
short chunk *and* redirect wording, because a real 5,000-token tutorial in the corpus
mentions deprecation in passing and must not be caught.

This is not cosmetic. In the unfiltered BM25 baseline, **161 of 400 result slots (40%) across
the 40 queries were duplicate alias pages**, and 39 of 40 queries had their top 10 change once
aliases were collapsed. Stubs cost only 1 slot. All four counts come from:

```
python -m eval.alias_slots            # 161 / 400, 39 of 40, 1 stub slot, and the 162 below
``` Judgments written against an uncleaned corpus
would have had to list every alias of every answer to score correctly.

## Retrieval

`retrieval/bm25.py` is the lexical baseline: SQLite's own `bm25()` over the FTS5 index, with
column weights favouring title and heading path over body text. Anything added later has to
beat it.

The awkward part is query translation. FTS5's MATCH grammar treats punctuation, hyphens and
its own keywords as syntax, so a natural-language question is not a valid query string.
`to_match_query` extracts identifier-like terms, drops stopwords, quotes each term and joins
them with `OR`, letting BM25 do the ranking. Dotted names are also split, so a query naming
`torch.nn.functional.cross_entropy` still matches pages that only say `cross_entropy`.

Alias collapsing and stub exclusion are on by default and can be switched off with
`--keep-aliases` / `--keep-stubs` to measure what they are worth. `eval.run` takes the same
two flags, so the ablation runs over the whole query set rather than one query: there they
switch off the behaviour *in the retriever*, which is the measurement, and are distinct from
`--no-canonical`, which switches off canonicalization in *scoring* so a judgment stops
matching an alias of the page it was written against. Only `bm25` and `dense` accept them.

### Dense retrieval

`retrieval/dense.py` is the semantic counterpart: sentence-transformer embeddings with exact
cosine search. It exists because BM25 built the judging pool by itself, and a pool built by
one retriever cannot measure another — a second retriever of a *different kind* is what makes
re-pooling worth doing.

**Model: `Alibaba-NLP/gte-modernbert-base`** (149M params, 768 dim, 8192-token context,
Apache-2.0, no `trust_remote_code`). Chosen on size and benchmarks, not downloads:

| model | params | max seq | BEIR nDCG@10 | CoIR (code) nDCG@10 |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 22.7M | 128 trained / 256 cap | ~41.9 | — |
| bge-base-en-v1.5 | 109M | 512 | 53.2 | — |
| snowflake-arctic-embed-m-v2.0 | 305M+ | 512 | ~55.5 | — |
| **gte-modernbert-base** | **149M** | **8192** | **55.33** | **79.31** |

- **55.33 BEIR at 149M** beats `bge-base-en-v1.5` (53.2) and matches
  `snowflake-arctic-embed-m-v2.0` (~55.5) at roughly half that model's size and without
  `trust_remote_code`.
- **CoIR 79.31 is the number that decides it.** CoIR is a 20-dataset *code* retrieval
  benchmark, and this corpus is API reference full of signatures and code blocks. The general
  BEIR average understates how much that matters here.
- **8192-token context removes a question instead of answering it.** Chunks are capped at 512
  `cl100k_base` tokens, but the model tokenizes with its own vocabulary and code-dense text
  expands under subword tokenization, so a 512-token model could truncate the long tail.
- **The popular default is disqualified on this corpus, not on taste.** `all-MiniLM-L6-v2`
  was trained at 128 tokens and truncates at 256 word pieces, while 61.7% of these chunks are
  longer than 128 tokens (mean 191.6) — it would discard most of the corpus body text.

Encoding is symmetric: this model takes no query prefix, unlike E5 (`query:`) or BGE. Each
chunk is embedded as `title / heading path / body`, the dense analogue of BM25's column
weighting — a chunk from the middle of a page is often uninterpretable without its heading
path. Stub exclusion and alias collapsing mirror BM25's defaults exactly, so the two
retrievers are compared to each other rather than to each other's corpus handling.

Vectors are L2-normalised at build time, so cosine similarity is a plain dot product and
search is one `numpy` matvec over a memory-mapped array — a few milliseconds, exact. An ANN
index would add a dependency and an approximation to buy nothing at this scale.

**Measured build** (RTX 4060 Ti, batch size 64, `python -m retrieval.dense build`):

```
embedded 11832 chunks (2266488 tokens) in 132.8s = 89.1 chunks/s, 17,062 tok/s
index: 11832 x 768 float32 -> 34.7 MiB data/dense/Alibaba-NLP__gte-modernbert-base.npy
       metadata 231.4 KiB data/dense/Alibaba-NLP__gte-modernbert-base.json
total on disk: 34.9 MiB
```

The index carries a fingerprint of the corpus it was built from, so a stale index is reported
rather than silently scored. `sentence-transformers` lives in `requirements-embed.txt` rather
than `requirements.txt` because it pulls in torch; the dense tests inject a numpy-only fake
encoder, so CI never installs it.

Dense was re-pooled rather than scored against the BM25-built pool: `eval/pool.py delta`
wrote the 211 dense top-10 candidates that carried no judgment, those were graded, and
`eval/pool.py append` added them to `eval/queries.yaml` without touching the original 400.
Results are in the table below.

## Evaluation

`eval/queries.yaml` holds 40 queries in four categories: `api_lookup` (12), `conceptual` (10),
`multi_hop` (8), `tutorial` (10). All 40 are judged by hand — **814 judgments** over three
pooling rounds, 465 of them relevant (303 at grade 1, 162 at grade 2), a mean of 20.4 judged
and 11.6 relevant pages per query. See the header of the file for the grade
scale (0/1/2) and the judgment unit (page URL, optionally narrowed with `anchor`; no judgment
uses an anchor yet, so every unit is currently a page). The harness treats unjudged pages as
grade 0.

`eval/run.py` accepts any object with `search(query, k) -> list[chunk_id]`, maps chunks to
judgment units through the store (a chunk counts as `url#anchor` when that section was judged,
else as its page), de-duplicates in rank order and reports recall@5, recall@10, MRR and
nDCG@10 (gain `2^g - 1`), overall and per category. Alias pages are collapsed to their
canonical URL on both sides, so a judgment naming `torch.optim.Adam` still matches a retriever
that returns `torch.optim.adam.Adam_class`, and the same content cannot be counted twice.
Queries without judgments are skipped and counted; `--include-unjudged` runs them anyway.

### Judging with a pool

Grading 11,832 chunks by hand is not feasible; grading what retrievers actually return is.
`eval/pool.py build` runs each retriever over every query, unions their top-k pages, and
writes `eval/pool.yaml` with a blank `grade` per candidate plus the title, section and a text
snippet, so a page can be graded without opening a browser. Fill the grades in, then
`eval/pool.py merge` folds them into `eval/queries.yaml`, preserving its comments and refusing
to overwrite existing judgments unless given `--force`.

The pool inherits the bias of the retrievers that built it. A relevant page that no retriever
surfaced is never graded and will look like a miss forever, and a pool built only from BM25
favours lexical matching. Re-run `build` after adding a retriever of a different kind and
grade only the candidates that are new. Pages you know are relevant should be added to
`queries.yaml` by hand rather than trusting the pool to be complete.

## Results

Run 2026-09-09 against `data/corpus.db` (11,832 chunks / 3,678 pages), 40 queries, **814
judgments** — the 400 from the BM25 pool, 211 from the dense delta pool, and 203 from the
agent re-pool (`eval/pool_agent.yaml`). 465 are relevant (grade > 0), a mean of 11.6 relevant
pages per query. `k=10`.

**Every number below moved when the pool grew, so this table replaces the 611-judgment one
rather than extending it.** No retriever changed and no ranking changed; the denominators did.

| retriever | group | R@5 | of ceiling | R@10 | of ceiling | MRR | nDCG@10 (resampled 95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| random (seed 0) | all | 0.016 | 0.03 | 0.025 | 0.03 | 0.064 | 0.02 [0.00, 0.04] |
|  | api_lookup | 0.031 | 0.05 | 0.039 | 0.04 | 0.079 |  |
|  | conceptual | 0.020 | 0.05 | 0.037 | 0.04 | 0.118 |  |
|  | multi_hop | 0.000 | 0.00 | 0.010 | 0.01 | 0.018 |  |
|  | tutorial | 0.005 | 0.01 | 0.010 | 0.01 | 0.031 |  |
| **bm25** | **all** | **0.392** | **0.80** | **0.675** | **0.81** | **0.988** | **0.80 [0.51, 0.69]** |
|  | api_lookup | 0.508 | 0.74 | 0.808 | 0.83 | 1.000 |  |
|  | conceptual | 0.355 | 0.79 | 0.618 | 0.75 | 1.000 |  |
|  | multi_hop | 0.337 | 0.90 | 0.569 | 0.76 | 1.000 |  |
|  | tutorial | 0.333 | 0.89 | 0.657 | 0.90 | 0.950 |  |
| **dense** | **all** | **0.371** | **0.76** | **0.588** | **0.71** | **0.988** | **0.76 [0.49, 0.66]** |
|  | api_lookup | 0.439 | 0.64 | 0.589 | 0.60 | 0.958 |  |
|  | conceptual | 0.352 | 0.78 | 0.538 | 0.65 | 1.000 |  |
|  | multi_hop | 0.324 | 0.87 | 0.603 | 0.80 | 1.000 |  |
|  | tutorial | 0.347 | 0.93 | 0.627 | 0.86 | 1.000 |  |
| **hybrid** (RRF) | **all** | **0.396** | **0.81** | **0.664** | **0.80** | **0.971** | **0.83 [0.52, 0.71]** |
|  | api_lookup | 0.520 | 0.75 | 0.720 | 0.74 | 0.944 |  |
|  | conceptual | 0.345 | 0.76 | 0.661 | 0.80 | 1.000 |  |
|  | multi_hop | 0.337 | 0.90 | 0.605 | 0.81 | 1.000 |  |
|  | tutorial | 0.346 | 0.93 | 0.647 | 0.88 | 0.950 |  |
| rerank(bm25) | all | 0.381 | 0.78 | 0.562 | 0.68 | 0.975 | 0.76 [0.45, 0.64] |
|  | api_lookup | 0.498 | 0.72 | 0.699 | 0.72 | 1.000 |  |
|  | conceptual | 0.337 | 0.75 | 0.542 | 0.66 | 1.000 |  |
|  | multi_hop | 0.365 | 0.97 | 0.518 | 0.69 | 0.938 |  |
|  | tutorial | 0.295 | 0.79 | 0.451 | 0.62 | 0.950 |  |
| rerank(hybrid) | all | 0.382 | 0.78 | 0.558 | 0.67 | 0.975 | 0.75 [0.45, 0.64] |
|  | api_lookup | 0.508 | 0.74 | 0.663 | 0.68 | 1.000 |  |
|  | conceptual | 0.338 | 0.75 | 0.543 | 0.66 | 1.000 |  |
|  | multi_hop | 0.365 | 0.97 | 0.534 | 0.71 | 0.938 |  |
|  | tutorial | 0.287 | 0.77 | 0.467 | 0.64 | 0.950 |  |
| **agent/hybrid** | **all** | **0.380** | **0.78** | **0.634** | **0.76** | **0.954** | **0.79 [0.50, 0.69]** |
|  | api_lookup | 0.477 | 0.69 | 0.702 | 0.72 | 0.903 |  |
|  | conceptual | 0.351 | 0.78 | 0.607 | 0.74 | 1.000 |  |
|  | multi_hop | 0.312 | 0.83 | 0.549 | 0.73 | 1.000 |  |
|  | tutorial | 0.346 | 0.93 | 0.647 | 0.88 | 0.933 |  |

**Read recall against its ceiling, not against 1.0.** With 11.6 relevant pages per query on
average, five slots cannot hold them all: the achievable recall@5 is 0.488 overall (0.690 for
`api_lookup`, down to 0.372 for `tutorial` and 0.375 for `multi_hop`). The "of ceiling"
columns are the fraction of what was reachable.

**Raw recall@5 fell for every system, and that is the pool growing, not the retrievers
regressing.** BM25 went 0.448 → 0.392 and hybrid 0.449 → 0.396 without a single ranking
changing: the re-pool added 203 judgments, 73 of them relevant, so the denominator of recall
grew while five slots stayed five. The of-ceiling column is the one that is comparable across
pool sizes, and there BM25 is flat (0.81 → 0.80) and hybrid is flat (0.81 → 0.81). This is
the same effect as BM25's R@10 falling from 1.000 to 0.772 when the dense delta landed, and
it has the same reading: the measurement is working.

**nDCG is quoted as the observed value with the resampled interval beside it, to two decimals
and no more.** For every system the observed value sits *above* the top of its own resampled
interval — that is limitation 3, unchanged by this round: re-grading would not scatter nDCG
around its current value, it would move it down. Ranking conclusions rest on recall against
ceiling, not on this column.

Dense still scores below BM25 (R@5 0.371 vs 0.392, R@10 0.588 vs 0.675). It keeps its lead on
`tutorial` of-ceiling recall@5 (0.93 vs 0.89) and on `multi_hop` of-ceiling recall@10 (0.80 vs
0.76), the two categories where queries describe a task rather than name a symbol.

### Pool coverage after the re-pool

The re-pool was meant to close the unjudged hole under the agent *and* under the hybrid, whose
published numbers had been running through a 10.5% hole. It did:

| system | unjudged @ 611 | unjudged @ 814 | unjudged pages @ 611 → @ 814 | pages the re-pool absorbed |
|---|---:|---:|---:|---:|
| bm25 | 0.0% | 0.0% | 0 → 0 | 0 |
| dense | 0.0% | 0.0% | 0 → 0 | 0 |
| **hybrid** | **10.5%** | **0.0%** | 40 → 0 | **40** |
| **agent/hybrid** | **20.0%** | **0.0%** | 75 → 0 | **75** |
| rerank(bm25) | 28.2% | 23.8% | 99 → 85 | 14 |
| rerank(hybrid) | 26.2% | 22.8% | 95 → 84 | 11 |
| random (seed 0) | 98.2% | 97.2% | 300 → 299 | 1 |

Both columns come from the same tool, run against the current judgments and against the
611-judgment file recovered from history:

```
python -m agent.coverage --offline --system rerank/bm25 --system rerank/hybrid \
    --system hybrid --system agent/hybrid --system random
git show 1ff9e3d:eval/queries.yaml > /tmp/q611.yaml    # the @ 611 column
python -m agent.coverage --offline --queries /tmp/q611.yaml --system rerank/bm25 ...
```

Per category the closure is complete for both pooled systems — agent/hybrid's worst category
was `multi_hop` at 30.0% unjudged, now 0.0%. **The four systems the re-pool was built from
(agent/bm25, agent/dense, agent/hybrid, hybrid) are now at 0% by construction, exactly as
BM25 and dense have been since the first round.**

**The rerankers are now the systems with a hole.** They were never pooled: they reorder a
top-100 that no pool ever saw, so they can promote a page into the top 10 that nothing has
graded. 23.8% and 22.8% of what they return is unjudged and is scored as grade 0. The
re-pool barely touched them: **14 and 11 of their unjudged pages** happened to be pooled for
another system, out of 99 and 95. Their rate fell mostly because the denominator of judged
pages grew around them, not because their own gap was addressed. **The reranking conclusions below are therefore the least trustworthy in
this README**, and in the direction of understating the rerankers.

Random's 97.2% is the control: it says what an unpooled system looks like.

### Hybrid: reciprocal rank fusion

`retrieval/hybrid.py` fuses BM25 and dense with reciprocal rank fusion — `1/(k + rank)`
summed across systems, `k = 60` from Cormack, Clarke & Buettcher, *"Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods"*, SIGIR 2009. RRF uses only
ranks, so it needs no way to compare a BM25 score against a cosine. **The constant is the
published one and was not tuned against this eval set** — 40 queries, optimistically biased,
pooled from the two systems being fused; anything fitted on it would be fitting pooling
artifacts. Fusion depth (100) is likewise fixed a priori. Fusion happens at the canonical-page
level, so one page cannot occupy two of the k slots.

### Paired comparison

The marginal intervals above are wide because each carries the full weight of judgment error.
But that error is *common to every system*: a page re-graded 1 → 0 stops counting for all of
them at once. `eval/paired.py` exploits this — per bootstrap draw it resamples the judgments
**once** and scores every system against that same resampled set, so shared error cancels in
the difference. 4,000 draws, Dirichlet transition rows, queries resampled with replacement.

The design works. Paired intervals are 2.6–5.3× narrower than naively combining two marginals:

| comparison | marginal A | marginal B | naive sum | **paired** | shrink |
|---|---:|---:|---:|---:|---:|
| hybrid − bm25 | 0.175 | 0.167 | 0.342 | **0.102** | 3.4× |
| hybrid − dense | 0.175 | 0.161 | 0.336 | **0.101** | 3.3× |
| bm25 − dense | 0.167 | 0.161 | 0.328 | **0.127** | 2.6× |
| rerank(hybrid) − hybrid | 0.168 | 0.175 | 0.343 | **0.112** | 3.1× |
| rerank(bm25) − bm25 | 0.168 | 0.167 | 0.335 | **0.114** | 2.9× |
| **agent/hybrid − hybrid** | 0.172 | 0.175 | 0.346 | **0.065** | **5.3×** |
| agent/hybrid − bm25 | 0.172 | 0.167 | 0.339 | **0.113** | 3.0× |

`agent/hybrid − hybrid` shrinks most because the two systems share a base retriever and agree
on most queries, so almost everything cancels — which is exactly the case a paired design is
built for, and it makes that comparison the sharpest test in this README.

**And the answer is still no. Hybrid does not beat either system.** Every paired interval
crosses zero, on both raw and of-ceiling recall@5, overall and in all four categories:

| comparison | recall@5 raw | recall@5 of ceiling | verdict |
|---|---|---|---|
| hybrid − bm25 | +0.013 [−0.037, +0.065] | +0.020 [−0.055, +0.090] | no difference |
| hybrid − dense | +0.013 [−0.036, +0.066] | +0.020 [−0.052, +0.093] | no difference |
| bm25 − dense | −0.000 [−0.064, +0.062] | +0.001 [−0.092, +0.092] | no difference |

Per-query wins on recall@5, with ties counted rather than split:

| comparison | wins | losses | ties |
|---|---:|---:|---:|
| hybrid > bm25 | 8 | 6 | 26 |
| hybrid > dense | 13 | 7 | 20 |
| bm25 > dense | 12 | 10 | 18 |

Hybrid's central estimate is positive against both, and it ties BM25 on 26 of 40 queries —
consistent with a small real gain, and equally consistent with none. A paired design that
resolves 2.6–3.4× finer than the marginals still cannot separate these systems, so the honest
reading is that **fusion buys nothing measurable here.** Nothing was adjusted in response:
no constant search, no depth search, no retriever changes.

MRR is reported as a secondary line only — hybrid − bm25 is +0.025 [−0.060, +0.118]. At 57.1%
of judged candidates relevant, MRR is saturated (every system finds *something* relevant at
rank 1 or 2 on almost every query) and is not expected to discriminate. It doesn't.

### Reranking

`retrieval/rerank.py` puts a cross-encoder over the top N of any base retriever. A bi-encoder
scores query and page independently; a cross-encoder reads the pair together and can attend
from one onto the other, which ranks better and costs one forward pass per candidate — so it
reorders, it cannot search.

**Model: `Alibaba-NLP/gte-reranker-modernbert-base`** (149M, 8192 ctx, Apache-2.0), chosen on
size and published reranking benchmarks weighted toward code:

| model | params | ctx | licence | BEIR | code retrieval |
|---|---:|---:|---|---:|---|
| ms-marco-MiniLM-L6-v2 | 22.7M | 512 | Apache-2.0 | — | none published |
| jina-reranker-v2-base-multilingual | 278M | 1024 | **CC-BY-NC-4.0** | 53.17 | CSN MRR@10 71.36 (3 tasks) |
| **gte-reranker-modernbert-base** | **149M** | **8192** | **Apache-2.0** | **56.73** | **CoIR 79.99 (20 tasks)** |

Best code retrieval of the three (CoIR 79.99 over 20 tasks vs jina's CodeSearchNet 71.36 over
3), best BEIR, at roughly half jina's size — and jina's CC-BY-NC licence rules it out of
anything but a prototype. `ms-marco-MiniLM-L6-v2` is the popular default and the wrong tool:
MS MARCO web passages, no published code numbers, 512-token limit. Same ModernBERT backbone as
the embedder, so both stages tokenize code-dense text alike.

**N = 100, fixed a priori and not searched** — the standard BEIR reranking depth, and already
the fusion depth of the hybrid, so both stages read the same candidate horizon.

**Cost** (RTX 4060 Ti, fp32, batch 32, N=100): 149.6M parameters, 571 MiB fp32. Per query
**1818 ms mean / 1852 ms median end-to-end**, of which 1760 ms is the cross-encoder — about
57 pairs/s. That is roughly 100× the latency of the fusion it sits on top of.

#### Reranking vs fusion, with respect to the pooling limitation

These two are not equally measurable here, and the difference matters for reading the results.

RRF fuses the two systems that *built* the pool, so it can only reorder pages that are already
judged and gets no credit for anything it surfaces outside the pool — its ceiling is capped by
construction. **A reranker is different in the part that counts.** Its job is precision within
the retrieved set: pushing the right pages from rank 7 to rank 2. Every page it moves is a
page some pooled retriever already returned and a human already graded, so that reordering is
*fully visible* to recall@5. Unlike fusion, a reranker's main claimed benefit is exactly the
thing this eval can see.

The caveat it still shares: a reranker reads only what the base hands it, so anything outside
the pool remains invisible, and it cannot rescue a page no first stage retrieved.

That makes the result below more informative than the fusion null — this eval *should* have
been able to detect the improvement, and it did not.

#### Result: reranking does not help, and probably hurts

Both point estimates are negative on both forms of recall@5:

| comparison | recall@5 raw | recall@5 of ceiling | verdict |
|---|---|---|---|
| rerank(hybrid) − hybrid | −0.020 [−0.076, +0.036] | −0.030 [−0.110, +0.052] | no difference |
| rerank(bm25) − bm25 | −0.009 [−0.064, +0.050] | −0.015 [−0.100, +0.068] | no difference |

Paired intervals still cross zero, so this is not a statistically clean loss. But the
per-query counts point the same way, and more sharply than the intervals do:

| comparison | wins | losses | ties |
|---|---:|---:|---:|
| rerank(hybrid) > hybrid | 7 | **11** | 22 |
| rerank(bm25) > bm25 | 7 | **13** | 20 |

The reranker loses on more queries than it wins. It is worst on `tutorial`
(rerank(hybrid) − hybrid = −0.071 raw, −0.120 of-ceiling) and roughly neutral on `api_lookup`
— it damages exactly the task-shaped queries where dense retrieval was contributing most. It
also drops recall@10 substantially (hybrid 0.664 → 0.558), which is expected: with N=100 the
cross-encoder is free to demote a correct page out of the top 10 entirely.

**One caveat is now specific to this comparison.** The paragraph above argues a reranker's
reordering is fully visible to the pool. That was true when the reranker was reading a pool
built from its own base; it is only partly true now. The rerankers are the only systems still
carrying an unjudged hole (23.8% and 22.8%), because they promote from a top-100 no pool has
ever graded, and every such page scores 0. The re-pool closed hybrid's and the agent's holes
and left theirs mostly open, so **this section understates the rerankers by an unknown amount
and is the least trustworthy result in this README.** Closing it needs a pool built from
rerank output, which has not been done.

Nothing was adjusted in response — no N search, no model search, no base changes. On this
corpus, at this judgment quality, a cross-encoder costing ~1.8 s per query buys nothing and
plausibly costs accuracy. The paired design is 2.9–3.1× finer than the marginals here, so
this is not merely an underpowered test.

### Limitations

These are the reasons not to read the table as a clean measurement of retrieval quality.

1. **The pool favours the four systems that built it.** The 814 judgments came from three
   rounds: BM25's top 10 (400), dense's top 10 (211), then a joint re-pool of agent/bm25,
   agent/dense, agent/hybrid and hybrid (203). BM25, dense, hybrid and agent/hybrid now all
   sit at 0% unjudged, so the head-to-heads between them are no longer confounded by coverage
   — that was the point of the third round. What remains: BM25 contributed every judgment in
   the first round and half the second, so it retains a residual advantage in *which* pages
   exist to be found at all; and a page that none of these six systems ever surfaced is still
   invisible and will look like a miss forever. **The rerankers were never pooled** and still
   run through a ~23% unjudged hole, which makes their numbers the least comparable in the
   table and biases them downward.

2. **The judgments are not reliable at the level the numbers imply.** Two blind re-grade
   probes measured how repeatable the grading is: grade 0 reproduced at 90%, grade 2 at 50%,
   grade 1 at 45% in the first probe and 30% in the second. Three-way Cohen's κ is **0.0943
   for the second probe** — barely above chance — against 0.3953 for the first and 0.3116
   pooled. The κ worth quoting is the per-probe pair, not the pooled figure: the probes
   differ in *both* the contrast used (1s against 0s versus 1s against 2s) and whether the
   written rubric in `eval/GRADING.md` existed, so pooling averages over the very difference
   in question. That same confound is why the drop in grade-1 stability cannot be attributed
   to the rubric or the contrast alone (two-sided Fisher exact p = 0.5145 on 9/20 against
   6/20 — there is not even a difference established to attribute).

   ```
   python -m eval.reliability     # matrices, reproduction rates, κ per probe, Fisher
   ```

3. **The judgment set is optimistically biased, not merely noisy.** Resampling all judgments
   under the measured re-grade transition matrices (4,000 draws, Dirichlet rows so the n=20
   behind each row is priced in) puts the observed MRR (0.988) and nDCG@10 (0.898 on the
   400-judgment set) *above the 97.5th percentile* of the resampled distribution. A re-grade
   would not scatter these numbers around their current values; it would move them down.
   recall@5 (0.578) sat inside its interval, because its denominator shrinks along with the
   relevant set — which is why recall@5 is the primary metric here.

4. **Every interval above is a floor.** The bootstrap treats grading error as independent
   across judgments. Real error is correlated within a query: a shift in how a query is read
   moves all of its candidates together. Correlated error produces wider swings than
   independent error, so the true uncertainty is larger than these intervals, probably by a
   lot.

5. **The 203 agent-pool judgments were graded over several review passes, not one sitting.**
   The earlier 611 were each graded in a single continuous session; these 203 were worked
   through across multiple passes separated by breaks. A grader's threshold drifts between
   sittings, and nothing here measures that drift — the re-grade probes were run against the
   611 and say nothing about within-round consistency for this batch. So the newest 203
   judgments are plausibly *less* internally consistent than the reliability figures in
   limitation 2 suggest, and they are exactly the judgments that carry the agent-vs-hybrid
   comparison.

6. **237 of the 303 grade-1 judgments predate the rubric, and the re-grade file covers
   only 159 of them.** Grade 1 arrived in three rounds — 159 from `eval/pool.yaml`, 78 from
   `eval/pool_delta.yaml`, 66 from `eval/pool_agent.yaml` — and `eval/GRADING.md` was written
   between the second and third, so the first 237 were graded without it.
   `eval/regrade_ones.yaml` was built from `pool.yaml` alone (`n: 159`, `source: pool.yaml`)
   and is still unfilled, which leaves **78 pre-rubric grade-1 judgments outside any re-grade
   plan**. Grade 1 is the least reproducible category and the largest.

7. **nDCG is reported but is not used to rank retrievers.** Its `2^g − 1` gain weights a
   grade 2 at 3× a grade 1, which amplifies exactly the 1↔2 boundary that reproduces worst
   (50% and 30–45%). It is quoted to two decimals with its interval attached and should not
   be read more precisely than that. Ranking conclusions rest on recall against ceiling.

To plug in a retriever, implement `search(query: str, k: int) -> list[str]` returning chunk ids
from `data/corpus.db` and pass it to `eval.run.evaluate(retriever, load_queries(), store.locate)`.

## Agent

`agent/` adds a layer above the retrievers: an LLM decides whether a query needs splitting,
splits it if so, retrieves for each sub-query and fuses the result lists. It is a retriever
like any other — `search(query, k) -> list[chunk_id]` — so it drops into `eval.run` and
`eval.pool` unchanged, and it is pluggable over BM25, dense or the hybrid (default hybrid).

The control flow is an explicit state graph, written out in `agent/graph.py` rather than
imported from a framework:

```
route ──(decompose)──> decompose ──┐
  │                                ├──> retrieve ──> fuse ──> END
  └──(direct)────────> direct ─────┘
```

Nodes take the state and return it; edges are either a node name or a function of the state,
so the whole graph is inspectable as data (`agent.graph.GRAPH`) and each node is testable on
its own. Routing is one model call returning a structured verdict — `{"route": ...,
"reason": ...}` — and every query's path, verdict and timings land in
`AgentRetriever.traces`.

**Constants are fixed a priori and asserted in `tests/test_agent.py`**, so tuning one against
eval scores fails CI rather than passing quietly:

| constant | value | |
|---|---|---|
| `MAX_SUB_QUERIES` | 3 | at most three sub-queries per query |
| `DECOMPOSITION_DEPTH` | 1 | sub-queries are never themselves decomposed |
| `RESULTS_PER_SUB_QUERY` | 10 | retrieved per sub-query, before fusion |
| `RRF_K` | 60 | imported from `retrieval.hybrid`, not redefined |
| `TEMPERATURE` | 0.0 | greedy decoding |

Fusion reuses `retrieval.hybrid.rrf_scores` — the same published constant and the same
formula that fuses BM25 with dense — and fuses at the level of the canonical page, so two
sub-queries that surface the same page from different chunks add their evidence instead of
splitting it across two slots. The fused list is truncated to the requested `k`.

The prompts in `agent/prompts.py` are a first draft and are frozen. They are not to be
iterated against eval scores: 40 queries whose judgments are pooled from the very retrievers
being compared would fit the noise, which is the same argument that keeps `RRF_K` at its
published value.

### Model and cache

`qwen2.5:7b` served locally by Ollama at `http://localhost:11434`, called over plain HTTP with
`requests` — no LangChain, no LangGraph, no client library. `requirements-agent.txt` keeps
this out of the core suite, as `requirements-embed.txt` does for the embedder. A missing
server or an unpulled model fails up front with the command that fixes it, before 40 queries'
worth of work rather than midway through.

Every response is cached to `agent/cache/ollama.json`, keyed by `sha256(model, prompt,
query)`, and **the cache is committed** (63 entries, 24K). Because the key covers the full
prompt text, editing a frozen prompt invalidates its entries rather than silently reusing
stale generations. `--offline` refuses to call Ollama at all, so a cache miss is an error
instead of a quiet live call — which is what makes the replay verifiable:

```
python -m agent.run --offline           # reproduces with no Ollama and no GPU
python -m agent.coverage --offline --baseline
```

Both commands above reproduce in a network-isolated namespace. Re-running live at temperature
0 reproduced the cached routing exactly on the categories tested.

### What it does, measured

40 queries, agent over hybrid, `k=10`:

| | |
|---|---|
| route | 23 decompose, 17 direct |
| sub-queries | 65 total, 1.62 per query |
| decomposition fallbacks | **0 / 23** |
| model calls | 63 (40 route + 23 decompose) |

**The decomposition fallback rate is 0/23 on these 40 queries.** That is the honest number and
it is lower than expected for a 7B local model; the credit belongs mostly to Ollama's
`format="json"` constrained decoding, not to the prompt. Constrained decoding guarantees
*valid JSON*, not *the requested schema*, so the parsers validate the schema themselves —
element by element for the sub-query list — and the fallback path is exercised by tests
rather than by production. It is a real path, not dead code, and the rate is a property worth
re-checking on any prompt or model change rather than a box that has been ticked.

Latency, split by where the time went:

| latency (s) | mean | median | max | total |
|---|---|---|---|---|
| model | 0.765 | 0.783 | 4.635 | 30.62 |
| retrieval | 0.058 | 0.053 | 0.296 | 2.33 |
| end to end | 0.824 | 0.833 | 4.931 | 32.95 |

**Model time is 92.9% of the work.** The 4.6 s maximum is the first call, paying the model
load; the median is the number to reason about. `total` is model + retrieval, deliberately
*not* the measured wall clock: on a cache hit the model half is the latency recorded when that
call originally ran live, so a replay's wall clock omits it entirely and would report a system
three orders of magnitude faster than the one that exists. `wall_seconds` keeps the measured
replay cost for anyone who wants it, and `--fresh-timings` takes both halves from one live
run.

### Judgment coverage: closed

The ablation could not be read before, because scoring the agent against a pool built from
BM25 and dense would have measured the pool. `agent/coverage.py` sized that hole: **20.0% of
the (query, page) pairs the agent returned had never been graded**, 30.0% on `multi_hop`, and
`eval.run` scores anything unjudged as grade 0. The hybrid was carrying a 10.5% hole of its
own and had been scored anyway.

`eval/pool_agent.yaml` pooled the four affected systems in one round — agent/bm25, agent/dense,
agent/hybrid and plain hybrid — into 203 candidates over 32 queries and 172 distinct pages,
each with title, section and snippet. It was graded (7 × grade 2, 66 × grade 1, 130 × grade 0)
and **appended**, not merged:

```bash
python -m eval.pool append --pool eval/pool_agent.yaml   # 611 -> 814, backup written
```

`append` only ever inserts lines, so all 611 earlier judgments survive byte for byte;
`merge` would have replaced each query's whole `judgments` block and destroyed them. It also
refuses to write a url a query already judges, since one page cannot hold two grades.

The hole is closed for both target systems, and per category as well as overall:

| system | @ 611 | @ 814 | worst category @ 611 | worst @ 814 |
|---|---:|---:|---|---|
| **hybrid** | 10.5% | **0.0%** | `api_lookup` 12.5% | 0.0% |
| **agent/hybrid** | 20.0% | **0.0%** | `multi_hop` 30.0% | 0.0% |
| rerank(bm25) | 28.2% | 23.8% | `tutorial` 37.0% | `tutorial` 31.0% |
| rerank(hybrid) | 26.3% | 22.8% | `tutorial` 33.0% | `tutorial` 29.0% |

**The agent and the hybrid now stand where BM25 and dense have stood since round one: 0%
unjudged, by construction.** The comparison below is no longer measuring pool coverage.

### The ablation: the agent does not beat plain retrieval

Agent over hybrid, `k=10`, 814 judgments, paired against its own base and against BM25:

| comparison | recall@5 raw | recall@5 of ceiling | verdict |
|---|---|---|---|
| **agent/hybrid − hybrid** | **−0.009 [−0.042, +0.023]** | **−0.020 [−0.076, +0.038]** | no difference |
| agent/hybrid − bm25 | +0.004 [−0.053, +0.061] | −0.000 [−0.085, +0.080] | no difference |

Both intervals cross zero, so neither is a statistically clean loss. But the point estimate
against the base it is built on is **negative**, and the per-query counts say the same thing:

| comparison | wins | losses | ties |
|---|---:|---:|---:|
| agent/hybrid > hybrid | 3 | **6** | **31** |
| agent/hybrid > bm25 | 8 | **11** | 21 |

**On 31 of 40 queries the agent returns a top-5 that scores identically to plain hybrid.** It
routes 17 of 40 to the direct path, where it *is* plain hybrid by construction; all 9 queries
whose score moves are on the decompose path, so of the 23 it actually decomposes it wins 3,
loses 6 and ties 14. The decomposition changes the answer on 9 queries and improves it on 3.

**`multi_hop` is where decomposition has a mechanism, and it is where the agent does worst:**

| category | agent/hybrid − hybrid (raw) | − hybrid (of ceiling) | − bm25 (raw) | − bm25 (of ceiling) |
|---|---|---|---|---|
| **multi_hop** | **−0.014 [−0.104, +0.072]** | **−0.025 [−0.176, +0.125]** | **−0.018 [−0.125, +0.081]** | **−0.050 [−0.225, +0.125]** |
| api_lookup | −0.010 [−0.077, +0.054] | −0.017 [−0.133, +0.067] | +0.001 [−0.129, +0.126] | −0.008 [−0.171, +0.150] |
| conceptual | +0.000 [−0.063, +0.056] | +0.000 [−0.120, +0.120] | −0.001 [−0.097, +0.095] | +0.000 [−0.140, +0.140] |
| tutorial | −0.009 [−0.063, +0.031] | −0.020 [−0.120, +0.060] | +0.030 [−0.061, +0.133] | +0.040 [−0.140, +0.200] |

Every category is negative or flat against hybrid, and `multi_hop` is the most negative of the
four on both forms of recall@5 — the one category where splitting a query into sub-queries was
supposed to pay. In the raw table it is the only category where the agent falls clearly below
its base (R@5 0.312 vs 0.337, of-ceiling 0.83 vs 0.90). The predicted mechanism does not
appear in the place it was predicted.

This is the sharpest comparison in this README: `agent/hybrid − hybrid` is the narrowest
paired interval of any pair (0.065 wide, **5.3×** tighter than naively summing the marginals),
because the two systems share a base and cancel almost everything. It is not underpowered.

**So: the agent layer costs 0.77 s of model time per query — 93% of its end-to-end latency,
against 0.06 s of retrieval — and returns no measurable retrieval gain over the hybrid it
wraps, and none over BM25.** Nothing was tuned in response — the
prompts are the frozen first draft, the routing threshold was never searched, and the fusion
is the same untuned RRF. What this measures is the agent as built, not the best agent
obtainable; a negative result on one 40-query set with κ = 0.09 judgments does not establish
that decomposition cannot help. It does establish that this one does not, here.

The one thing the agent does not lose on is recall@10 relative to the rerankers (0.634 vs
0.562/0.558) — it is the only system besides BM25 and hybrid that keeps most of its top-10
recall, because fusing sub-query lists adds pages rather than reordering a fixed set.

### The re-pool file

`eval/pool_agent.yaml` is that pool, in the same format `eval/pool.py` writes and reads —
**203 candidates over 40 queries, 172 distinct pages**, now fully graded:

```bash
python -m agent.coverage --offline \
  --system agent/bm25 --system agent/dense --system agent/hybrid --system hybrid \
  --out eval/pool_agent.yaml
```

`make_grading_page.py` turned that pool into the interface the 203 candidates were actually
graded in — one self-contained HTML page carrying the rubric, each candidate's title, section
path and snippet, and a link to the live doc, so nothing else had to be open while grading.
What it emits is `agent_pool_grades.txt`, one `candidate-id: grade` line per row; those
grades go into the `grade:` fields of `eval/pool_agent.yaml`, which
`python -m eval.pool append` then folds into `eval/queries.yaml` (append, never merge — merge
would replace a query's whole judgment list):

```bash
python3 make_grading_page.py eval/pool_agent.yaml grading.html   # then grade in a browser
```

`--system` is repeatable, and the systems are pooled in **one** `build_pool` call rather than
merged afterwards, so a page four systems returned is one candidate carrying four `found_by`
entries, not four rows. The overlap is large: 270 unjudged pairs summed over the four systems
collapsed to 203 once pooled.

| system | retrieved | unjudged | rate | distinct pages |
|---|---|---|---|---|
| agent/bm25 | 400 | 72 | 18.0% | 65 |
| agent/dense | 400 | 76 | 19.0% | 72 |
| agent/hybrid | 400 | 80 | 20.0% | 75 |
| hybrid | 400 | 42 | 10.5% | 40 |
| **union** | 743 | **203** | 27.3% | **172** |

8 of 40 queries were already fully covered, so the 203 candidates fall across the other 32;
the worst were q14 (60.0%), q24 (58.3%) and q10 (55.6%). Grading came back 130 × 0, 66 × 1
and 7 × 2 — **64% of the pages the pool was missing turned out to be irrelevant anyway**,
which is why closing a 20% hole moved the agent's recall so little. The 73 relevant additions
raised the mean relevant pages per query from 9.8 to 11.6 and pushed every system's raw
recall@5 down.

`--out` refuses to overwrite an existing file (`--force` overrides). A graded pool is not
recoverable, and every other `eval/*.yaml` is one. Nothing in `agent/` modifies a judgment
file.

## Generation

`generate/` answers a query from a retriever's top-k chunks with `qwen2.5:7b` over Ollama —
the same local setup, client and cache machinery as the agent (`agent/llm.py`). The stage is
deliberately thin: it does not re-rank, re-retrieve or reformulate, so a difference in what
comes out is attributable to the retriever that fed it rather than to the generator.

Every sentence must cite the passages it came from as `[C1]`, and when the passages do not
contain the answer the model must emit `INSUFFICIENT_CONTEXT` and refuse.

```
python -m generate.run --retriever hybrid -v            # answers, live
python -m generate.faithfulness --all --offline         # the table below, from the cache
```

**Fixed a priori, asserted in `tests/test_generate.py`.** k = 10 (the harness's k, so
retrieval and generation see the same depth), temperature 0, `num_ctx` 8192, `num_predict`
400, at most 8 claims per answer, the refusal marker's exact spelling, and the mismatch
rotation of +20. The three prompts are pinned **by SHA-256**: editing one fails a test with a
message saying to re-measure rather than to re-word. None of these was chosen by looking at a
result, and none was moved after seeing one — including the two that, as it turns out, would
have improved the headline number.

The cache (`generate/cache/generation.json`, 807 entries, 440 KB) is keyed by
`(model, prompt, query, retriever)` and committed, so `--offline` replays all of this with no
Ollama and no GPU. Building it cost 29.2 minutes of model time and 1.53 M tokens.

### Faithfulness: is each claim in the answer stated by the retrieved passages?

No new human judgments. Each answer is split into atomic claims by one model call that never
sees the passages, and each claim is then checked against the passages by a second call that
never sees the question. Keeping them apart is the point: a decomposer that could see the
passages would shape claims to check out, and a checker that could see the question would
drift from "is this stated" to "does this answer the question".

200 answers over 40 queries × 5 retrievers produced **721 claims**.

| retriever | claims | judged | support, unjudged = unsupported | support, judged only |
|---|---:|---:|---:|---:|
| bm25 | 131 | 83 | **0.542** [0.46, 0.63] | **0.855** [0.79, 0.92] |
| dense | 166 | 134 | **0.639** [0.55, 0.72] | **0.791** [0.73, 0.85] |
| hybrid | 131 | 104 | **0.656** [0.56, 0.76] | **0.827** [0.75, 0.89] |
| rerank | 155 | 111 | **0.600** [0.50, 0.70] | **0.838** [0.78, 0.89] |
| agent | 138 | 100 | **0.616** [0.51, 0.73] | **0.850** [0.78, 0.91] |

Intervals are a 4,000-draw bootstrap resampling **queries**, not claims, because claims
inside one answer stand or fall together. Every table in this section, including the
by-category and citation tables below, is printed by:

```
python -m generate.faithfulness --all --offline --json record.json
python -m generate.report record.json
```

**Two columns, because the checker did not judge 26% of the claims.** On answers with three
or more claims it routinely returns a `verdicts` list shorter than the claim list — 189 of
721 claims came back unjudged. This is not output truncation: the longest support response
was 188 tokens against a 400-token cap. The left column counts an unjudged claim as
unsupported, the right one drops it. The true rate is bracketed by the pair; neither column
is "the" faithfulness rate.

**The two readings reverse the ranking, so do not read one.** BM25 is last on the left
(0.542) and first on the right (0.855), because its answers had the largest share of claims
left unjudged (83 of 131, against dense's 134 of 166). The apparent spread between retrievers
is mostly a checker artifact tracking claims-per-answer, not a difference in how faithfully
each retriever's context gets used. Every interval on the left overlaps every other.

#### By query category

Same two readings, `unjudged = unsupported` / `judged only`, with the claim count behind each
cell. Ten queries or fewer sit behind every cell, so these are indicative, not separating.

| retriever | api_lookup | conceptual | multi_hop | tutorial |
|---|---|---|---|---|
| bm25 | 0.64 / 0.84 (n=33) | 0.57 / 0.81 (n=30) | 0.46 / 0.79 (n=24) | 0.50 / 0.96 (n=44) |
| dense | 0.58 / 0.71 (n=38) | 0.59 / 0.79 (n=37) | 0.59 / 0.76 (n=27) | 0.72 / 0.85 (n=64) |
| hybrid | 0.62 / 0.83 (n=32) | 0.56 / 0.68 (n=27) | 0.59 / 0.76 (n=27) | 0.78 / 0.95 (n=45) |
| rerank | 0.65 / 0.87 (n=40) | 0.67 / 0.86 (n=45) | 0.42 / 0.64 (n=33) | 0.62 / 0.96 (n=37) |
| agent | 0.76 / 0.91 (n=38) | 0.60 / 0.78 (n=30) | 0.42 / 0.73 (n=26) | 0.61 / 0.90 (n=44) |

Pooled over all five, the category ordering is the one thing here that survives both
readings:

| category | claims | judged | unjudged = unsupported | judged only | refusals |
|---|---:|---:|---:|---:|---:|
| api_lookup | 181 | 142 | 0.652 | 0.831 | 9/60 |
| conceptual | 169 | 129 | 0.604 | 0.791 | 0/50 |
| multi_hop | 137 | 93 | **0.496** | **0.731** | 5/40 |
| tutorial | 234 | 168 | **0.654** | **0.911** | 1/50 |

**`multi_hop` is the worst category under both readings and `tutorial` the best.** That is
the result you would predict from the task: a multi-hop answer has to join facts that live on
different pages, and the join itself — the sentence that connects them — is exactly the kind
of claim no single passage states. Tutorial answers mostly restate one prose passage and
stay inside it. `multi_hop` also has the lowest judged share (93 of 137), so its two readings
are the furthest apart of any category, and it is where the checker defect bites hardest.

`api_lookup` produced 9 of the 15 refusals on real context and `conceptual` none, which cuts
against what recall@k says about those categories: `api_lookup` is the top category by R@10
for both BM25 (0.808) and the hybrid (0.720), and `conceptual` is below it for both. A
retriever finding the right page and the generator then declining to answer from it is a
distinct failure from retrieval missing it, and only this stage can see the difference.

#### Citations

The model never cited a passage outside the ten it was given: **0 dangling citations out of
726**, in every retriever. Two answers of the 183 scored carried no citation at all, which
the prompt forbids; both are counted in the support rate like any other answer, since an
uncited sentence is still a claim that either is or is not in the passages.

| retriever | scored answers | citations | dangling | rate |
|---|---:|---:|---:|---:|
| bm25 | 34 | 145 | 0 | 0.000 |
| dense | 38 | 168 | 0 | 0.000 |
| hybrid | 38 | 123 | 0 | 0.000 |
| rerank | 35 | 148 | 0 | 0.000 |
| agent | 38 | 142 | 0 | 0.000 |

What the failures look like, from the lowest-scoring hybrid answer (q20, `.contiguous()`):
the first claim quotes the docstring and is supported; the three that follow — when you
*need* to call it — are the model completing the topic from prior knowledge. The passages
never say them. That is the failure mode the metric exists to catch, and it is invisible to
recall@k, which scores that same retrieval as a hit.

### Refusal: does it decline when the passages cannot answer?

Measuring this needs contexts that provably lack the answer, without labelling any. Each
query is re-asked over the passages retrieved for a **different** query, under a fixed
rotation of +20 over the 40. A pair is dropped when the donor passages contain a page the
*existing* judgments already mark relevant (grade ≥ 1) for the recipient query — that check
reads `eval/queries.yaml` and writes nothing.

| retriever | refusals on real context | probe pairs | dropped | correct refusal |
|---|---:|---:|---:|---:|
| bm25 | 5/40 | 36 | 4 | **36/36 = 1.000** |
| dense | 2/40 | 35 | 5 | **35/35 = 1.000** |
| hybrid | 2/40 | 34 | 6 | **34/34 = 1.000** |
| rerank | 4/40 | 35 | 5 | **34/35 = 0.971** |
| agent | 2/40 | 35 | 5 | **35/35 = 1.000** |

174 of 175 probe pairs refused. **Read this as a floor, not a score**: mismatched context is
the easy case. A page about `torch.topk` offered against a question on random seeding is
obviously off-topic, and a model that refuses there may still answer confidently from a
passage that is plausibly on-topic but silent on the specific question. That harder case —
near-miss retrieval — is the one that matters in production and is not measured here.

It also refuses on genuinely retrieved context 15 times out of 200 (q06, "set the random seed
for CPU and every GPU at once", among them). Whether those are correct abstentions or lost
answers is not established: it would need the graded judgments for those queries read against
the answer, which is a different measurement.

### Latency and tokens

Per query, **measured on the live run**, on an RTX 4060 Ti.

The two halves of this table reproduce differently, which matters for anyone re-running it.
Model timings are stored with the response, so `generate.report` prints the same figures from
the cache. **Retrieval timings are measured fresh on every run and are not reproducible from
the cache at all** — most sharply for the agent, which retrieves *by calling the model*, so an
`--offline` replay serves those calls from disk and reports 59 ms where the live run took
4,043 ms. The replay's other retrieval figures drift for the ordinary reason that they are
re-measured: bm25 5 ms against 9 ms live, rerank 1,945 ms against 2,356 ms. `generate.report`
prints a warning when it detects an all-cached record, so the distinction is hard to miss.

| retriever | answer min / p50 / p90 / max | retrieval p50 | end-to-end p50 | prompt tok | completion tok | tok/query |
|---|---|---:|---:|---:|---:|---:|
| bm25 | 1.1 / **3.2** / 4.4 / 13.2 s | 9 ms | 3.2 s | 2,759 | 93 | 2,853 |
| dense | 1.2 / **2.7** / 5.6 / 6.8 s | 35 ms | 2.8 s | 2,204 | 107 | 2,312 |
| hybrid | 0.8 / **2.8** / 4.5 / 7.4 s | 39 ms | 2.8 s | 2,574 | 101 | 2,675 |
| rerank | 1.6 / **3.2** / 5.2 / 10.2 s | 2,356 ms | 5.6 s | 2,654 | 96 | 2,749 |
| agent | 3.8 / **5.4** / 7.7 / 11.3 s | 4,043 ms | 9.5 s | 2,442 | 99 | 2,541 |

Retrieval latency spans three orders of magnitude — 9 ms for BM25 against 2.4 s for the
reranker and 4.0 s for the agent — while the generation cost barely moves, because all five
send the model the same ten passages. End-to-end that makes the agent about 3× the hybrid
per query (9.5 s against 2.8 s) for a faithfulness rate inside the hybrid's interval, on top
of the retrieval result that it does not beat plain retrieval either. Faithfulness scoring costs a further two calls per
answer, which is why the full sweep is 807 cached calls rather than 200.

### What this measures, and what it does not

1. **Faithfulness is groundedness in retrieved text, not correctness.** A claim counts as
   supported when a retrieved passage states it. An answer that faithfully reproduces a
   wrong, outdated or irrelevant chunk scores 1.0, and an answer that is entirely correct
   about PyTorch but says something the passages omit scores 0. Nothing here checks whether
   the retrieved page was the right page — that is what the 814 hand judgments and
   `eval/run.py` measure, on a different axis. **Neither number substitutes for the other,
   and a system can be excellent on one and useless on the other.**

2. **The checker is the same model family as the generator, so it shares its blind spots.**
   `qwen2.5:7b` judges `qwen2.5:7b`. Where the generator misreads a passage, the checker is
   disposed to misread it the same way and call the claim supported. This biases the support
   rate **up** by an amount nothing here bounds. The rates are comparable *between*
   retrievers, which share the checker, and are not absolute levels. An independent checker —
   a different family, or a human pass over a sample — is the only thing that would fix this,
   and neither has been run.

3. **The checker left 26% of claims unjudged**, and which claims go unjudged tracks how many
   claims an answer has, which differs by retriever. That is enough to reverse the ranking
   between the two readings above. Until it is fixed, this measurement cannot rank retrievers
   by faithfulness at all — it can only say that all five sit somewhere in a wide band.

4. **The refusal probe measures the easy case.** Mismatched contexts are off-topic in an
   obvious way. The near-miss — a plausibly related passage that does not contain the answer —
   is untested, and is where a refusal mechanism actually earns its keep.

5. **One model, one run, 40 queries.** Temperature is 0, so a cached replay is exact, but a
   fresh run is reproducible only up to Ollama's numerics. Nothing here is a claim about
   generation in general; it is a claim about this model on this corpus at this k.

6. **Category cells are small.** The by-category tables above rest on 8-12 queries and
   24-64 claims per cell. The pooled category ordering (`multi_hop` worst, `tutorial` best)
   holds under both readings and is worth something; the per-retriever cells are not
   separating and should not be read as one retriever beating another on a category.

7. **The claims are the model's own decomposition.** An answer split into 3 claims and the
   same answer split into 6 are not scored on the same denominator, and the decomposer is
   never checked against a human split. The claim counts in the table above are inputs to the
   metric, not properties of the answers.

## Next

1. ~~Grade `eval/pool.yaml` and merge it.~~ Done 2026-09-08.
2. ~~Add a semantic retriever, re-pool, grade the new candidates.~~ Done: 611 judgments, and
   BM25's tautological R@10 of 1.000 is now 0.675.
3. ~~Grade `eval/pool_agent.yaml`, then run the ablation.~~ Done 2026-09-09: 814 judgments,
   hybrid's and agent/hybrid's unjudged rates both closed to 0.0%, and the ablation says the
   agent does not beat plain retrieval.
4. **Re-grade the pre-rubric grade-1 judgments.** There are 303 grade-1 judgments in all;
   **237 predate `eval/GRADING.md`** (159 from `pool.yaml`, 78 from `pool_delta.yaml`) and 66
   came after it with the agent round. `eval/regrade_ones.yaml` is built and unfilled but
   covers only the 159 from `pool.yaml`, so finishing it still leaves 78 pre-rubric grade-1
   judgments un-re-graded; a second file would be needed for those. Grade 1 is the largest
   category and the least reproducible, so until this is done limitation 3 stands and the
   metrics stay optimistically biased.
5. **Pool the rerankers.** They are now the only systems with an open unjudged hole (23.8%
   and 22.8%), because they promote out of a top-100 nothing has graded. Until that round is
   run, the reranking result is biased against them by an unknown amount and is the weakest
   claim in this README.
6. Re-run the blind probe after the grade-1 re-grade, holding the contrast fixed this time, so
   the rubric's effect can be separated from the contrast effect. A probe that also samples
   the agent-round judgments would measure the cross-sitting drift in limitation 5, which
   nothing currently does.
7. Decide what alias pages should ultimately be: collapsed as they are now, merged into one
   page at chunk time, or dropped. The data to decide is in `pages.canonical_url`.
   Re-measured on the current 814 judgments, the cost of not collapsing is a drop in R@10
   from **0.675 to 0.443** for BM25 (R@5 0.392 -> 0.343, nDCG@10 0.802 -> 0.659) and from
   **0.588 to 0.381** for dense (R@5 0.371 -> 0.318, nDCG@10 0.761 -> 0.621). With aliases
   kept, 162 of BM25's 400 top-10 slots are a page it had already returned
   (`python -m eval.alias_slots`), which is where the recall goes. That count is 162 rather
   than the 161 quoted under [Corpus quality](#corpus-quality) because this arm still
   excludes stubs, and the one slot a stub occupies displaces a duplicate. Both arms reproduce from the committed CLI:

   ```
   # R@10 over all 40 queries, 814 judgments
   python -m eval.run --retriever bm25                   # 0.675  aliases collapsed
   python -m eval.run --retriever bm25  --keep-aliases   # 0.443  aliases kept
   python -m eval.run --retriever dense                  # 0.588  aliases collapsed
   python -m eval.run --retriever dense --keep-aliases   # 0.381  aliases kept
   ```

   The earlier figure for this ablation, 1.000 -> 0.652, was measured on the retired
   400-judgment pool that BM25 had built by itself: its 1.000 was pool-induced, not a
   retrieval result, so the drop it quoted was against a ceiling that never existed.

   **Hybrid is absent because it cannot be ablated this way.** `HybridRetriever._pages`
   resolves every candidate through `pages.canonical_url` before fusion, unconditionally, so
   RRF already scores one entry per canonical page and there is no switch to turn off.
   Collapsing is structural in the hybrid, not a policy it applies. `rerank` and `agent` are
   absent for a weaker reason: both read a base retriever that `eval.run` builds with the
   default policy, so the flags would not reach the stage being measured. Passing
   `--keep-aliases` to any of the four is an error rather than a silent no-op, since a
   silently ignored flag reads as an ablation that measured nothing.
8. **Fix the faithfulness checker before quoting a faithfulness ranking.** It leaves 26% of
   claims unjudged, the share differs by retriever, and the two defensible ways of handling
   that reverse the order of the five systems. The fix is not a prompt edit chased against
   the score: judge one claim per call, or have the checker echo the claim text it is
   judging, then re-measure everything in the generation section. Until then that section
   supports "all five sit in a wide band" and nothing narrower.
9. **Check refusal on near-miss context, not just mismatched context.** 174 of 175 mismatched
   pairs refused, which measures the easy case. The case that matters is a passage on the
   right topic that is silent on the specific question; building that set needs the existing
   grade-0 judgments, which are already written and unused for this.
10. **Get an independent faithfulness checker.** Same-family checking biases the support rate
    up by an unbounded amount. A human pass over a sample of the 721 claims would bound it,
    and is the smallest thing that would turn these rates into levels rather than contrasts.
