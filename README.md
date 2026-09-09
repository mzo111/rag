# rag — PyTorch docs retrieval: corpus + evaluation harness

Retrieval over PyTorch's documentation, built in order: corpus, then measurement, then
retrievers. The evaluation set was written before any retriever existed so nothing could be
tuned against it. **The relevance judgments are still blank** and are the next thing needed.

```
corpus/fetch.py   download docs (stable = 2.14.0) + tutorials HTML -> data/raw/ (gitignored)
corpus/chunk.py   HTML -> text -> chunks (pure, unit-tested)      -> data/chunks.jsonl
corpus/quality.py alias-page and deprecated-stub detection (pure)
corpus/store.py   SQLite + FTS5 schema and idempotent loader       -> data/corpus.db
retrieval/bm25.py BM25 baseline over the FTS5 index
eval/queries.yaml 40 queries, relevance judgments to be filled in by hand
eval/metrics.py   recall@k, MRR, nDCG@10 (pure functions, hand-computed tests)
eval/run.py       harness: any object with search(query, k) -> metrics table
eval/pool.py      build a judging pool from retriever output, merge grades back
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

CI (`.github/workflows/ci.yml`): `ruff check`, `ruff format --check`, `pytest -q`.

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
aliases were collapsed. Stubs cost only 1 slot. Judgments written against an uncleaned corpus
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
`--keep-aliases` / `--keep-stubs` to measure what they are worth.

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
`multi_hop` (8), `tutorial` (10). All 40 are now judged by hand — 400 judgments, 10 per query,
294 of them relevant (159 at grade 1, 135 at grade 2). See the header of the file for the grade
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

Run 2026-09-08 against `data/corpus.db` (11,832 chunks / 3,678 pages), 40 queries, **611
judgments** — the original 400 from the BM25 pool plus 211 appended from the dense delta pool.
392 are relevant (grade > 0), a mean of 9.8 relevant pages per query. `k=10`.

| retriever | group | R@5 | of ceiling | R@10 | of ceiling | MRR | nDCG@10 (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| random (seed 0) | all | 0.015 | 0.03 | 0.027 | 0.03 | 0.056 | 0.02 [0.01, 0.03] |
| | api_lookup | 0.032 | 0.04 | 0.042 | 0.04 | 0.079 | |
| | conceptual | 0.017 | 0.03 | 0.033 | 0.04 | 0.084 | |
| | multi_hop | 0.000 | 0.00 | 0.011 | 0.01 | 0.018 | |
| | tutorial | 0.007 | 0.02 | 0.014 | 0.02 | 0.031 | |
| **bm25** | **all** | **0.448** | **0.81** | **0.772** | **0.84** | **0.988** | **0.82 [0.58, 0.72]** |
| | api_lookup | 0.536 | 0.72 | 0.858 | 0.87 | 1.000 | |
| | conceptual | 0.432 | 0.81 | 0.745 | 0.78 | 1.000 | |
| | multi_hop | 0.430 | 0.90 | 0.724 | 0.79 | 1.000 | |
| | tutorial | 0.373 | 0.90 | 0.736 | 0.90 | 0.950 | |
| **dense** | **all** | **0.423** | **0.76** | **0.677** | **0.73** | **0.988** | **0.78 [0.56, 0.69]** |
| | api_lookup | 0.474 | 0.64 | 0.642 | 0.65 | 0.958 | |
| | conceptual | 0.412 | 0.77 | 0.626 | 0.66 | 1.000 | |
| | multi_hop | 0.413 | 0.87 | 0.769 | 0.83 | 1.000 | |
| | tutorial | 0.381 | 0.92 | 0.696 | 0.85 | 1.000 | |
| **hybrid** (RRF) | **all** | **0.449** | **0.81** | **0.718** | **0.78** | **0.971** | **0.83 [0.56, 0.72]** |
| | api_lookup | 0.552 | 0.75 | 0.753 | 0.76 | 0.944 | |
| | conceptual | 0.410 | 0.77 | 0.692 | 0.73 | 1.000 | |
| | multi_hop | 0.430 | 0.90 | 0.730 | 0.79 | 1.000 | |
| | tutorial | 0.381 | 0.92 | 0.693 | 0.85 | 0.950 | |
| rerank(bm25) | all | 0.432 | 0.78 | 0.625 | 0.68 | 0.975 | 0.76 [0.50, 0.65] |
| | api_lookup | 0.540 | 0.73 | 0.743 | 0.75 | 1.000 | |
| | conceptual | 0.393 | 0.74 | 0.640 | 0.67 | 1.000 | |
| | multi_hop | 0.452 | 0.95 | 0.621 | 0.67 | 0.938 | |
| | tutorial | 0.324 | 0.78 | 0.472 | 0.58 | 0.950 | |
| rerank(hybrid) | all | 0.432 | 0.78 | 0.630 | 0.68 | 0.975 | 0.76 [0.50, 0.66] |
| | api_lookup | 0.549 | 0.74 | 0.707 | 0.72 | 1.000 | |
| | conceptual | 0.395 | 0.74 | 0.643 | 0.68 | 1.000 | |
| | multi_hop | 0.438 | 0.92 | 0.652 | 0.71 | 0.938 | |
| | tutorial | 0.323 | 0.78 | 0.505 | 0.62 | 0.950 | |

**Read recall against its ceiling, not against 1.0.** With 9.8 relevant pages per query on
average, five slots cannot hold them all: the achievable recall@5 is 0.555 overall (0.741 for
`api_lookup`, down to 0.415 for `tutorial`, which now averages 12.3 relevant pages). The
"of ceiling" columns are the fraction of what was reachable. BM25 at 0.448 raw is 0.81 of
ceiling; its weakest category by raw recall (`tutorial`, 0.373) is among its strongest once
the ceiling is applied (0.90).

**BM25's recall@10 fell from 1.000 to 0.772. That is the pooling bias being corrected, not a
regression.** The retriever did not change and its rankings are identical. The earlier 1.000
was an artifact: every judgment came from BM25's own top 10, so BM25 could not miss anything
that existed. The pool now also contains dense-only pages that BM25 does not retrieve, and
those are exactly the misses the old number could not see. A drop here is the measurement
starting to work.

Dense scores below BM25 on this judgment set (R@5 0.423 vs 0.448, R@10 0.677 vs 0.772), but
the comparison is still tilted — see limitation 1. It leads on `multi_hop` recall@10 (0.769 vs
0.724) and on `tutorial` of-ceiling recall@5 (0.92 vs 0.90), the two categories where queries
describe a task rather than name a symbol.

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

The design works. Paired intervals are 2.2–3.0× narrower than naively combining two marginals:

| comparison | marginal A | marginal B | naive sum | **paired** | shrink |
|---|---:|---:|---:|---:|---:|
| hybrid − bm25 | 0.175 | 0.165 | 0.340 | **0.114** | 3.0× |
| hybrid − dense | 0.175 | 0.163 | 0.338 | **0.121** | 2.8× |
| bm25 − dense | 0.165 | 0.163 | 0.328 | **0.150** | 2.2× |
| rerank(hybrid) − hybrid | 0.175 | 0.175 | 0.350 | **0.127** | 2.8× |
| rerank(bm25) − bm25 | 0.177 | 0.165 | 0.343 | **0.133** | 2.6× |

**And the answer is still no. Hybrid does not beat either system.** Every paired interval
crosses zero, on both raw and of-ceiling recall@5, overall and in all four categories:

| comparison | recall@5 raw | recall@5 of ceiling | verdict |
|---|---|---|---|
| hybrid − bm25 | +0.014 [−0.044, +0.070] | +0.019 [−0.060, +0.090] | no difference |
| hybrid − dense | +0.015 [−0.046, +0.075] | +0.020 [−0.055, +0.095] | no difference |
| bm25 − dense | +0.002 [−0.075, +0.075] | +0.003 [−0.094, +0.099] | no difference |

Per-query wins on recall@5, with ties counted rather than split:

| comparison | wins | losses | ties |
|---|---:|---:|---:|
| hybrid > bm25 | 8 | 6 | 26 |
| hybrid > dense | 13 | 7 | 20 |
| bm25 > dense | 12 | 10 | 18 |

Hybrid's central estimate is positive against both, and it ties BM25 on 26 of 40 queries —
consistent with a small real gain, and equally consistent with none. A paired design that
resolves 2.2–3.0× finer than the marginals still cannot separate these systems, so the honest
reading is that **fusion buys nothing measurable here.** Nothing was adjusted in response:
no constant search, no depth search, no retriever changes.

MRR is reported as a secondary line only — hybrid − bm25 is +0.028 [−0.063, +0.120]. At 65.7%
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
| rerank(hybrid) − hybrid | −0.024 [−0.084, +0.044] | −0.039 [−0.115, +0.049] | no difference |
| rerank(bm25) − bm25 | −0.016 [−0.079, +0.054] | −0.028 [−0.114, +0.061] | no difference |

Paired intervals still cross zero, so this is not a statistically clean loss. But the
per-query counts point the same way, and more sharply than the intervals do:

| comparison | wins | losses | ties |
|---|---:|---:|---:|
| rerank(hybrid) > hybrid | 6 | **13** | 21 |
| rerank(bm25) > bm25 | 7 | **15** | 18 |

The reranker loses on twice as many queries as it wins. It is worst on `tutorial`
(rerank(hybrid) − hybrid = −0.076 raw, −0.120 of-ceiling) and roughly neutral on `api_lookup`
— it damages exactly the task-shaped queries where dense retrieval was contributing most. It
also drops recall@10 substantially (hybrid 0.718 → 0.630), which is expected: with N=100 the
cross-encoder is free to demote a correct page out of the top 10 entirely.

Nothing was adjusted in response — no N search, no model search, no base changes. On this
corpus, at this judgment quality, a cross-encoder costing ~1.8 s per query buys nothing and
plausibly costs accuracy. The paired design is 2.6–2.8× finer than the marginals here, so
this is not merely an underpowered test.

### Limitations

These are the reasons not to read the table as a clean measurement of retrieval quality.

1. **The pool still favours BM25.** The original 400 judgments came from BM25's top 10 alone;
   the delta added dense's top 10. Two retrievers is better than one, but BM25 contributed
   every judgment in the first round and half the second, so a page neither retriever surfaced
   is still invisible, and BM25 keeps a residual advantage this does not remove. Hybrid fuses exactly the two systems
   that built the pool, so it can only reorder pages already judged — it gets no credit for
   finding anything new, because nothing it finds is new by construction.

2. **The judgments are not reliable at the level the numbers imply.** Two blind re-grade
   probes measured how repeatable the grading is: grade 0 reproduced at 90%, grade 2 at 50%,
   grade 1 at 30–45%. Three-way Cohen's κ = 0.09 — barely above chance. The two probes differ
   in *both* the contrast used (1s against 0s versus 1s against 2s) and whether the written
   rubric in `eval/GRADING.md` existed, so the change in grade-1 stability cannot be
   attributed to either one alone (Fisher exact p = 0.514).

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

5. **The 159 grade-1 judgments predate the rubric.** They were graded before
   `eval/GRADING.md` was written and have not been re-graded. `eval/regrade_ones.yaml` is
   built and unfilled. Grade 1 is the least reproducible category, and it is the largest.

6. **nDCG is reported but is not used to rank retrievers.** Its `2^g − 1` gain weights a
   grade 2 at 3× a grade 1, which amplifies exactly the 1↔2 boundary that reproduces worst
   (50% and 30–45%). It is quoted to two decimals with its interval attached and should not
   be read more precisely than that. Ranking conclusions rest on recall against ceiling.

To plug in a retriever, implement `search(query: str, k: int) -> list[str]` returning chunk ids
from `data/corpus.db` and pass it to `eval.run.evaluate(retriever, load_queries(), store.locate)`.

## Next

1. ~~Grade `eval/pool.yaml` and merge it.~~ Done 2026-09-08.
2. ~~Add a semantic retriever, re-pool, grade the new candidates.~~ Done: 611 judgments, and
   BM25's tautological R@10 of 1.000 is now 0.772.
3. **Re-grade the 159 pre-rubric grade-1 judgments** (`eval/regrade_ones.yaml`, built and
   unfilled). Grade 1 is the largest category and the least reproducible; until it is redone
   against `eval/GRADING.md`, limitation 3 stands and the metrics stay optimistically biased.
4. Re-run the blind probe after that re-grade, holding the contrast fixed this time, so the
   rubric's effect can be separated from the contrast effect.
5. Decide what alias pages should ultimately be: collapsed as they are now, merged into one
   page at chunk time, or dropped. The data to decide is in `pages.canonical_url`, and the
   measured cost of not collapsing them is a drop in R@10 from 1.000 to 0.652 on the
   400-judgment set.
