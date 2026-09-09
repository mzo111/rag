"""Prompts for the retrieval agent. Written once, first draft, deliberately frozen.

These are **not** to be edited in response to evaluation scores. Iterating a prompt against
40 queries whose judgments are optimistically biased and pooled from the very retrievers
being compared would fit the noise, not the task - the same argument that keeps the RRF
constant at its published value and the reranker's N at 100.

Both prompts ask for JSON and are called with Ollama's ``format="json"`` mode. That is the
documented way to get structured output and is fixed a priori; the parsers in
:mod:`agent.graph` still validate the schema themselves, because constrained decoding
guarantees *valid JSON*, not *the JSON you asked for*.

The ``{...}`` placeholders are filled with :meth:`str.format`, so literal braces in the
examples are doubled.
"""

from __future__ import annotations

# One cheap call, one structured verdict: does this query need splitting at all?
ROUTE_PROMPT = """\
You are routing a search query for a PyTorch documentation search engine.

Decide whether the query must be split into sub-questions before searching.

Answer "decompose" only when a complete answer needs information from two or more
distinct documentation pages - for example the query compares two things, names two
different APIs, or chains a task across separate topics.

Answer "direct" when the query names a single API, asks for one fact, or describes one
task that a single page would answer.

Query: {query}

Reply with JSON only, no prose:
{{"route": "decompose", "reason": "<8 words or fewer>"}}
or
{{"route": "direct", "reason": "<8 words or fewer>"}}
"""

# Depth 1: sub-queries are never themselves decomposed.
DECOMPOSE_PROMPT = """\
You are splitting a PyTorch documentation search query into independent sub-queries.

Rules:
- Produce at most {max_sub_queries} sub-queries.
- Each sub-query must be searchable on its own, with no pronoun referring to another.
- Keep the vocabulary of PyTorch documentation: API names, concepts, task words.
- Do not answer the query, and do not introduce topics the query did not mention.
- If the query really has only one part, return it unchanged as a single sub-query.

Query: {query}

Reply with JSON only, no prose:
{{"sub_queries": ["first sub-query", "second sub-query"]}}
"""
