"""Prompts for the generation stage. Written once, first draft, deliberately frozen.

Same rule as :mod:`agent.prompts`: these are **not** edited in response to any number this
repo reports. The faithfulness rate below is a measurement of this prompt, and a prompt
revised until the rate improves measures nothing but the revision. If a prompt is wrong,
that is a finding to write down, not a knob to turn.

Three prompts, three separate calls, deliberately not merged:

* :data:`ANSWER_PROMPT` answers from retrieved passages, or refuses.
* :data:`CLAIMS_PROMPT` splits an answer into atomic claims. It never sees the passages, so
  it cannot be influenced by whether a claim is going to check out.
* :data:`SUPPORT_PROMPT` judges claims against the passages. It never sees the question, so
  it cannot substitute "answers the question" for "is stated in the passages".

All three ask for JSON and run under Ollama's ``format="json"``; the parsers in
:mod:`generate.run` and :mod:`generate.faithfulness` still validate the schema, because
constrained decoding guarantees valid JSON, not the JSON that was asked for.

The ``{...}`` placeholders are filled with :meth:`str.format`, so literal braces are doubled.
"""

from __future__ import annotations

# The exact string the answerer must emit when the passages do not contain the answer.
# Fixed a priori and asserted in tests: the refusal rate is only meaningful if refusal has
# one spelling that cannot drift.
REFUSAL_MARKER = "INSUFFICIENT_CONTEXT"

ANSWER_PROMPT = """\
You answer questions about PyTorch using only the numbered passages below.

Rules:
- Use only the passages. Do not use anything you know about PyTorch that is not in them.
- Every sentence of the answer must cite the passage it came from, as [C1], or [C2][C5]
  when it draws on more than one.
- Cite only passages that state what the sentence says.
- If the passages do not contain the answer, refuse: set "refused" to true, set "answer" to
  "{refusal}", and cite nothing. A partial answer to a different question is still a refusal.
- Do not guess, and do not fill a gap in the passages from memory.
- At most 120 words.

Passages:
{context}

Question: {query}

Reply with JSON only, no prose:
{{"refused": false, "answer": "<answer, every sentence carrying a [C<n>] citation>"}}
or
{{"refused": true, "answer": "{refusal}"}}
"""

CLAIMS_PROMPT = """\
You split an answer into atomic claims.

Rules:
- Each claim is one self-contained statement that can be checked on its own.
- Resolve references: replace "it", "this" and "that" with the thing being referred to.
- Drop the [C<n>] citation markers from the claim text.
- Use only what the answer says. Do not add, correct or complete anything.
- Do not judge whether a claim is true.
- At most {max_claims} claims.

Answer: {answer}

Reply with JSON only, no prose:
{{"claims": ["first claim", "second claim"]}}
"""

SUPPORT_PROMPT = """\
You check whether numbered passages support each of the claims below.

A claim is supported only if a passage states it. This is about the passages, not about
PyTorch: a claim that is true but that no passage states is NOT supported. A claim that
contradicts the passages is NOT supported.

Passages:
{context}

Claims:
{claims}

Judge every claim, in order, by its number. Give the passage that supports it, or null when
none does.

Reply with JSON only, no prose:
{{"verdicts": [{{"claim": 1, "supported": true, "passage": 3}}, \
{{"claim": 2, "supported": false, "passage": null}}]}}
"""
