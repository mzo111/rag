"""Corpus quality analysis: alias pages and deprecated stubs.

Two properties of the PyTorch docs distort retrieval evaluation, and both are recorded
here rather than fixed by deleting pages, so the decision stays reversible.

**Alias pages.** Sphinx autodoc publishes the same object at several URLs, e.g.
``torch.optim.Adam`` and ``torch.optim.adam.Adam_class``, or ``torch.nn.CrossEntropyLoss``
and ``torch.nn.modules.loss.CrossEntropyLoss``. Their text is identical apart from the
dotted path in signatures. A retriever spends several of its top-k slots on the same
content, and recall depends on whether the judge listed every alias. Detection requires all
three of:

1. identical page text after collapsing dotted identifiers to their last segment;
2. *path compatibility* - one URL's dotted path is a subsequence of the other's, so
   ``torch.einsum`` matches ``torch.functional.einsum`` but ``torch.cuda.current_device``
   does not match ``torch.xpu.current_device``;
3. the same device namespace, so ``torch.Event`` does not match ``torch.mtia.Event``.

Conditions 2 and 3 matter: 18 groups have byte-identical normalized text but document
genuinely different APIs, and text alone would merge them.

**Deprecated stubs.** ~29 pages are redirect notices ("This tutorial was deprecated",
"PyTorch Mobile is no longer actively supported") with no content. They match queries
lexically and answer nothing. Detection requires a single short chunk *and* redirect
wording, because a full-length tutorial may mention deprecation in passing.

All functions are pure. ``corpus.store`` persists the result on the ``pages`` table.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# A dotted identifier: keep only the final segment (torch.nn.modules.loss.X -> X).
DOTTED = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)+([A-Za-z_][A-Za-z0-9_]*)\b")
# Device namespaces that must match exactly for two pages to be aliases.
DEVICE = re.compile(r"\b(cuda|xpu|mtia|mps|cpu|hpu|npu)\b")

STUB_PATTERNS = (
    "was deprecated",
    "has been deprecated",
    "have been deprecated",
    "has been moved",
    "have been moved",
    "no longer actively supported",
    "redirecting in",
    "redirecting now",
    "will redirect",
    "you will be redirected",
)
STUB_MAX_TOKENS = 80

STATUS_OK = "ok"
STATUS_STUB = "stub"


def normalize_identifiers(text: str) -> str:
    """Collapse dotted identifiers to their last segment, so alias pages compare equal."""
    return DOTTED.sub(r"\1", text)


def page_signature(texts: Iterable[str]) -> str:
    """Stable hash of a page's chunk texts after identifier normalization."""
    joined = "\n".join(texts)
    return hashlib.sha256(normalize_identifiers(joined).encode("utf-8")).hexdigest()


def path_segments(url: str) -> list[str]:
    """Dotted segments of a page's identifier: torch.optim.Adam.html -> [torch, optim, Adam].

    Non-autodoc pages (tutorials, notes) yield their path, which never matches another
    page's segments as a subsequence, so they are never treated as aliases.
    """
    base = url.rstrip("/").split("/")[-1]
    base = base.removesuffix(".html").removesuffix("_class")
    return base.split(".") if base else []


def is_subsequence(a: Sequence[str], b: Sequence[str]) -> bool:
    """True if every element of a appears in b in order."""
    it = iter(b)
    return all(x in it for x in a)


def devices(segments: Sequence[str]) -> set[str]:
    return set(DEVICE.findall(".".join(segments)))


def alias_compatible(url_a: str, url_b: str) -> bool:
    """True if two same-text pages are plausibly the same object under different paths."""
    a, b = path_segments(url_a), path_segments(url_b)
    if devices(a) != devices(b):
        return False
    return is_subsequence(a, b) or is_subsequence(b, a)


def canonical_of(group: Sequence[str]) -> str:
    """Pick the canonical URL of an alias group: shortest path, ties broken alphabetically."""
    return min(group, key=lambda u: (len(u), u))


def count_text_only_collisions(pages: Mapping[str, Sequence[str]]) -> int:
    """Signature buckets that the path and device checks split into more than one cluster.

    These are the pages that would be merged by identical text alone and must not be:
    `torch.cuda.current_device` against `torch.xpu.current_device`, `torch.Event` against
    `torch.mtia.Event`. The count is the argument for conditions 2 and 3 not being optional,
    so it is measured rather than asserted.
    """
    by_sig: dict[str, list[str]] = {}
    for url, texts in pages.items():
        by_sig.setdefault(page_signature(texts), []).append(url)
    split = 0
    for urls in by_sig.values():
        if len(urls) < 2:
            continue
        clusters: list[list[str]] = []
        for url in sorted(urls):
            for cluster in clusters:
                if all(alias_compatible(url, other) for other in cluster):
                    cluster.append(url)
                    break
            else:
                clusters.append([url])
        if len(clusters) > 1:
            split += 1
    return split


def find_alias_groups(pages: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Group URLs whose pages are aliases of one another. Singletons are omitted."""
    by_sig: dict[str, list[str]] = {}
    for url, texts in pages.items():
        by_sig.setdefault(page_signature(texts), []).append(url)

    groups: list[list[str]] = []
    for urls in by_sig.values():
        if len(urls) < 2:
            continue
        # Same text is not enough: split into path-compatible clusters.
        clusters: list[list[str]] = []
        for url in sorted(urls):
            for cluster in clusters:
                if all(alias_compatible(url, other) for other in cluster):
                    cluster.append(url)
                    break
            else:
                clusters.append([url])
        groups.extend(c for c in clusters if len(c) > 1)
    return sorted(groups, key=lambda g: canonical_of(g))


def is_stub(texts: Sequence[str], n_tokens: int) -> bool:
    """True for deprecation/redirect placeholder pages with no real content."""
    if len(texts) != 1 or n_tokens > STUB_MAX_TOKENS:
        return False
    low = texts[0].lower()
    return any(p in low for p in STUB_PATTERNS)


@dataclass
class QualityReport:
    canonical: dict[str, str] = field(default_factory=dict)  # url -> canonical url
    alias_groups: list[list[str]] = field(default_factory=list)
    stubs: set[str] = field(default_factory=set)
    # Buckets identical text alone would have merged wrongly; see count_text_only_collisions.
    text_only_collisions: int = 0

    @property
    def n_alias_pages(self) -> int:
        return sum(len(g) for g in self.alias_groups)

    @property
    def n_redundant(self) -> int:
        """Pages that duplicate a canonical page."""
        return self.n_alias_pages - len(self.alias_groups)

    def status_of(self, url: str) -> str:
        return STATUS_STUB if url in self.stubs else STATUS_OK

    def summary(self) -> str:
        return (
            f"alias groups: {len(self.alias_groups)} covering {self.n_alias_pages} pages "
            f"({self.n_redundant} redundant); deprecated stubs: {len(self.stubs)}; "
            f"identical text but different APIs: {self.text_only_collisions} groups"
        )


def analyze(pages: Mapping[str, Sequence[str]], tokens: Mapping[str, int]) -> QualityReport:
    """Compute alias groups and stub pages for a corpus.

    ``pages`` maps url -> chunk texts in order; ``tokens`` maps url -> total token count.
    """
    groups = find_alias_groups(pages)
    canonical: dict[str, str] = {}
    for group in groups:
        head = canonical_of(group)
        for url in group:
            canonical[url] = head
    stubs = {url for url, texts in pages.items() if is_stub(texts, tokens.get(url, 0))}
    return QualityReport(
        canonical=canonical,
        alias_groups=groups,
        stubs=stubs,
        text_only_collisions=count_text_only_collisions(pages),
    )
