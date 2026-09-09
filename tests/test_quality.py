from corpus.quality import (
    STATUS_OK,
    STATUS_STUB,
    alias_compatible,
    analyze,
    canonical_of,
    find_alias_groups,
    is_stub,
    is_subsequence,
    normalize_identifiers,
    page_signature,
    path_segments,
)

BASE = "https://docs.pytorch.org/docs/2.14/generated/"


def u(name: str) -> str:
    return BASE + name + ".html"


# --- identifier normalization ---------------------------------------------------


def test_normalize_collapses_dotted_identifiers():
    assert normalize_identifiers("class torch.nn.modules.loss.CrossEntropyLoss(weight=None)") == (
        "class CrossEntropyLoss(weight=None)"
    )
    assert normalize_identifiers("torch.optim.Adam and torch.optim.adam.Adam") == "Adam and Adam"
    assert normalize_identifiers("no identifiers here") == "no identifiers here"
    # a sentence-ending period is not a dotted identifier
    assert normalize_identifiers("Returns a Tensor. Next line") == "Returns a Tensor. Next line"


def test_page_signature_ignores_module_path_only():
    a = ["class torch.nn.CrossEntropyLoss(x)", "Computes loss."]
    b = ["class torch.nn.modules.loss.CrossEntropyLoss(x)", "Computes loss."]
    c = ["class torch.nn.CrossEntropyLoss(x)", "Computes something else."]
    assert page_signature(a) == page_signature(b)
    assert page_signature(a) != page_signature(c)


# --- path compatibility ---------------------------------------------------------


def test_path_segments():
    assert path_segments(u("torch.optim.Adam")) == ["torch", "optim", "Adam"]
    assert path_segments(u("torch.optim.adam.Adam_class")) == ["torch", "optim", "adam", "Adam"]
    assert path_segments("https://docs.pytorch.org/tutorials/beginner/x.html") == ["x"]


def test_is_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert is_subsequence([], ["a"])


def test_alias_compatible_accepts_inserted_module_segments():
    assert alias_compatible(u("torch.optim.Adam"), u("torch.optim.adam.Adam_class"))
    assert alias_compatible(
        u("torch.nn.CrossEntropyLoss"), u("torch.nn.modules.loss.CrossEntropyLoss")
    )
    assert alias_compatible(u("torch.einsum"), u("torch.functional.einsum"))
    assert alias_compatible(u("torch.cuda.manual_seed"), u("torch.cuda.random.manual_seed"))


def test_alias_compatible_rejects_different_apis():
    # same trailing name, different device namespace: genuinely different functions
    assert not alias_compatible(u("torch.cuda.current_device"), u("torch.xpu.current_device"))
    assert not alias_compatible(u("torch.Event"), u("torch.mtia.Event"))
    # unrelated names
    assert not alias_compatible(u("torch.topk"), u("torch.sort"))


def test_canonical_is_shortest_then_alphabetical():
    assert canonical_of([u("torch.optim.adam.Adam_class"), u("torch.optim.Adam")]) == u(
        "torch.optim.Adam"
    )
    same_len = [u("torch.bbb"), u("torch.aaa")]
    assert canonical_of(same_len) == u("torch.aaa")


# --- grouping -------------------------------------------------------------------


def test_find_alias_groups_groups_only_compatible_pages():
    pages = {
        u("torch.optim.Adam"): ["class torch.optim.Adam(params)", "Implements Adam."],
        u("torch.optim.adam.Adam_class"): [
            "class torch.optim.adam.Adam(params)",
            "Implements Adam.",
        ],
        # identical text after normalization but a different device: must not group
        u("torch.cuda.current_device"): ["torch.cuda.current_device()", "Return the index."],
        u("torch.xpu.current_device"): ["torch.xpu.current_device()", "Return the index."],
        u("torch.topk"): ["torch.topk(input, k)", "Returns the k largest elements."],
    }
    groups = find_alias_groups(pages)
    assert groups == [[u("torch.optim.Adam"), u("torch.optim.adam.Adam_class")]]


def test_find_alias_groups_ignores_singletons():
    assert find_alias_groups({u("torch.topk"): ["unique text"]}) == []
    assert find_alias_groups({}) == []


# --- stubs ----------------------------------------------------------------------


def test_is_stub_detects_redirect_placeholders():
    assert is_stub(["This tutorial was deprecated. Redirecting in 3 seconds..."], 20)
    assert is_stub(["This page has been moved. Redirecting now..."], 10)
    assert is_stub(["PyTorch Mobile is no longer actively supported."], 25)


def test_is_stub_rejects_real_pages_that_mention_deprecation():
    # a full tutorial that happens to discuss a deprecated API is not a stub
    assert not is_stub(["This API was deprecated in 1.9 " + "word " * 100], 500)
    # multi-chunk pages are never stubs
    assert not is_stub(["This tutorial was deprecated.", "More content"], 20)
    assert not is_stub(["Ordinary content with no redirect wording"], 20)


# --- analyze --------------------------------------------------------------------


def test_analyze_reports_canonicals_and_stubs():
    pages = {
        u("torch.optim.Adam"): ["class torch.optim.Adam(p)", "Implements Adam."],
        u("torch.optim.adam.Adam_class"): ["class torch.optim.adam.Adam(p)", "Implements Adam."],
        u("torch.topk"): ["torch.topk(input, k)"],
        "https://docs.pytorch.org/tutorials/old.html": [
            "This tutorial was deprecated. Redirecting in 3 seconds"
        ],
    }
    tokens = {k: 50 for k in pages}
    tokens[u("torch.topk")] = 10
    r = analyze(pages, tokens)
    assert r.canonical[u("torch.optim.adam.Adam_class")] == u("torch.optim.Adam")
    assert r.canonical[u("torch.optim.Adam")] == u("torch.optim.Adam")
    assert u("torch.topk") not in r.canonical  # not an alias
    assert r.stubs == {"https://docs.pytorch.org/tutorials/old.html"}
    assert r.n_alias_pages == 2 and r.n_redundant == 1
    assert r.status_of(u("torch.topk")) == STATUS_OK
    assert r.status_of("https://docs.pytorch.org/tutorials/old.html") == STATUS_STUB
    assert "alias groups: 1" in r.summary() and "stubs: 1" in r.summary()
