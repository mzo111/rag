import numpy as np
import pytest

from eval.paired import (
    RECALL_K,
    QueryRun,
    bootstrap,
    ci,
    per_query_scores,
    transition_counts,
    win_loss_tie,
)

A, B, C = "pageA", "pageB", "pageC"


def run(ranked, keys, grades, cat="api_lookup", qid="q01"):
    return QueryRun(qid, cat, list(ranked), list(keys), np.array(grades, dtype=int))


# --- per-query scoring ----------------------------------------------------------


def test_recall_and_ceiling_when_relevant_fits_in_k():
    r = run([A, B], [A, B, C], [2, 1, 0])
    s = per_query_scores([r], [r.grades])
    assert s["recall5"][0] == pytest.approx(1.0)
    assert s["recall5_ceil"][0] == pytest.approx(1.0)  # 2 relevant, ceiling is 1.0


def test_ceiling_rescales_when_more_relevant_than_k():
    # 10 relevant pages, only 5 slots: raw recall caps at 0.5, which IS the ceiling
    keys = [f"p{i}" for i in range(10)]
    r = run(keys[:5], keys, [1] * 10)
    s = per_query_scores([r], [r.grades])
    assert s["recall5"][0] == pytest.approx(0.5)
    assert s["recall5_ceil"][0] == pytest.approx(1.0)


def test_ceiling_is_nan_when_a_draw_leaves_no_relevant_page():
    r = run([A], [A, B], [0, 0])
    s = per_query_scores([r], [r.grades])
    assert s["recall5"][0] == 0.0
    assert np.isnan(s["recall5_ceil"][0])


def test_scores_follow_the_drawn_grades_not_the_originals():
    r = run([A], [A, B], [2, 0])
    flipped = [np.array([0, 2])]  # A becomes irrelevant, B relevant but unretrieved
    assert per_query_scores([r], flipped)["recall5"][0] == 0.0


def test_recall_k_is_five():
    assert RECALL_K == 5


# --- the pairing itself ---------------------------------------------------------


def _cache(rank_a, rank_b, rank_c=None):
    keys, grades = [A, B, C], [2, 1, 0]
    mk = lambda rk: [run(rk, keys, grades)]  # noqa: E731
    cache = {"bm25": mk(rank_a), "dense": mk(rank_b)}
    cache["hybrid"] = mk(rank_c if rank_c is not None else rank_a)
    return cache


def test_identical_systems_have_exactly_zero_paired_difference():
    # the whole point of pairing: shared judgment error cancels completely
    cache = _cache([A, B], [A, B], [A, B])
    res = bootstrap(cache, transition_counts(), draws=50, seed=1)
    for key, v in res["diff"].items():
        assert np.all(v == 0.0), key


def test_a_strictly_better_system_never_loses_under_any_draw():
    # hybrid retrieves both judged pages, dense retrieves neither
    cache = _cache([A, B], [], [A, B])
    res = bootstrap(cache, transition_counts(), draws=50, seed=1)
    assert np.all(res["diff"][("hybrid", "dense", "all", "recall5")] >= 0.0)


def test_marginal_intervals_are_wider_than_the_paired_interval():
    """Pairing pays off when systems are positively correlated, which real ones are.

    Five of six queries are retrieved identically, so shared judgment error cancels on
    those and only the sixth contributes to the difference - while both marginals still
    carry the full judgment noise of all six.
    """
    keys, grades = [A, B, C], [2, 1, 0]
    same = [run([A, B], keys, grades, qid=f"q{i:02d}") for i in range(5)]
    cache = {
        "bm25": [*same, run([A], keys, grades, qid="q05")],
        "dense": [*same, run([B], keys, grades, qid="q05")],
    }
    cache["hybrid"] = cache["bm25"]
    res = bootstrap(cache, transition_counts(), draws=800, seed=2)
    lo, _, hi = ci(res["diff"][("bm25", "dense", "all", "recall5")])
    mlo, _, mhi = ci(res["marginal"][("bm25", "all", "recall5")])
    assert (hi - lo) < (mhi - mlo)


def test_pairing_can_widen_when_systems_are_anticorrelated():
    """The honest converse: pairing is not a free narrowing.

    Two systems that never retrieve the same page have differences that swing more than
    either system alone. Recorded so the shrink seen on real data is read as an empirical
    property of these retrievers, not a guarantee of the method.
    """
    keys, grades = [A, B, C], [2, 1, 0]
    cache = {
        "bm25": [run([A], keys, grades), run([B], keys, grades, qid="q02")],
        "dense": [run([B], keys, grades), run([A], keys, grades, qid="q02")],
    }
    cache["hybrid"] = cache["bm25"]
    res = bootstrap(cache, transition_counts(), draws=800, seed=2)
    lo, _, hi = ci(res["diff"][("bm25", "dense", "all", "recall5")])
    mlo, _, mhi = ci(res["marginal"][("bm25", "all", "recall5")])
    assert (hi - lo) > (mhi - mlo)


def test_bootstrap_is_reproducible_for_a_seed():
    cache = _cache([A, B], [B, A])
    a = bootstrap(cache, transition_counts(), draws=30, seed=7)
    b = bootstrap(cache, transition_counts(), draws=30, seed=7)
    assert np.array_equal(
        a["diff"][("bm25", "dense", "all", "recall5")],
        b["diff"][("bm25", "dense", "all", "recall5")],
    )


# --- transition matrix ----------------------------------------------------------


def test_transition_counts_are_three_rows_of_twenty():
    counts = transition_counts()
    assert counts.shape == (3, 3)
    assert counts.sum(1).tolist() == [20.0, 20.0, 20.0]


def test_transition_rows_normalise_to_probabilities():
    M = transition_counts()
    M = M / M.sum(1, keepdims=True)
    assert np.allclose(M.sum(1), 1.0)
    assert M[0, 0] > M[1, 1]  # grade 0 reproduces better than grade 1


# --- win / loss / tie -----------------------------------------------------------


def test_win_loss_tie_counts_ties_explicitly():
    cache = _cache([A, B], [A, B], [A, B])
    w = win_loss_tie(cache)[("hybrid", "bm25")]
    assert w == {"wins": 0, "losses": 0, "ties": 1, "n": 1}


def test_win_loss_tie_splits_wins_and_losses():
    keys, grades = [A, B, C], [2, 1, 0]
    cache = {
        "bm25": [run([C], keys, grades), run([A, B], keys, grades, qid="q02")],
        "dense": [run([A], keys, grades), run([C], keys, grades, qid="q02")],
    }
    cache["hybrid"] = cache["bm25"]
    w = win_loss_tie(cache)[("bm25", "dense")]
    assert (w["wins"], w["losses"], w["ties"], w["n"]) == (1, 1, 0, 2)
