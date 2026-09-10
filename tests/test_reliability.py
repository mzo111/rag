import pytest

from eval.reliability import (
    PROBE_1,
    PROBE_2,
    cohens_kappa,
    confusion,
    fisher_exact_two_sided,
    load_pairs,
    reproduction,
)


def test_confusion_counts_original_by_regrade():
    m = confusion([(0, 0), (0, 1), (1, 1), (2, 0)])
    assert m[0, 0] == 1 and m[0, 1] == 1 and m[1, 1] == 1 and m[2, 0] == 1
    assert m.sum() == 4


def test_perfect_agreement_is_kappa_one():
    k, observed, _ = cohens_kappa([(0, 0), (1, 1), (2, 2), (1, 1)])
    assert k == pytest.approx(1.0) and observed == pytest.approx(1.0)


def test_agreement_at_chance_is_kappa_zero():
    """Both raters spread identically and independently: observed equals chance."""
    pairs = [(a, b) for a in (0, 1) for b in (0, 1)]
    k, observed, chance = cohens_kappa(pairs)
    assert k == pytest.approx(0.0) and observed == pytest.approx(chance)


def test_total_disagreement_is_negative():
    k, _, _ = cohens_kappa([(0, 1), (1, 0), (0, 1), (1, 0)])
    assert k < 0


def test_kappa_of_nothing_is_zero_rather_than_an_error():
    assert cohens_kappa([]) == (0.0, 0.0, 0.0)


def test_kappa_is_zero_when_one_grade_is_the_only_one_used():
    """Chance agreement is 1.0 there, so the ratio is undefined and must not divide by zero."""
    k, observed, chance = cohens_kappa([(1, 1), (1, 1)])
    assert k == 0.0 and observed == 1.0 and chance == 1.0


@pytest.mark.parametrize(
    "table,expected",
    [
        ((3, 1, 1, 3), 0.4857),  # a table with a published two-sided value
        ((9, 11, 6, 14), 0.5145),  # the probes' grade-1 comparison
        ((10, 0, 0, 10), 0.0000),  # complete separation
    ],
)
def test_fisher_matches_known_values(table, expected):
    assert fisher_exact_two_sided(*table) == pytest.approx(expected, abs=5e-4)


def test_fisher_is_one_when_a_margin_is_empty():
    assert fisher_exact_two_sided(0, 0, 0, 0) == 1.0
    assert fisher_exact_two_sided(0, 5, 0, 5) == 1.0


def test_fisher_is_symmetric_under_swapping_rows():
    assert fisher_exact_two_sided(9, 11, 6, 14) == pytest.approx(
        fisher_exact_two_sided(6, 14, 9, 11)
    )


def test_reproduction_counts_only_the_grade_asked_for():
    pairs = [(1, 1), (1, 0), (2, 2), (1, 1)]
    assert reproduction(pairs, 1) == (2, 3)
    assert reproduction(pairs, 2) == (1, 1)
    assert reproduction(pairs, 0) == (0, 0)


# --- the committed probes ------------------------------------------------------------------


def test_both_probes_are_fully_graded():
    for probe in (PROBE_1, PROBE_2):
        pairs = load_pairs(*probe)
        assert len(pairs) == 40


def test_each_probe_shows_twenty_of_each_of_its_two_grades():
    """The probe design: 20 candidates per original grade, two grades per probe."""
    assert sorted(reproduction(load_pairs(*PROBE_1), g)[1] for g in (0, 1, 2)) == [0, 20, 20]
    assert sorted(reproduction(load_pairs(*PROBE_2), g)[1] for g in (0, 1, 2)) == [0, 20, 20]
