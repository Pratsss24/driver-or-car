"""Tests for the driver-or-car pipeline.

The anchors here are of three kinds, in descending order of how much
they are worth.

**Algebra.** A forest on n nodes with c components has exactly n - c
edges and no cycles; a design with one column dropped per component has
rank n - c; centring within a component makes that component sum to
zero. These have one right answer and no tolerance.

**Recovery of a planted truth.** Lap times are generated with known
driver effects, known team paces and known noise, and the pipeline has
to give the planted numbers back. This is the only kind of test that
can catch a sign error, and two of them below exist because it did.

**Refusal.** A measurement that cannot be made has to be declined and
counted, not returned as a number. Every refusal branch in
`contrast.build` is reached by a fixture designed to trip it.

No test compares against saved output. A golden file only proves the
code still does what it did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import contrast                                            # noqa: E402
import effects                                             # noqa: E402
import graph                                               # noqa: E402
import linear                                              # noqa: E402
import stints                                              # noqa: E402


# --------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------

class FakeResult:
    """Just enough of data.Result for graph.build."""

    def __init__(self, season, round_, driver, team):
        self.season, self.round = season, round_
        self.driver, self.team = driver, team


def results_of(pairings: dict[str, list[tuple[str, str]]],
               races: int = 4) -> list[FakeResult]:
    """One entry per driver per race, for the teams described."""
    out = []
    for round_ in range(1, races + 1):
        for team, members in pairings.items():
            for one, two in members:
                out.append(FakeResult(2023, round_, one, team))
                out.append(FakeResult(2023, round_, two, team))
    return out


def laps_of(*, drivers: dict[str, float], team: str, round_: int = 1,
            laps: int = 40, base: float = 90.0, wear: float = 0.03,
            burn: float = -0.02, noise: float = 0.0, seed: int = 0,
            stop_lap: int = 20) -> list[stints.StintLap]:
    """Laps for one team in one race, with each driver's pace planted.

    `drivers` maps a name to how many seconds a lap slower than the team
    base that driver is, so a *negative* value is a quicker driver. The
    pipeline reports gaps the other way round, and keeping these two
    conventions visibly different is deliberate.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name, offset in drivers.items():
        for lap in range(1, laps + 1):
            stint, age = (1, lap) if lap <= stop_lap else (2, lap - stop_lap)
            seconds = (base + offset + wear * age + burn * lap
                       + (rng.normal(0, noise) if noise else 0.0))
            rows.append(stints.StintLap(
                season=2023, round=round_, race=f"race {round_}",
                circuit=f"circuit{round_}", driver=name, team=team,
                lap=lap, stint=stint, age=age, seconds=seconds,
                position=1))
    return rows


# --------------------------------------------------------------------
# the graph: what is answerable, before anything is measured
# --------------------------------------------------------------------

def test_a_forest_has_no_cycles_and_the_count_is_exact():
    comparison = graph.build(results_of({
        "A": [("ann", "bob")], "B": [("cal", "dee")],
        "C": [("eve", "fay")]}))
    shape = comparison.summary()
    assert shape == {"drivers": 6, "pairings": 3, "components": 3,
                     "identified": 3, "free": 3, "cycles": 0,
                     "largest component": 2, "isolated pairs": 3}
    # n - c edges exactly, which is what makes it a forest.
    assert shape["pairings"] == shape["drivers"] - shape["components"]


def test_a_mid_season_swap_joins_two_drivers_into_one_component():
    results = results_of({"A": [("ann", "bob")]}, races=2)
    results += [FakeResult(2023, 3, "ann", "A"),
                FakeResult(2023, 3, "cal", "A")]
    comparison = graph.build(results)
    shape = comparison.summary()
    assert shape["drivers"] == 3 and shape["components"] == 1
    assert shape["cycles"] == 0            # a path, still a tree
    assert comparison.path("bob", "cal") == ["bob", "ann", "cal"]


def test_a_cycle_appears_only_when_a_third_pairing_closes_the_triangle():
    results = results_of({"A": [("ann", "bob")]}, races=1)
    results += [FakeResult(2023, 2, "bob", "A"),
                FakeResult(2023, 2, "cal", "A"),
                FakeResult(2023, 3, "ann", "A"),
                FakeResult(2023, 3, "cal", "A")]
    comparison = graph.build(results)
    assert comparison.summary()["cycles"] == 1
    # And with a cycle every comparison has a second route to check it.
    assert comparison.path("ann", "cal") is not None


def test_unconnected_drivers_have_no_path_however_hard_you_look():
    comparison = graph.build(results_of({
        "A": [("ann", "bob")], "B": [("cal", "dee")]}))
    assert comparison.path("ann", "dee") is None
    assert comparison.path("ann", "ann") == ["ann"]


def test_the_design_drops_exactly_one_column_per_component():
    comparison = graph.build(results_of({
        "A": [("ann", "bob")], "B": [("cal", "dee")]}))
    rows = [contrast.Contrast(2023, 1, "c", "A", "ann", "bob", 0.2, 0.01,
                              30, 30),
            contrast.Contrast(2023, 1, "c", "B", "cal", "dee", 0.1, 0.01,
                              30, 30)]
    matrix, names = graph.design(rows, comparison)
    assert len(names) == 2                 # 4 drivers, 2 components
    assert np.linalg.matrix_rank(matrix) == len(names)
    # Every row has one +1 and one -1 among the columns it touches, or a
    # lone +/-1 where the other driver is the dropped baseline.
    assert set(matrix.sum(axis=1)) <= {-1.0, 0.0, 1.0}


def test_centring_makes_each_component_sum_to_zero_separately():
    comparison = graph.build(results_of({
        "A": [("ann", "bob")], "B": [("cal", "dee")]}))
    centred = graph.centre(
        {"ann": 1.0, "bob": 0.0, "cal": 5.0, "dee": 5.4}, comparison)
    assert centred["ann"] == pytest.approx(0.5)
    assert centred["bob"] == pytest.approx(-0.5)
    assert centred["cal"] + centred["dee"] == pytest.approx(0.0)
    # The 5-second offset of the second component vanishes entirely,
    # which is the point: it was never comparable to the first.
    assert abs(centred["cal"]) < 1.0


# --------------------------------------------------------------------
# the contrasts: measuring one pairing in one race
# --------------------------------------------------------------------

def test_a_planted_teammate_gap_comes_back_out():
    rows = laps_of(drivers={"ann": 0.0, "bob": 0.25}, team="A")
    found, census = contrast.build(rows, {})
    assert census["kept: contrasts"] == 1
    assert found[0].quicker == "ann" and found[0].slower == "bob"
    assert found[0].gap == pytest.approx(0.25, abs=1e-6)


def test_the_quicker_driver_is_named_first_whichever_way_round_it_is():
    rows = laps_of(drivers={"zoe": 0.0, "amy": 0.40}, team="A")
    found, _ = contrast.build(rows, {})
    # Alphabetically amy comes first; the quicker driver is zoe.
    assert found[0].quicker == "zoe"
    assert found[0].gap > 0                # gaps are always positive
    assert found[0].pair == ("amy", "zoe")


def test_laps_in_traffic_are_held_out_and_counted():
    rows = laps_of(drivers={"ann": 0.0, "bob": 0.20}, team="A")
    held = {(2023, 1, "bob"): {lap: ("x",) for lap in range(1, 21)}}
    found, census = contrast.build(rows, held)
    assert census["held out: in traffic"] == 20
    assert census["kept: contrasts"] == 1
    # The gap is unchanged, because the held-out laps carried no penalty
    # in this fixture - the holding out must not itself bias the answer.
    assert found[0].gap == pytest.approx(0.20, abs=1e-6)


def test_holding_out_traffic_removes_a_penalty_that_would_be_mistaken_for_pace():
    rows = laps_of(drivers={"ann": 0.0, "bob": 0.0}, team="A")
    stuck = []
    for row in rows:
        if row.driver == "bob" and row.lap <= 20:
            row = stints.StintLap(**{**row.__dict__,
                                     "seconds": row.seconds + 0.5})
        stuck.append(row)

    loose, _ = contrast.build(stuck, {})
    assert loose[0].gap > 0.2              # the traffic looks like pace

    held = {(2023, 1, "bob"): {lap: ("x",) for lap in range(1, 21)}}
    clean, _ = contrast.build(stuck, held)
    assert clean[0].gap == pytest.approx(0.0, abs=1e-6)


def test_a_team_race_with_one_classified_driver_is_refused_not_guessed():
    rows = laps_of(drivers={"ann": 0.0}, team="A")
    found, census = contrast.build(rows, {})
    assert found == []
    assert census["refused: only one driver classified"] == 1


def test_too_few_clear_laps_is_refused():
    rows = laps_of(drivers={"ann": 0.0, "bob": 0.2}, team="A", laps=40)
    held = {(2023, 1, "bob"): {lap: ("x",) for lap in range(1, 36)}}
    found, census = contrast.build(rows, held)
    assert found == []
    assert census["refused: too few clear laps from one of them"] == 1


def test_a_car_that_never_stopped_has_age_equal_to_lap_and_is_refused():
    # With no stop, tyre age *is* the lap number, so the two columns are
    # the same vector and no fit can tell them apart. This has to be
    # refused as collinearity rather than reported as a pace difference.
    rows = laps_of(drivers={"ann": 0.0, "bob": 0.2}, team="A",
                   laps=30, stop_lap=99)
    found, census = contrast.build(rows, {})
    assert found == []
    assert census["refused: age and lap not separable"] == 1


def test_an_implausibly_large_gap_is_refused_rather_than_believed():
    rows = laps_of(drivers={"ann": 0.0, "bob": 9.0}, team="A")
    found, census = contrast.build(rows, {})
    assert found == []
    assert census["refused: a gap too large to be pace"] == 1


def test_pooling_a_pair_averages_the_races_and_counts_the_wins():
    rows = []
    for round_, offset in enumerate([0.30, 0.10, -0.20, 0.20], start=1):
        rows += laps_of(drivers={"ann": 0.0, "bob": offset}, team="A",
                        round_=round_)
    found, _ = contrast.build(rows, {})
    pooled = contrast.by_pair(found)
    assert len(pooled) == 1
    row = pooled[0]
    assert row["races"] == 4
    assert row["quicker"] == "ann"
    assert row["gap"] == pytest.approx(np.mean([0.30, 0.10, -0.20, 0.20]),
                                       abs=1e-6)
    assert row["won"] == 3                 # ann was quicker in three


def test_reliability_is_one_when_the_gap_never_changes_and_low_when_it_is_noise():
    steady, noisy = [], []
    teams = (("A", 0.4), ("B", -0.3), ("C", 0.1),
             ("D", 0.25), ("E", -0.15), ("F", 0.05))
    for round_ in range(1, 13):
        for team, gap in teams:
            steady += laps_of(drivers={f"{team}1": 0.0, f"{team}2": gap},
                              team=team, round_=round_)
    found, _ = contrast.build(steady, {})
    assert contrast.reliability(found)["reliability"] == pytest.approx(
        1.0, abs=1e-6)

    rng = np.random.default_rng(7)
    for round_ in range(1, 13):
        for team, _ in teams:
            noisy += laps_of(
                drivers={f"{team}1": 0.0, f"{team}2": rng.normal(0, 0.5)},
                team=team, round_=round_)
    found, _ = contrast.build(noisy, {})
    # Every pairing has the same expected gap of zero, so nothing here
    # persists and the reliability must not claim that anything does.
    assert contrast.reliability(found)["reliability"] < 0.5


# --------------------------------------------------------------------
# the decomposition
# --------------------------------------------------------------------

def _fitted(paces: dict[str, dict[str, float]], teams: dict[str, str]):
    return {"pace": {(2023, int(race)): values
                     for race, values in paces.items()},
            "teams": {(2023, int(race)): dict(teams) for race in paces},
            "pace error": 0.0}


def test_decompose_recovers_a_planted_between_and_within_spread():
    teams = {"a1": "A", "a2": "A", "b1": "B", "b2": "B",
             "c1": "C", "c2": "C"}
    # Team means -1, 0, +1; within each team the two cars sit +/-0.25.
    paces = {}
    for race in range(1, 9):
        paces[str(race)] = {"a1": -1.25, "a2": -0.75,
                            "b1": -0.25, "b2": 0.25,
                            "c1": 0.75, "c2": 1.25}
    split = effects.decompose(_fitted(paces, teams))
    assert split["races"] == 8
    assert split["pairs"] == 24
    assert split["team sd"] == pytest.approx(1.0, abs=1e-9)
    # The within-team term is sqrt(E[(d1-d2)^2] / 2), a second moment
    # about zero rather than a variance, because a teammate gap has no
    # reason to average zero. A fixed difference of 0.5 gives 0.5/sqrt(2).
    assert split["driver sd observed"] == pytest.approx(
        0.5 / np.sqrt(2), abs=1e-9)


def test_decompose_recovers_a_planted_driver_sd_when_the_gaps_are_random():
    rng = np.random.default_rng(11)
    teams = {"a1": "A", "a2": "A", "b1": "B", "b2": "B",
             "c1": "C", "c2": "C"}
    paces = {}
    for race in range(1, 4001):
        values = {}
        for team, middle in (("a", -1.0), ("b", 0.0), ("c", 1.0)):
            # Drawn independently. Making them mirror each other instead
            # would double the difference and the estimator would be
            # right about a quantity nobody asked for.
            values[team + "1"] = middle + rng.normal(0, 0.25)
            values[team + "2"] = middle + rng.normal(0, 0.25)
        paces[str(race)] = values
    split = effects.decompose(_fitted(paces, teams))
    # Each driver deviates from the team mean with sd 0.25, and that is
    # what the estimator is built to return.
    assert split["driver sd observed"] == pytest.approx(0.25, abs=0.01)
    assert split["team sd"] == pytest.approx(1.0, abs=0.03)


def test_persistent_halves_the_gap_signal_because_a_gap_is_a_difference():
    split = {"team variance": 0.64, "driver variance observed": 0.25,
             "team sd": 0.8, "driver sd observed": 0.5}
    shares = effects.persistent(split, {"signal sd": 0.4}, pace_error=0.0)
    # var(gap) = 2 * var(driver), so driver sd = 0.4 / sqrt(2).
    assert shares["driver sd"] == pytest.approx(0.4 / np.sqrt(2), abs=1e-9)
    # Everything left over is the day, not the driver.
    assert shares["race to race sd"] == pytest.approx(
        np.sqrt(0.25 - 0.08), abs=1e-9)
    assert sum(shares[key] for key in ("team share", "driver share",
                                       "race to race share",
                                       "measurement share")) == (
        pytest.approx(1.0))


def test_the_measurement_term_comes_out_of_the_day_not_the_driver():
    split = {"team variance": 0.64, "driver variance observed": 0.25,
             "team sd": 0.8, "driver sd observed": 0.5}
    loose = effects.persistent(split, {"signal sd": 0.4}, pace_error=0.0)
    tight = effects.persistent(split, {"signal sd": 0.4}, pace_error=0.2)
    assert tight["driver sd"] == pytest.approx(loose["driver sd"])
    assert tight["race to race sd"] < loose["race to race sd"]


def test_the_race_to_race_term_cannot_go_negative():
    split = {"team variance": 0.64, "driver variance observed": 0.01,
             "team sd": 0.8, "driver sd observed": 0.1}
    shares = effects.persistent(split, {"signal sd": 1.0}, pace_error=0.5)
    assert shares["race to race sd"] == 0.0


# --------------------------------------------------------------------
# shrinkage, the sign test and the false-discovery control
# --------------------------------------------------------------------

def test_shrinking_pulls_toward_zero_and_never_past_it():
    rows = [{"gap": 0.4}, {"gap": 0.1}, {"gap": -0.3}]
    out = effects.shrink(rows, 0.565)
    assert [row["shrunk"] for row in out] == pytest.approx(
        [0.226, 0.0565, -0.1695])
    for row in out:
        assert abs(row["shrunk"]) <= abs(row["gap"])
        assert np.sign(row["shrunk"]) == np.sign(row["gap"])


def test_a_reliability_of_one_leaves_the_gaps_alone():
    out = effects.shrink([{"gap": 0.4}], 1.0)
    assert out[0]["shrunk"] == pytest.approx(0.4)


def test_the_sign_test_matches_the_binomial_by_hand():
    out = effects.sign_test([{"won": 8, "races": 8},
                             {"won": 7, "races": 14}])
    assert out[0]["p"] == pytest.approx(0.5 ** 8)
    assert out[1]["p"] == pytest.approx(
        float(stats.binomtest(7, 14, 0.5, alternative="greater").pvalue))


def test_benjamini_hochberg_matches_the_definition_on_a_worked_case():
    # Sorted p-values against the i/m * alpha line, alpha = 0.05, m = 4:
    # thresholds 0.0125, 0.025, 0.0375, 0.05. The largest rank passing is
    # 2 (0.020 <= 0.025), so the first two reject - including 0.020 even
    # though it fails its own Bonferroni threshold.
    rows = [{"p": 0.040}, {"p": 0.004}, {"p": 0.020}, {"p": 0.700}]
    out = effects.discoveries(rows, alpha=0.05)
    assert [row["survives"] for row in out] == [False, True, True, False]


def test_benjamini_hochberg_rejects_nothing_when_nothing_is_small():
    rows = [{"p": 0.20}, {"p": 0.30}, {"p": 0.90}]
    assert not any(row["survives"]
                   for row in effects.discoveries(rows, alpha=0.05))


def test_the_correction_is_stricter_than_a_bare_count_of_p_under_five_per_cent():
    rows = [{"p": 0.004}, {"p": 0.030}, {"p": 0.045}, {"p": 0.047},
            {"p": 0.400}, {"p": 0.600}, {"p": 0.700}, {"p": 0.800},
            {"p": 0.900}, {"p": 0.950}, {"p": 0.960}, {"p": 0.990}]
    out = effects.discoveries(rows, alpha=0.05)
    assert sum(row["p"] < 0.05 for row in rows) == 4
    assert sum(row["survives"] for row in out) == 1


# --------------------------------------------------------------------
# driver effects
# --------------------------------------------------------------------

def test_fit_effects_recovers_planted_differences_within_a_component():
    # A three-driver chain: ann with bob, then ann with cal.
    results = [FakeResult(2023, 1, "ann", "A"), FakeResult(2023, 1, "bob", "A"),
               FakeResult(2023, 2, "ann", "A"), FakeResult(2023, 2, "cal", "A")]
    comparison = graph.build(results)
    rows = [contrast.Contrast(2023, 1, "c", "A", "ann", "bob", 0.30, 0.01,
                              30, 30),
            contrast.Contrast(2023, 2, "c", "A", "ann", "cal", 0.60, 0.01,
                              30, 30)]
    found = effects.fit_effects(rows, comparison)["effects"]
    # ann is 0.30 quicker than bob and 0.60 quicker than cal, so on a
    # scale centred at zero: ann +0.30, bob 0.00, cal -0.30.
    assert found["ann"] == pytest.approx(0.30, abs=1e-6)
    assert found["bob"] == pytest.approx(0.00, abs=1e-6)
    assert found["cal"] == pytest.approx(-0.30, abs=1e-6)


def test_positive_is_quicker_which_is_the_opposite_sign_to_a_lap_time():
    # This exists because the first version of fit_effects negated the
    # coefficients, so the quicker driver came out negative and the whole
    # table read backwards against its own caption.
    results = [FakeResult(2023, 1, "ann", "A"), FakeResult(2023, 1, "bob", "A")]
    comparison = graph.build(results)
    rows = [contrast.Contrast(2023, 1, "c", "A", "ann", "bob", 0.4, 0.01,
                              30, 30)]
    found = effects.fit_effects(rows, comparison)["effects"]
    assert found["ann"] > 0 > found["bob"]


def test_effects_in_different_components_are_centred_separately():
    results = results_of({"A": [("ann", "bob")], "B": [("cal", "dee")]},
                         races=1)
    comparison = graph.build(results)
    rows = [contrast.Contrast(2023, 1, "c", "A", "ann", "bob", 0.40, 0.01,
                              30, 30),
            contrast.Contrast(2023, 1, "c", "B", "cal", "dee", 0.10, 0.01,
                              30, 30)]
    found = effects.fit_effects(rows, comparison)["effects"]
    assert found["ann"] + found["bob"] == pytest.approx(0.0, abs=1e-9)
    assert found["cal"] + found["dee"] == pytest.approx(0.0, abs=1e-9)
    # Nothing ties the two blocks together, so the second block's gap
    # being smaller says nothing about cal against ann.
    assert found["ann"] == pytest.approx(0.20, abs=1e-6)
    assert found["cal"] == pytest.approx(0.05, abs=1e-6)


def test_car_pace_recovers_planted_paces_and_refuses_a_thin_race():
    rows = []
    planted = {"a1": -0.6, "a2": -0.4, "b1": -0.1, "b2": 0.1,
               "c1": 0.4, "c2": 0.6}
    for team in ("A", "B", "C"):
        rows += laps_of(
            drivers={name: value for name, value in planted.items()
                     if name.startswith(team.lower())}, team=team)
    fitted, census = effects.car_pace(rows, {})
    assert census["fitted races"] == 1
    found = fitted["pace"][(2023, 1)]
    for name, value in planted.items():
        assert found[name] == pytest.approx(value, abs=1e-6)

    thin = laps_of(drivers={"a1": 0.0, "a2": 0.2}, team="A")
    _, census = effects.car_pace(thin, {})
    assert census["fitted races"] == 0
    assert census["refused: too few cars"] == 1


# --------------------------------------------------------------------
# the regression underneath
# --------------------------------------------------------------------

def test_ols_recovers_a_planted_line():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = 2.0 + 0.5 * x + rng.normal(0, 0.05, size=400)
    fit = linear.ols(y, np.column_stack([np.ones(400), x]),
                     ("base", "slope"), np.arange(400))
    assert fit["base"] == pytest.approx(2.0, abs=0.02)
    assert fit["slope"] == pytest.approx(0.5, abs=0.02)


def test_clustering_widens_the_error_when_the_rows_repeat_themselves():
    rng = np.random.default_rng(5)
    shocks = rng.normal(0, 1.0, size=20)
    x, y, group = [], [], []
    for cluster, shock in enumerate(shocks):
        for _ in range(20):
            value = rng.normal()
            x.append(value)
            y.append(0.3 * value + shock + rng.normal(0, 0.1))
            group.append(cluster)
    design = np.column_stack([np.ones(len(x)), x])
    plain = linear.ols(np.array(y), design, ("base", "slope"),
                       np.arange(len(y)))
    clustered = linear.ols(np.array(y), design, ("base", "slope"),
                           np.array(group))
    assert clustered.error("base") > plain.error("base")
