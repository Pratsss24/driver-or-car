"""How much is the car, how much is the driver, and how much is neither.

Three things this does, in increasing order of how much they are worth.

**The decomposition.** Within each race, every car gets a pace fitted on
its own clear-air laps with tyre age and fuel load taken out. Those paces
split into a team level and a driver deviation from it, and the two spreads
answer the question in the title directly. This is a nested analysis of
variance and nothing cleverer; the care is all in what goes into it.

**The correction.** The within-team spread is measured with error, and a
measured spread is always too wide: noise adds variance, it never removes
any. The split-half reliability of the teammate gaps says how much to take
back out, and the corrected number is the one to quote. Without this step
the driver's share is overstated, in a way that no amount of extra data
fixes because the bias does not shrink with the sample.

**The shrinkage.** The same correction applied per pair. A gap measured
over eight races is partly real and partly the luck of those eight races,
so the honest estimate pulls it toward zero by the reliability. James and
Stein proved this beats the raw estimates in total squared error, and here
it does something more useful than that: it stops a table of twelve
numbers from implying twelve findings when the data supports about half
of one.

The sign test sits alongside, because it assumes nothing at all. If a
driver is quicker than their teammate, they should be quicker in more than
half the races, and a count of races needs no model to be believed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats

import graph
import linear

MIN_CARS = 6          # cars a race needs before its spreads mean anything
COLUMNS = 3           # base, wear, burn


def car_pace(clean, traffic) -> tuple[dict, dict]:
    """Each car's pace in a race, with tyre age and fuel load taken out.

    One joint fit per race, with a dummy per car, so every car in the
    race is measured against the same wear and burn curve rather than
    against its own. Within a race that is the right choice: the tyres
    and the fuel effect are properties of the afternoon, and letting each
    car have its own would absorb into the curve exactly the pace
    difference being looked for.
    """
    races: dict[tuple[int, int], list] = defaultdict(list)
    for row in clean:
        if row.lap in traffic.get(row.key, ()):
            continue
        races[(row.season, row.round)].append(row)

    pace: dict[tuple[int, int], dict[str, float]] = {}
    teams: dict[tuple[int, int], dict[str, str]] = {}
    census = {"races seen": len(races), "refused: too few cars": 0,
              "fitted races": 0}
    precision: list[float] = []
    for race, rows in sorted(races.items()):
        drivers = sorted({row.driver for row in rows})
        if len(drivers) < MIN_CARS:
            census["refused: too few cars"] += 1
            continue
        index = {name: number for number, name in enumerate(drivers)}
        design = np.zeros((len(rows), COLUMNS + len(drivers) - 1))
        design[:, 0] = 1.0
        design[:, 1] = [row.age for row in rows]
        design[:, 2] = [row.lap for row in rows]
        for position, row in enumerate(rows):
            slot = index[row.driver]
            if slot:                       # the first driver is the baseline
                design[position, COLUMNS + slot - 1] = 1.0
        if np.linalg.matrix_rank(design) < design.shape[1]:
            census["refused: too few cars"] += 1
            continue

        target = np.array([row.seconds for row in rows])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ beta
        spread = float(residual @ residual) / max(
            len(rows) - design.shape[1], 1)
        precision.append(float(np.sqrt(
            spread * len(drivers) / max(len(rows), 1))))
        values = {drivers[0]: 0.0}
        for name, slot in index.items():
            if slot:
                values[name] = float(beta[COLUMNS + slot - 1])
        middle = float(np.mean(list(values.values())))
        pace[race] = {name: value - middle for name, value in values.items()}
        teams[race] = {row.driver: row.team for row in rows}
        census["fitted races"] += 1
    # How precisely one car's pace in one race is known. It matters
    # because the race-to-race term below is only interesting if it is
    # bigger than the error in measuring it, and it is.
    return {"pace": pace, "teams": teams,
            "pace error": float(np.median(precision)) if precision
            else float("nan")}, census


def decompose(fitted: dict) -> dict:
    """Split the spread of car paces into a team part and a driver part.

    Per race, so that a season's worth of development does not leak into
    the team term, then pooled. The driver part uses the difference
    between two cars of the same team, whose variance is twice a single
    driver's, hence the halving.
    """
    between: list[float] = []
    within: list[float] = []
    races = 0
    for race, values in fitted["pace"].items():
        teams = fitted["teams"][race]
        grouped: dict[str, list[float]] = defaultdict(list)
        for name, value in values.items():
            grouped[teams[name]].append(value)
        means = [float(np.mean(one)) for one in grouped.values()]
        if len(means) < 2:
            continue
        races += 1
        between.append(float(np.var(means, ddof=1)))
        for one in grouped.values():
            if len(one) == 2:
                within.append((one[0] - one[1]) ** 2 / 2.0)

    team_variance = float(np.mean(between)) if between else float("nan")
    driver_variance = float(np.mean(within)) if within else float("nan")
    return {"races": races, "pairs": len(within),
            "team sd": float(np.sqrt(team_variance)),
            "driver sd observed": float(np.sqrt(driver_variance)),
            "team variance": team_variance,
            "driver variance observed": driver_variance}


def persistent(split: dict, trust: dict,
               pace_error: float = 0.0) -> dict:
    """Separate what persists about a driver from what happened that day.

    The within-team spread measured race by race is not driver skill. It
    is skill *plus* everything that befalls one car and not the other -
    damage, a strategy call, a lap of traffic the detector missed, and
    the ordinary error in fitting a pace to forty laps. Those do not
    persist, and the split-half of the teammate gaps is what separates
    them: whatever reappears in the other half of the season is the
    driver, and whatever does not is the day.

    A teammate gap is a difference of two driver effects, so its signal
    variance is twice a single driver's. Hence the halving, which is the
    only step here that is arithmetic rather than judgement.

    `pace_error` is how precisely one car's pace in one race is known,
    and it is taken out of the race-to-race term rather than left in it.
    A term that turned out to be mostly the error in measuring it would
    be worth nothing, so it is worth checking that this one is not.
    """
    driver_variance = (trust["signal sd"] ** 2) / 2.0
    measurement = pace_error ** 2
    race_variance = max(split["driver variance observed"] - driver_variance
                        - measurement, 0.0)
    team_variance = split["team variance"]
    total = team_variance + driver_variance + race_variance + measurement
    return {"team sd": float(np.sqrt(team_variance)),
            "driver sd": float(np.sqrt(driver_variance)),
            "race to race sd": float(np.sqrt(race_variance)),
            "measurement sd": float(pace_error),
            "ratio": float(np.sqrt(team_variance / driver_variance)),
            "race against driver": float(np.sqrt(race_variance
                                                 / driver_variance)),
            "team share": team_variance / total,
            "driver share": driver_variance / total,
            "race to race share": race_variance / total,
            "measurement share": measurement / total,
            "within team observed sd": split["driver sd observed"]}


def shrink(pairs: list[dict], reliability: float) -> list[dict]:
    """Pull every teammate gap toward zero by the reliability.

    The same factor for all of them, because the reliability was measured
    across pairs rather than within one. A per-pair factor would need each
    pair's own split-half, and with a dozen races each those are noisier
    than the thing they would be correcting.
    """
    out = []
    for row in pairs:
        out.append(row | {"shrunk": row["gap"] * reliability})
    return out


def discoveries(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    """Benjamini-Hochberg over the twelve sign tests.

    Twelve tests at five per cent expect about half a false positive
    between them, so a bare count of significant pairs overstates what
    was found. This controls the share of the rejections that are false
    rather than the chance of any, which is the right trade when the
    output is a table somebody will read top to bottom.
    """
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    total = len(rows)
    passed: set[int] = set()
    for rank, position in enumerate(order, start=1):
        if rows[position]["p"] <= alpha * rank / total:
            passed = set(order[:rank])
    return [row | {"survives": index in passed}
            for index, row in enumerate(rows)]


def sign_test(pairs: list[dict]) -> list[dict]:
    """How often the quicker driver was quicker, against a coin.

    Assumes nothing about how lap times are distributed, which matters
    when a handful of races carry damage or a strategy call that no
    covariate in the fit knows about.
    """
    out = []
    for row in pairs:
        wins, races = row["won"], row["races"]
        out.append(row | {
            "p": float(stats.binomtest(wins, races, 0.5,
                                       alternative="greater").pvalue)})
    return out


def fit_effects(contrasts, comparison: graph.Comparison) -> dict:
    """Driver effects from the contrasts, one component at a time.

    Least squares on the contrast design, with errors clustered on the
    pair - the races of one pairing are the same two drivers seen again,
    not fresh evidence about them. The result is centred within each
    component by `graph.centre`, and comparing across components is not
    a thing this returns, because it is not a thing the data contains.
    """
    matrix, names = graph.design(contrasts, comparison)
    if not names:
        return {"names": (), "effects": {}}
    signed = np.array([one.gap for one in contrasts])
    pairs = sorted({one.pair for one in contrasts})
    index = {pair: number for number, pair in enumerate(pairs)}
    group = np.array([index[one.pair] for one in contrasts])

    fit = linear.ols(signed, matrix, names, group,
                     label="driver effects from teammate contrasts")
    # The design carries +1 for the quicker slot, so a larger coefficient
    # is a quicker driver. Kept that way round: every number below reads
    # as seconds a lap *quicker* than the component average, which is the
    # opposite sign to a lap time and so is worth saying twice.
    values = {name: fit[name] for name in names}
    for group_names in comparison.components():
        for name in group_names:
            values.setdefault(name, 0.0)
    return {"names": names, "fit": fit,
            "effects": graph.centre(values, comparison)}
