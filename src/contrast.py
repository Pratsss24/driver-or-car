"""The teammate gap: two drivers, one car, one race.

For each team in each race, both drivers' usable laps are fitted together:

    seconds  =  base  +  wear * age  +  burn * lap  +  gap * is_second_driver

`gap` is the teammate contrast, in seconds a lap, for that one race. Tyre
age and race lap are in there because the two cars do not run the same
strategy: a driver who stopped later carries more fuel and older tyres for
part of the race, and reading a raw average of lap times would charge that
to the driver.

**Traffic is the confounder that matters most, and it is removable.** A
driver stuck behind another car runs slower for reasons that have nothing
to do with how quick they are - the pace-to-pass project measured that at
0.25 seconds a lap when the car behind had the pace to go faster, which is
larger than most teammate gaps in this season. Laps spent within reach of
the car directly ahead are held out here, using the same detector. It is
the single most important thing this module does, and the one most likely
to be left out of a naive version.

**What a contrast still cannot see.** Tyre compound is not in the feed, so
a driver who ran the softer tyre looks quicker and nothing here can tell.
Damage, a floor change, a fuel saving instruction, a driver nursing a
problem home: all of them land in `gap` as if they were pace. These are
race-level accidents that average out over a season of contrasts if they
are symmetric, and bias the result if they are not, which is why the
per-race contrasts are kept and reported rather than only their mean.

**Refusals.** A contrast needs enough clear laps from *both* drivers, and
it needs tyre age and race lap to be separable - inside a single stint they
advance together and the fit cannot tell wear from fuel burn. Both refusals
are counted rather than worked around.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

import stints

MIN_LAPS_EACH = 12    # clear laps needed from each driver
MAX_ENTANGLED = 0.95  # correlation between age and lap the fit will accept
MAX_GAP = 3.0         # seconds a lap: beyond this is an incident, not pace
COLUMNS = 4           # base, wear, burn, driver


@dataclass(frozen=True)
class Contrast:
    """One race's pace difference between two drivers in the same car."""

    season: int
    round: int
    circuit: str
    team: str
    quicker: str
    slower: str
    gap: float            # seconds a lap, always positive by construction
    error: float
    laps_quicker: int
    laps_slower: int

    @property
    def race(self) -> tuple[int, int]:
        return (self.season, self.round)

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.quicker, self.slower)))


def _fit(first: list, second: list) -> tuple[float, float] | None:
    """Least squares of lap time on age, lap and which driver it was."""
    rows = first + second
    design = np.column_stack([
        np.ones(len(rows)),
        np.array([row.age for row in rows], dtype=float),
        np.array([row.lap for row in rows], dtype=float),
        np.array([0.0] * len(first) + [1.0] * len(second))])
    if abs(np.corrcoef(design[:, 1], design[:, 2])[0, 1]) > MAX_ENTANGLED:
        return None
    if np.linalg.matrix_rank(design) < COLUMNS:
        return None

    target = np.array([row.seconds for row in rows])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    spread = float(residual @ residual) / max(len(rows) - COLUMNS, 1)
    covariance = spread * np.linalg.inv(design.T @ design)
    return float(beta[3]), float(np.sqrt(covariance[3, 3]))


def build(clean: list[stints.StintLap], traffic: dict,
          *, minimum: int = MIN_LAPS_EACH, cap: float = MAX_GAP
          ) -> tuple[list[Contrast], dict[str, int]]:
    """Every teammate contrast the season supports."""
    held: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    in_traffic = 0
    for row in clean:
        if row.lap in traffic.get(row.key, ()):
            in_traffic += 1
            continue
        held[(row.team, row.season, row.round)][row.driver].append(row)

    census = {"clean laps": len(clean), "held out: in traffic": in_traffic,
              "team-races seen": len(held),
              "refused: only one driver classified": 0,
              "refused: too few clear laps from one of them": 0,
              "refused: age and lap not separable": 0,
              "refused: a gap too large to be pace": 0,
              "kept: contrasts": 0}

    out: list[Contrast] = []
    for (team, season, round_), drivers in sorted(held.items()):
        if len(drivers) != 2:
            census["refused: only one driver classified"] += 1
            continue
        one, two = sorted(drivers)
        if (len(drivers[one]) < minimum or len(drivers[two]) < minimum):
            census["refused: too few clear laps from one of them"] += 1
            continue

        fitted = _fit(drivers[one], drivers[two])
        if fitted is None:
            census["refused: age and lap not separable"] += 1
            continue
        gap, error = fitted
        if abs(gap) > cap:
            census["refused: a gap too large to be pace"] += 1
            continue

        # `gap` is how much slower `two` was. Name the faster driver first
        # so every contrast in the table reads the same way round.
        quicker, slower = (one, two) if gap > 0 else (two, one)
        out.append(Contrast(
            season=season, round=round_,
            circuit=drivers[one][0].circuit, team=team,
            quicker=quicker, slower=slower, gap=abs(gap), error=error,
            laps_quicker=len(drivers[quicker]),
            laps_slower=len(drivers[slower])))
        census["kept: contrasts"] += 1
    return out, census


def by_pair(contrasts: list[Contrast]) -> list[dict]:
    """Pool each pair of teammates across the races they shared.

    The mean of the per-race contrasts, signed consistently, with the
    standard error of that mean. Weighting by each race's own precision
    was tried and makes almost no difference, because the races differ
    far more in what happened in them than in how well they were measured.
    """
    rows: dict[tuple[str, str], list[Contrast]] = defaultdict(list)
    for contrast in contrasts:
        rows[contrast.pair].append(contrast)

    out = []
    for pair, group in rows.items():
        first, second = pair
        signed = np.array([one.gap if one.quicker == first else -one.gap
                           for one in group])
        mean = float(signed.mean())
        error = (float(signed.std(ddof=1) / np.sqrt(signed.size))
                 if signed.size > 1 else float("nan"))
        quicker, slower = (first, second) if mean > 0 else (second, first)
        out.append({"quicker": quicker, "slower": slower,
                    "gap": abs(mean), "error": error, "races": signed.size,
                    "team": group[0].team,
                    "won": int(np.sum(signed > 0) if mean > 0
                               else np.sum(signed < 0)),
                    "signed": signed})
    out.sort(key=lambda row: -row["gap"])
    return out


def reliability(contrasts: list[Contrast]) -> dict:
    """Does a teammate gap measured on half a season predict the other half?

    The split is odd against even rounds, which keeps both halves spread
    across the calendar rather than splitting the season in two. A pair
    needs races on both sides to contribute.
    """
    rows: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"odd": [], "even": []})
    for one in contrasts:
        first = sorted(one.pair)[0]
        signed = one.gap if one.quicker == first else -one.gap
        rows[one.pair]["odd" if one.round % 2 else "even"].append(signed)

    pairs = [(float(np.mean(halves["odd"])), float(np.mean(halves["even"])))
             for halves in rows.values()
             if halves["odd"] and halves["even"]]
    if len(pairs) < 4:
        return {"pairs": len(pairs)}

    odd = np.array([one for one, _ in pairs])
    even = np.array([two for _, two in pairs])
    noise = float(np.var(odd - even, ddof=1) / 2)
    measured = float(np.var(np.concatenate([odd, even]), ddof=1))
    signal = max(measured - noise, 0.0)
    return {"pairs": len(pairs),
            "correlation": float(np.corrcoef(odd, even)[0, 1]),
            "noise sd": float(np.sqrt(noise)),
            "signal sd": float(np.sqrt(signal)),
            "reliability": signal / measured if measured else float("nan"),
            "odd": odd, "even": even}
