"""How much of a lap time is the car and how much is the driver.

Run with no arguments; the season is cached, so a second run costs nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts                                              # noqa: E402
import contrast                                            # noqa: E402
import data                                                # noqa: E402
import effects                                             # noqa: E402
import follow                                              # noqa: E402
import gaps                                                # noqa: E402
import graph                                               # noqa: E402
import report                                              # noqa: E402
import stints                                              # noqa: E402

SEASONS = (2023,)
OUT = ROOT / "out"

_log: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _log.append(line)


def main() -> None:
    laps, stops, results = data.load(SEASONS)
    say(f"{len(laps):,} lap times, {len(results):,} race entries")

    # ---- what can be compared with what, before any measuring ----------
    comparison = graph.build(results)
    shape = comparison.summary()
    say()
    say("who shared a car with whom, which decides what is answerable")
    for key, value in shape.items():
        say(f"  {key:46s} {value:6d}")
    say(f"  {shape['drivers']} drivers on a design of rank "
        f"{shape['identified']}, so {shape['free']} driver effects are free "
        f"parameters that this season cannot pin down at all")
    say(f"  nodes minus components is "
        f"{shape['drivers'] - shape['components']}, and there are "
        f"{shape['pairings']} pairings, so the graph is a forest: "
        f"{shape['cycles']} cycles, and therefore no pair of drivers whose "
        f"comparison can be checked by a second route")
    for group in comparison.components():
        say(f"    {len(group)}: {', '.join(group)}")

    far = comparison.path("max_verstappen", "hamilton")
    say(f"  a path from Verstappen to Hamilton: {far}")
    say("  there is none, and no statistic can manufacture one")

    # ---- the teammate contrasts -----------------------------------------
    times = gaps.elapsed(laps)
    traffic = follow.in_traffic(laps, times)
    clean, _ = stints.build(laps, stops, results)
    contrasts, census = contrast.build(clean, traffic)
    say()
    say("teammate contrasts: one car, two drivers, one race")
    for key, value in sorted(census.items()):
        say(f"  {key:46s} {value:6d}")
    say(f"  laps spent within {follow.REACH:.0f} s of the car ahead are held "
        f"out, because being stuck behind someone costs about a quarter of "
        f"a second a lap and is not a property of the driver")

    # ---- how much is the car --------------------------------------------
    fitted, pace_census = effects.car_pace(clean, traffic)
    split = effects.decompose(fitted)
    trust = contrast.reliability(contrasts)
    say()
    say(f"each car's pace is known to +/- {fitted['pace error']:.3f} s a lap "
        f"in a given race, over {pace_census['fitted races']} races")
    say(f"the spread of car paces within a race splits into")
    say(f"  between teams            {split['team sd']:.3f} s a lap")
    say(f"  within a team            {split['driver sd observed']:.3f} s a "
        f"lap, over {split['pairs']} team-races")
    say()
    say("but the within-team number is not the driver. It is the driver "
        "plus whatever befell one car and not the other that afternoon")
    say(f"  the same teammate gap measured on odd rounds against even "
        f"rounds correlates {trust['correlation']:.3f} over "
        f"{trust['pairs']} pairs")
    say(f"  reliability {trust['reliability']:.3f}: barely more than half "
        f"of what separates one pairing from another comes back in the "
        f"other half of the season")

    shares = effects.persistent(split, trust, fitted["pace error"])
    say()
    say("so the spread of car paces within a race decomposes as")
    for key in ("team", "race to race", "driver", "measurement"):
        name = f"{key} sd" if key != "driver" else "driver sd"
        say(f"  {key:16s} {shares[name]:.3f} s a lap   "
            f"{shares[key + ' share']:6.1%}")
    say(f"  the car is {shares['ratio']:.1f} times the driver in seconds a "
        f"lap, and {shares['ratio'] ** 2:.0f} times in variance")
    say(f"  and what happens to a car on the day is "
        f"{shares['race against driver']:.1f} times the driver, which is "
        f"the part nobody ranks")

    # ---- the pairs themselves --------------------------------------------
    pairs = effects.discoveries(effects.sign_test(
        effects.shrink(contrast.by_pair(contrasts), trust["reliability"])))
    say()
    say("every teammate pairing the season supports")
    say(f"  {'quicker':16s} {'slower':17s} {'raw':>7s} {'shrunk':>8s} "
        f"{'ahead':>9s} {'p':>8s}")
    for row in pairs:
        mark = " *" if row["survives"] else ""
        say(f"  {row['quicker']:16s} {row['slower']:17s} "
            f"{row['gap']:+7.3f} {row['shrunk']:+8.3f} "
            f"{row['won']:5d}/{row['races']:<3d} {row['p']:8.4f}{mark}")
    survivors = [row for row in pairs if row["survives"]]
    say(f"  the sign test asks only how often the quicker driver was "
        f"quicker, which needs no model to believe")
    say(f"  {sum(1 for row in pairs if row['p'] < 0.05)} pairings reach "
        f"p<0.05, but twelve tests expect some of that. Controlling the "
        f"false discovery rate leaves {len(survivors)}: "
        f"{', '.join(row['quicker'] + ' over ' + row['slower'] for row in survivors)}")
    say(f"  shrinking by the reliability pulls every gap toward zero, "
        f"because a gap over a dozen races is partly those dozen races")

    # ---- driver effects, within components only --------------------------
    found = effects.fit_effects(contrasts, comparison)
    say()
    say("driver effects, in seconds a lap quicker than their own "
        "component's average. Positive is quicker. Comparing a number "
        "here with one from another block is meaningless")
    for group in comparison.components():
        say(f"  {'-' * 44}")
        for name in sorted(group, key=lambda one: -found["effects"][one]):
            say(f"  {name:22s} {found['effects'][name]:+.3f}")

    # ---- output ------------------------------------------------------------
    figures = charts.draw(
        OUT, comparison=comparison, shape=shape, contrasts=contrasts,
        pairs=pairs, trust=trust, shares=shares, split=split,
        effects=found["effects"])
    path = report.build(
        comparison=comparison, shape=shape, census=census,
        pace_census=pace_census, contrasts=contrasts, pairs=pairs,
        trust=trust, shares=shares, split=split, effects=found["effects"],
        pace_error=fitted["pace error"], seasons=SEASONS, figures=figures,
        out_path=OUT / "report.html")

    say()
    say(f"report  {path}")
    for name, figure in figures.items():
        say(f"figure  {figure}")
    (OUT / "_run.txt").write_text("\n".join(_log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
