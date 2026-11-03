"""The report, as a printable PDF built out of matplotlib pages.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper

PIECES = (("the car the team built", "team"),
          ("what happened on the day", "race to race"),
          ("the driver", "driver"),
          ("error in measuring it", "measurement"))


def _counts(census: dict) -> list[list[str]]:
    return [[key, f"{value:,}"] for key, value
            in sorted(census.items(), key=lambda item: -item[1])]


def _pairs(pairs) -> tuple[list[str], list[list[str]]]:
    rows = [[row["quicker"], row["slower"], f"{row['gap']:+.3f}",
             f"{row['shrunk']:+.3f}", f"{row['won']}/{row['races']}",
             f"{row['p']:.4f}", "yes" if row["survives"] else "no"]
            for row in pairs]
    return (["quicker", "slower", "raw", "shrunk", "ahead", "p",
             "survives"], rows)


def _effects(comparison, effects: dict) -> tuple[list[str], list[list[str]]]:
    rows = []
    for group in sorted(comparison.components(), key=len, reverse=True):
        if len(group) < 3:
            continue
        for name in sorted(group, key=lambda one: -effects[one]):
            rows.append([name, f"{effects[name]:+.3f}"])
    return ["driver", "s a lap"], rows


def build(*, comparison, shape, census, pace_census, contrasts, pairs, trust,
          shares, split, effects, pace_error, seasons, figures,
          out_path: Path) -> Path:
    years = ", ".join(str(one) for one in seasons)
    survivors = [row for row in pairs if row["survives"]]
    survivor_text = "; ".join(
        f"{row['quicker']} over {row['slower']} (p {row['p']:.4f})"
        for row in survivors)
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "How much of a lap time is the car",
               f"Formula 1, {years}. {len(contrasts)} teammate "
               f"comparisons from {pace_census['fitted races']} races, "
               f"with every lap spent stuck behind another car held "
               f"out.") as paper:

        paper.cover(
            headline=f"The car is {shares['ratio']:.1f} times the driver",
            small=f"{shares['team sd']:.3f} s a lap between teams against "
                  f"{shares['driver sd']:.3f} s a lap between teammates "
                  f"— {shares['ratio'] ** 2:.0f} times in variance. "
                  f"And what happens to a car on a given afternoon is "
                  f"{shares['race against driver']:.1f} times the driver: "
                  f"the term nobody ranks.",
            tiles=[("the car, in seconds a lap",
                    f"{shares['team sd']:.3f}"),
                   ("the driver", f"{shares['driver sd']:.3f}"),
                   ("car over driver", f"{shares['ratio']:.1f}×"),
                   ("pairings that survive",
                    f"{len(survivors)} of {len(pairs)}")],
            blurb="A driver is only measurable against someone who had "
                  "the same car. Each driver's pace is fitted on their "
                  "own clear-air laps with tyre age and fuel load taken "
                  "out, and the difference between two teammates on one "
                  "afternoon is one comparison.",
            figure_path=figures["variance"])

        paper.section(
            "First: what can be compared with what",
            note="Drivers are nodes, a shared team-race is an edge, and "
                 "the shape of that graph decides which questions have "
                 "answers before any statistic is computed.",
            warn=f"The graph is a forest. A forest on {shape['drivers']} "
                 f"nodes with {shape['components']} components has "
                 f"exactly {shape['drivers'] - shape['components']} edges, "
                 f"and there are {shape['pairings']} — so "
                 f"{shape['cycles']} cycles. Every comparison rests on a "
                 f"single pairing with no second route to check it "
                 f"against, and {shape['free']} of the "
                 f"{shape['drivers']} driver effects are free parameters "
                 f"this season cannot pin down at all. There is no chain "
                 f"of shared cars from Verstappen to Hamilton, and no "
                 f"statistic can manufacture one.",
            figure_path=figures["graph"],
            table=(["property", "count"],
                   [[key, f"{value:,}"] for key, value in shape.items()]))

        paper.section(
            "Who can be compared with whom",
            table=(["size", "drivers who can be compared"],
                   [[str(len(group)), ", ".join(group)]
                    for group in sorted(comparison.components(), key=len,
                                        reverse=True)]))

        paper.section(
            "The comparisons themselves",
            note="One car, two drivers, one afternoon.",
            body="Laps within two seconds of the car ahead are held out, "
                 "because being held up costs about a quarter of a second "
                 "a lap and is a property of the traffic rather than the "
                 "driver.",
            table=(["step", "laps or races"], _counts(census)))

        paper.section(
            "Splitting the spread",
            note=f"Within a race, the spread of car paces splits into a "
                 f"part between teams ({split['team sd']:.3f} s a lap) "
                 f"and a part within a team "
                 f"({split['driver sd observed']:.3f} s a lap, over "
                 f"{split['pairs']} team-races).",
            body=f"The within-team part is not the driver. It is the "
                 f"driver plus damage, a strategy call, a lap of traffic "
                 f"the detector missed, and the ordinary error in fitting "
                 f"a pace to forty laps.\n\n"
                 f"The same teammate gap measured on odd rounds against "
                 f"even rounds correlates {trust['correlation']:.3f} over "
                 f"{trust['pairs']} pairings, a reliability of "
                 f"{trust['reliability']:.3f}. Barely more than half of "
                 f"what separates one pairing from another comes back in "
                 f"the other half of the season. That fraction is what "
                 f"tells the persistent part from the day, and a car's "
                 f"pace in one race is itself only known to "
                 f"±{pace_error:.3f} s a lap, which is taken out "
                 f"separately.",
            figure_path=figures["reliability"],
            table=(["what it is", "s a lap", "share of variance"],
                   [[label, f"{shares[key + ' sd']:.3f}",
                     f"{shares[key + ' share']:.1%}"]
                    for label, key in PIECES]))

        paper.section(
            "Every pairing the season supports",
            note="The sign test asks only how often the quicker driver "
                 "was quicker, which needs no model to believe. Shrinking "
                 "by the reliability pulls every gap toward zero, because "
                 "a gap over a dozen races is partly those dozen races.",
            warn=f"{sum(1 for row in pairs if row['p'] < 0.05)} pairings "
                 f"reach p<0.05, but {len(pairs)} tests expect some of "
                 f"that. Controlling the false discovery rate leaves "
                 f"{len(survivors)}: {survivor_text}. The rest of the "
                 f"table is a table, not a set of findings.",
            figure_path=figures["pairs"], table=_pairs(pairs))

        paper.section(
            "Driver effects, and why they do not form one list",
            note="Seconds a lap quicker than the driver's own component "
                 "average. Positive is quicker, which is the opposite "
                 "sign to a lap time.",
            warn=f"Comparing a number here with one from another block is "
                 f"meaningless. Each component is centred on its own "
                 f"mean, and nothing in the season ties one block's zero "
                 f"to another's. A single table ordering all "
                 f"{shape['drivers']} drivers would be inventing "
                 f"{shape['free']} numbers the data does not contain.",
            figure_path=figures["effects"],
            table=_effects(comparison, effects),
            after="Only the components with three or more drivers are "
                  "shown, because in a pair the two numbers are the same "
                  "comparison written twice.")

        paper.section(
            "What this does not do",
            body="It does not rank drivers across teams, measure anything "
                 "in qualifying, or separate a driver from the car's "
                 "suitability to their style — a car built around "
                 "one driver flatters that driver, and that appears here "
                 "as driver skill. It uses one season, which is why the "
                 "race-to-race term is as large as it is, and it has no "
                 "tyre compound, because the source does not record one.",
            figure_path=figures["spread"],
            after="Lap times, pit stops and results from the Jolpica "
                  "mirror of the Ergast schema.")

    return out
