"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


def _count_rows(census: dict) -> str:
    return "".join(f"<tr><td>{key}</td><td>{value:,}</td></tr>"
                   for key, value in sorted(census.items(),
                                            key=lambda item: -item[1]))


def _pair_row(row: dict) -> str:
    mark = ' class="good"' if row["survives"] else ""
    return (f"<tr><td>{row['quicker']}</td><td>{row['slower']}</td>"
            f"<td>{row['gap']:+.3f}</td><td>{row['shrunk']:+.3f}</td>"
            f"<td>{row['won']}/{row['races']}</td>"
            f"<td{mark}>{row['p']:.4f}</td>"
            f"<td>{'yes' if row['survives'] else 'no'}</td></tr>")


def build(*, comparison, shape, census, pace_census, contrasts, pairs, trust,
          shares, split, effects, pace_error, seasons, figures,
          out_path: Path) -> Path:
    tiles = [
        ("the car, in seconds a lap", f"{shares['team sd']:.3f}"),
        ("the driver", f"{shares['driver sd']:.3f}"),
        ("car over driver", f"{shares['ratio']:.1f}&times;"),
        ("pairings that survive", f"{sum(1 for r in pairs if r['survives'])}"
                                  f" of {len(pairs)}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    shape_html = "".join(f"<tr><td>{key}</td><td>{value:,}</td></tr>"
                         for key, value in shape.items())
    components_html = "".join(
        f"<tr><td>{len(group)}</td><td>{', '.join(group)}</td></tr>"
        for group in sorted(comparison.components(), key=len, reverse=True))

    pieces = [("the car the team built", "team"),
              ("what happened on the day", "race to race"),
              ("the driver", "driver"),
              ("error in measuring it", "measurement")]
    split_html = "".join(
        f"<tr><td>{label}</td>"
        f"<td>{shares[key + ' sd']:.3f}</td>"
        f"<td>{shares[key + ' share']:.1%}</td></tr>"
        for label, key in pieces)

    effects_html = ""
    for group in sorted(comparison.components(), key=len, reverse=True):
        if len(group) < 3:
            continue
        for name in sorted(group, key=lambda one: -effects[one]):
            effects_html += (f"<tr><td>{name}</td>"
                             f"<td>{effects[name]:+.3f}</td></tr>")

    years = ", ".join(str(one) for one in seasons)
    survivors = [row for row in pairs if row["survives"]]
    survivor_text = "; ".join(
        f"{row['quicker']} over {row['slower']} (p {row['p']:.4f})"
        for row in survivors)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>How much of a lap time is the car</title><style>{CSS}</style></head>
<body>
<h1>How much of a lap time is the car, and how much is the driver</h1>
<p class="sub">Formula 1, {years}. {len(contrasts)} teammate comparisons
from {pace_census['fitted races']} races, with every lap spent stuck
behind another car held out.</p>
{tiles_html}

<h2>First: what can be compared with what</h2>
<p>A driver is only measurable against someone who had the same car.
Drivers are nodes, a shared team-race is an edge, and the shape of that
graph decides which questions have answers before any statistic is
computed.</p>
{_img(figures['graph'])}
<table><tr><th>property</th><th>count</th></tr>{shape_html}</table>
<div class="note"><b>The graph is a forest.</b> A forest on
{shape['drivers']} nodes with {shape['components']} components has
exactly {shape['drivers'] - shape['components']} edges, and there are
{shape['pairings']} &mdash; so {shape['cycles']} cycles. Every
comparison rests on a single pairing with no second route to check it
against, and {shape['free']} of the {shape['drivers']} driver effects
are free parameters this season cannot pin down at all. There is no
chain of shared cars from Verstappen to Hamilton, and no statistic can
manufacture one.</div>
<table><tr><th>size</th><th>drivers who can be compared</th></tr>
{components_html}</table>

<h2>The comparisons themselves</h2>
<p>One car, two drivers, one afternoon. Each driver's pace is fitted on
their own clear-air laps with tyre age and fuel load taken out, and the
difference between the two is one comparison.</p>
<table><tr><th>step</th><th>laps or races</th></tr>
{_count_rows(census)}</table>
<p>Laps within two seconds of the car ahead are held out, because being
held up costs about a quarter of a second a lap and is a property of
the traffic rather than the driver.</p>

<h2>Splitting the spread</h2>
{_img(figures['variance'])}
<p>Within a race, the spread of car paces splits into a part between
teams ({split['team sd']:.3f} s a lap) and a part within a team
({split['driver sd observed']:.3f} s a lap, over {split['pairs']}
team-races). The within-team part is <em>not</em> the driver. It is the
driver plus damage, a strategy call, a lap of traffic the detector
missed, and the ordinary error in fitting a pace to forty laps.</p>
{_img(figures['reliability'])}
<p>The same teammate gap measured on odd rounds against even rounds
correlates {trust['correlation']:.3f} over {trust['pairs']} pairings, a
reliability of {trust['reliability']:.3f}. Barely more than half of what
separates one pairing from another comes back in the other half of the
season. That fraction is what tells the persistent part from the day,
and a car's pace in one race is itself only known to
&plusmn;{pace_error:.3f} s a lap, which is taken out separately.</p>
<table><tr><th>what it is</th><th>s a lap</th><th>share of variance</th>
</tr>{split_html}</table>
<div class="note"><b>The car is {shares['ratio']:.1f} times the driver
in seconds a lap, and {shares['ratio'] ** 2:.0f} times in variance.</b>
And what happens to a car on a given afternoon is
{shares['race against driver']:.1f} times the driver &mdash; the term
nobody ranks, and the one that makes a single-season driver ranking
mostly a ranking of luck.</div>

<h2>Every pairing the season supports</h2>
{_img(figures['pairs'])}
<table><tr><th>quicker</th><th>slower</th><th>raw</th><th>shrunk</th>
<th>ahead</th><th>p</th><th>survives</th></tr>
{"".join(_pair_row(row) for row in pairs)}</table>
<p>The sign test asks only how often the quicker driver was quicker,
which needs no model to believe. Shrinking by the reliability pulls
every gap toward zero, because a gap over a dozen races is partly those
dozen races.</p>
{_img(figures['spread'])}
<div class="note"><b>{sum(1 for r in pairs if r['p'] < 0.05)} pairings
reach p&lt;0.05, but {len(pairs)} tests expect some of that.</b>
Controlling the false discovery rate leaves {len(survivors)}:
{survivor_text}. The rest of the table is a table, not a set of
findings.</div>

<h2>Driver effects, and why they do not form one list</h2>
{_img(figures['effects'])}
<p>Seconds a lap quicker than the driver's own component average.
Positive is quicker, which is the opposite sign to a lap time. Only the
components with three or more drivers are shown, because in a pair the
two numbers are the same comparison written twice.</p>
<table><tr><th>driver</th><th>s a lap</th></tr>{effects_html}</table>
<div class="note"><b>Comparing a number here with one from another
block is meaningless.</b> Each component is centred on its own mean,
and nothing in the season ties one block's zero to another's. A single
table ordering all {shape['drivers']} drivers would be inventing
{shape['free']} numbers the data does not contain.</div>

<h2>What this does not do</h2>
<p>It does not rank drivers across teams, measure anything in
qualifying, or separate a driver from the car's suitability to their
style &mdash; a car built around one driver flatters that driver, and
that appears here as driver skill. It uses one season, which is why the
race-to-race term is as large as it is, and it has no tyre compound,
because the source does not record one.</p>

<p class="sub">Generated by <code>run.py</code>. Lap times, pit stops
and results from the Jolpica mirror of the Ergast schema. The cleaning
rules are carried over from the stint-degradation, undercut-effect and
pace-to-pass projects.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
