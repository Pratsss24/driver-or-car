"""Six figures, written straight to PNG so the README renders on GitHub."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

INK = "#16181d"
MUTED = "#5b6472"
GRID = "#e3e6ea"
GOOD = "#17803d"
BAD = "#b4232c"
LINE = "#1f5fa8"
SOFT = "#aab4c0"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 12, "figure.dpi": 130, "savefig.bbox": "tight"})


def _frame(axis, title: str, subtitle: str = "") -> None:
    axis.set_title(title, loc="left", pad=22 if subtitle else 8,
                   fontweight="bold")
    if subtitle:
        axis.text(0, 1.02, subtitle, transform=axis.transAxes,
                  fontsize=9, color=MUTED, va="bottom")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _label(name: str) -> str:
    return name.replace("max_verstappen", "verstappen").replace(
        "kevin_magnussen", "magnussen").replace("_", " ")


def comparison_graph(path: Path, comparison, shape) -> Path:
    """The whole argument in one picture: ten islands, no bridges."""
    groups = comparison.components()
    figure, axis = plt.subplots(figsize=(8.6, 4.8))

    neighbours = comparison.neighbours()
    spots: dict[str, tuple[float, float]] = {}
    big = [one for one in groups if len(one) > 2]
    small = [one for one in groups if len(one) == 2]

    for index, group in enumerate(small):
        column, row = index % 5, index // 5
        x = column * 2.1
        y = -row * 1.5
        spots[group[0]] = (x, y + 0.30)
        spots[group[1]] = (x, y - 0.30)

    for group in big:
        centre = next(name for name in group
                      if len(neighbours.get(name, ())) > 1)
        others = [name for name in group if name != centre]
        spots[centre] = (10.6, -0.75)
        for index, name in enumerate(others):
            angle = np.pi * (0.5 - index / max(len(others) - 1, 1))
            spots[name] = (10.6 + 1.25 * np.cos(angle),
                           -0.75 + 0.95 * np.sin(angle))

    for one, two in comparison.edges:
        if one in spots and two in spots:
            axis.plot([spots[one][0], spots[two][0]],
                      [spots[one][1], spots[two][1]],
                      color=SOFT, linewidth=1.6, zorder=1)
    hubs = {name for name, near in neighbours.items() if len(near) > 1}
    for name, (x, y) in spots.items():
        axis.scatter([x], [y], s=70, color=LINE, zorder=2)
        # A hub has a spoke going off to its right, so its own label would
        # sit on top of that neighbour's. Put the hub's label underneath.
        if name in hubs:
            axis.text(x, y - 0.22, _label(name), fontsize=8, ha="center",
                      va="top", color=INK, zorder=3)
        else:
            axis.text(x + 0.12, y, _label(name), fontsize=8, va="center",
                      color=INK, zorder=3)

    axis.set_xlim(-0.6, 13.4)
    axis.set_ylim(-2.5, 1.1)
    axis.axis("off")
    axis.set_title("Twenty-two drivers, ten islands", loc="left", pad=22,
                   fontweight="bold")
    axis.text(0, 1.02,
              f"an edge is a shared car; {shape['cycles']} cycles, so no "
              f"comparison has a second route to check it against",
              transform=axis.transAxes, fontsize=9, color=MUTED, va="bottom")
    axis.text(0, -0.04,
              f"{shape['drivers']} driver effects on a design of rank "
              f"{shape['identified']}: {shape['free']} of them are free "
              f"parameters no amount of 2023 data can fix.",
              transform=axis.transAxes, fontsize=9, color=BAD, va="top")
    out = path / "graph.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def variance(path: Path, shares) -> Path:
    """Where the spread of car paces actually comes from."""
    figure, axis = plt.subplots(figsize=(8.4, 2.9))
    parts = [("the car", shares["team share"], LINE),
             ("the day", shares["race to race share"], "#d9a441"),
             ("the driver", shares["driver share"], GOOD),
             ("measuring it", shares["measurement share"], SOFT)]
    left = 0.0
    for name, share, colour in parts:
        axis.barh([0], [100 * share], left=[100 * left], color=colour,
                  height=0.52)
        middle = 100 * (left + share / 2)
        if share > 0.12:
            axis.text(middle, 0, f"{name}\n{100 * share:.1f}%",
                      ha="center", va="center", color="white", fontsize=10,
                      fontweight="bold")
        else:
            # Too thin to write inside. A leader down to a line of its
            # own, at staggered depths so the last two do not collide.
            depth = -0.52 if name == parts[-1][0] else -0.88
            axis.plot([middle, middle], [-0.26, depth + 0.06],
                      color=colour, linewidth=1.2)
            axis.text(middle - 1.2, depth, f"{name} {100 * share:.1f}%",
                      ha="right", va="center", fontsize=9, color=colour)
        left += share
    axis.set_yticks([])
    axis.set_xlim(0, 100)
    axis.set_ylim(-1.15, 0.45)
    axis.set_xlabel("share of the variance in car pace within a race")
    axis.set_title("The driver is four and a half per cent of it",
                   loc="left", pad=22, fontweight="bold")
    axis.text(0, 1.06,
              f"car {shares['team sd']:.2f} s a lap, the day "
              f"{shares['race to race sd']:.2f}, the driver "
              f"{shares['driver sd']:.2f}",
              transform=axis.transAxes, fontsize=9, color=MUTED, va="bottom")
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    out = path / "variance.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def pair_gaps(path: Path, pairs) -> Path:
    """Every teammate gap, raw against shrunk, and which ones survive."""
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    rows = list(reversed(pairs))
    spots = np.arange(len(rows))
    raw = [row["gap"] for row in rows]
    shrunk = [row["shrunk"] for row in rows]

    for spot, one, two in zip(spots, raw, shrunk):
        axis.plot([two, one], [spot, spot], color=SOFT, linewidth=1.4,
                  zorder=1)
    axis.scatter(raw, spots, s=46, color=SOFT, zorder=2, label="as measured")
    axis.scatter(shrunk, spots,
                 s=58, zorder=3, label="shrunk by the reliability",
                 color=[GOOD if row["survives"] else BAD for row in rows])

    axis.set_yticks(spots)
    axis.set_yticklabels(
        [f"{_label(row['quicker'])} over {_label(row['slower'])}"
         for row in rows], fontsize=9)
    axis.axvline(0, color=MUTED, linewidth=0.8)
    for spot, row in zip(spots, rows):
        axis.text(max(raw) * 1.03, spot,
                  f"{row['won']}/{row['races']}", fontsize=8,
                  color=MUTED, va="center")
    axis.set_xlim(-0.02, max(raw) * 1.16)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_title("Two of twelve survive", loc="left", pad=22,
                   fontweight="bold")
    axis.text(0, 1.02,
              "green survives a false-discovery correction over the twelve "
              "sign tests; red does not",
              transform=axis.transAxes, fontsize=9, color=MUTED, va="bottom")
    axis.set_xlabel("teammate gap, seconds a lap")
    axis.legend(frameon=False, fontsize=9, loc="lower right",
                bbox_to_anchor=(0.87, 0.01))
    out = path / "pairs.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def reliability(path: Path, trust) -> Path:
    """Does half a season predict the other half?"""
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    odd, even = trust["odd"], trust["even"]
    edge = float(max(np.abs(np.concatenate([odd, even]))) * 1.2)
    axis.plot([-edge, edge], [-edge, edge], color=SOFT, linewidth=1.2,
              linestyle="--", zorder=1)
    axis.axhline(0, color=GRID, linewidth=1.0)
    axis.axvline(0, color=GRID, linewidth=1.0)
    axis.scatter(odd, even, s=62, color=LINE, zorder=3)
    axis.set_xlim(-edge, edge)
    axis.set_ylim(-edge, edge)
    _frame(axis, "Half a season against the other half",
           f"each point is one teammate pairing, r = "
           f"{trust['correlation']:.2f}")
    axis.grid(color=GRID, linewidth=0.8)
    axis.set_xlabel("gap over the odd rounds, seconds a lap")
    axis.set_ylabel("gap over the even rounds, seconds a lap")
    axis.text(0.03, 0.96,
              f"reliability {trust['reliability']:.2f}\n"
              f"signal {trust['signal sd']:.2f} s a lap\n"
              f"noise  {trust['noise sd']:.2f}",
              transform=axis.transAxes, fontsize=9, va="top", color=INK,
              bbox=dict(facecolor="#f6f8fa", edgecolor=GRID, pad=6))
    out = path / "reliability.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def race_spread(path: Path, pairs) -> Path:
    """How much a single race says, which is less than it looks."""
    figure, axis = plt.subplots(figsize=(8.2, 5.0))
    rows = list(reversed(pairs))
    for spot, row in enumerate(rows):
        # `signed` is oriented on the alphabetically first of the pair;
        # flip it where that is not the quicker driver, so every row
        # reads the same way round as its label.
        first = sorted((row["quicker"], row["slower"]))[0]
        shown = (row["signed"] if row["quicker"] == first
                 else -row["signed"])
        axis.scatter(shown, np.full(shown.size, spot) +
                     np.random.default_rng(spot).normal(0, 0.07, shown.size),
                     s=26, color=SOFT, zorder=2)
        axis.scatter([row["gap"]], [spot], s=70, color=LINE, zorder=3,
                     marker="D")
    axis.axvline(0, color=INK, linewidth=1.0)
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels(
        [f"{_label(row['quicker'])} over {_label(row['slower'])}"
         for row in rows], fontsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_title("One race says almost nothing", loc="left", pad=22,
                   fontweight="bold")
    axis.text(0, 1.02,
              "grey is a single race, blue is the season's mean; most "
              "pairings cross zero repeatedly",
              transform=axis.transAxes, fontsize=9, color=MUTED, va="bottom")
    axis.set_xlabel("that race's gap, seconds a lap")
    out = path / "spread.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def effects_chart(path: Path, comparison, effects) -> Path:
    """Driver effects, drawn so the components cannot be read across."""
    groups = comparison.components()
    figure, axis = plt.subplots(figsize=(8.0, 5.6))
    spot = 0.0
    ticks, labels, boundaries = [], [], []
    for group in groups:
        for name in sorted(group, key=lambda one: effects[one]):
            axis.barh([spot], [effects[name]],
                      color=GOOD if effects[name] > 0 else BAD, height=0.6)
            ticks.append(spot)
            labels.append(_label(name))
            spot += 1
        boundaries.append(spot - 0.5)
        spot += 0.7

    for edge in boundaries[:-1]:
        axis.axhline(edge, color=GRID, linewidth=1.0)
    axis.axvline(0, color=INK, linewidth=1.0)
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels, fontsize=8.5)
    axis.set_ylim(-1.0, spot - 0.4)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_title("Ten separate scales, not one table", loc="left", pad=22,
                   fontweight="bold")
    axis.text(0, 1.015,
              "each block is centred on its own average; a number here "
              "cannot be compared with one in another block",
              transform=axis.transAxes, fontsize=9, color=MUTED, va="bottom")
    axis.set_xlabel("seconds a lap quicker than the block's average")
    out = path / "effects.png"
    figure.savefig(out)
    plt.close(figure)
    return out


def draw(path: Path, **kwargs) -> dict[str, Path]:
    path.mkdir(parents=True, exist_ok=True)
    return {
        "graph": comparison_graph(path, kwargs["comparison"],
                                  kwargs["shape"]),
        "variance": variance(path, kwargs["shares"]),
        "pairs": pair_gaps(path, kwargs["pairs"]),
        "reliability": reliability(path, kwargs["trust"]),
        "spread": race_spread(path, kwargs["pairs"]),
        "effects": effects_chart(path, kwargs["comparison"],
                                 kwargs["effects"]),
    }
