"""Which drivers can be compared with which, and which cannot at all.

A teammate contrast is the only clean comparison in Formula 1: two drivers,
one car, one race. Everything else is confounded by machinery that differs
by seconds a lap, where the drivers differ by tenths.

That makes the set of available comparisons a **graph**. Drivers are nodes
and a shared team-race is an edge. A driver effect is identified against
another only if a path of edges runs between them, and the whole question
of "who is the quickest driver" is a question about this graph before it is
a question about any data.

In a single season the graph is almost nothing. Nine teams ran the same two
drivers all year, which is nine isolated edges: two nodes and no way out.
One team ran four drivers, giving a four-node component. Twenty-two drivers,
twelve edges, ten components - so twenty-two driver effects sit on a design
of rank twelve, and ten of them are free parameters that no amount of
2023 data can pin down.

**It is worse than disconnected: it is a forest.** A graph with `n` nodes
and `c` components needs at least `n - c` edges to connect its components
internally, and has a cycle only if it has more. Here `n - c` is 22 - 10 =
12, which is exactly the number of edges. There is not one cycle in it. A
cycle would mean two independent routes between the same pair of drivers,
and comparing them would be a consistency check on the whole method. There
are none available, so no such check can be run - not because nobody
thought of it, but because the season does not contain one.

What survives is comparison *within* a component, and `centre` is where
that is enforced: effects are measured against their own component's mean
and are meaningless across components. A table that ranks all twenty-two
is not a stronger analysis of this data. It is a different assumption
about the cars, presented as a result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Comparison:
    """The graph of who shared a car with whom."""

    edges: dict[tuple[str, str], int] = field(default_factory=dict)

    @property
    def nodes(self) -> list[str]:
        return sorted({name for pair in self.edges for name in pair})

    def neighbours(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = defaultdict(set)
        for one, two in self.edges:
            out[one].add(two)
            out[two].add(one)
        return dict(out)

    def components(self) -> list[list[str]]:
        """Groups of drivers connected by some chain of shared cars."""
        neighbours = self.neighbours()
        unseen = set(self.nodes)
        found: list[list[str]] = []
        while unseen:
            start = min(unseen)
            stack, group = [start], []
            unseen.discard(start)
            while stack:
                here = stack.pop()
                group.append(here)
                for other in neighbours.get(here, ()):
                    if other in unseen:
                        unseen.discard(other)
                        stack.append(other)
            found.append(sorted(group))
        return sorted(found, key=lambda one: (-len(one), one[0]))

    def component_of(self) -> dict[str, int]:
        return {name: number
                for number, group in enumerate(self.components())
                for name in group}

    @property
    def cycles(self) -> int:
        """Independent cycles: edges minus nodes plus components.

        Zero means every comparison rests on a single chain of shared
        cars, with no second route to check it against.
        """
        return len(self.edges) - len(self.nodes) + len(self.components())

    @property
    def rank(self) -> int:
        """How many driver effects the contrasts actually identify."""
        return len(self.nodes) - len(self.components())

    def summary(self) -> dict:
        groups = self.components()
        return {"drivers": len(self.nodes), "pairings": len(self.edges),
                "components": len(groups), "identified": self.rank,
                "free": len(self.nodes) - self.rank,
                "cycles": self.cycles,
                "largest component": max(len(one) for one in groups),
                "isolated pairs": sum(1 for one in groups if len(one) == 2)}

    def path(self, start: str, end: str) -> list[str] | None:
        """A chain of shared cars linking two drivers, if one exists."""
        if start == end:
            return [start]
        neighbours = self.neighbours()
        seen = {start}
        queue: list[list[str]] = [[start]]
        while queue:
            route = queue.pop(0)
            for other in sorted(neighbours.get(route[-1], ())):
                if other in seen:
                    continue
                if other == end:
                    return route + [other]
                seen.add(other)
                queue.append(route + [other])
        return None


def build(results) -> Comparison:
    """Every pair of drivers who shared a team in the same race."""
    together: dict[tuple, set[str]] = defaultdict(set)
    for row in results:
        together[(row.team, row.season, row.round)].add(row.driver)

    edges: dict[tuple[str, str], int] = defaultdict(int)
    for members in together.values():
        names = sorted(members)
        for index, one in enumerate(names):
            for two in names[index + 1:]:
                edges[(one, two)] += 1
    return Comparison(edges=dict(edges))


def design(contrasts, comparison: Comparison
           ) -> tuple[np.ndarray, tuple[str, ...]]:
    """One row per contrast: +1 for the faster slot, -1 for the other.

    The columns are drivers, with one dropped per component to pin that
    component's level. Dropping one per component rather than one overall
    is the whole identification argument expressed as a matrix: each
    component floats freely against the others and no arithmetic can
    bring them onto a common scale.
    """
    groups = comparison.components()
    held: set[str] = {group[0] for group in groups}
    names = tuple(name for name in comparison.nodes if name not in held)
    index = {name: number for number, name in enumerate(names)}

    matrix = np.zeros((len(contrasts), len(names)))
    for row, contrast in enumerate(contrasts):
        if contrast.quicker in index:
            matrix[row, index[contrast.quicker]] += 1.0
        if contrast.slower in index:
            matrix[row, index[contrast.slower]] -= 1.0
    return matrix, names


def centre(values: dict[str, float], comparison: Comparison
           ) -> dict[str, float]:
    """Shift each component so its own drivers average zero.

    After this the numbers are readable within a component and still
    meaningless across one, which is the honest state of affairs.
    """
    out: dict[str, float] = {}
    for group in comparison.components():
        here = [values.get(name, 0.0) for name in group]
        middle = float(np.mean(here)) if here else 0.0
        for name in group:
            out[name] = values.get(name, 0.0) - middle
    return out
