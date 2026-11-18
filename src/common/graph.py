"""Directed road graph of one trip: network distances between positions on arcs.

Shared by the HMM transition model (``src/hmm``) and by the route-continuity
metric (``src/evaluation``). Both need the same question answered:

    "Driving legally along the directed network, how far is it from position
     f_a on arc a to position f_b on arc b?"

A position is (arc_id, offset_frac) with offset_frac in [0, 1] measured from
the arc's from-node. Node-to-node distances come from a bounded Dijkstra
search (``heapq``), cached per (source node, distance budget). The budget keeps
each search local: at 1 Hz a vehicle moves tens of metres per step, so only a
few nodes are ever expanded.

Arcs are straight segments between two nodes (the ``.arcs`` format has no
intermediate geometry), so arc length = Euclidean length in the local metric
projection.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Tuple

import numpy as np

INF = float("inf")


class RoadGraph:
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        self.nodes_xy = np.asarray(nodes_xy, dtype=float)
        self.arcs = np.asarray(arcs, dtype=np.int64)
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.length_m = np.hypot(self.nodes_xy[b, 0] - self.nodes_xy[a, 0], self.nodes_xy[b, 1] - self.nodes_xy[a, 1])
        n_nodes = len(self.nodes_xy)
        # Adjacency list: node -> list of (neighbour, length). Duplicate arcs keep the shortest.
        best: Dict[Tuple[int, int], float] = {}
        for (u, v), length in zip(self.arcs.tolist(), self.length_m.tolist()):
            if u == v:
                continue  # self-loops never help a shortest path
            if length < best.get((u, v), INF):
                best[(u, v)] = length
        self.adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        for (u, v), length in best.items():
            self.adjacency[u].append((v, length))
        self._cache: Dict[Tuple[int, float], Dict[int, float]] = {}
        # Equivalence key: arcs with the same (from, to) node pair are the same road piece.
        self.arc_key = self.arcs[:, 0] * (n_nodes + 1) + self.arcs[:, 1]

    # ------------------------------------------------------------------ #
    def node_distances(self, source: int, budget_m: float) -> Dict[int, float]:
        """Shortest network distance from ``source`` to every node reachable within ``budget_m``."""
        key = (int(source), float(budget_m))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        dist: Dict[int, float] = {int(source): 0.0}
        heap = [(0.0, int(source))]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, INF):
                continue
            for v, w in self.adjacency[u]:
                nd = d + w
                if nd <= budget_m and nd < dist.get(v, INF):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) > 200_000:  # bound memory on huge trips
            self._cache.clear()
        self._cache[key] = dist
        return dist

    def route_distance(self, arc_a: int, frac_a: float, arc_b: int, frac_b: float, budget_m: float) -> float:
        """Network distance from (arc_a, frac_a) to (arc_b, frac_b); ``inf`` if not reachable within budget.

        On the same arc (or an equivalent duplicate) the absolute along-arc distance is
        returned: at 1 Hz a stationary vehicle's GPS jitters back and forth, which must not
        count as an impossible move.
        """
        if arc_a == arc_b or self.arc_key[arc_a] == self.arc_key[arc_b]:
            return abs(frac_b - frac_a) * self.length_m[arc_a]
        rest_a = (1.0 - frac_a) * self.length_m[arc_a]
        into_b = frac_b * self.length_m[arc_b]
        remaining = budget_m - rest_a - into_b
        if remaining < 0:
            return INF
        mid = self.node_distances(int(self.arcs[arc_a, 1]), _bucket(budget_m)).get(int(self.arcs[arc_b, 0]), INF)
        total = rest_a + mid + into_b
        return total if total <= budget_m else INF

    def equivalent(self, arc_a: int, arc_b: int) -> bool:
        return bool(self.arc_key[arc_a] == self.arc_key[arc_b])


def _bucket(budget_m: float) -> float:
    """Round a budget up to a few fixed values so the Dijkstra cache is reused across steps."""
    for b in (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0):
        if budget_m <= b:
            return b
    return float(np.ceil(budget_m / 1000.0) * 1000.0)
