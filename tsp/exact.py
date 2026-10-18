"""Exact solvers and a lower bound, used as ground truth on small instances.

* brute_force  : enumerate all (n-1)!/2 tours (uninformed, exhaustive).
* held_karp    : dynamic programming over subsets, O(n^2 2^n).
* astar_tsp    : graph search over states (set of visited cities, last city).
                 with use_heuristic=False it is Uniform-Cost Search (UCS);
                 with use_heuristic=True it is A* with the admissible
                 'minimum spanning tree' heuristic.
* one_tree_lower_bound : classical lower bound (MST without city 0 + the two
                 cheapest edges at city 0); lets us bound the optimality gap
                 on instances that are too large to solve exactly.
"""
from __future__ import annotations

import heapq
import itertools
import time
from typing import Dict, List, Tuple

from .core import tour_length


def brute_force(d) -> Tuple[float, List[int], int]:
    n = len(d)
    best, best_t, count = float("inf"), None, 0
    for perm in itertools.permutations(range(1, n)):
        if perm[0] > perm[-1]:            # skip mirror images
            continue
        count += 1
        t = (0,) + perm
        L = tour_length(d, t)
        if L < best:
            best, best_t = L, list(t)
    return best, best_t, count


def held_karp(d) -> Tuple[float, List[int]]:
    n = len(d)
    INF = float("inf")
    full = 1 << (n - 1)                       # subsets of cities 1..n-1
    dp = [[INF] * (n - 1) for _ in range(full)]
    parent = [[-1] * (n - 1) for _ in range(full)]
    for j in range(n - 1):
        dp[1 << j][j] = d[0][j + 1]
    for mask in range(1, full):
        for j in range(n - 1):
            if not (mask >> j) & 1 or dp[mask][j] == INF:
                continue
            base = dp[mask][j]
            for k in range(n - 1):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                cand = base + d[j + 1][k + 1]
                if cand < dp[nm][k]:
                    dp[nm][k] = cand
                    parent[nm][k] = j
    last_mask = full - 1
    best, bj = INF, -1
    for j in range(n - 1):
        cand = dp[last_mask][j] + d[j + 1][0]
        if cand < best:
            best, bj = cand, j
    tour, mask, j = [], last_mask, bj
    while j != -1:
        tour.append(j + 1)
        pj = parent[mask][j]
        mask ^= 1 << j
        j = pj
    tour.append(0)
    tour.reverse()
    return best, tour


def mst_cost(d, nodes: List[int]) -> float:
    """Prim's algorithm on the complete graph restricted to `nodes`."""
    if len(nodes) <= 1:
        return 0.0
    in_tree = {nodes[0]}
    dist = {v: d[nodes[0]][v] for v in nodes[1:]}
    total = 0.0
    while dist:
        v = min(dist, key=dist.get)
        total += dist.pop(v)
        for u in dist:
            if d[v][u] < dist[u]:
                dist[u] = d[v][u]
    return total


def one_tree_lower_bound(d) -> float:
    n = len(d)
    mst = mst_cost(d, list(range(1, n)))
    two = sorted(d[0][j] for j in range(1, n))[:2]
    return mst + two[0] + two[1]


def astar_tsp(d, use_heuristic: bool = True) -> dict:
    """UCS (use_heuristic=False) or A* with MST heuristic (True).

    State = (visited bitmask, last city); start = city 0 visited.
    Goal  = all cities visited AND back at city 0.
    """
    t0 = time.perf_counter()
    n = len(d)
    all_mask = (1 << n) - 1
    h_cache: Dict[Tuple[int, int], float] = {}

    def h(mask: int, last: int) -> float:
        if not use_heuristic:
            return 0.0
        key = (mask, last)
        if key in h_cache:
            return h_cache[key]
        unvisited = [c for c in range(n) if not (mask >> c) & 1]
        if not unvisited:
            val = d[last][0]
        else:
            nodes = unvisited + [0] + ([last] if last != 0 else [])
            val = mst_cost(d, nodes)
        h_cache[key] = val
        return val

    start = (1, 0)
    goal = (-1, -1)                              # virtual state "tour closed"
    best_g = {start: 0.0}
    parent = {start: None}
    counter = itertools.count()
    frontier = [(h(*start), next(counter), 0.0, start)]
    expanded = generated = 0
    max_frontier = 1
    closed = set()
    while frontier:
        f, _, g, state = heapq.heappop(frontier)
        if state in closed or g > best_g[state] + 1e-12:
            continue
        if state == goal:
            path, s = [], parent[state]
            while s is not None:
                path.append(s[1])
                s = parent[s]
            path.reverse()
            return {"length": g, "tour": path, "expanded": expanded,
                    "generated": generated, "max_frontier": max_frontier,
                    "seconds": time.perf_counter() - t0}
        closed.add(state)
        expanded += 1
        mask, last = state
        if mask == all_mask:
            succ = [(goal, d[last][0])]
        else:
            succ = [((mask | (1 << c), c), d[last][c])
                    for c in range(n) if not (mask >> c) & 1]
        for s2, cost in succ:
            g2 = g + cost
            if s2 not in best_g or g2 < best_g[s2] - 1e-12:
                best_g[s2] = g2
                parent[s2] = state
                hv = 0.0 if s2 == goal else h(*s2)
                heapq.heappush(frontier, (g2 + hv, next(counter), g2, s2))
                generated += 1
        max_frontier = max(max_frontier, len(frontier))
    raise RuntimeError("no solution")
