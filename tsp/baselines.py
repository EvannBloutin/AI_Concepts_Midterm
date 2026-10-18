"""Simple baselines used as reference points."""
from __future__ import annotations

import random
import time

from .core import random_tour, tour_length
from .result import Result


def random_search(d, budget: int, seed: int, history_points: int = 200) -> Result:
    """Best of `budget` random tours (no search guidance at all)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(d)
    best_len, best_tour = float("inf"), None
    history, step, nxt = [], max(1, budget // history_points), 0
    for evals in range(1, budget + 1):
        if evals >= nxt:
            history.append((evals, best_len))
            nxt += step
        t = random_tour(n, rng)
        L = tour_length(d, t)
        if L < best_len:
            best_len, best_tour = L, t
    history.append((budget, best_len))
    return Result(best_tour, best_len, budget, time.perf_counter() - t0, history)


def nearest_neighbour(d, start: int = 0) -> Result:
    """Greedy constructive heuristic: always go to the closest unvisited city."""
    t0 = time.perf_counter()
    n = len(d)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda c: d[last][c])
        unvisited.remove(nxt)
        tour.append(nxt)
    L = tour_length(d, tour)
    return Result(tour, L, n * n, time.perf_counter() - t0, [(n * n, L)])


def best_nearest_neighbour(d) -> Result:
    """Nearest neighbour from every start city, keep the best (n runs)."""
    t0 = time.perf_counter()
    runs = [nearest_neighbour(d, s) for s in range(len(d))]
    best = min(runs, key=lambda r: r.length)
    return Result(best.tour, best.length, len(d) ** 3, time.perf_counter() - t0,
                  [(len(d) ** 3, best.length)])
