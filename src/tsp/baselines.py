"""Simple baselines to compare with."""

import time

from .core import random_tour, tour_length
from .result import Result, SearchRun


def random_search(dist, budget, seed):
    """Generate `budget` random tours and keep the best one."""
    run = SearchRun(budget, seed)

    for _ in range(budget):
        run.evals += 1
        run.record()
        tour = random_tour(len(dist), run.rng)
        run.update(tour, tour_length(dist, tour))
    return run.result(dist)


def nearest_neighbour(dist, start=0):
    """Greedy: always go to the closest city not visited yet."""
    start_time = time.perf_counter()
    n = len(dist)
    tour = [start]
    unvisited = set(range(n)) - {start}

    while unvisited:
        last = tour[-1]
        closest = min(unvisited, key=lambda city: dist[last][city])
        tour.append(closest)
        unvisited.remove(closest)
    length = tour_length(dist, tour)
    seconds = time.perf_counter() - start_time
    return Result(tour, length, n * n, seconds, [(n * n, length)])


def best_nearest_neighbour(dist):
    """Nearest neighbour from every start city, keep the best tour."""
    start_time = time.perf_counter()
    n = len(dist)
    runs = [nearest_neighbour(dist, start) for start in range(n)]
    best = min(runs, key=lambda result: result.length)
    seconds = time.perf_counter() - start_time

    return Result(best.tour, best.length, n ** 3, seconds,
                  [(n ** 3, best.length)])
