"""Hill climbing for the TSP (greedy local search, lecture 'Local Search').

* Neighbourhood : 2-opt (see core.two_opt_pairs).
* Variant       : *first-choice* hill climbing - neighbours are examined in a
                  random order and the first improving one is taken (AIMA 3rd
                  ed., sec. 4.1.1).  Steepest ascent would need n(n-3)/2
                  evaluations for every single step, which wastes the budget.
* Local optimum : detected exactly - a complete cycle through all neighbours
                  without any improvement.
* Random restart: when a local optimum is reached and budget remains, start
                  again from a new random tour ("if you want to pick one local
                  search algorithm, learn this one!").
"""
from __future__ import annotations

import random
import time

from .core import (apply_two_opt, random_tour, tour_length, two_opt_delta,
                   two_opt_pairs)
from .result import Result


def hill_climbing(d, budget: int, seed: int, restarts: bool = True,
                  history_points: int = 200) -> Result:
    """Hill climbing with (optional) random restarts.

    budget   : maximum number of candidate evaluations (1 per neighbour tested
               or per random tour generated).
    restarts : False -> a single climb (stops at the first local optimum).
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(d)
    pairs = two_opt_pairs(n)
    size = len(pairs)

    evals = 0
    best_len = float("inf")
    best_tour = None
    history = []
    step = max(1, budget // history_points)
    nxt = 0
    n_restarts = 0
    local_optima = []                    # length of every local optimum reached

    while evals < budget:
        tour = random_tour(n, rng)
        cur = tour_length(d, tour)
        evals += 1
        n_restarts += 1
        order = pairs[:]
        rng.shuffle(order)
        k = 0
        fails = 0
        while fails < size and evals < budget:
            if evals >= nxt:
                history.append((evals, min(best_len, cur)))
                nxt += step
            i, j = order[k]
            k += 1
            if k == size:
                k = 0
            evals += 1
            delta = two_opt_delta(d, tour, i, j)
            if delta < -1e-12:
                apply_two_opt(tour, i, j)
                cur += delta
                fails = 0
            else:
                fails += 1
        if fails >= size:
            local_optima.append(cur)
        if cur < best_len:
            best_len, best_tour = cur, tour[:]
        if not restarts:
            break

    history.append((evals, best_len))
    best_len = tour_length(d, best_tour)          # remove floating-point drift
    return Result(best_tour, best_len, evals, time.perf_counter() - t0, history,
                  {"restarts": n_restarts, "local_optima": local_optima})
