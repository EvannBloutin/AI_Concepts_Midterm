"""Simulated annealing for the TSP (lecture 'Local Search').

Pseudo-code (minimisation, DeltaE = new_length - current_length):

    T <- T0
    repeat
        pick a random 2-opt neighbour
        if DeltaE < 0:                     accept
        else:                              accept with probability exp(-DeltaE / T)
        T <- alpha * T                     (geometric cooling schedule)
    until the evaluation budget is used

T0 is calibrated from the instance: it is chosen so that an average
*uphill* move taken from a random tour is accepted with probability `p0`.
alpha is chosen so that the temperature reaches `t_end_ratio * T0` exactly
at the end of the budget (so the schedule adapts to the budget).
"""
from __future__ import annotations

import math
import random
import time

from .core import (apply_two_opt, random_tour, tour_length, two_opt_delta,
                   two_opt_pairs)
from .result import Result


def estimate_t0(d, rng: random.Random, p0: float, samples: int = 200) -> float:
    n = len(d)
    pairs = two_opt_pairs(n)
    tour = random_tour(n, rng)
    ups = []
    for _ in range(samples):
        i, j = pairs[rng.randrange(len(pairs))]
        delta = two_opt_delta(d, tour, i, j)
        if delta > 0:
            ups.append(delta)
    mean_up = sum(ups) / len(ups) if ups else 1.0
    return -mean_up / math.log(p0)


def simulated_annealing(d, budget: int, seed: int, p0: float = 0.3,
                        t_end_ratio: float = 1e-3,
                        history_points: int = 200) -> Result:
    t0_clock = time.perf_counter()
    rng = random.Random(seed)
    n = len(d)
    pairs = two_opt_pairs(n)
    size = len(pairs)

    T0 = estimate_t0(d, rng, p0)
    T = T0
    alpha = t_end_ratio ** (1.0 / max(1, budget))   # T_end = t_end_ratio * T0

    tour = random_tour(n, rng)
    cur = tour_length(d, tour)
    best_len, best_tour = cur, tour[:]
    history = []
    step = max(1, budget // history_points)
    nxt = 0
    accepted = 0
    exp = math.exp
    rand = rng.random
    randrange = rng.randrange

    for evals in range(1, budget + 1):
        if evals >= nxt:
            history.append((evals, best_len))
            nxt += step
        i, j = pairs[randrange(size)]
        delta = two_opt_delta(d, tour, i, j)
        if delta < 0 or rand() < exp(-delta / T):
            apply_two_opt(tour, i, j)
            cur += delta
            accepted += 1
            if cur < best_len - 1e-12:
                best_len, best_tour = cur, tour[:]
        T *= alpha

    history.append((budget, best_len))
    best_len = tour_length(d, best_tour)
    return Result(best_tour, best_len, budget, time.perf_counter() - t0_clock, history,
                  {"T0": T0, "T_end": T0 * t_end_ratio, "accepted": accepted})
