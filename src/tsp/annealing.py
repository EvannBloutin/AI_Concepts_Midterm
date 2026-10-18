"""Simulated annealing.

    T = T0
    repeat budget times:
        pick a random 2-opt move, delta = change of length
        if delta < 0: accept
        else:         accept with probability exp(-delta / T)
        T = alpha * T

T0 is chosen so that an average uphill move is accepted with probability p0.
alpha is chosen so that T = t_end_ratio * T0 at the end of the budget.
"""

import math

from .core import (EPS, apply_two_opt, random_tour, tour_length,
                   two_opt_delta, two_opt_pairs)
from .result import SearchRun, with_defaults

SA_DEFAULTS = {"p0": 0.3, "t_end_ratio": 1e-3}
T0_SAMPLES = 200


def start_temperature(dist, rng, p0):
    """T0 such that exp(-average_uphill_delta / T0) = p0."""
    moves = two_opt_pairs(len(dist))
    tour = random_tour(len(dist), rng)
    uphill = []

    for _ in range(T0_SAMPLES):
        i, j = moves[rng.randrange(len(moves))]
        delta = two_opt_delta(dist, tour, i, j)
        if delta > 0:
            uphill.append(delta)
    mean_uphill = sum(uphill) / len(uphill) if uphill else 1.0
    return -mean_uphill / math.log(p0)


def simulated_annealing(dist, budget, seed, params=None):
    """params: {"p0": ..., "t_end_ratio": ...} (see SA_DEFAULTS)."""
    run = SearchRun(budget, seed, EPS, with_defaults(SA_DEFAULTS, params))
    moves = two_opt_pairs(len(dist))
    temp = start_temperature(dist, run.rng, run.params["p0"])
    alpha = run.params["t_end_ratio"] ** (1.0 / max(1, budget))
    tour = random_tour(len(dist), run.rng)
    length = tour_length(dist, tour)

    run.update(tour, length)
    for _ in range(budget):
        run.evals += 1
        run.record()
        i, j = moves[run.rng.randrange(len(moves))]
        delta = two_opt_delta(dist, tour, i, j)
        if delta < 0 or run.rng.random() < math.exp(-delta / temp):
            apply_two_opt(tour, i, j)
            length += delta
            run.update(tour, length)
        temp *= alpha
    return run.result(dist)
