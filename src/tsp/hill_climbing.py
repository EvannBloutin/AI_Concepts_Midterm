"""Hill climbing with random restarts.

Start from a random tour, improve it with 2-opt moves until no move helps
(local optimum), then start again from a new random tour while budget is
left. One evaluation = one random tour or one 2-opt move tested.
"""

from .core import local_search_2opt, random_tour
from .result import SearchRun


def hill_climbing(dist, budget, seed, restarts=True):
    """Hill climbing; with restarts=False it does a single climb only."""
    run = SearchRun(budget, seed)

    while run.budget_left() > 0:
        tour = random_tour(len(dist), run.rng)
        run.evals += 1
        length, used = local_search_2opt(dist, tour, run.rng,
                                         run.budget_left())
        run.evals += used
        run.update(tour, length)
        run.record()
        if not restarts:
            break
    return run.result(dist)
