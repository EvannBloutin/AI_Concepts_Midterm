"""Result of a solver, and the state shared by every search run."""

import random
import time
from dataclasses import dataclass, field

from .core import tour_length

HISTORY_POINTS = 200


@dataclass
class Result:
    """What every solver returns."""
    tour: list
    length: float
    evals: int
    seconds: float
    history: list = field(default_factory=list)


def with_defaults(defaults, params):
    """Copy of `defaults` where the keys given in `params` are replaced."""
    merged = dict(defaults)

    merged.update(params or {})
    return merged


class SearchRun:
    """State of one run: parameters, random generator, budget, number of
    evaluations used, best tour found so far and history of the best."""

    def __init__(self, budget, seed, eps=0.0, params=None):
        """eps: a new tour must be shorter than the best by more than eps."""
        self.params = params or {}
        self.rng = random.Random(seed)
        self.budget = budget
        self.evals = 0
        self.eps = eps
        self.best_tour = None
        self.best_len = float("inf")
        self.history = []
        self.step = max(1, budget // HISTORY_POINTS)
        self.next_record = 0
        self.start_time = time.perf_counter()

    def budget_left(self):
        """Number of evaluations we are still allowed to do."""
        return self.budget - self.evals

    def record(self):
        """Save (evals, best length) about HISTORY_POINTS times per run."""
        if self.evals >= self.next_record:
            self.history.append((self.evals, self.best_len))
            self.next_record += self.step

    def update(self, tour, length):
        """Keep a copy of `tour` if it is the best one so far."""
        if length < self.best_len - self.eps:
            self.best_tour = tour[:]
            self.best_len = length

    def result(self, dist):
        """Build the final Result (the length is recomputed exactly)."""
        final_evals = min(self.evals, self.budget)
        length = tour_length(dist, self.best_tour)
        seconds = time.perf_counter() - self.start_time

        self.history.append((final_evals, self.best_len))
        return Result(self.best_tour, length, self.evals, seconds,
                      self.history)
