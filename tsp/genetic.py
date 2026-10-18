"""Genetic algorithm (and a memetic variant) for the TSP.

Follows the pseudo-code of the lecture (selection -> crossover -> mutation)
with a permutation encoding:

* individual  : a tour (permutation of the cities)
* fitness     : tour length (lower is better; equivalent to "higher fitness
                = better" with fitness = 1 / length)
* selection   : tournament of size k (roulette-wheel selection on 1/length is
                almost uniform for tours of similar length)
* crossover   : order crossover (OX) - a random segment of parent 1 is kept,
                the remaining cities are copied in the order of parent 2
* mutation    : inversion of a random segment (a random 2-opt style move)
* replacement : generational with elitism

Memetic variant (extension): each child is, with probability `ls_prob`,
improved by a short first-choice 2-opt local search (at most `ls_cap`
neighbour evaluations).  Improved children are written back
(Lamarckian evolution).  Every evaluation made by the local search is
charged to the same budget as in all the other algorithms.
"""
from __future__ import annotations

import random
import time
from typing import List

from .core import local_search_2opt, random_tour, tour_length
from .result import Result


def order_crossover(p1: List[int], p2: List[int], rng: random.Random) -> List[int]:
    """OX: keep p1[a..b], fill the other positions with p2's relative order."""
    n = len(p1)
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b:
        a, b = b, a
    segment = p1[a:b + 1]
    in_seg = set(segment)
    rest = [c for c in p2 if c not in in_seg]
    return rest[:a] + segment + rest[a:]


def inversion_mutation(t: List[int], rng: random.Random) -> None:
    n = len(t)
    a = rng.randrange(n)
    b = rng.randrange(n)
    if a > b:
        a, b = b, a
    t[a:b + 1] = t[a:b + 1][::-1]


def genetic_algorithm(d, budget: int, seed: int, pop_size: int = 100,
                      cx_prob: float = 0.9, mut_prob: float = 0.2,
                      tournament_k: int = 3, elite: int = 2,
                      ls_prob: float = 0.0, ls_cap: int = 0,
                      history_points: int = 200) -> Result:
    """GA (ls_prob = 0) or memetic algorithm (ls_prob > 0)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = len(d)
    rand = rng.random
    randrange = rng.randrange

    pop = [random_tour(n, rng) for _ in range(pop_size)]
    lens = [tour_length(d, t) for t in pop]
    evals = pop_size
    bi = min(range(pop_size), key=lens.__getitem__)
    best_len, best_tour = lens[bi], pop[bi][:]
    history = []
    step = max(1, budget // history_points)
    nxt = 0
    generations = 0

    def select() -> int:
        w = randrange(pop_size)
        for _ in range(tournament_k - 1):
            c = randrange(pop_size)
            if lens[c] < lens[w]:
                w = c
        return w

    while evals < budget:
        ranked = sorted(range(pop_size), key=lens.__getitem__)
        new_pop = [pop[i][:] for i in ranked[:elite]]
        new_lens = [lens[i] for i in ranked[:elite]]
        while len(new_pop) < pop_size and evals < budget:
            if evals >= nxt:
                history.append((evals, best_len))
                nxt += step
            p1, p2 = pop[select()], pop[select()]
            child = order_crossover(p1, p2, rng) if rand() < cx_prob else p1[:]
            if rand() < mut_prob:
                inversion_mutation(child, rng)
            length = tour_length(d, child)
            evals += 1
            if ls_prob > 0 and rand() < ls_prob:
                length, used, _ = local_search_2opt(d, child, length, rng, ls_cap)
                evals += used
            if length < best_len - 1e-12:
                best_len, best_tour = length, child[:]
            new_pop.append(child)
            new_lens.append(length)
        if len(new_pop) < pop_size:        # budget exhausted in mid-generation
            break
        pop, lens = new_pop, new_lens
        generations += 1

    history.append((min(evals, budget), best_len))
    best_len = tour_length(d, best_tour)
    return Result(best_tour, best_len, evals, time.perf_counter() - t0, history,
                  {"generations": generations})


def memetic_algorithm(d, budget: int, seed: int, pop_size: int = 100,
                      ls_prob: float = 0.2, ls_cap=None, **kw) -> Result:
    """Memetic algorithm = GA + short Lamarckian 2-opt local search."""
    if ls_cap is None:
        ls_cap = len(d)                 # at most n neighbour evaluations per child
    return genetic_algorithm(d, budget, seed, pop_size=pop_size,
                             ls_prob=ls_prob, ls_cap=ls_cap, **kw)
