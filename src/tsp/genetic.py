"""Genetic algorithm (and memetic variant) for the TSP.

* individual : a tour
* fitness    : tour length (shorter is better)
* selection  : tournament (pick k random tours, keep the shortest)
* crossover  : order crossover (OX)
* mutation   : reverse a random part of the tour
* elitism    : the best `elite` tours are copied to the next generation

Memetic variant: some children are also improved by a short 2-opt local
search. Its evaluations count in the same budget.
"""

from .core import EPS, local_search_2opt, random_tour, tour_length
from .result import SearchRun, with_defaults

GA_DEFAULTS = {
    "pop_size": 100,
    "cx_prob": 0.9,
    "mut_prob": 0.2,
    "tournament_k": 3,
    "elite": 2,
    "ls_prob": 0.0,
    "ls_cap": 0,
}


def random_segment(n, rng):
    """Two random positions a <= b."""
    a = rng.randrange(n)
    b = rng.randrange(n)

    return min(a, b), max(a, b)


def order_crossover(parent1, parent2, rng):
    """Keep parent1[a..b] in place, fill the other positions with the
    cities of parent2 in their order."""
    a, b = random_segment(len(parent1), rng)
    segment = parent1[a:b + 1]
    in_segment = set(segment)
    rest = [c for c in parent2 if c not in in_segment]

    return rest[:a] + segment + rest[a:]


def inversion_mutation(tour, rng):
    """Reverse tour[a..b] (in place)."""
    a, b = random_segment(len(tour), rng)

    tour[a:b + 1] = reversed(tour[a:b + 1])


def tournament(lengths, k, rng):
    """Index of the shortest tour among k randomly chosen ones."""
    winner = rng.randrange(len(lengths))

    for _ in range(k - 1):
        other = rng.randrange(len(lengths))
        if lengths[other] < lengths[winner]:
            winner = other
    return winner


def make_child(pop, lengths, params, rng):
    """Selection of two parents, then crossover and mutation."""
    parent1 = pop[tournament(lengths, params["tournament_k"], rng)]
    parent2 = pop[tournament(lengths, params["tournament_k"], rng)]

    if rng.random() < params["cx_prob"]:
        child = order_crossover(parent1, parent2, rng)
    else:
        child = parent1[:]
    if rng.random() < params["mut_prob"]:
        inversion_mutation(child, rng)
    return child


def evaluate_child(dist, child, params, rng):
    """Length of the child, improved by local search in the memetic case.
    Returns (length, evaluations_used)."""
    length = tour_length(dist, child)

    if params["ls_prob"] > 0 and rng.random() < params["ls_prob"]:
        length, used = local_search_2opt(dist, child, rng, params["ls_cap"])
        return length, 1 + used
    return length, 1


def next_generation(dist, pop, lengths, run):
    """Elitism + new children until the population is full or the budget
    is used. Returns (new_pop, new_lengths)."""
    elite = run.params["elite"]
    ranked = sorted(range(len(pop)), key=lambda i: lengths[i])
    new_pop = [pop[i][:] for i in ranked[:elite]]
    new_lengths = [lengths[i] for i in ranked[:elite]]

    while len(new_pop) < len(pop) and run.budget_left() > 0:
        run.record()
        child = make_child(pop, lengths, run.params, run.rng)
        length, used = evaluate_child(dist, child, run.params, run.rng)
        run.evals += used
        run.update(child, length)
        new_pop.append(child)
        new_lengths.append(length)
    return new_pop, new_lengths


def genetic_algorithm(dist, budget, seed, params=None):
    """GA (ls_prob = 0) or memetic algorithm (ls_prob > 0).
    params: see GA_DEFAULTS."""
    run = SearchRun(budget, seed, EPS, with_defaults(GA_DEFAULTS, params))
    pop = [random_tour(len(dist), run.rng)
           for _ in range(run.params["pop_size"])]
    lengths = [tour_length(dist, tour) for tour in pop]
    best = lengths.index(min(lengths))

    run.evals = len(pop)
    run.update(pop[best], lengths[best])
    while run.budget_left() > 0:
        new_pop, new_lengths = next_generation(dist, pop, lengths, run)
        if len(new_pop) < len(pop):
            break
        pop = new_pop
        lengths = new_lengths
    return run.result(dist)


def memetic_algorithm(dist, budget, seed, params=None):
    """GA + short 2-opt local search on some children (at most n moves)."""
    defaults = {"ls_prob": 0.2, "ls_cap": len(dist)}

    return genetic_algorithm(dist, budget, seed,
                             with_defaults(defaults, params))
