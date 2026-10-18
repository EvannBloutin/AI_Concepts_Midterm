##
## EPITECH PROJECT, 2026
## AI_Concepts_Midterm
## File description:
## Unit tests of the heuristics (HC, SA, GA, MA, baselines)
##

"""Unit tests of the heuristics. Run from the project root with:

    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from tsp.annealing import simulated_annealing  # noqa: E402
from tsp.baselines import (best_nearest_neighbour,  # noqa: E402
                           nearest_neighbour, random_search)
from tsp.core import (is_valid_tour, local_search_2opt,  # noqa: E402
                      random_tour, tour_length, two_opt_delta,
                      two_opt_pairs)
from tsp.data import JAVA_12, get_instances, java_instance  # noqa: E402
from tsp.exact import held_karp  # noqa: E402
from tsp.genetic import genetic_algorithm, memetic_algorithm  # noqa: E402
from tsp.hill_climbing import hill_climbing  # noqa: E402

J30 = get_instances()["Java-30"]


class TestHeuristics(unittest.TestCase):
    """Valid tours, budget, local optimum, reproducibility, quality."""

    def check(self, res):
        """The tour is valid and its reported length is correct."""
        self.assertTrue(is_valid_tour(res.tour, J30.n))
        self.assertAlmostEqual(res.length, tour_length(J30.d, res.tour),
                               places=6)

    def test_all_solvers_return_valid_tours(self):
        """Every solver returns a valid tour with the right length."""
        budget = 20000

        for solver in (hill_climbing, simulated_annealing, genetic_algorithm,
                       memetic_algorithm, random_search):
            self.check(solver(J30.d, budget, 1))
        self.check(nearest_neighbour(J30.d))
        self.check(best_nearest_neighbour(J30.d))

    def test_budget_is_respected(self):
        """MA may go over by at most one local search (n moves)."""
        budget = 15000

        self.assertLessEqual(hill_climbing(J30.d, budget, 2).evals, budget)
        self.assertEqual(simulated_annealing(J30.d, budget, 2).evals, budget)
        self.assertLessEqual(genetic_algorithm(J30.d, budget, 2).evals,
                             budget)
        self.assertLessEqual(memetic_algorithm(J30.d, budget, 2).evals,
                             budget + J30.n)

    def test_single_hill_climb_ends_in_2opt_local_optimum(self):
        """After one climb, no 2-opt move can shorten the tour."""
        res = hill_climbing(J30.d, 10 ** 7, 5, restarts=False)

        for i, j in two_opt_pairs(J30.n):
            self.assertGreaterEqual(two_opt_delta(J30.d, res.tour, i, j),
                                    -1e-9)

    def test_local_search_never_worsens(self):
        """Local search returns a shorter (or equal) tour, exact length."""
        rng = random.Random(7)
        tour = random_tour(30, rng)
        before = tour_length(J30.d, tour)
        after, _ = local_search_2opt(J30.d, tour, rng, 5000)

        self.assertLessEqual(after, before)
        self.assertAlmostEqual(after, tour_length(J30.d, tour), places=6)

    def test_history_is_monotone(self):
        """The best length so far can only go down."""
        for solver in (hill_climbing, simulated_annealing,
                       genetic_algorithm):
            values = [best for _, best in solver(J30.d, 20000, 3).history]
            self.assertTrue(all(b <= a + 1e-9
                                for a, b in zip(values, values[1:])))

    def test_same_seed_same_result(self):
        """Same seed = same tour."""
        for solver in (simulated_annealing, genetic_algorithm):
            first = solver(J30.d, 20000, 11)
            second = solver(J30.d, 20000, 11)
            self.assertEqual(first.tour, second.tour)

    def test_heuristics_reach_near_optimum_on_small_instance(self):
        """On 12 cities: never below the optimum, within 5 % of it and
        better than random search."""
        inst = java_instance(JAVA_12, "Java-12")
        optimum, _ = held_karp(inst.d)
        budget = 120000
        random_len = random_search(inst.d, 2000, 0).length

        for res in (hill_climbing(inst.d, budget, 0),
                    simulated_annealing(inst.d, budget, 0),
                    genetic_algorithm(inst.d, budget, 0, {"mut_prob": 0.5}),
                    memetic_algorithm(inst.d, budget, 0, {"ls_prob": 0.5})):
            self.assertGreaterEqual(res.length, optimum - 1e-9)
            self.assertLessEqual(res.length, optimum * 1.05)
            self.assertLessEqual(res.length, random_len)


if __name__ == "__main__":
    unittest.main(verbosity=2)
