##
## EPITECH PROJECT, 2026
## AI_Concepts_Midterm
## File description:
## Unit tests of the building blocks (distances, 2-opt, GA operators)
##

"""Unit tests of the building blocks. Run from the project root with:

    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from tsp.core import (apply_two_opt, distance_matrix,  # noqa: E402
                      euclidean, haversine_km, is_valid_tour, random_tour,
                      tour_length, two_opt_delta, two_opt_pairs)
from tsp.data import (JAVA_12, JAVA_CITIES, get_instances,  # noqa: E402
                      uniform_instance)
from tsp.genetic import inversion_mutation, order_crossover  # noqa: E402

SQUARE = [(0, 0), (0, 1), (1, 1), (1, 0)]


class TestBuildingBlocks(unittest.TestCase):
    """Distances, tour length, 2-opt move and GA operators."""

    def test_haversine_known_distance(self):
        """Jakarta - Surabaya great-circle distance is about 660 km."""
        km = haversine_km(JAVA_CITIES["Jakarta"], JAVA_CITIES["Surabaya"])

        self.assertTrue(650 < km < 675, km)

    def test_distance_matrix_symmetric_zero_diagonal(self):
        """d[i][i] = 0 and d[i][j] = d[j][i]."""
        inst = uniform_instance(15, seed=1)

        for i in range(inst.n):
            self.assertEqual(inst.d[i][i], 0)
            for j in range(inst.n):
                self.assertAlmostEqual(inst.d[i][j], inst.d[j][i])

    def test_tour_length_square(self):
        """Around the unit square = 4, with two diagonals = 2 + 2 sqrt(2)."""
        dist = distance_matrix(SQUARE, euclidean)

        self.assertAlmostEqual(tour_length(dist, [0, 1, 2, 3]), 4.0)
        self.assertAlmostEqual(tour_length(dist, [0, 2, 1, 3]),
                               2 + 2 * 2 ** 0.5)

    def test_instances_have_expected_sizes(self):
        """The four instances have 12, 30, 50 and 100 cities."""
        instances = get_instances()
        names = ("Java-12", "Java-30", "U-50", "U-100")

        self.assertEqual([instances[name].n for name in names],
                         [12, 30, 50, 100])
        self.assertEqual(len(set(JAVA_12)), 12)
        self.assertEqual(len(JAVA_CITIES), 30)

    def test_two_opt_delta_matches_recomputation(self):
        """The O(1) delta equals the difference of the full lengths."""
        for n in (5, 8, 20):
            inst = uniform_instance(n, seed=n)
            tour = random_tour(n, random.Random(0))
            before = tour_length(inst.d, tour)
            for i, j in two_opt_pairs(n):
                delta = two_opt_delta(inst.d, tour, i, j)
                moved = tour[:]
                apply_two_opt(moved, i, j)
                self.assertTrue(is_valid_tour(moved, n))
                self.assertAlmostEqual(tour_length(inst.d, moved) - before,
                                       delta, places=7)

    def test_two_opt_neighbourhood_size(self):
        """There are n(n-3)/2 different 2-opt moves."""
        for n in (5, 12, 30):
            self.assertEqual(len(two_opt_pairs(n)), n * (n - 3) // 2)

    def test_crossover_and_mutation_keep_permutation(self):
        """Children and mutants are still valid tours."""
        rng = random.Random(3)

        for _ in range(300):
            n = rng.randint(4, 40)
            parent1 = random_tour(n, rng)
            parent2 = random_tour(n, rng)
            child = order_crossover(parent1, parent2, rng)
            self.assertTrue(is_valid_tour(child, n))
            inversion_mutation(child, rng)
            self.assertTrue(is_valid_tour(child, n))


if __name__ == "__main__":
    unittest.main(verbosity=2)
