"""Unit tests of the exact methods. Run from the project root with:

    python -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from tsp.astar import astar_tsp  # noqa: E402
from tsp.core import is_valid_tour, tour_length  # noqa: E402
from tsp.data import uniform_instance  # noqa: E402
from tsp.exact import (brute_force, held_karp,  # noqa: E402
                       one_tree_lower_bound)


class TestExact(unittest.TestCase):
    """Brute force, Held-Karp, UCS and A* must find the same optimum."""

    def test_exact_methods_agree(self):
        """Same optimum, valid tours, and A* expands fewer states."""
        for seed in range(5):
            inst = uniform_instance(8, seed=seed)
            best, _, count = brute_force(inst.d)
            hk_len, hk_tour = held_karp(inst.d)
            astar = astar_tsp(inst.d, use_heuristic=True)
            ucs = astar_tsp(inst.d, use_heuristic=False)
            self.assertEqual(count, math.factorial(7) // 2)
            self.assertAlmostEqual(hk_len, best)
            self.assertAlmostEqual(astar["length"], best)
            self.assertAlmostEqual(ucs["length"], best)
            self.assertAlmostEqual(tour_length(inst.d, hk_tour), best)
            self.assertAlmostEqual(tour_length(inst.d, astar["tour"]), best)
            self.assertTrue(is_valid_tour(astar["tour"], 8))
            self.assertTrue(is_valid_tour(ucs["tour"], 8))
            self.assertLessEqual(astar["expanded"], ucs["expanded"])

    def test_lower_bound_is_below_optimum(self):
        """The 1-tree bound is never above the optimum."""
        for seed in range(5):
            inst = uniform_instance(9, seed=seed)
            optimum, _ = held_karp(inst.d)
            self.assertLessEqual(one_tree_lower_bound(inst.d),
                                 optimum + 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
