"""Unit tests (standard library only).  Run from the project root with:

    python -m unittest discover -s tests -v

They check (1) the building blocks, (2) that the exact methods agree with
each other, and (3) that every heuristic returns a valid tour, reports the
correct length, respects its budget and behaves as the theory says.
"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsp.annealing import simulated_annealing                      # noqa: E402
from tsp.baselines import best_nearest_neighbour, nearest_neighbour, random_search  # noqa: E402
from tsp.core import (apply_two_opt, distance_matrix, euclidean, haversine_km,  # noqa: E402
                      is_valid_tour, local_search_2opt, random_tour, tour_length,
                      two_opt_delta, two_opt_pairs)
from tsp.data import JAVA_12, JAVA_CITIES, get_instances, java_instance, uniform_instance  # noqa: E402
from tsp.exact import astar_tsp, brute_force, held_karp, one_tree_lower_bound  # noqa: E402
from tsp.genetic import (genetic_algorithm, inversion_mutation, memetic_algorithm,  # noqa: E402
                         order_crossover)
from tsp.hill_climbing import hill_climbing                         # noqa: E402

INST = get_instances()
J30 = INST["Java-30"]


class TestBuildingBlocks(unittest.TestCase):
    def test_haversine_known_distance(self):
        # Jakarta - Surabaya great-circle distance is about 660 km
        d = haversine_km(JAVA_CITIES["Jakarta"], JAVA_CITIES["Surabaya"])
        self.assertTrue(650 < d < 675, d)

    def test_distance_matrix_symmetric_zero_diagonal(self):
        inst = uniform_instance(15, seed=1)
        for i in range(inst.n):
            self.assertEqual(inst.d[i][i], 0)
            for j in range(inst.n):
                self.assertAlmostEqual(inst.d[i][j], inst.d[j][i])

    def test_tour_length_square(self):
        d = distance_matrix([(0, 0), (0, 1), (1, 1), (1, 0)], euclidean)
        self.assertAlmostEqual(tour_length(d, [0, 1, 2, 3]), 4.0)
        self.assertAlmostEqual(tour_length(d, [0, 2, 1, 3]), 2 + 2 * 2 ** 0.5)

    def test_instances_have_expected_sizes(self):
        self.assertEqual([INST[k].n for k in ("Java-12", "Java-30", "U-50", "U-100")],
                         [12, 30, 50, 100])
        self.assertEqual(len(set(JAVA_12)), 12)
        self.assertEqual(len(JAVA_CITIES), 30)

    def test_two_opt_delta_matches_recomputation(self):
        for n in (5, 8, 20):
            inst = uniform_instance(n, seed=n)
            tour = random_tour(n, random.Random(0))
            before = tour_length(inst.d, tour)
            for (i, j) in two_opt_pairs(n):
                delta = two_opt_delta(inst.d, tour, i, j)
                t2 = tour[:]
                apply_two_opt(t2, i, j)
                self.assertTrue(is_valid_tour(t2, n))
                self.assertAlmostEqual(tour_length(inst.d, t2) - before, delta, places=7)

    def test_two_opt_neighbourhood_size(self):
        for n in (5, 12, 30):
            self.assertEqual(len(two_opt_pairs(n)), n * (n - 3) // 2)

    def test_crossover_and_mutation_keep_permutation(self):
        rng = random.Random(3)
        for _ in range(300):
            n = rng.randint(4, 40)
            a, b = random_tour(n, rng), random_tour(n, rng)
            c = order_crossover(a, b, rng)
            self.assertTrue(is_valid_tour(c, n))
            inversion_mutation(c, rng)
            self.assertTrue(is_valid_tour(c, n))


class TestExact(unittest.TestCase):
    def test_exact_methods_agree(self):
        for seed in range(5):
            inst = uniform_instance(8, seed=seed)
            bf, _, count = brute_force(inst.d)
            hk, hk_tour = held_karp(inst.d)
            a = astar_tsp(inst.d, use_heuristic=True)
            u = astar_tsp(inst.d, use_heuristic=False)
            self.assertEqual(count, 5040 // 2)               # (n-1)!/2 distinct tours
            self.assertAlmostEqual(hk, bf)
            self.assertAlmostEqual(a["length"], bf)
            self.assertAlmostEqual(u["length"], bf)
            self.assertAlmostEqual(tour_length(inst.d, hk_tour), bf)
            self.assertAlmostEqual(tour_length(inst.d, a["tour"]), bf)
            self.assertTrue(is_valid_tour(a["tour"], 8) and is_valid_tour(u["tour"], 8))
            self.assertLessEqual(a["expanded"], u["expanded"])  # informed search never worse

    def test_lower_bound_is_below_optimum(self):
        for seed in range(5):
            inst = uniform_instance(9, seed=seed)
            opt, _ = held_karp(inst.d)
            self.assertLessEqual(one_tree_lower_bound(inst.d), opt + 1e-9)


class TestHeuristics(unittest.TestCase):
    def check(self, res, inst=J30):
        self.assertTrue(is_valid_tour(res.tour, inst.n))
        self.assertAlmostEqual(res.length, tour_length(inst.d, res.tour), places=6)

    def test_all_solvers_return_valid_tours(self):
        B = 20000
        for res in (hill_climbing(J30.d, B, 1), simulated_annealing(J30.d, B, 1),
                    genetic_algorithm(J30.d, B, 1), memetic_algorithm(J30.d, B, 1),
                    random_search(J30.d, B, 1), nearest_neighbour(J30.d),
                    best_nearest_neighbour(J30.d)):
            self.check(res)

    def test_budget_is_respected(self):
        B = 15000
        self.assertLessEqual(hill_climbing(J30.d, B, 2).evals, B)
        self.assertEqual(simulated_annealing(J30.d, B, 2).evals, B)
        self.assertLessEqual(genetic_algorithm(J30.d, B, 2).evals, B)
        self.assertLessEqual(memetic_algorithm(J30.d, B, 2).evals, B + J30.n)

    def test_single_hill_climb_ends_in_2opt_local_optimum(self):
        res = hill_climbing(J30.d, 10 ** 7, 5, restarts=False)
        for (i, j) in two_opt_pairs(J30.n):
            self.assertGreaterEqual(two_opt_delta(J30.d, res.tour, i, j), -1e-9)

    def test_local_search_never_worsens(self):
        rng = random.Random(7)
        t = random_tour(30, rng)
        L = tour_length(J30.d, t)
        L2, used, _ = local_search_2opt(J30.d, t, L, rng, 5000)
        self.assertLessEqual(L2, L)
        self.assertAlmostEqual(L2, tour_length(J30.d, t), places=6)

    def test_history_is_monotone(self):
        for res in (hill_climbing(J30.d, 20000, 3), simulated_annealing(J30.d, 20000, 3),
                    genetic_algorithm(J30.d, 20000, 3)):
            vals = [v for _, v in res.history]
            self.assertTrue(all(b <= a + 1e-9 for a, b in zip(vals, vals[1:])))

    def test_same_seed_same_result(self):
        a, b = simulated_annealing(J30.d, 20000, 11), simulated_annealing(J30.d, 20000, 11)
        self.assertEqual(a.tour, b.tour)
        c, e = genetic_algorithm(J30.d, 20000, 11), genetic_algorithm(J30.d, 20000, 11)
        self.assertEqual(c.tour, e.tour)

    def test_heuristics_reach_near_optimum_on_small_instance(self):
        inst = java_instance(JAVA_12, "Java-12")
        opt, _ = held_karp(inst.d)
        B = 120000
        rs = random_search(inst.d, 2000, 0).length
        for res in (hill_climbing(inst.d, B, 0), simulated_annealing(inst.d, B, 0),
                    genetic_algorithm(inst.d, B, 0, mut_prob=0.5),
                    memetic_algorithm(inst.d, B, 0, ls_prob=0.5)):
            self.assertGreaterEqual(res.length, opt - 1e-9)   # can never beat the optimum
            self.assertLessEqual(res.length, opt * 1.05)      # within 5 % on 12 cities
            self.assertLessEqual(res.length, rs)              # better than random search


if __name__ == "__main__":
    unittest.main(verbosity=2)
