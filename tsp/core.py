"""Core TSP utilities: distances, tour length and the 2-opt move.

A *state* is a permutation of the cities (a closed tour).
The *successor function* of local search is the 2-opt move: choose two
edges, remove them and reconnect the tour by reversing the segment
between them.  This is the TSP analogue of "move one queen inside its
column" in the 8-queens example of the local-search lecture.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0088

Point = Tuple[float, float]
Matrix = List[List[float]]


def haversine_km(p: Point, q: Point) -> float:
    """Great-circle distance in km between two (lat, lon) points in degrees."""
    lat1, lon1 = map(math.radians, p)
    lat2, lon2 = map(math.radians, q)
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def euclidean(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def distance_matrix(points: Sequence[Point],
                    metric: Callable[[Point, Point], float]) -> Matrix:
    """Full symmetric distance matrix as a list of lists (fast scalar access)."""
    n = len(points)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d[i][j] = d[j][i] = metric(points[i], points[j])
    return d


def tour_length(d: Matrix, tour: Sequence[int]) -> float:
    """Length of the closed tour (includes the edge back to the start)."""
    total = 0.0
    prev = tour[-1]
    for c in tour:
        total += d[prev][c]
        prev = c
    return total


def is_valid_tour(tour: Sequence[int], n: int) -> bool:
    return len(tour) == n and sorted(tour) == list(range(n))


def random_tour(n: int, rng: random.Random) -> List[int]:
    t = list(range(n))
    rng.shuffle(t)
    return t


# 2-opt move
def two_opt_pairs(n: int) -> List[Tuple[int, int]]:
    """All distinct 2-opt moves (i, j) with i < j-1.

    Move (i, j) removes edges (t[i], t[i+1]) and (t[j], t[j+1 mod n]) and
    reverses the segment t[i+1..j].  The pair (0, n-1) is excluded because
    the two edges would share the city t[0] (the move changes nothing).
    The neighbourhood size is n(n-3)/2.
    """
    return [(i, j) for i in range(n - 1) for j in range(i + 2, n)
            if not (i == 0 and j == n - 1)]


def two_opt_delta(d: Matrix, t: Sequence[int], i: int, j: int) -> float:
    """Change of tour length if move (i, j) were applied (O(1))."""
    n = len(t)
    a, b = t[i], t[i + 1]
    c, e = t[j], t[(j + 1) % n]
    return d[a][c] + d[b][e] - d[a][b] - d[c][e]


def apply_two_opt(t: List[int], i: int, j: int) -> None:
    """Apply move (i, j) in place."""
    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]


def local_search_2opt(d: Matrix, tour: List[int], cur_len: float,
                      rng: random.Random, max_evals: int):
    """First-choice hill climbing with the 2-opt neighbourhood (in place).

    Neighbours are scanned cyclically in a random order; the first improving
    move is applied.  The search stops at a true 2-opt local optimum (a whole
    cycle of N neighbours without improvement) or when `max_evals` neighbour
    evaluations have been used.

    Returns (new_length, evaluations_used, reached_local_optimum).
    """
    n = len(tour)
    order = two_opt_pairs(n)
    rng.shuffle(order)
    size = len(order)
    k = 0
    fails = 0
    used = 0
    while fails < size and used < max_evals:
        i, j = order[k]
        k += 1
        if k == size:
            k = 0
        used += 1
        delta = two_opt_delta(d, tour, i, j)
        if delta < -1e-12:
            apply_two_opt(tour, i, j)
            cur_len += delta
            fails = 0
        else:
            fails += 1
    return cur_len, used, fails >= size
