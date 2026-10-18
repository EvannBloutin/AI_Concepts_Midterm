"""Exact methods (only for small instances) and a lower bound.

* brute_force          : try all (n-1)!/2 tours.
* held_karp            : dynamic programming over subsets of cities.
* one_tree_lower_bound : the optimal tour can never be shorter than this.
(UCS and A* are in astar.py.)

Subsets of cities are stored as bitmasks: city c is in the set if bit c
is 1. Example: 0b1010 = {1, 3}.
"""

import itertools

from .core import tour_length


def brute_force(dist):
    """Returns (best_length, best_tour, number_of_tours_tried).
    A tour and its reverse are the same: perm[0] > perm[-1] is skipped."""
    best_len = float("inf")
    best_tour = None
    count = 0

    for perm in itertools.permutations(range(1, len(dist))):
        if perm[0] <= perm[-1]:
            count += 1
            tour = [0] + list(perm)
            length = tour_length(dist, tour)
            if length < best_len:
                best_len = length
                best_tour = tour
    return best_len, best_tour, count


def extend_path(dist, cost, parent, state):
    """Try to go from state = (visited, last) to every unvisited city."""
    visited, last = state

    for nxt in range(1, len(dist)):
        if visited & (1 << nxt):
            continue
        new_state = (visited | (1 << nxt), nxt)
        new_cost = cost[state] + dist[last][nxt]
        if new_state not in cost or new_cost < cost[new_state]:
            cost[new_state] = new_cost
            parent[new_state] = last


def held_karp_table(dist):
    """cost[(visited, last)] = shortest path that starts at city 0, visits
    exactly the cities of `visited` (0 not included) and ends at `last`.
    A subset is a smaller number than the sets containing it, so the masks
    are processed in increasing order (even masks: city 0 not inside)."""
    n = len(dist)
    cost = {(1 << c, c): dist[0][c] for c in range(1, n)}
    parent = {}

    for visited in range(2, 1 << n, 2):
        for last in range(1, n):
            if (visited, last) in cost:
                extend_path(dist, cost, parent, (visited, last))
    return cost, parent


def held_karp_tour(parent, last, n):
    """Follow the parents backwards from `last` to rebuild the tour."""
    visited = (1 << n) - 2
    tour = []

    while last is not None:
        tour.append(last)
        previous = parent.get((visited, last))
        visited -= 1 << last
        last = previous
    tour.append(0)
    tour.reverse()
    return tour


def held_karp(dist):
    """Returns (best_length, best_tour): best path + way back to city 0."""
    n = len(dist)
    all_cities = (1 << n) - 2
    cost, parent = held_karp_table(dist)
    best_len, last = min((cost[(all_cities, c)] + dist[c][0], c)
                         for c in range(1, n))

    return best_len, held_karp_tour(parent, last, n)


def mst_cost(dist, nodes):
    """Cost of the minimum spanning tree of `nodes` (Prim's algorithm).
    to_tree[v] = cheapest edge between the tree and v."""
    to_tree = {v: dist[nodes[0]][v] for v in nodes[1:]}
    total = 0.0

    while to_tree:
        v = min(to_tree, key=to_tree.get)
        total += to_tree.pop(v)
        for u in to_tree:
            to_tree[u] = min(to_tree[u], dist[v][u])
    return total


def one_tree_lower_bound(dist):
    """MST of cities 1..n-1 + the two shortest edges at city 0."""
    n = len(dist)
    two_shortest = sorted(dist[0][j] for j in range(1, n))[:2]
    mst = mst_cost(dist, list(range(1, n)))

    return mst + two_shortest[0] + two_shortest[1]
