"""Uniform-cost search (UCS) and A* for the TSP.

State = (visited cities as a bitmask, current city). We start at city 0.
Goal  = every city visited and back at city 0.
UCS is A* with h = 0. A* uses the minimum spanning tree heuristic.
"""

import heapq
import itertools
import time

from .exact import mst_cost

START = (1, 0)


def goal_state(n):
    """All the cities are visited and we are back at city 0."""
    return ((1 << n) - 1, 0)


def mst_heuristic(dist, state):
    """Admissible estimate of the remaining cost: MST of the unvisited
    cities, city 0 and the current city."""
    visited, last = state
    unvisited = [c for c in range(len(dist)) if not visited & (1 << c)]
    nodes = unvisited + [0]

    if not unvisited:
        return dist[last][0]
    if last != 0:
        nodes.append(last)
    return mst_cost(dist, nodes)


def successors(dist, state):
    """Go to an unvisited city, or go home once all are visited."""
    n = len(dist)
    visited, _ = state

    if visited == (1 << n) - 1:
        return [goal_state(n)]
    return [(visited | (1 << c), c) for c in range(n)
            if not visited & (1 << c)]


def rebuild_tour(parent, goal):
    """Follow the parents backwards from the goal to the start."""
    tour = []
    state = parent[goal]

    while state is not None:
        tour.append(state[1])
        state = parent[state]
    tour.reverse()
    return tour


class AStar:
    """Graph search with a priority queue ordered by f = g + h."""

    def __init__(self, dist, use_heuristic):
        """With use_heuristic=False, h = 0 and this is UCS."""
        self.dist = dist
        self.goal = goal_state(len(dist))
        self.h_values = {} if use_heuristic else None
        self.best_g = {START: 0.0}
        self.parent = {START: None}
        self.counter = itertools.count()
        self.frontier = []
        self.closed = set()
        self.stats = {"expanded": 0, "generated": 0, "max_frontier": 1}
        self.push(START, 0.0)

    def h(self, state):
        """Heuristic value, computed only once per state."""
        if self.h_values is None or state == self.goal:
            return 0.0
        if state not in self.h_values:
            self.h_values[state] = mst_heuristic(self.dist, state)
        return self.h_values[state]

    def push(self, state, g):
        """Add a state to the frontier (the counter breaks the ties)."""
        entry = (g + self.h(state), next(self.counter), g, state)

        heapq.heappush(self.frontier, entry)

    def expand(self, state, g):
        """Add to the frontier every successor reached by a shorter path."""
        self.closed.add(state)
        self.stats["expanded"] += 1
        for new_state in successors(self.dist, state):
            new_g = g + self.dist[state[1]][new_state[1]]
            old_g = self.best_g.get(new_state, float("inf"))
            if new_g < old_g - 1e-12:
                self.best_g[new_state] = new_g
                self.parent[new_state] = state
                self.push(new_state, new_g)
                self.stats["generated"] += 1
        self.stats["max_frontier"] = max(self.stats["max_frontier"],
                                         len(self.frontier))

    def solve(self):
        """Pop the most promising state until the goal is reached.
        Returns the length of the optimal tour."""
        while self.frontier:
            _, _, g, state = heapq.heappop(self.frontier)
            if state == self.goal:
                return g
            if state not in self.closed:
                self.expand(state, g)
        raise RuntimeError("no solution found")


def astar_tsp(dist, use_heuristic=True):
    """UCS (use_heuristic=False) or A* (use_heuristic=True).
    Returns a dict: length, tour, expanded, generated, max_frontier,
    seconds."""
    start_time = time.perf_counter()
    search = AStar(dist, use_heuristic)
    length = search.solve()
    output = dict(search.stats)

    output["length"] = length
    output["tour"] = rebuild_tour(search.parent, search.goal)
    output["seconds"] = time.perf_counter() - start_time
    return output
