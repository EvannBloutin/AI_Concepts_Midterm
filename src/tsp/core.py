"""Basic TSP tools: distances, tour length and the 2-opt move.

A tour is a list of city indices, e.g. [0, 3, 1, 2]. The tour is closed:
after the last city we go back to the first one.

The 2-opt move (i, j) removes the edges (t[i], t[i+1]) and (t[j], t[j+1])
and reconnects the tour by reversing the part t[i+1..j].
"""

import math

EARTH_RADIUS_KM = 6371.0088
EPS = 1e-12


def haversine_km(p, q):
    """Distance in km between two (latitude, longitude) points."""
    lat1 = math.radians(p[0])
    lon1 = math.radians(p[1])
    lat2 = math.radians(q[0])
    lon2 = math.radians(q[1])
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2)
         * math.sin((lon2 - lon1) / 2) ** 2)

    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def euclidean(p, q):
    """Straight-line distance between two (x, y) points."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


def distance_matrix(points, metric):
    """dist[i][j] = distance between city i and city j."""
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = metric(points[i], points[j])
            dist[j][i] = dist[i][j]
    return dist


def tour_length(dist, tour):
    """Length of the closed tour (k = 0 gives the last -> first edge)."""
    total = 0.0

    for k in range(len(tour)):
        total += dist[tour[k - 1]][tour[k]]
    return total


def is_valid_tour(tour, n):
    """True if the tour visits every city 0..n-1 exactly once."""
    return len(tour) == n and sorted(tour) == list(range(n))


def random_tour(n, rng):
    """The cities 0..n-1 in a random order."""
    tour = list(range(n))

    rng.shuffle(tour)
    return tour


def two_opt_pairs(n):
    """All the 2-opt moves (i, j): there are n(n-3)/2 of them.
    (0, n-1) is skipped: both edges touch t[0], nothing would change."""
    moves = []

    for i in range(n - 1):
        for j in range(i + 2, n):
            if i != 0 or j != n - 1:
                moves.append((i, j))
    return moves


def two_opt_delta(dist, tour, i, j):
    """How much the tour length changes if move (i, j) is applied."""
    a = tour[i]
    b = tour[i + 1]
    c = tour[j]
    e = tour[(j + 1) % len(tour)]

    return dist[a][c] + dist[b][e] - dist[a][b] - dist[c][e]


def apply_two_opt(tour, i, j):
    """Apply move (i, j): reverse the part tour[i+1..j] (in place)."""
    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])


def local_search_2opt(dist, tour, rng, max_evals):
    """Apply improving 2-opt moves (random order) until a local optimum or
    `max_evals` tested moves. Returns (new_length, moves_tested)."""
    moves = two_opt_pairs(len(tour))
    length = tour_length(dist, tour)
    fails = 0
    used = 0

    rng.shuffle(moves)
    while fails < len(moves) and used < max_evals:
        i, j = moves[used % len(moves)]
        used += 1
        delta = two_opt_delta(dist, tour, i, j)
        if delta < -EPS:
            apply_two_opt(tour, i, j)
            length += delta
            fails = 0
        else:
            fails += 1
    return length, used
