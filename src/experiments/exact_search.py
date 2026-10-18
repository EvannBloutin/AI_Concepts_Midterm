"""E1: exact search on small instances (brute force, UCS, A*, Held-Karp).
E6: reference ("best known") solution of every instance.
"""

import math

from tsp.annealing import simulated_annealing
from tsp.astar import astar_tsp
from tsp.data import JAVA_12, java_instance
from tsp.exact import brute_force, held_karp, one_tree_lower_bound

from .common import SA_PARAMS, log, save, timed

E1_SIZES = range(6, 13)
MAX_BRUTE_FORCE = 10
REFERENCE_SEED0 = 7000


def add_search_stats(row, prefix, search):
    """Copy the UCS / A* statistics into the row."""
    row[prefix + "_expanded"] = search["expanded"]
    row[prefix + "_seconds"] = search["seconds"]
    row[prefix + "_len"] = search["length"]
    row[prefix + "_max_frontier"] = search["max_frontier"]


def exact_row(n):
    """Solve the first n cities of Java-12 with every exact method."""
    inst = java_instance(JAVA_12[:n], f"Java-{n}")
    row = {"n": n, "tours": math.factorial(n - 1) // 2}

    if n <= MAX_BRUTE_FORCE:
        (_, _, count), seconds = timed(brute_force, inst.d)
        row["bf_seconds"] = seconds
        row["bf_tours"] = count
    (optimum, _), seconds = timed(held_karp, inst.d)
    row["hk_seconds"] = seconds
    row["optimum"] = optimum
    add_search_stats(row, "ucs", astar_tsp(inst.d, use_heuristic=False))
    add_search_stats(row, "astar", astar_tsp(inst.d, use_heuristic=True))
    return row


def exp_exact(config):
    """E1: how far can we go with exact methods?"""
    rows = []

    log("E1 exact search")
    for n in E1_SIZES:
        row = exact_row(n)
        rows.append(row)
        log(f"   n={n} optimum={row['optimum']:.1f} "
            f"UCS={row['ucs_expanded']} A*={row['astar_expanded']}")
    save(config, "e1_exact.json", rows)


def reference_entry(inst, config):
    """Exact optimum for Java-12, else best of long SA runs (10x budget)."""
    lower_bound = one_tree_lower_bound(inst.d)
    budget = 10 * config["budget_factor"] * inst.n
    long_runs = config["long_runs"]

    if inst.name == "Java-12":
        optimum, tour = held_karp(inst.d)
        return {"best_known": optimum, "tour": tour,
                "source": "Held-Karp (exact optimum)",
                "lower_bound": lower_bound, "proven": True}
    runs = [simulated_annealing(inst.d, budget, REFERENCE_SEED0 + r,
                                SA_PARAMS) for r in range(long_runs)]
    best = min(runs, key=lambda res: res.length)
    return {"best_known": best.length, "tour": best.tour,
            "source": f"best of {long_runs} SA runs with 10x budget",
            "lower_bound": lower_bound, "proven": False}


def exp_reference(instances, config):
    """E6: best known solution of every instance (needed by E4)."""
    ref = {}

    log("E6 reference solutions (long SA runs)")
    for name, inst in instances.items():
        ref[name] = reference_entry(inst, config)
        if not ref[name]["proven"]:
            log(f"   {name}: {ref[name]['best_known']:.2f}")
    save(config, "reference.json", ref)
    return ref
