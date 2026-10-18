"""E4: hill climbing, single climbs vs random restarts.

p = probability that one climb ends within tau of the best known tour.
Theory: about 1/p restarts are needed. We check it with real restarts.
"""

import statistics

from tsp.hill_climbing import hill_climbing

from .common import log, save

E4_INSTANCES = ("Java-30", "U-50")
TAUS = (0.0, 0.01, 0.02, 0.05)
FIRST_HIT_RUNS = 100
MIN_P = 0.01
NO_LIMIT = 10 ** 9


def single_climb(dist, seed):
    """One hill climb without restart, until a local optimum."""
    return hill_climbing(dist, NO_LIMIT, seed, restarts=False)


def restarts_until_success(dist, target, run):
    """Climb again and again until a tour of length <= target is found.
    Returns (number_of_climbs, total_evaluations)."""
    climbs = 0
    total_evals = 0

    while True:
        res = single_climb(dist, (90_000 + run) * 1000 + climbs)
        climbs += 1
        total_evals += res.evals
        if res.length <= target:
            return climbs, total_evals


def threshold_stats(dist, lengths, target):
    """p for this target, and the real restarts (if p is not too small)."""
    p = sum(1 for length in lengths if length <= target) / len(lengths)
    first_hit = []

    if p >= MIN_P:
        first_hit = [restarts_until_success(dist, target, run)
                     for run in range(FIRST_HIT_RUNS)]
    return {"p": p, "first_hit": first_hit}


def hc_entry(inst, best, n_climbs):
    """All the E4 results for one instance."""
    climbs = [single_climb(inst.d, 50_000 + r) for r in range(n_climbs)]
    lengths = [res.length for res in climbs]
    evals = [res.evals for res in climbs]
    entry = {"climbs": n_climbs, "lengths": lengths, "evals": evals,
             "best_known": best,
             "mean_evals_per_climb": statistics.mean(evals),
             "thresholds": {}}

    for tau in TAUS:
        target = best * (1 + tau) + 1e-9
        entry["thresholds"][str(tau)] = threshold_stats(inst.d, lengths,
                                                        target)
    return entry


def exp_hc_restarts(instances, config, ref):
    """E4 on Java-30 and U-50 (ref = reference solutions from E6)."""
    out = {}

    log("E4 single hill climbs vs random restarts")
    for name in E4_INSTANCES:
        entry = hc_entry(instances[name], ref[name]["best_known"],
                         config["climbs"])
        out[name] = entry
        log(f"   {name}: " + ", ".join(
            f"p(<={tau}) = {stats['p']:.3f}"
            for tau, stats in entry["thresholds"].items()))
    save(config, "e4_hc.json", out)
