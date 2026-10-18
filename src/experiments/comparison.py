"""E2: every algorithm on every instance, `runs` seeds each.
E3: effect of the evaluation budget on U-50.
"""

import statistics

from tsp.baselines import best_nearest_neighbour
from tsp.core import is_valid_tour, tour_length
from tsp.data import uniform_instance

from .common import ALGOS, SEED0, log, run_algo, save

RS_MAX_RUNS = 10
E3_FACTORS = (100, 1_000, 10_000, 100_000)
E3_ALGOS = ("HC-RR", "SA", "GA")
GA_MAX_FACTOR = 10_000


def run_row(inst, algo, seed, res):
    """One line of e2_runs.json (checks the result first)."""
    assert is_valid_tour(res.tour, inst.n)
    assert abs(res.length - tour_length(inst.d, res.tour)) < 1e-6
    return {"instance": inst.name, "algo": algo, "seed": seed,
            "length": res.length, "evals": res.evals,
            "seconds": res.seconds}


def keep_best_tour(best_tours, key, res):
    """Remember the shortest tour found for `key` = "instance|algo"."""
    if key not in best_tours or res.length < best_tours[key]["length"]:
        best_tours[key] = {"length": res.length, "tour": res.tour}


def run_all_seeds(inst, algo, config, outputs):
    """All the runs of one algorithm on one instance. Random search is a
    weak baseline, so it gets at most RS_MAX_RUNS runs."""
    key = f"{inst.name}|{algo}"
    budget = config["budget_factor"] * inst.n
    n_runs = config["runs"]
    lengths = []

    if algo == "RS":
        n_runs = min(n_runs, RS_MAX_RUNS)
    outputs["history"][key] = []
    for seed in range(SEED0, SEED0 + n_runs):
        res = run_algo(algo, inst.d, budget, seed)
        outputs["rows"].append(run_row(inst, algo, seed, res))
        outputs["history"][key].append(res.history)
        keep_best_tour(outputs["best_tours"], key, res)
        lengths.append(res.length)
    log(f"   {inst.name:8s} {algo:6s} "
        f"median={statistics.median(lengths):10.2f} best={min(lengths):10.2f}")


def exp_main(instances, config):
    """E2: the files are saved after every instance (partial results)."""
    outputs = {"rows": [], "history": {}, "best_tours": {}}

    log("E2 main comparison")
    for inst in instances.values():
        nn = best_nearest_neighbour(inst.d)
        outputs["rows"].append(run_row(inst, "NN", -1, nn))
        keep_best_tour(outputs["best_tours"], f"{inst.name}|NN", nn)
        for algo in ALGOS:
            run_all_seeds(inst, algo, config, outputs)
        save(config, "e2_runs.json", outputs["rows"])
        save(config, "e2_history.json", outputs["history"])
        save(config, "e2_best_tours.json", outputs["best_tours"])


def budget_rows(dist, algo, factor, runs):
    """`runs` runs of `algo` with budget = factor * n."""
    rows = []

    for seed in range(SEED0, SEED0 + runs):
        res = run_algo(algo, dist, factor * len(dist), seed)
        rows.append({"factor": factor, "algo": algo, "seed": seed,
                     "length": res.length, "seconds": res.seconds})
    return rows


def exp_budget(config):
    """E3: the GA is skipped above GA_MAX_FACTOR (too slow in Python)."""
    inst = uniform_instance(50, seed=50)
    rows = []

    log("E3 budget effect (U-50)")
    for factor in E3_FACTORS:
        for algo in E3_ALGOS:
            if algo != "GA" or factor <= GA_MAX_FACTOR:
                rows += budget_rows(inst.d, algo, factor,
                                    config["budget_runs"])
        log(f"   factor {factor} done")
    save(config, "e3_budget.json", rows)
