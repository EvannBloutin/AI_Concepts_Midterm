"""Run every experiment of the report and store raw results in results/.

Usage:   python run_experiments.py            (full run, ~15-25 min)
         python run_experiments.py --quick    (smoke test, ~1 min)

Experiments
  E1  exact search on small instances      (brute force / UCS / A* / Held-Karp)
  E2  main comparison on 4 instances       (RS, HC-RR, SA, GA, MA + NN baseline)
  E3  effect of the evaluation budget      (HC-RR, SA, GA on U-50)
  E4  hill climbing: single climbs vs. random restarts (lecture analysis)
  E5  parameter sensitivity                (SA schedule, GA population/mutation)
  E6  reference ("best known") solutions   (very long SA runs)
All runs are seeded, so the results can be reproduced exactly.
"""
import argparse
import json
import os
import random
import statistics as st
import time

from tsp.annealing import simulated_annealing
from tsp.baselines import best_nearest_neighbour, random_search
from tsp.core import is_valid_tour, tour_length
from tsp.data import get_instances, java_instance, JAVA_CITIES, uniform_instance
from tsp.exact import (astar_tsp, brute_force, held_karp, one_tree_lower_bound)
from tsp.genetic import genetic_algorithm, memetic_algorithm
from tsp.hill_climbing import hill_climbing

OUT = "results"
BUDGET_FACTOR = 10_000            # evaluations per city
SEED0 = 1000                      # run r uses seed SEED0 + r (same for every algorithm)

# default parameters, chosen on an independent pilot instance (uniform, n=50, seed 999)
PARAMS = {
    "SA": dict(p0=0.3, t_end_ratio=1e-3),
    "GA": dict(pop_size=100, mut_prob=0.5, cx_prob=0.9, tournament_k=3, elite=2),
    "MA": dict(pop_size=100, mut_prob=0.5, ls_prob=0.5),
}
ALGOS = {
    "RS": lambda d, B, s: random_search(d, B, s),
    "HC-RR": lambda d, B, s: hill_climbing(d, B, s),
    "SA": lambda d, B, s: simulated_annealing(d, B, s, **PARAMS["SA"]),
    "GA": lambda d, B, s: genetic_algorithm(d, B, s, **PARAMS["GA"]),
    "MA": lambda d, B, s: memetic_algorithm(d, B, s, **PARAMS["MA"]),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f)


# E1 exact
def exp_exact():
    log("E1 exact search")
    rows = []
    for n in range(6, 13):
        cities = ["Jakarta", "Serang", "Bandung", "Cirebon", "Purwokerto", "Semarang",
                  "Yogyakarta", "Solo", "Madiun", "Surabaya", "Malang", "Banyuwangi"][:n]
        inst = java_instance(cities, f"Java-{n}")
        row = {"n": n, "tours": _fact(n - 1) // 2}
        if n <= 10:
            t = time.perf_counter()
            opt, _, cnt = brute_force(inst.d)
            row["bf_seconds"] = time.perf_counter() - t
            row["bf_tours"] = cnt
        t = time.perf_counter()
        hk, _ = held_karp(inst.d)
        row["hk_seconds"] = time.perf_counter() - t
        u = astar_tsp(inst.d, use_heuristic=False)
        a = astar_tsp(inst.d, use_heuristic=True)
        row.update(optimum=hk, ucs_expanded=u["expanded"], ucs_seconds=u["seconds"],
                   astar_expanded=a["expanded"], astar_seconds=a["seconds"],
                   ucs_len=u["length"], astar_len=a["length"],
                   ucs_max_frontier=u["max_frontier"], astar_max_frontier=a["max_frontier"])
        rows.append(row)
        log(f"   n={n} optimum={hk:.1f} UCS={u['expanded']} A*={a['expanded']}")
    save("e1_exact.json", rows)


def _fact(k):
    r = 1
    for i in range(2, k + 1):
        r *= i
    return r


# E6 reference
def exp_reference(instances, runs_long):
    log("E6 reference solutions (long SA runs)")
    ref = {}
    for name, inst in instances.items():
        if name == "Java-12":
            opt, tour = held_karp(inst.d)
            ref[name] = {"best_known": opt, "tour": tour, "source": "Held-Karp (exact optimum)",
                         "lower_bound": one_tree_lower_bound(inst.d), "proven": True}
            continue
        B = 100 * BUDGET_FACTOR * inst.n // 10      # 10x the standard budget
        best, tour = float("inf"), None
        for r in range(runs_long):
            res = simulated_annealing(inst.d, B, 7000 + r, **PARAMS["SA"])
            if res.length < best:
                best, tour = res.length, res.tour
        ref[name] = {"best_known": best, "tour": tour,
                     "source": f"best of {runs_long} SA runs with 10x budget",
                     "lower_bound": one_tree_lower_bound(inst.d), "proven": False}
        log(f"   {name}: {best:.2f}")
    save("reference.json", ref)
    return ref


# E2 main
def exp_main(instances, runs):
    log("E2 main comparison")
    paths = [os.path.join(OUT, f) for f in ("e2_runs.json", "e2_history.json", "e2_best_tours.json")]
    if all(os.path.exists(p) for p in paths):          # resume after an interruption
        rows, hist, tours = (json.load(open(p)) for p in paths)
    else:
        rows, hist, tours = [], {}, {}
    done = {r["instance"] for r in rows if r["algo"] == "MA"}
    for name, inst in instances.items():
        if len(done) and name in done and sum(1 for r in rows if r["instance"] == name and r["algo"] == "MA") >= runs:
            continue
        rows = [r for r in rows if r["instance"] != name]
        d, n = inst.d, inst.n
        B = BUDGET_FACTOR * n
        nn = best_nearest_neighbour(d)
        rows.append(dict(instance=name, algo="NN", seed=-1, length=nn.length,
                         evals=nn.evals, seconds=nn.seconds))
        tours[f"{name}|NN"] = {"length": nn.length, "tour": nn.tour}
        for algo, fn in ALGOS.items():
            hist[f"{name}|{algo}"] = []
            n_runs = min(runs, 10) if algo == "RS" else runs   # RS is only a floor baseline
            for r in range(n_runs):
                res = fn(d, B, SEED0 + r)
                assert is_valid_tour(res.tour, n)
                assert abs(res.length - tour_length(d, res.tour)) < 1e-6
                rows.append(dict(instance=name, algo=algo, seed=SEED0 + r, length=res.length,
                                 evals=res.evals, seconds=res.seconds))
                hist[f"{name}|{algo}"].append(res.history)
                key = f"{name}|{algo}"
                if key not in tours or res.length < tours[key]["length"]:
                    tours[key] = {"length": res.length, "tour": res.tour}
            lens = [x["length"] for x in rows if x["instance"] == name and x["algo"] == algo]
            log(f"   {name:8s} {algo:6s} median={st.median(lens):10.2f} best={min(lens):10.2f}")
        save("e2_runs.json", rows)                      # checkpoint after every instance
        save("e2_history.json", hist)
        save("e2_best_tours.json", tours)


# E3 budget effect
def exp_budget(runs):
    log("E3 budget effect (U-50)")
    inst = uniform_instance(50, seed=50)
    rows = []
    for factor in (100, 1_000, 10_000, 100_000):
        B = factor * inst.n
        for algo in ("HC-RR", "SA", "GA"):
            if algo == "GA" and factor > 10_000:
                continue                              # too slow in pure Python
            for r in range(runs):
                res = ALGOS[algo](inst.d, B, SEED0 + r)
                rows.append(dict(factor=factor, algo=algo, seed=SEED0 + r,
                                 length=res.length, seconds=res.seconds))
        log(f"   factor {factor} done")
    save("e3_budget.json", rows)


# E4 hill climbing
def exp_hc_restarts(instances, climbs, ref):
    """Lecture analysis: p = P(one climb succeeds); expected restarts = 1/p."""
    log("E4 single hill climbs vs random restarts")
    out = {}
    for name in ("Java-30", "U-50"):
        inst = instances[name]
        best = ref[name]["best_known"]
        lens, evs = [], []
        for r in range(climbs):
            res = hill_climbing(inst.d, 10 ** 9, 50_000 + r, restarts=False)
            lens.append(res.length)
            evs.append(res.evals)
        entry = dict(climbs=climbs, lengths=lens, evals=evs, best_known=best,
                     mean_evals_per_climb=st.mean(evs), thresholds={})
        for tau in (0.0, 0.01, 0.02, 0.05):
            ok = [L <= best * (1 + tau) + 1e-9 for L in lens]
            p = sum(ok) / len(ok)
            first_hit = []
            if p >= 0.01:                  # empirical check: restarts until first success
                for r in range(100):
                    total, restarts = 0, 0
                    while True:
                        res = hill_climbing(inst.d, 10 ** 9, (90_000 + r) * 1000 + restarts,
                                            restarts=False)
                        restarts += 1
                        total += res.evals
                        if res.length <= best * (1 + tau) + 1e-9:
                            break
                    first_hit.append((restarts, total))
            entry["thresholds"][str(tau)] = dict(p=p, first_hit=first_hit)
        out[name] = entry
        log(f"   {name}: " + ", ".join(f"p(<={t}) = {v['p']:.3f}" for t, v in entry["thresholds"].items()))
    save("e4_hc.json", out)


# E5 sensitivity
def exp_sensitivity(runs):
    log("E5 parameter sensitivity (Java-30)")
    inst = java_instance(list(JAVA_CITIES), "Java-30")
    B = BUDGET_FACTOR * inst.n
    out = {"SA": [], "GA": []}
    for p0 in (0.8, 0.3, 0.1):
        for te in (1e-1, 1e-2, 1e-3, 1e-4):
            lens = [simulated_annealing(inst.d, B, SEED0 + r, p0=p0, t_end_ratio=te).length
                    for r in range(runs)]
            out["SA"].append(dict(p0=p0, t_end_ratio=te, lengths=lens))
    log("   SA done")
    for pop in (20, 50, 100, 200):
        for mut in (0.05, 0.2, 0.5, 0.8):
            lens = [genetic_algorithm(inst.d, B, SEED0 + r, pop_size=pop, mut_prob=mut,
                                      cx_prob=0.9, tournament_k=3, elite=2).length
                    for r in range(runs)]
            out["GA"].append(dict(pop=pop, mut=mut, lengths=lens))
        log(f"   GA pop={pop} done")
    save("e5_sensitivity.json", out)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.quick:
        OUT = "results_quick"
    os.makedirs(OUT, exist_ok=True)
    runs = 3 if args.quick else args.runs
    global BUDGET_FACTOR
    if args.quick:
        BUDGET_FACTOR = 500
    instances = get_instances()
    only = set(args.only.split(",")) if args.only else {"E1", "E6", "E2", "E3", "E4", "E5"}
    t0 = time.time()
    if "E1" in only:
        exp_exact()
    if "E6" in only:
        ref = exp_reference(instances, runs_long=2 if args.quick else 5)
    else:
        ref = json.load(open(os.path.join(OUT, "reference.json")))
    if "E2" in only:
        exp_main(instances, runs)
    if "E4" in only:
        exp_hc_restarts(instances, 50 if args.quick else 1000, ref)
    if "E3" in only:
        exp_budget(3 if args.quick else 20)
    if "E5" in only:
        exp_sensitivity(2 if args.quick else 10)
    log(f"finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
