"""E5: how much do the parameters of SA and GA matter? (Java-30)"""

from tsp.annealing import simulated_annealing
from tsp.data import JAVA_CITIES, java_instance
from tsp.genetic import genetic_algorithm

from .common import SEED0, log, save

SA_P0 = (0.8, 0.3, 0.1)
SA_T_END = (1e-1, 1e-2, 1e-3, 1e-4)
GA_POP = (20, 50, 100, 200)
GA_MUT = (0.05, 0.2, 0.5, 0.8)


def sa_grid(dist, budget, runs):
    """SA with every (p0, t_end_ratio) pair."""
    out = []

    for p0 in SA_P0:
        for t_end in SA_T_END:
            params = {"p0": p0, "t_end_ratio": t_end}
            lengths = [simulated_annealing(dist, budget, seed, params).length
                       for seed in range(SEED0, SEED0 + runs)]
            out.append({"p0": p0, "t_end_ratio": t_end, "lengths": lengths})
    return out


def ga_grid(dist, budget, runs):
    """GA with every (population size, mutation probability) pair."""
    out = []

    for pop in GA_POP:
        for mut in GA_MUT:
            params = {"pop_size": pop, "mut_prob": mut, "cx_prob": 0.9,
                      "tournament_k": 3, "elite": 2}
            lengths = [genetic_algorithm(dist, budget, seed, params).length
                       for seed in range(SEED0, SEED0 + runs)]
            out.append({"pop": pop, "mut": mut, "lengths": lengths})
        log(f"   GA pop={pop} done")
    return out


def exp_sensitivity(config):
    """E5 with the normal budget."""
    inst = java_instance(list(JAVA_CITIES), "Java-30")
    budget = config["budget_factor"] * inst.n
    runs = config["sensitivity_runs"]
    out = {}

    log("E5 parameter sensitivity (Java-30)")
    out["SA"] = sa_grid(inst.d, budget, runs)
    log("   SA done")
    out["GA"] = ga_grid(inst.d, budget, runs)
    save(config, "e5_sensitivity.json", out)
