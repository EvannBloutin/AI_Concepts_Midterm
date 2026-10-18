##
## EPITECH PROJECT, 2026
## AI_Concepts_Midterm
## File description:
## Settings and helpers shared by all the experiments
##

"""Settings and helpers shared by all the experiments.

`config` is a dict built by run_experiments.py with the output folder,
the budget factor and the number of runs of each experiment.
"""

import json
import os
import time

from tsp.annealing import simulated_annealing
from tsp.baselines import random_search
from tsp.genetic import genetic_algorithm, memetic_algorithm
from tsp.hill_climbing import hill_climbing

SEED0 = 1000

SA_PARAMS = {"p0": 0.3, "t_end_ratio": 1e-3}
GA_PARAMS = {"pop_size": 100, "mut_prob": 0.5, "cx_prob": 0.9,
             "tournament_k": 3, "elite": 2}
MA_PARAMS = {"pop_size": 100, "mut_prob": 0.5, "ls_prob": 0.5}

SOLVERS = {
    "RS": random_search,
    "HC-RR": hill_climbing,
    "SA": simulated_annealing,
    "GA": genetic_algorithm,
    "MA": memetic_algorithm,
}
PARAMS = {"SA": SA_PARAMS, "GA": GA_PARAMS, "MA": MA_PARAMS}
ALGOS = list(SOLVERS)


def run_algo(algo, dist, budget, seed):
    """Run the solver called `algo` with its default parameters.
    Run r always uses seed SEED0 + r, for every algorithm."""
    if algo in PARAMS:
        return SOLVERS[algo](dist, budget, seed, PARAMS[algo])
    return SOLVERS[algo](dist, budget, seed)


def log(message):
    """Print a message with the current time."""
    print(time.strftime("%H:%M:%S"), message, flush=True)


def save(config, filename, data):
    """Write `data` as JSON in the output folder."""
    with open(os.path.join(config["out"], filename), "w") as file:
        json.dump(data, file)


def load(config, filename):
    """Read a JSON file of the output folder."""
    with open(os.path.join(config["out"], filename)) as file:
        return json.load(file)


def timed(function, argument):
    """Run function(argument) and return (result, seconds)."""
    start = time.perf_counter()
    result = function(argument)

    return result, time.perf_counter() - start
