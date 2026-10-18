#!/usr/bin/env python3
##
## EPITECH PROJECT, 2026
## AI_Concepts_Midterm
## File description:
## Run every experiment of the report and save the results
##

"""Run every experiment of the report and save the raw results.

Usage (from the project root):
    python src/run_experiments.py              (full run -> results/)
    python src/run_experiments.py --quick      (smoke test -> results_quick/)
    python src/run_experiments.py --only E2,E3

Experiments
  E1  exact search on small instances      (brute force / UCS / A* / HK)
  E2  main comparison on 4 instances       (RS, HC-RR, SA, GA, MA + NN)
  E3  effect of the evaluation budget      (HC-RR, SA, GA on U-50)
  E4  hill climbing: single climbs vs random restarts
  E5  parameter sensitivity                (SA schedule, GA pop/mutation)
  E6  reference ("best known") solutions   (long SA runs)
All runs are seeded, so the results can be reproduced exactly.
"""

import argparse
import os
import time

from experiments.common import load, log
from experiments.comparison import exp_budget, exp_main
from experiments.exact_search import exp_exact, exp_reference
from experiments.restarts import exp_hc_restarts
from experiments.sensitivity import exp_sensitivity
from tsp.data import get_instances

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FULL_CONFIG = {
    "out": os.path.join(ROOT, "results"),
    "budget_factor": 10_000,
    "runs": 30,
    "long_runs": 5,
    "climbs": 1000,
    "budget_runs": 20,
    "sensitivity_runs": 10,
}
QUICK_CONFIG = {
    "out": os.path.join(ROOT, "results_quick"),
    "budget_factor": 500,
    "runs": 3,
    "long_runs": 2,
    "climbs": 50,
    "budget_runs": 3,
    "sensitivity_runs": 2,
}


def parse_args():
    """Read the command line options."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--quick", action="store_true",
                        help="small smoke test")
    parser.add_argument("--runs", type=int, default=FULL_CONFIG["runs"],
                        help="runs per algorithm in E2")
    parser.add_argument("--only", default="E1,E6,E2,E3,E4,E5",
                        help="comma-separated list of experiments")
    return parser.parse_args()


def make_config(args):
    """Settings of the run (--runs is ignored with --quick)."""
    if args.quick:
        return dict(QUICK_CONFIG)
    config = dict(FULL_CONFIG)
    config["runs"] = args.runs
    return config


def run_selected(only, config):
    """Run the experiments listed in `only`. E6 runs before E2 / E4
    because E4 needs the reference solutions."""
    instances = get_instances()

    if "E1" in only:
        exp_exact(config)
    if "E6" in only:
        ref = exp_reference(instances, config)
    else:
        ref = load(config, "reference.json")
    if "E2" in only:
        exp_main(instances, config)
    if "E4" in only:
        exp_hc_restarts(instances, config, ref)
    if "E3" in only:
        exp_budget(config)
    if "E5" in only:
        exp_sensitivity(config)


def main():
    """Entry point."""
    args = parse_args()
    config = make_config(args)
    start = time.time()

    os.makedirs(config["out"], exist_ok=True)
    run_selected(args.only.split(","), config)
    log(f"finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
