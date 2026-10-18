# Short delivery routes on Java: Hill Climbing vs Simulated Annealing vs Genetic Algorithm (TSP)

Midterm project - *Artificial Intelligence Concepts* (EK234201).

We solve the Travelling Salesman Problem (TSP) on real Indonesian cities (island of Java) and
on random instances, and compare **three local-search algorithms taught in class**
(Hill Climbing with random restarts, Simulated Annealing, Genetic Algorithm) plus a memetic
extension, against baselines (random search, nearest neighbour) and against **exact** solutions
(brute force, Uniform-Cost Search, A* with an MST heuristic, Held-Karp) on small instances.

## Repository layout

```
src/                       all the source code
  run_experiments.py       entry point: runs E1-E6 (seeded) -> results/*.json
  tsp/                     the solvers
    core.py                distances, tour length, 2-opt move, 2-opt local search
    result.py              Result + SearchRun (budget, best tour, history of a run)
    data.py                instances: Java-12, Java-30 (real cities), U-50, U-100
    hill_climbing.py       hill climbing (first-choice, 2-opt) with random restarts
    annealing.py           simulated annealing (geometric cooling)
    genetic.py             genetic algorithm (OX crossover) + memetic variant
    baselines.py           random search, nearest neighbour
    exact.py               brute force, Held-Karp, 1-tree lower bound
    astar.py               UCS / A* (MST heuristic)
  experiments/             one file per group of experiments
    common.py              shared settings (seeds, parameters) and helpers
    exact_search.py        E1 exact search, E6 reference solutions
    comparison.py          E2 main comparison, E3 budget effect
    restarts.py            E4 single climbs vs random restarts
    sensitivity.py         E5 parameter sensitivity
tests/                     16 unit tests (test_core, test_exact, test_heuristics)
results/                   raw results (JSON) used by the report
report/                    figures and final report (PDF)
```

The code follows the Epitech coding style adapted to Python: functions of at
most 20 lines and 4 parameters, at most 10 functions per file, 80 columns, no
comments inside functions (docstrings instead), no global variables modified.

Solvers take `(dist, budget, seed, params)` where `params` is an optional dict
(see `SA_DEFAULTS` and `GA_DEFAULTS`), e.g.
`genetic_algorithm(inst.d, 30000, 1, {"pop_size": 50})`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -v   # 1. unit tests (about 2 s)  -> must say OK
python src/run_experiments.py --quick     # 2. smoke test (about 1 min) -> results_quick/
python src/run_experiments.py             # 3. full experiments (about 40-50 min, 1 CPU core)
```

Every run is seeded (`seed = 1000 + run`), so a full run reproduces the numbers of the report
exactly (wall-clock times depend on your machine; all other numbers do not).

## Fair comparison

All algorithms get the same budget of **10 000 x n candidate evaluations** (one evaluation = one
neighbour move tested, one child evaluated or one random tour generated). Wall-clock time is
reported as well, because evaluating a whole tour (GA) costs more than evaluating a 2-opt delta (HC, SA).
