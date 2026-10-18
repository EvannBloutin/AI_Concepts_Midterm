# Short delivery routes on Java: Hill Climbing vs Simulated Annealing vs Genetic Algorithm (TSP)

Midterm project - *Artificial Intelligence Concepts* (EK234201).

We solve the Travelling Salesman Problem (TSP) on real Indonesian cities (island of Java) and
on random instances, and compare **three local-search algorithms taught in class**
(Hill Climbing with random restarts, Simulated Annealing, Genetic Algorithm) plus a memetic
extension, against baselines (random search, nearest neighbour) and against **exact** solutions
(brute force, Uniform-Cost Search, A* with an MST heuristic, Held-Karp) on small instances.

## Repository layout

```
tsp/
  core.py           distances, tour length, 2-opt move (state / successor function)
  data.py           instances: Java-12, Java-30 (real cities), U-50, U-100 (random)
  hill_climbing.py  hill climbing (first-choice, 2-opt) with random restarts
  annealing.py      simulated annealing (geometric cooling)
  genetic.py        genetic algorithm (OX crossover) + memetic variant
  baselines.py      random search, nearest neighbour
  exact.py          brute force, UCS / A* (MST heuristic), Held-Karp, 1-tree lower bound
tests/test_tsp.py   16 unit tests
run_experiments.py  all experiments (E1-E6), seeded -> results/*.json
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s tests -v   # 1. unit tests (about 2 s)  -> must say OK
python run_experiments.py --quick         # 2. smoke test (about 1 min) -> results_quick/
python run_experiments.py                 # 3. full experiments (about 30-40 min, 1 CPU core)
```

Every run is seeded (`seed = 1000 + run`), so a full run reproduces the numbers of the report
exactly (wall-clock times depend on your machine; all other numbers do not).

## Fair comparison

All algorithms get the same budget of **10 000 x n candidate evaluations** (one evaluation = one
neighbour move tested, one child evaluated or one random tour generated). Wall-clock time is
reported as well, because evaluating a whole tour (GA) costs more than evaluating a 2-opt delta (HC, SA).
