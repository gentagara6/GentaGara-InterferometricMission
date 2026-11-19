"""
Compare four search algorithms on the same budget of score evaluations.

Run:  python compare.py                 (all three instances, about 5 minutes)
      python compare.py --budget 5000   (quicker, rougher)
Writes results/comparison.json, which visualize.py turns into a chart.

The four algorithms, from simplest to the one solver.py uses:
  1. random      : random positions AND random velocities (like experiments.py, Experiment 3)
  2. random+vel  : random positions, velocities CALCULATED by least squares (Q4/Q5 idea)
  3. hill-climb  : local search, accept only better-or-equal moves (temperature 0)
  4. annealing   : simulated annealing (temperature starts at the instance default)
Every algorithm gets the same number of evaluations, so the comparison is fair.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from solver import DEFAULTS, Solver

INSTANCES = list(DEFAULTS)
ALGORITHMS = ["random", "random+vel", "hill-climb", "annealing"]
CHECKPOINTS = 100   # how many points of each curve we keep


def run(task):
    name, algo, seed, budget = task
    s = Solver(name, seed=seed)
    rng = s.rng
    every = budget // CHECKPOINTS

    if algo in ("random", "random+vel"):
        best, curve = 0.0, [0.0]
        for it in range(budget):
            pos = rng.uniform(-1, 1, (s.N, 3))
            if algo == "random":
                vel = rng.uniform(-10, 10, (s.N, 3))
            else:
                vel = s.velocity_for(pos)
            best = min(best, s.score(pos, vel))
            if (it + 1) % every == 0:
                curve.append(best)
    else:
        temp = 0.0 if algo == "hill-climb" else DEFAULTS[name][1]
        s.anneal(budget, temp, verbose=False)
        hist = dict(s.history)
        # keep the best-so-far at the same checkpoints as the random searches
        steps = sorted(hist)
        curve, j, last = [], 0, hist[0]
        for c in range(0, budget + 1, every):
            while j < len(steps) and steps[j] <= c:
                last = hist[steps[j]]
                j += 1
            curve.append(last)
    print(f"  done: {name:28s} {algo:11s} seed {seed}   best {curve[-1]:.4f}", flush=True)
    return name, algo, seed, curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=20_000, help="evaluations per run")
    parser.add_argument("--seeds", type=int, default=3, help="runs per algorithm")
    parser.add_argument("--workers", type=int, default=4, help="parallel processes")
    args = parser.parse_args()

    tasks = [(n, a, seed, args.budget) for n in INSTANCES for a in ALGORITHMS
             for seed in range(args.seeds)]
    print(f"{len(tasks)} runs, {args.budget} evaluations each")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, tasks))

    out = {"budget": args.budget, "seeds": args.seeds, "checkpoints": CHECKPOINTS,
           "algorithms": ALGORITHMS, "instances": {}}
    for name, algo, seed, curve in results:
        out["instances"].setdefault(name, {}).setdefault(algo, []).append(curve)

    path = Path(__file__).parent / "results" / "comparison.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
