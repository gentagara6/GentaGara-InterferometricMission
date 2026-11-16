"""
Experiments with the GRAIL evaluator.

Run:  python experiments.py
Each experiment has a SETTINGS block marked  >>> CHANGE ME <<<.
Edit those values, run again, and compare the output.
To run only some experiments, edit RUN at the bottom of the file.
"""
import numpy as np

from grail_eval import GrailProblem              # the evaluator (official scoring, SciPy physics)


def cell_to_pos(cell, g):
    """Centre of grid square number `cell` (0 .. g-1) as a position in (-1, 1)."""
    return (cell - g // 2 + 0.5) * 2 / g


def build_x(positions, velocities):
    """Build the chromosome from per-satellite lists.
    positions  = [(x, y, z), ...]  one tuple per satellite
    velocities = [(vx, vy, vz), ...]
    Returns x in the official order [all dx, all dy, all dz, all dvx, all dvy, all dvz]."""
    table = np.hstack([np.array(positions, float), np.array(velocities, float)])  # N rows x 6 columns
    return table.T.ravel()    # transpose -> 6 rows (dx row, dy row, ...) -> flatten


def report(P, x):
    """Print the 3x3 fill-factor table, the score, and the best possible score."""
    g = P.grid_size
    best_one_plane = (P.n_sat * (P.n_sat - 1) + 1) / (2 * g - 1) ** 2
    ff = P.fill_factors(x)
    print("            XY       XZ       YZ       sum")
    for name, row in zip(["t0", "t1", "t2"], ff):
        print(f"  {name}    " + "  ".join(f"{v:.4f}" for v in row) + f"   {row.sum():.4f}")
    print(f"  fitness = {P.fitness(x)[0]:.4f}   (best possible = {-3 * best_one_plane:.4f})")


# ===========================================================================
# EXPERIMENT 1: place satellites by hand, no velocity
# Question: a perfect pattern at t0. Does it survive one and two orbits?
# ===========================================================================
def experiment_1():
    print("\n=== Experiment 1: hand-placed satellites ===")
    P = GrailProblem("tiny-observatory-in-dro")
    g = P.grid_size

    # >>> CHANGE ME <<<  grid squares (0..10 on the 11x11 grid; 5 is the centre/mothership)
    cells = [(5, 5, 5),
             (6, 5, 7),
             (8, 7, 6),
             (5, 9, 8),
             (2, 6, 4)]
    # >>> CHANGE ME <<<  velocities (same for every satellite here; try 0, then small values)
    velocity = (0.0, 0.0, 0.0)

    positions = [tuple(cell_to_pos(c, g) for c in cell) for cell in cells]
    x = build_x(positions, [velocity] * len(cells))
    report(P, x)


# ===========================================================================
# EXPERIMENT 2: one satellite, try many velocities
# Question: which start velocities bring it back near where it started?
# This is the homework from last time: look for a pattern in the output.
# ===========================================================================
def experiment_2():
    print("\n=== Experiment 2: one satellite, sweep one velocity component ===")
    P = GrailProblem("tiny-observatory-in-dro")

    # >>> CHANGE ME <<<
    start_position = (-0.3, 0.0, 0.0)    # try moving it along x, then along y, then along z
    which = 1                           # velocity component to sweep: 0 = vx, 1 = vy, 2 = vz
    values = np.linspace(-1, 0, 21)     # steps of 0.05, from 0 to 2
    # Use a copy of the problem where only satellite 1 is "real":
    # the other satellites sit at the centre with zero velocity, so they never move.
    print("  velocity   pos at t0              pos at 1 period         pos at 2 periods")
    for v in values:
        vel = [0.0, 0.0, 0.0]
        vel[which] = v
        x = build_x([start_position] + [(0, 0, 0)] * (P.n_sat - 1),
                    [tuple(vel)] + [(0, 0, 0)] * (P.n_sat - 1))
        pos = P.positions(x)[:, 0]               # satellite 1 at the three times
        pos[1:] /= P.inflation_factor            # as seen on the enlarged grid
        on = ["on " if np.all(np.abs(p) < 1) else "OFF" for p in pos]
        cols = [f"{o} ({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f})" for o, p in zip(on, pos)]
        print(f"  {v:7.3f}    " + "   ".join(cols))


# ===========================================================================
# EXPERIMENT 3: how good is pure random guessing?
# Question: how does the best score change with more tries, or smaller velocities?
# ===========================================================================
def experiment_3():
    print("\n=== Experiment 3: random search ===")
    # >>> CHANGE ME <<<
    problem = "tiny-observatory-in-dro"   # or "large-observatory-in-dro", "large-observatory-in-halo"
    tries = 5000                          # more tries = slower but better?
    max_velocity = 10.0                   # official bound is 10; try 1, 0.1, 0.01, 0
    seed = 0                              # change to get different random numbers

    P = GrailProblem(problem)
    rng = np.random.default_rng(seed)
    N = P.n_sat
    best_x, best_f = None, 0.0
    for _ in range(tries):
        x = np.concatenate([rng.uniform(-1, 1, 3 * N),                        # positions
                            rng.uniform(-max_velocity, max_velocity, 3 * N)])  # velocities
        f = P.fitness(x)[0]
        if f < best_f:                    # lower is better
            best_x, best_f = x, f
    print(f"  best of {tries} random tries: {best_f:.4f}")
    if best_x is not None:
        report(P, best_x)


# ===========================================================================
# EXPERIMENT 4: change the rules of the game
# Question: how much does the grid enlargement (1.23) actually help?
# ===========================================================================
def experiment_4():
    print("\n=== Experiment 4: change the inflation factor ===")
    # >>> CHANGE ME <<<
    inflation = 1.23                      # official value; try 1.0 (no help) and 3.0 (lots of help)

    P = GrailProblem("tiny-observatory-in-dro", inflation_factor=inflation)
    rng = np.random.default_rng(0)
    N = P.n_sat
    scores = [P.fitness(np.concatenate([rng.uniform(-1, 1, 3 * N), np.zeros(3 * N)]))[0]
              for _ in range(2000)]       # zero velocity, random positions
    print(f"  inflation {inflation}: best {min(scores):.4f}, "
          f"share of solutions scoring 0: {np.mean(np.array(scores) == 0):.0%}")


RUN = [experiment_1, experiment_2, experiment_3, experiment_4]   # >>> CHANGE ME <<< remove ones you don't want

if __name__ == "__main__":
    for exp in RUN:
        exp()