"""
GRAIL solver: search the positions, calculate the velocities.

Run:
    python solver.py tiny            # 5 satellites, about 10 seconds
    python solver.py large-dro       # 40 satellites, about 2 minutes
    python solver.py large-halo      # 40 satellites, unstable orbit (hardest)
Options:
    --iters 50000   number of search steps
    --seed 3        a different seed gives a different random search
    --temp 0.01     starting temperature of simulated annealing

The idea (from the study questions):
  * Q4: at t0 a satellite is exactly at its start position, so the 3 position
    numbers set the t0 pattern directly.
  * Q4/Q5: the map is linear, so for a given start position we can CALCULATE
    the velocity that keeps the satellite as close as possible to its start
    square at t1 and t2 (least squares, the "best compromise").
  * Q2: trying everything is impossible, so we SEARCH the positions with
    simulated annealing: change one satellite a little, keep the change if
    the score gets better (and sometimes even if it gets a bit worse, so the
    search doesn't get stuck).
The result is checked with the official scoring logic in grail_eval.py and
saved as a submission file in solutions/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from grail_eval import GrailProblem

SHORT_NAMES = {
    "tiny": "tiny-observatory-in-dro",
    "large-dro": "large-observatory-in-dro",
    "large-halo": "large-observatory-in-halo",
}
# Sensible defaults per instance: (search steps, starting temperature)
DEFAULTS = {
    "tiny-observatory-in-dro": (20_000, 0.02),
    "large-observatory-in-dro": (100_000, 0.01),
    "large-observatory-in-halo": (100_000, 0.01),
}


class Solver:
    def __init__(self, problem_name, seed=0):
        self.P = GrailProblem(problem_name)
        self.N = self.P.n_sat
        self.g = self.P.grid_size
        self.rng = np.random.default_rng(seed)

        # The 3 STMs, keeping only the 3 position rows (x, y, z).
        # At t1 and t2 we divide by the inflation factor 1.23 right here,
        # so the matrices give positions "as seen on the grid".
        R = self.P.stms[:, :3, :].copy()      # shape (3 times, 3 rows, 6 columns)
        R[1:] /= self.P.inflation_factor
        self.R = R

        # ---- Calculating the velocity (Q4 and Q5) -----------------------------
        # Position at time k = R_k[:, :3] @ p  +  R_k[:, 3:] @ v
        #                      (start position)    (start velocity)
        # We want the position at t1 and at t2 to equal p (stay in the same square).
        # That's 6 equations for the 3 unknowns in v:
        #       A @ v = (I - B) @ p
        # with A = velocity columns of t1 and t2 stacked, B = position columns.
        # More equations than unknowns -> least squares via the pseudo-inverse.
        A = np.vstack([R[1][:, 3:], R[2][:, 3:]])        # 6 x 3
        B = np.vstack([R[1][:, :3], R[2][:, :3]])        # 6 x 3
        I = np.vstack([np.eye(3), np.eye(3)])            # 6 x 3
        self.V = np.linalg.pinv(A) @ (I - B)             # 3 x 3:  v = V @ p

    # ------------------------------------------------------------------------
    def velocity_for(self, p):
        """Best-compromise start velocity for start position(s) p, within bounds."""
        return np.clip(p @ self.V.T, -10, 10)

    def score(self, pos, vel):
        """Same result as the official fitness, but faster.

        Instead of the 2D autocorrelation we count the distinct arrows directly:
        all differences between occupied squares, plus the zero arrow.
        """
        g = self.g
        state = np.hstack([pos, vel])                    # N x 6
        sums = []
        for k in range(3):
            pts = state @ self.R[k].T                    # N x 3 positions on the grid
            pts = pts[np.all(np.abs(pts) < 1, axis=1)]   # drop satellites off the grid
            if len(pts) == 0:
                return 0.0                               # empty snapshot -> score 0
            cells = (pts * g / 2 + g // 2).astype(int)   # same snapping as the official code
            total = 0
            for a, b in ((0, 1), (0, 2), (1, 2)):        # XY, XZ, YZ shadows
                occupied = np.unique(cells[:, a] * g + cells[:, b])   # one id per square
                u, w = occupied // g, occupied % g
                arrows = (u[:, None] - u[None, :]) * (2 * g) + (w[:, None] - w[None, :])
                total += len(np.unique(arrows))         # distinct arrows (includes zero)
            sums.append(total)
        return -min(sums) / (2 * g - 1) ** 2            # worst snapshot, minus sign

    # ------------------------------------------------------------------------
    def random_move(self, pos, vel):
        """Change ONE satellite. Returns new copies of pos and vel."""
        pos, vel = pos.copy(), vel.copy()
        i = self.rng.integers(self.N)
        r = self.rng.random()
        if r < 0.5:      # nudge: move it a little, recalculate its velocity
            pos[i] = np.clip(pos[i] + self.rng.normal(0, 0.1, 3), -1, 1)
            vel[i] = self.velocity_for(pos[i])
        elif r < 0.7:    # jump: put it somewhere random, recalculate its velocity
            pos[i] = self.rng.uniform(-1, 1, 3)
            vel[i] = self.velocity_for(pos[i])
        else:            # fine-tune: keep the position, adjust the velocity a little
            vel[i] = np.clip(vel[i] + self.rng.normal(0, 0.3, 3), -10, 10)
        return pos, vel

    def anneal(self, iters, temp0, verbose=True):
        """Simulated annealing over the satellites' start states.
        With temp0 = 0 it becomes plain hill climbing (only better-or-equal moves)."""
        pos = self.rng.uniform(-0.3, 0.3, (self.N, 3))   # start near the centre
        vel = self.velocity_for(pos)
        cur = self.score(pos, vel)
        best, best_pos, best_vel = cur, pos, vel
        self.history = [(0, best)]                      # (step, best score so far), for plots
        record_every = max(1, iters // 200)
        start = time.time()
        for it in range(iters):
            temp = temp0 * (1 - it / iters) + 1e-9      # temperature falls to ~0
            new_pos, new_vel = self.random_move(pos, vel)
            new = self.score(new_pos, new_vel)
            # Lower is better. Always accept better or equal; accept worse
            # with a probability that shrinks as the temperature falls.
            if new <= cur or self.rng.random() < np.exp((cur - new) / temp):
                pos, vel, cur = new_pos, new_vel, new
                if cur < best:
                    best, best_pos, best_vel = cur, pos, vel
            if (it + 1) % record_every == 0:
                self.history.append((it + 1, best))
            if verbose and (it + 1) % max(1, iters // 10) == 0:
                print(f"  step {it + 1:>7}/{iters}   current {cur:.4f}   best {best:.4f}"
                      f"   ({time.time() - start:.0f} s)")
        return best_pos, best_vel

    def to_x(self, pos, vel):
        """Pack into the official order [all dx, all dy, all dz, all dvx, all dvy, all dvz]."""
        return np.concatenate([pos.T.ravel(), vel.T.ravel()])


def main():
    parser = argparse.ArgumentParser(description="GRAIL solver")
    parser.add_argument("problem", choices=list(SHORT_NAMES))
    parser.add_argument("--iters", type=int)
    parser.add_argument("--temp", type=float)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    name = SHORT_NAMES[args.problem]
    iters = args.iters or DEFAULTS[name][0]
    temp = args.temp or DEFAULTS[name][1]
    print(f"{name}: {iters} steps, temperature {temp}, seed {args.seed}")

    solver = Solver(name, seed=args.seed)
    pos, vel = solver.anneal(iters, temp)
    x = solver.to_x(pos, vel)

    # Check with the official scoring logic (grail_eval.py)
    official = solver.P.fitness(x)[0]
    print(f"\nofficial fitness: {official:.4f}")
    print(solver.P.fill_factors(x).round(4), "  <- rows t0,t1,t2; columns XY,XZ,YZ")

    # Save a submission file (only if it beats the file already saved)
    out = Path(__file__).parent / "solutions" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        old = solver.P.fitness(json.loads(out.read_text())[0]["decisionVector"])[0]
        if old <= official:
            print(f"kept the saved solution ({old:.4f} is at least as good)")
            return
    submission = [{
        "decisionVector": x.tolist(),
        "problem": name,
        "challenge": "spoc-3-interferometric-mission",
    }]
    out.write_text(json.dumps(submission))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
