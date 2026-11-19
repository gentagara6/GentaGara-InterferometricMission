"""
Build results/grail_results.html: one page that shows every solution and the
algorithm comparison. Open the file in any browser.

Run:  python visualize.py
Needs: solutions/*.json (from solver.py) and, for the algorithm chart,
       results/comparison.json (from compare.py).

What the page shows, per instance:
  * an animation of the swarm over two orbits (from the STMs at 97 moments)
  * the 3 x 3 snapshots (t0, t1, t2 x XY, XZ, YZ) with the occupied squares
  * the "arrow map" of any snapshot: which baselines are covered (= fill factor)
and, for all instances: the convergence of the four algorithms and the
one-orbit eigenvalues of the DRO and halo orbits.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from grail_eval import MU, PROBLEMS, GrailProblem, _cr3bp_with_stm

HERE = Path(__file__).parent
FRAMES_PER_ORBIT = 48
LABELS = {
    "tiny-observatory-in-dro": "Tiny · DRO",
    "large-observatory-in-dro": "Large · DRO",
    "large-observatory-in-halo": "Large · Halo",
}


def stms_over_time(ic, period, n_frames):
    """STMs at n_frames equally spaced moments from 0 to 2 periods."""
    t = np.linspace(0, 2 * period, n_frames)
    start = np.concatenate([ic, np.eye(6).ravel()])
    sol = solve_ivp(_cr3bp_with_stm, (0, t[-1]), start, t_eval=t, args=(MU,),
                    method="DOP853", rtol=1e-13, atol=1e-13)
    return sol.y[6:].T.reshape(n_frames, 6, 6)


def instance_data(name):
    P = GrailProblem(name)
    g, N = P.grid_size, P.n_sat
    x = np.array(json.loads((HERE / "solutions" / f"{name}.json").read_text())[0]["decisionVector"])
    state = P.decode(x)                                   # N x 6

    # positions "as seen on the grid" at the three scored snapshots
    snaps = []
    for k, pts in enumerate(P.positions(x)):
        if k:
            pts = pts / P.inflation_factor
        on = np.all(np.abs(pts) < 1, axis=1)
        cells = (pts * g / 2 + g // 2).astype(int)
        snaps.append({
            "cells": [c.tolist() if o else None for c, o in zip(cells, on)],
            "onGrid": int(on.sum()),
        })

    # animation: grid grows from 1 to 1.23 during the first orbit, then stays
    p = PROBLEMS[name]
    n_frames = 2 * FRAMES_PER_ORBIT + 1
    stms = stms_over_time(p["ic"], p["period"], n_frames)
    frames = []
    for f, stm in enumerate(stms):
        grow = 1 + (P.inflation_factor - 1) * min(f / FRAMES_PER_ORBIT, 1)
        pos = (state @ stm[:3].T) / grow
        frames.append(np.round(pos, 3).tolist())

    ff = P.fill_factors(x)
    sums = ff.sum(axis=1)
    best_plane = (N * (N - 1) + 1) / (2 * g - 1) ** 2
    return {
        "id": name, "label": LABELS[name], "N": N, "g": g,
        "fitness": round(float(P.fitness(x)[0]), 4),
        "bound": round(-3 * best_plane, 4),
        "ff": np.round(ff, 4).tolist(),
        "worst": int(np.argmin(sums)),
        "snaps": snaps,
        "frames": frames,
        "framesPerOrbit": FRAMES_PER_ORBIT,
    }


def eigen_data():
    out = {}
    for label, name in (("DRO", "large-observatory-in-dro"), ("Halo", "large-observatory-in-halo")):
        stm = GrailProblem(name).stms[1]                  # one-orbit STM
        mags = sorted(np.abs(np.linalg.eigvals(stm)), reverse=True)
        out[label] = [round(float(m), 3) for m in mags]
    return out


def main():
    data = {
        "instances": [instance_data(n) for n in LABELS],
        "eigen": eigen_data(),
        "comparison": None,
    }
    comp = HERE / "results" / "comparison.json"
    if comp.exists():
        data["comparison"] = json.loads(comp.read_text())
    else:
        print("note: results/comparison.json not found, run compare.py for the algorithm chart")

    template = (HERE / "results_template.html").read_text(encoding="utf-8")
    html = template.replace("__GRAIL_DATA__", json.dumps(data, separators=(",", ":")))
    out = HERE / "results" / "grail_results.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"saved to {out}  ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
