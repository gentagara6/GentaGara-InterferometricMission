# GRAIL – SpOC 3 Interferometric Mission

Course project for *Design and Analysis of Algorithms*, based on the GRAIL challenge
from ESA's Space Optimisation Competition (GECCO 2024):
https://optimize.esa.int/challenge/spoc-3-interferometric-mission/About

Satellites fly near a mothership in an Earth–Moon orbit. For each satellite we choose a
start position and a start velocity. The goal is a "Golomb pattern": all baselines (arrows
between satellites) different, at three times: at the start, after 1 orbit and after 2 orbits.
The score is the worst of the three snapshots. Lower fitness is better.

## Results

| Instance | Satellites | Fitness | Best possible |
|---|---|---|---|
| tiny-observatory-in-dro | 5 | **−0.1429** | −0.1429 (optimal) |
| large-observatory-in-dro | 40 | **−1.40** | −2.79 (upper bound, probably unreachable) |
| large-observatory-in-halo | 40 | **−1.23** | −2.79 (upper bound, probably unreachable) |

Solutions are in `solutions/` in the competition's submission format.

## Method: search the positions, calculate the velocities

1. **Physics shortcut.** The mothership's orbit is simulated once. This gives three 6×6
   state transition matrices (STMs). A satellite's position at each snapshot is
   STM × (its 6 start numbers), so the map from start state to positions is **linear**.
2. **Calculate the velocities.** At t0 a satellite is at its start position. Its 3 velocity
   numbers then have to keep it on the grid at t1 and t2 (6 targets, 3 unknowns: an
   overdetermined system). We solve it by **least squares**: the velocity that keeps the
   satellite as close as possible to its start square.
3. **Search the positions.** Trying every placement is impossible (441 choose 40 ≈ 10^57).
   Instead we use **simulated annealing**: repeatedly change one satellite, keep the change
   if the score improves, and sometimes keep a slightly worse change early on, so the
   search doesn't get stuck.
4. **Fast scoring.** Inside the search, the distinct arrows are counted directly instead of
   with the 2D autocorrelation. The final answer is always re-checked with the official
   scoring logic.

## Files

| File | What it does |
|---|---|
| `solver.py` | The solver described above |
| `grail_eval.py` | Official ESA scoring logic, with SciPy instead of heyoka for the physics, plus comments |
| `experiments.py` | Small experiments used to understand the problem |
| `recipes.py` | Prints the STM "recipes" for y at t1 and t2 |
| `grail_study_guide.html` | Visual study notes |
| `solutions/*.json` | Best solutions found |

## Run

```bash
pip install -r requirements.txt
python solver.py tiny          # about 10 s
python solver.py large-dro     # about 2 min
python solver.py large-halo    # about 2 min
```

Use `--seed N` to try a different random search and `--iters N` for a longer one.
A new solution file is only written when it beats the saved one.

## Notes

- The official code uses heyoka to build the STMs. heyoka isn't available for Windows with
  Python 3.14, so `grail_eval.py` builds them with SciPy. It gives the same scores as the
  official code (tested on 6,000 random solutions).
- The website formula says "max" over the three times, but the official code uses "min":
  the worst snapshot sets the score.
