"""
GRAIL evaluator: the official ESA scoring logic, using SciPy, with comments.

Source: github.com/esa/SpOC3 -> "Challenge 2 Interferometric Mission/problems/golomb_*.py"
The scoring is identical to the official file (tested on thousands of solutions).
The only change: the official file builds the STMs (see below) with heyoka; this
file builds them with SciPy. Both agree to many decimal places, so scores match.

Install:   pip install numpy scipy

THE BIG PICTURE
---------------
1. At startup, SciPy simulates the MOTHERSHIP's orbit once and produces three
   6x6 matrices called STMs (state transition matrices), one for each
   observation time (t = 0, 1 period, 2 periods).
2. An STM answers: "if a satellite starts a tiny bit away from the mothership,
   where is it relative to the mothership later?"  Answer = STM x (start state).
3. To score a solution x, we multiply every satellite's start state by each STM,
   put the resulting positions on a grid, and count distinct baselines.
So the solver is only used ONCE (step 1). Scoring itself is just matrix maths.
"""

import numpy as np                    # arrays and linear algebra
import scipy.signal                   # provides correlate(), used to find all baselines
from scipy.integrate import solve_ivp # SciPy's ODE solver (replaces heyoka)

# ---------------------------------------------------------------------------
# PROBLEM SETTINGS (copied from the bottom of the three official files)
# ---------------------------------------------------------------------------
MU = 0.01215058560962404     # Moon mass / (Earth mass + Moon mass). The only
                             # physical constant the Earth-Moon model needs.

DRO_IC = [0.896508460944940632764, 0., 0.,              # mothership start position (x, y, z)
          0.000000000000013951082, 0.474817948848534454598, 0.]  # and velocity (vx, vy, vz)
DRO_PERIOD = 2.6905181697222775                          # time for one full DRO loop

HALO_IC = [1.0829551779304256e+00, -6.9232801936027592e-27, -2.0231744561698364e-01,
           9.7888791827480806e-15, -2.0102644884016105e-01, 2.4744866465838825e-14]
HALO_PERIOD = 2.383491010514447                          # time for one full halo loop

PROBLEMS = {
    #  name                          satellites  grid   mothership orbit
    "tiny-observatory-in-dro":   dict(N=5,  grid_size=11, ic=DRO_IC,  period=DRO_PERIOD),
    "large-observatory-in-dro":  dict(N=40, grid_size=21, ic=DRO_IC,  period=DRO_PERIOD),
    "large-observatory-in-halo": dict(N=40, grid_size=21, ic=HALO_IC, period=HALO_PERIOD),
}


# ---------------------------------------------------------------------------
# PART 1: PHYSICS. Build the three STMs with SciPy (runs once)
# ---------------------------------------------------------------------------
def _cr3bp_with_stm(t, s, mu):
    """Rate of change of everything we integrate: 6 state numbers + 36 STM entries.

    The model is the CR3BP (circular restricted 3-body problem): a tiny spacecraft
    moves under the gravity of Earth and Moon. Coordinates rotate with the
    Earth-Moon line, so Earth and Moon stay fixed on the x-axis.
    """
    x, y, z, vx, vy, vz = s[:6]                 # mothership position and velocity
    phi = s[6:].reshape(6, 6)                   # the current STM (36 numbers -> 6x6)

    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2)   # distance to Earth (Earth sits at x = -mu)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2 + z**2)  # distance to Moon (Moon sits at x = 1-mu)

    # Newton's law in rotating coordinates. "2*vy", "-2*vx" = Coriolis effect;
    # "+x", "+y" = centrifugal effect (both appear because the frame rotates).
    ax = 2 * vy + x - (1 - mu) * (x + mu) / r1**3 - mu * (x + mu - 1) / r2**3
    ay = -2 * vx + y - (1 - mu) * y / r1**3 - mu * y / r2**3
    az = -(1 - mu) * z / r1**3 - mu * z / r2**3

    # Variational equations: how a SMALL nudge grows over time.
    # A = Jacobian = "how does each rate of change react to a tiny change in each
    # state variable". heyoka computes it symbolically; here it is written out.
    def gravity_gradient(dx, r, m):             # how one body's pull changes with position
        p = np.array([dx, y, z])
        return m * (3 * np.outer(p, p) / r**5 - np.eye(3) / r**3)
    G = (gravity_gradient(x + mu, r1, 1 - mu)       # Earth's part
         + gravity_gradient(x + mu - 1, r2, mu)     # Moon's part
         + np.diag([1.0, 1.0, 0.0]))                # centrifugal part
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)       # position changes because of velocity
    A[3:, :3] = G               # velocity changes because of gravity (depends on position)
    A[3, 4], A[4, 3] = 2.0, -2.0  # Coriolis: velocity changes because of velocity

    dphi_dt = A @ phi           # the rule for how the STM changes: dPhi/dt = A * Phi
    return np.concatenate([[vx, vy, vz, ax, ay, az], dphi_dt.ravel()])


def make_stms(ic, period, n_meas=3, mu=MU):
    """Simulate the mothership and return the STMs at t = 0, 1 period, 2 periods."""
    t_grid = np.linspace(0, period * (n_meas - 1), n_meas)  # [0, 1 period, 2 periods]
    start = np.concatenate([ic, np.eye(6).ravel()])  # mothership state + identity matrix
                                                     # (at t = 0 nothing has changed yet)
    sol = solve_ivp(_cr3bp_with_stm,                 # the equations above
                    (0, t_grid[-1]),                 # integrate from 0 to 2 periods
                    start,
                    t_eval=t_grid,                   # record the state at these three times
                    args=(mu,),
                    method="DOP853",                 # a high-accuracy Runge-Kutta method
                    rtol=1e-13, atol=1e-13)          # very strict error tolerance
    # sol.y has one COLUMN per time; .T makes it one row per time (42 numbers each).
    # Drop the 6 mothership numbers and reshape the remaining 36 into a 6x6 matrix.
    return sol.y[6:].T.reshape(n_meas, 6, 6)


# ---------------------------------------------------------------------------
# PART 2: SCORING. The fitness function (runs every time you test a solution)
# ---------------------------------------------------------------------------
class GrailProblem:
    def __init__(self, name, scaling_factor=1e-4, inflation_factor=1.23):
        p = PROBLEMS[name]                       # look up the settings for this instance
        self.name = name
        self.n_sat = p["N"]                      # number of satellites
        self.grid_size = p["grid_size"]          # grid is grid_size x grid_size squares
        self.scaling_factor = scaling_factor     # shrinks your numbers so the STM (a small-nudge
                                                 # approximation) is valid; cancels out later
        self.inflation_factor = inflation_factor # grid enlargement at t = 1 and t = 2 periods
        self.stms = make_stms(p["ic"], p["period"])  # SciPy runs here, once

    def get_bounds(self):
        """Allowed range of each number in x: positions in [-1, 1], velocities in [-10, 10]."""
        n = self.n_sat
        return [-1.0] * 3 * n + [-10.0] * 3 * n, [1.0] * 3 * n + [10.0] * 3 * n

    def decode(self, x):
        """Turn the flat list x into a table with one row per satellite.

        x = [dx1..dxN, dy1..dyN, dz1..dzN, dvx1..dvxN, dvy1..dvyN, dvz1..dvzN]
        reshape(6, N) makes 6 rows (dx row, dy row, ...); .T flips it so that
        row i = (dx_i, dy_i, dz_i, dvx_i, dvy_i, dvz_i) for satellite i.
        """
        return np.asarray(x, dtype=float).reshape(6, self.n_sat).T

    def positions(self, x):
        """Positions of all satellites at each observation time. Shape (3 times, N, 3 coords)."""
        start = self.decode(x) * self.scaling_factor   # shrink to a "small nudge"
        out = []
        for stm in self.stms:                          # one STM per observation time
            later = (stm @ start.T).T                  # new state = STM x start state, for every satellite
            out.append(later[:, :3] / self.scaling_factor)   # keep x, y, z only; undo the shrinking
        return np.array(out)

    def fill_factors(self, x):
        """Fill factor for every (time, plane). Returns a 3x3 table:
        rows = t0, t1, t2   columns = XY, XZ, YZ."""
        g = self.grid_size
        table = []
        for k, pts in enumerate(self.positions(x)):    # k = 0, 1, 2 (the three times)

            if k != 0:                                 # at the later times the grid is
                pts = pts / self.inflation_factor      # 1.23x bigger = positions look 1.23x smaller

            # Drop satellites outside the grid: every coordinate must be strictly in (-1, 1).
            pts = pts[np.max(pts, axis=1) < 1]
            pts = pts[np.min(pts, axis=1) > -1]

            # Convert a position in (-1, 1) into a square number 0 .. g-1.
            # Example with g = 11: -0.99 -> 0,   0.0 -> 5 (centre),   0.99 -> 10.
            cells = (pts * g / 2 + int(g / 2)).astype(int)

            # A 3D box of g*g*g cells: 1 = a satellite is here, 0 = empty.
            # Two satellites in the same cell just set the same 1 (they count as one).
            box = np.zeros((g, g, g))
            for i, j, l in cells:
                box[i, j, l] = 1

            row = []
            # Take the three "shadows" (projections) of the 3D box:
            #   max over axis 2 squashes z away -> XY plane, and so on.
            for plane in (box.max(axis=2), box.max(axis=1), box.max(axis=0)):   # XY, XZ, YZ
                # Autocorrelation: slide the plane over a copy of itself in every possible
                # direction. Output cell (a, b) is nonzero exactly when some pair of
                # satellites is separated by the arrow (a, b). So the number of nonzero
                # cells = number of DISTINCT baselines (both directions + the zero arrow).
                C = scipy.signal.correlate(plane, plane, mode="full")   # size (2g-1) x (2g-1)
                C[abs(C) < 1e-8] = 0          # clean up tiny rounding errors (FFT maths)
                # Fill factor = distinct baselines / all possible baselines.
                row.append(np.count_nonzero(C) / C.size)
            table.append(row)
        return np.array(table)

    def fitness(self, x):
        """The official score: add the 3 planes at each time, take the WORST time,
        and flip the sign because the platform minimises. Lower is better."""
        per_time = self.fill_factors(x).sum(axis=1)    # [sum at t0, sum at t1, sum at t2]
        return [-min(per_time)]


if __name__ == "__main__":
    P = GrailProblem("tiny-observatory-in-dro")
    x = np.random.default_rng(0).uniform(*P.get_bounds())
    print("fitness of a random solution:", P.fitness(x))