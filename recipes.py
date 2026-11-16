import numpy as np
from grail_eval import GrailProblem

P = GrailProblem("tiny-observatory-in-dro")
np.set_printoptions(precision=2, suppress=True)

# Each recipe has 6 numbers, in this order: [dx, dy, dz, dvx, dvy, dvz]
print("recipe for y at t1:", P.stms[1][1] / 1.23)
print("recipe for y at t2:", P.stms[2][1] / 1.23)