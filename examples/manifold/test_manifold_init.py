#/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sdu_estimators

import pymanopt

import time

###

sphere = sdu_estimators.Sphere_3()

##

v1 = np.array([0.74766, -0.232124, 0.622192])

v2 = np.array([0.504459, 0.695816, -0.511235])

v3 = v2 * 2

start = time.time_ns()

d = sphere.dist(v1, v2)
projv = sphere.projection(v1, v3)
retr = sphere.retraction(v1, v3)
expv = sphere.exp(v1, v3)
logv = sphere.log(v1, v2)

end = time.time_ns()
duration = end - start
print(f"Duration {duration*1e-6:.4f} ms")
#
# print("dist", d)
# print("projv", projv)
# print("retr", retr)
# print("expv", expv)
# print("logv", logv)

##
sphere_manopt = pymanopt.manifolds.Sphere(3)

start = time.time_ns()

d2 = sphere_manopt.dist(v1, v2)
projv2 = sphere_manopt.projection(v1, v3)
retr2 = sphere_manopt.retraction(v1, v3)
expv2 = sphere_manopt.exp(v1, v3)
logv2 = sphere_manopt.log(v1, v2)

end = time.time_ns()
duration = end - start
print(f"Duration {duration*1e-6:.4f} ms")

# print("dist2", d2)