#!/usr/bin/env python3

import sdu_estimators

import numpy as np
import matplotlib.pyplot as plt


def main():
    dt = 0.002
    tend = 50 / dt

    gamma = np.array([[10.], [10.]])

    ell = 1.
    r = 1.

    theta_init = np.zeros((2, 1))
    theta_true = np.array([[1.], [2.]])

    solver = sdu_estimators._sdu_estimators.Gradient(dt, gamma, theta_init, ell, r)

    all_theta_est = np.zeros((tend, 2))

    for i in range(tend):
        t = i * dt
        phi = np.array([[np.sin(t)], np.cos(t)])
        y = phi.T @ theta_true

        solver.step(y, phi)

        all_theta_est[i, :] = solver.get_estimate()

    print(all_theta_est)


if __name__ == "__main__":
    main()