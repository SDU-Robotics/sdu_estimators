#!/usr/bin/env python3

import sdu_estimators

import numpy as np
import matplotlib.pyplot as plt

def main():
    dt = 0.01
    tend = int(50 / dt)

    gamma = np.array([[10.], [10.]]).flatten() * 2e6

    ell = 10.
    r = 1.

    theta_init = np.zeros((2, 1)).flatten()
    theta_true = np.array([[1.], [2.]]).flatten()

    intg_method = sdu_estimators.integrator.RK4

    reg_ext = sdu_estimators.regressor_extensions.Kreisselmeier_1x2(dt, ell, intg_method)

    solver = sdu_estimators.parameter_estimators.DREM(dt, gamma, theta_init, reg_ext, r, intg_method)

    all_theta_est = np.zeros((tend, 2))

    tvec = []

    for i in range(tend):
        t = i * dt
        tvec.append(t)
        phi = np.array([[np.sin(t)], [np.cos(t)]])
        y = phi.T @ theta_true

        solver.step(y.flatten(), phi.flatten())

        all_theta_est[i, :] = solver.get_estimate()

    print(all_theta_est)

    plt.figure()
    plt.plot(tvec, all_theta_est, label="Estimated")
    plt.hlines(theta_true, tvec[0], tvec[-1], linestyles='dashed', label='Actual')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
