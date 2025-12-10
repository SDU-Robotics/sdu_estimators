#!/usr/bin/env python3

import sdu_estimators

import numpy as np
import matplotlib.pyplot as plt


def main():
    print(sdu_estimators.parameter_estimators.__doc__)
    print(sdu_estimators.parameter_estimators.__name__)
    print(dir(sdu_estimators.parameter_estimators))

    dt = 0.001
    tend = int(50 / dt)

    gamma = 0.5 * np.array([[1], [1]]).flatten()

    r = 1

    theta_init = np.zeros((1, 2)).flatten()
    theta_true = np.array([[1.], [2.]]).flatten()

    method = sdu_estimators.integrator.IntegrationMethod.RK4
    print(method)

    # solver = sdu_estimators.parameter_estimators.GradientEstimator_4x2(dt, gamma, theta_init, r, method)
    solver = sdu_estimators.parameter_estimators.GradientEstimator_1x2(dt, gamma, theta_init, r, method)

    theta_lower_bound = np.array((1.4, 1.5))
    theta_upper_bound = np.array((1.5, 2.5))
    solver.set_theta_bounds(theta_lower_bound, theta_upper_bound)

    # solver = sdu_estimators.parameter_estimators.GradientEstimator_1x2(dt, gamma, theta_init, r, method)

    all_theta_est = np.zeros((tend, 2))

    tvec = []

    for i in range(tend):
        t = i * dt
        tvec.append(t)
        phi = np.array([[np.sin(t)], [np.cos(t)]]).flatten()
        # phi = np.array([[2*np.cos(t), -np.cos(t+1), 3*np.cos(2*t+1/2.), 2*np.cos(t/3.+1)],
        #                 [np.cos(2*t), np.cos(t/2.), 2*np.cos(3*t/2.+3/4.), -3*np.cos(4*t/3.)]])
        y = phi.T @ theta_true

        solver.step(y.flatten(), phi)

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
