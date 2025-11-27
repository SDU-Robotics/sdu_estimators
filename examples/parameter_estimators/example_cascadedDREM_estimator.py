#!/usr/bin/env python3

import sdu_estimators

import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    dt = 0.001
    tend = int(12 / dt)

    a = 10.

    theta_true = np.array([3., 0.])

    DIM_N = 4
    DIM_P = 2
    method = sdu_estimators.integrator.IntegrationMethod.RK4

    # solver_cascade = sdu_estimators.parameter_estimators.CascadedDREM(dt, a, DIM_N, DIM_P, method)
    # solver_standard = sdu_estimators.parameter_estimators.CascadedDREM(dt, a, DIM_N, DIM_P, method)

    # sdu_estimators.parameter_estimators.CascadedDREM_4x2(dt, a, method) # if this line is not run, then solver_cascade will return nan estimates.
    solver_cascade = sdu_estimators.parameter_estimators.CascadedDREM_4x2(dt, a, method)
    solver_standard = sdu_estimators.parameter_estimators.CascadedDREM_4x2(dt, a, method)

    all_theta_est_cascade = np.zeros((tend, 2))
    all_theta_est_standard = np.zeros((tend, 2))
    all_theta_true = np.zeros((tend, 2))

    tvec = []

    t_begin = time.time_ns()

    for i in range(tend):
        t = i * dt
        tvec.append(t)
        
        phi = np.array([[2.*np.cos(t), -np.cos(t+1.), 3.*np.cos(2.*t+1./2.), 2.*np.cos(t/3. + 1.)],
                        [np.cos(2.*t), np.cos(t/2.), 2.*np.cos(3.*t/2. + 3./4.), -3.*np.cos(4.*t/3.)]])

        dphi = np.array([[-2.*np.sin(t), np.sin(t+1.), -6.*np.sin(2.*t+1./2.), -2.*np.sin(t/3. + 1.)/3],
                        [-2.*np.sin(2.*t), -np.sin(t/2.)/2., -3.*np.sin(3.*t/2. + 3./4.), 4.*np.sin(4.*t/3.)]])

        # dtheta = np.array([
        #     theta_true[1],
        #     (2. - theta_true[0]**2) * theta_true[1] / 3. - theta_true[0]
        # ])

        get_dtheta = lambda theta_ : np.array([
            theta_[1],
            (2. - theta_[0]**2) * theta_[1] / 3. - theta_[0]
        ])

        # theta_true = theta_true + dt * dtheta
        theta_true = sdu_estimators.integrator.integrate_2x1(
            theta_true,
            get_dtheta,
            dt,
            method
        )

        dtheta = get_dtheta(theta_true)
        # print(dtheta)

        y = phi.T @ theta_true
        dy = dphi.T @ theta_true + phi.T @ dtheta

        # print("y", y)
        # print("dy", dy)
        # print("phi", phi)
        # print("dphi", dphi)
        
        # print(dy)
        # print(dphi)

        solver_cascade.set_dy_dphi(dy, dphi)
        solver_cascade.step(y, phi)

        solver_standard.set_dy_dphi(np.zeros_like(dy), np.zeros_like(dphi))
        solver_standard.step(y, phi)

        all_theta_est_cascade[i, :] = solver_cascade.get_estimate()
        all_theta_est_standard[i, :] = solver_standard.get_estimate()

        # print("Cascade:", all_theta_est_cascade[i, :])
        # print("Standard:", all_theta_est_standard[i, :])
        
        all_theta_true[i, :] = theta_true

    t_end = time.time_ns()
    t_dur = t_end - t_begin

    print(f"Loop duration: {1e-6 * t_dur} ms.")
    print(f"Time per iteration: {1e-6 * t_dur/tend} ms/it")

    plt.figure()
    plt.plot(tvec, all_theta_est_cascade, label="Estimated, cascade")
    plt.plot(tvec, all_theta_est_standard, label="Estimated, standard")
    plt.plot(tvec, all_theta_true, label="Actual")
    # plt.ylim([-5,5])
    plt.legend()
    plt.grid()

    ##
    err_norm_cascade = np.linalg.norm(all_theta_est_cascade - all_theta_true, axis=1)
    err_norm_standard = np.linalg.norm(all_theta_est_standard - all_theta_true, axis=1)

    plt.figure()
    plt.plot(tvec, err_norm_cascade)
    plt.plot(tvec, err_norm_standard)
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
