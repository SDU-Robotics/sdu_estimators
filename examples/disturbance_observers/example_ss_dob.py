#!/usr/bin/env python3

import sdu_estimators
import numpy as np
import matplotlib.pyplot as plt


def main():
    A = np.array([[0, 0],
                  [-1, -1]])

    B = np.array([[0.],
                  [1.]])

    C = np.eye(2, dtype=np.float64)

    D = np.eye(2, dtype=np.float64)

    # Ld = np.array([[1., 1.]])
    # L = np.array([[1., 1.]])

    Ld = 5e1*np.eye(2, dtype=np.float64)
    L = 5e1*np.eye(2, dtype=np.float64)

    print(A)
    print(B)
    print(C)
    print(D)

    dt = 1e-3
    end_time = 2  # s
    tvec = np.arange(0, end_time, dt)

    intg_method = sdu_estimators.integrator.IntegrationMethod.Euler

    dob = sdu_estimators.disturbance_observers.StateSpaceDisturbanceObserver_2x1x2(A, B.flatten(), D, Ld, L, dt, intg_method)

    all_x = np.zeros((tvec.shape[0], 2))
    all_z = np.zeros((tvec.shape[0], 2))
    all_u = np.zeros((tvec.shape[0], 1))
    all_d = np.zeros((tvec.shape[0], 2))
    all_d_est = np.zeros((tvec.shape[0], 2))
    all_d_est_cpp = np.zeros((tvec.shape[0], 2))


    for i in range(tvec.shape[0] - 1):
        t = dt * i

        u = np.sin(10*t)
        u = np.array([[u]])
        # switching disturbance
        if t < 1. or t > 1.5:
            d = np.array([[0.], [0.]])
        else:
            d = np.array([[0.1], [0.05]])
        
        get_dx = lambda x : A @ x + B @ u + D @ d

        all_x[i + 1, :] = all_x[i, :] + dt * get_dx(all_x[i, :].reshape((2, 1))).T
        # all_x[i + 1, :] = sdu_estimators.integrator.integrate_2x1(all_x[i, :], get_dx, dt, intg_method)

        z = all_z[i, :].reshape((2, 1))
        x = all_x[i, :].reshape((2, 1))
        dz = -Ld @ D @ (z + L @ x) - Ld @ (A @ x + B @ u)

        all_z[i + 1, :] = (z + dt * dz).flatten()

        all_d_est[i, :] = (z + Ld @ x).flatten()

        all_u[i, :] = u.flatten()
        all_d[i, :] = d.flatten()
        
        dob.step(x.flatten(), u.flatten())
        all_d_est_cpp[i, :] = dob.get_disturbance_estimate()


    print(all_x)
    print(all_d)
    print(all_d_est)

    plt.figure()
    plt.plot(tvec, all_x)
    plt.grid()

    plt.figure()
    plt.plot(tvec, all_u)
    plt.grid()
    plt.title("u")

    plt.figure()
    plt.plot(tvec, all_d)
    plt.plot(tvec, all_d_est, '--')
    plt.plot(tvec, all_d_est_cpp, '-.')
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
