import sdu_estimators

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import os


def test_gradient():
    import numpy as np
    import time

    print(dir(sdu_estimators))

    dt = 0.002
    tend = int(50./dt)
    gamma = np.array([0.5, 0.5, 0.5])/2
    theta_init = np.array([0, 0, 0])
    theta_true = np.array([1, 2, 3])
    r = 0.5

    integration_method = sdu_estimators.IntegrationMethod.Euler
    GradientEstimator = sdu_estimators.GradientEstimator_1x3(dt, gamma, theta_init, r, integration_method)
    print(GradientEstimator)
    print(GradientEstimator.get_estimate())

    # print(help(sdu_estimators.GradientEstimator.step))

    all_theta_est = np.zeros((tend, 3))

    before = time.time()

    for i in range(tend):
        t = i * dt
        phi = np.array([np.sin(t), np.cos(t), 1.])
        y = phi.T @ theta_true.reshape([3, 1])
        GradientEstimator.step(y, phi)

        all_theta_est[i, :] = GradientEstimator.get_estimate()

    after = time.time()
    print(f"duration: {(after - before)*1000} ms")

    print(GradientEstimator.get_estimate())
    if(os.name != 'nt'):
        plt.figure()
        plt.plot(all_theta_est)
        plt.grid()


def test_KRE():
    import numpy as np
    import time

    print(dir(sdu_estimators))

    dt = 0.002
    tend = int(50./dt)
    gamma = np.array([0.5, 0.5, 0.5])
    theta_init = np.array([0, 0, 0])
    theta_true = np.array([1, 2])
    r = 0.5

    # GradientEstimator = sdu_estimators.GradientEstimator(dt, gamma, theta_init, r)
    # print(GradientEstimator)
    # print(GradientEstimator.get_estimate())

    ell = 1.
    KRE = sdu_estimators.Kreisselmeier_1x2(dt, ell)

    all_Y = np.zeros((tend, 2))

    before = time.time()

    for i in range(tend):
        t = i * dt
        phi = np.array([np.sin(t), np.cos(t)])
        y = phi.T @ theta_true.reshape([2, 1])
        KRE.step(y, phi)

        all_Y[i, :] = KRE.getY()

    after = time.time()
    print(f"duration: {(after - before)*1000} ms")
    if(os.name != 'nt'):
        plt.figure()
        plt.plot(all_Y)

def test_DREM():
    import numpy as np
    import time

    print(dir(sdu_estimators))

    dt = 0.002
    tend = int(50./dt)
    tvec = np.arange(0, tend*dt, dt)
    gamma = np.ones(3,) * 500
    theta_init = np.array([0, 0, 0])
    theta_true = np.array([1, 2, 3])
    r = 0.5
    ell = 1.

    KRE = sdu_estimators.Kreisselmeier_1x3(dt, ell)
    # solver = sdu_estimators.DREM_1x2(dt, gamma, theta_init, KRE, r)
    solver = sdu_estimators.DREM(dt, gamma, theta_init, KRE)
    print(solver)
    # print(solver.get_estimate())
    # print(help(sdu_estimators.DREM_1x2.step))

    all_theta_est = np.zeros((tend, theta_true.shape[0]))

    before = time.time()

    for i in range(tend):
        t = i * dt
        phi = np.array([np.sin(t), np.cos(t), 1])
        y = phi.T @ theta_true.reshape([3, 1])
        solver.step(y, phi)

        all_theta_est[i, :] = solver.get_estimate()

    after = time.time()
    print(f"duration: {(after - before)*1000} ms")

    print(solver.get_estimate())

    if(os.name != 'nt'):
        plt.figure()
        plt.plot(tvec, all_theta_est)
        plt.grid()


def main():
    # test_sdu_estimators()
    test_gradient()
    # test_KRE()
    test_DREM()

  
    plt.show()
    

if __name__ == "__main__":
    main()