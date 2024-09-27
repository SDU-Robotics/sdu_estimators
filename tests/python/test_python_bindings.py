import sdu_estimators


def test_sdu_estimators():
    assert sdu_estimators.add_one(1) == 2
    assert sdu_estimators.one_plus_one() == 2


def test_gradient():
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    print(dir(sdu_estimators))

    dt = 0.002
    tend = int(50./dt)
    gamma = np.array([0.5, 0.5, 0.5])
    theta_init = np.array([0, 0, 0])
    theta_true = np.array([1, 2, 3])
    r = 0.5

    GradientEstimator = sdu_estimators.GradientEstimator(dt, gamma, theta_init, r)
    print(GradientEstimator)
    print(GradientEstimator.get_estimate())

    all_theta_est = np.zeros((tend, 3))

    before = time.time()

    for i in range(tend):
        t = i * dt
        phi = np.array([np.sin(t), np.cos(t), 1.])
        y = phi.T @ theta_true.reshape([3, 1]);
        GradientEstimator.step(y, phi)

        all_theta_est[i, :] = GradientEstimator.get_estimate()

    after = time.time()
    print(f"duration: {(after - before)*1000} ms")

    print(GradientEstimator.get_estimate())

    plt.figure()
    plt.plot(all_theta_est)

    plt.show()


def main():
    test_sdu_estimators()
    test_gradient()
    

if __name__ == "__main__":
    main()