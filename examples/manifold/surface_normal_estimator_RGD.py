#!/usr/bin/env python3
import copy
import numpy as np

import sdu_estimators

def skew_symmetric(vector):
    x = vector[0].item()
    y = vector[1].item()
    z = vector[2].item()
    Sv = np.zeros((3, 3))
    Sv[1, 0] = z
    Sv[2, 0] = -y
    Sv[0, 1] = -z
    Sv[2, 1] = x
    Sv[0, 2] = y
    Sv[1, 2] = -x
    return Sv


class SurfaceNormalEstimatorRGD:
    """ SurfaceNormalEstimatorRGD Class
    Args:
    """

    def __init__(self, dt, gamma, p_tcp_tip, initial_n=None):
        self.dt = dt
        self.gamma = gamma
        self.p_tcp_tip = p_tcp_tip
        self.Delta = 0.

        if initial_n is None:
            initial_n = [0, 0, 1]

        self.initial_n = np.reshape(np.asarray(initial_n), (3, 1))
        self.nc = None

        self.sphere_manifold = sdu_estimators.Sphere_3()
        self.sphere_est = sdu_estimators.GradientEstimatorSphere_1x3(dt, gamma, self.initial_n.flatten())

        self.reset()

    def reset(self):
        self.Phi_f = np.zeros((3, 3))
        self.y_f = np.zeros((3, 1))
        self.nc = self.initial_n

    def update(self, R_base_tcp, twist, wrench):
        v = twist[:3]
        v = np.reshape(v, (3, 1))
        omega = twist[3:]
        omega = np.reshape(omega, (3, 1))
        contact_force = wrench[:3]

        v_tip = v + skew_symmetric(omega) @ R_base_tcp @ self.p_tcp_tip

        #
        #nabla = np.cross(contact_force.flatten(), v_tip.flatten()).reshape((3, 1))

        # estimation
        # y = np.array([[0], [0]])
        # phi = np.hstack([v_tip, nabla])
        y = np.array([[0]])
        phi = v_tip

        self.sphere_est.step(y.flatten(), phi.flatten())

        self.nc = self.sphere_est.get_estimate()

        # egrad = -phi @ (y - phi.T @ self.nc)
        # egrad = np.linalg.inv(np.eye(3) + phi @ phi.T) @ egrad
        #
        # # from Euclidean gradient to Riemannian gradient
        # rgrad = self.sphere_manifold.euclidean_to_riemannian_gradient(self.nc.flatten(), egrad.flatten())
        #
        # # Integrate using manifold
        # self.nc = self.sphere_manifold.retraction(
        #     self.nc.flatten(),
        #     (-self.dt * self.gamma * rgrad).flatten()
        # )

    def get_estimate(self):
        return copy.copy(self.nc)


if __name__ == '__main__':
    # import pandas as pd
    import matplotlib.pyplot as plt
    import time

    data_path = "data1.csv"
    data = np.loadtxt(data_path, delimiter=',', skiprows=1)
    # data = np.vstack([data, data, data])
    fs = 500.
    dt = 1. / fs
    # time = data[:, 0]

    velocity = data[:, 1:]
    velocity = np.vstack([velocity] * 1)
    print(velocity)

    tvec = np.arange(0, velocity.shape[0] * dt, dt)

    # velocity += np.array([0, 0, 1e-6])
    twists = np.hstack((velocity, np.zeros((velocity.shape))))
    twists += 0 * np.random.randn(*twists.shape)
    velocity = twists[:, 0:3]
    R_base_tcp = np.eye(3)

    tmp = np.pi/4.
    #    def __init__(self, dt, gamma, p_tcp_tip, initial_n=None, ell=0., r=1.):
    # normal_estimator = SurfaceNormalEstimatorDREM(dt, gamma=1e39, p_tcp_tip=np.zeros((3, 1)),
    normal_estimator = SurfaceNormalEstimatorRGD(dt, gamma=200, p_tcp_tip=np.zeros((3, 1)),
                                                  initial_n=[0, np.sin(tmp), np.cos(tmp)])
    normals = np.empty((0, 3), float)

    all_Delta = []
    # H = np.diag([0, 0, 1e-6])
    # H = np.diag([0, 0, 1e-4])
    H = np.diag([0, 0, 0])

    wrench = np.zeros((6, 1))

    start = time.time_ns()
    for twist in twists:
        normal_estimator.update(R_base_tcp, twist, wrench)
        normal = normal_estimator.get_estimate().reshape((1, 3))
        normals = np.append(normals, normal, axis=0)
        all_Delta.append(float(normal_estimator.Delta))

    end = time.time_ns()
    dur = end - start
    print(f"Duration:  {dur * 1e-9:.4f} s")
    print(f"Duration per iterations: {dur/twists.shape[0]*1e-6:.4f} ms/it")

    fig1, axes1 = plt.subplots(nrows=1, ncols=3)
    axes1[0].plot(tvec, normals[:, 0], label="x")
    axes1[0].set_ylim(-1.2, 1.2)
    axes1[1].plot(tvec, normals[:, 1], label="y")
    axes1[1].set_ylim(-1.2, 1.2)
    axes1[2].plot(tvec, normals[:, 2], label="z")
    axes1[2].set_ylim(-1.2, 1.2)
    [ax.grid() for ax in axes1]
    plt.suptitle("Nc Estimate")

    fig2, axes2 = plt.subplots(nrows=1, ncols=3)
    axes2[0].plot(tvec, velocity[:, 0], label="x")
    axes2[1].plot(tvec, velocity[:, 1], label="y")
    axes2[2].plot(tvec, velocity[:, 2], label="z")
    plt.suptitle("Velocity")

    plt.figure()
    plt.plot(tvec, all_Delta)

    plt.show()
