import sdu_estimators

import numpy as np


class RobotModel:
    def __init__(self, dof):
        self.dof = dof

    def get_inertia(self, q):
        return np.eye(self.dof)

    def get_coriolis(self, q, qd):
        return np.eye(self.dof)

    def get_gravity(self, q):
        tmp = np.ones(self.dof).reshape(self.dof, 1)
        print(tmp)
        return tmp    

    def get_friction(self, qd):
        return np.ones(self.dof).reshape(self.dof, 1)


class TwoLinkRobot(RobotModel):
    """
    Implements a two link robot with revolute joints.
    """
    def __init__(self, l1, l2, m1, m2, Il1, Il2, a1, a2):
        RobotModel.__init__(self, 2)

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.Il1 = Il1
        self.Il2 = Il2
        self.a1 = a1
        self.a2 = a2

        self.g = 9.82

    def get_inertia(self, q):
        b11 = (self.Il1 + self.m1 * self.l1**2 + self.Il2 
            + self.m2 * (self.a1**2 + self.l2**2 + 2 * self.a1 * self.l2 * np.cos(q[1]))
        )
        b12 = self.Il2 + self.m2 * (self.l2**2 + self.a1 * self.l2 * np.cos(q[1]))
        b22 = self.Il2 + self.m2 * self.l2**2

        return np.array([[b11, b12], 
                         [b12, b22]])

    def get_coriolis(self, q, qd):
        h = -self.m1 * self.a1 * self.l2 * np.sin(q[1])

        C = np.zeros([2, 2])
        C[0, 0] = h * qd[1]
        C[0, 1] = h * (qd[0] + qd[1])
        C[1, 0] = -h * qd[0]

        return C

    def get_gravity(self, q):
        grav = np.ones(self.dof).reshape(self.dof, 1)

        grav[0] = (
            (self.m1 * self.l1 + self.m2 * self.a1) * self.g * np.cos(q[0]) 
            + self.m2 * self.l2 * self.g * np.cos(q[0] + q[1])
        )

        grav[1] = self.m2 * self.l2 * self.g * np.cos(q[0] + q[1])

        return grav

    def get_friction(self, qd):
        return np.zeros(self.dof).reshape(self.dof, 1)



def get_position_and_velocity(t, size):
    q = np.sin(t) * np.ones(size)
    qd = np.cos(t) * np.ones(size)
    return q, qd

def measure_torque(t, size):
    tau = np.sin(t) * np.ones(size)
    return tau

def main():
    dt = 0.001
    dof = 2

    K = np.ones(dof)

    # robot_model = RobotModel(dof)
    l1 = 0.5
    l2 = 0.5
    m1 = 50.
    m2 = 50. 
    Il1 = 10.
    Il2 = 10.
    a1 = 1.
    a2 = 1.

    robot_model = TwoLinkRobot(l1, l2, m1, m2, Il1, Il2, a1, a2)

    observer = sdu_estimators.disturbance_observers.MomentumObserver(
        robot_model.get_inertia,
        robot_model.get_coriolis,
        robot_model.get_gravity,
        robot_model.get_friction,
        dt, 
        K
    )

    for i in range(int(1 / dt)):
        t = i * dt
        q, qd = get_position_and_velocity(t, dof)
        tau_m = measure_torque(t, dof)
        observer.update(q, qd, tau_m)
        print(f"Estimated torques at time {t}: {observer.estimated_torques().flatten()}")

main()