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
        return np.eye(self.dof).reshape(self.dof, 1)

    def get_friction(self, qd):
        return np.eye(self.dof).reshape(self.dof, 1)

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

    robot_model = RobotModel(dof)
    observer = sdu_estimators.MomentumObserver(robot_model.get_inertia,
                                               robot_model.get_coriolis,
                                               robot_model.get_gravity,
                                               robot_model.get_friction,
                                               dt, np.ones(dof))

    for i in range(int(1 / dt)):
        t = i * dt
        q, qd = get_position_and_velocity(t, dof)
        tau_m = measure_torque(t, dof)
        observer.update(q, qd, tau_m)
        print(f"Estimated torques at time {t}: {observer.estimated_torques().flatten()}")

main()