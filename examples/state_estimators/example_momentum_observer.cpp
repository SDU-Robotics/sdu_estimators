#include <Eigen/Core>
#include <iostream>
#include <sdu_estimators/state_estimators/momentum_observer.hpp>


/**
 * @brief Dummy robot model for testing the momentum observer.
 */
class RobotModel
{
 public:
  RobotModel(int dof) : dof(dof)
  {
  }

  Eigen::MatrixXd get_inertia_matrix(const Eigen::VectorXd& q) const
  {
    return Eigen::MatrixXd::Identity(dof, dof);
  }

  Eigen::MatrixXd get_coriolis(const Eigen::VectorXd& q, const Eigen::VectorXd& qd) const
  {
    return Eigen::MatrixXd::Identity(dof, dof);
  }

  Eigen::MatrixXd get_gravity(const Eigen::VectorXd& q) const
  {
    return Eigen::MatrixXd::Identity(dof, 1);
  }

  Eigen::MatrixXd get_friction(const Eigen::VectorXd& qd) const
  {
    return Eigen::MatrixXd::Identity(dof, 1);
  }

 private:
  int dof;
};

/**
 * @brief Placeholder function to simulate getting joint positions and velocities.
 * @param t 
 * @param size 
 * @return std::pair<Eigen::VectorXd, Eigen::VectorXd> 
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> getPositionAndVelocity(double t, size_t size)
{
  Eigen::VectorXd q(size);
  Eigen::VectorXd qd(size);

  // create a sinus path for all joints, let the speed be 0.25 radians per
  // second
  for (size_t i = 0; i < size; i++)
  {
    q[i] = 0.7 * M_PI * sin(t * M_PI + i);
    qd[i] = 0.5 * M_PI * cos(t * M_PI + i);
  }
  return { q, qd };
}

/**
 * @brief Placeholder function to simulate measuring joint torques.
 * 
 * @param t 
 * @param size 
 * @return Eigen::VectorXd 
 */
Eigen::VectorXd measureTorque(double t,size_t size) {
  Eigen::VectorXd tau(size);

  // create a sinus path for all joints, let the speed be 0.25 radians per
  // second
  for (size_t i = 0; i < size; i++)
  {
    tau[i] = -3 * M_PI * M_PI * cos(t * M_PI + i);

  }
  return tau;
}

int main()
{
  double dt = 0.01;
  size_t dof = 3;
  sdu_estimators::state_estimators::MomentumObserver observer(
      std::make_shared<RobotModel>(dof), dt, Eigen::VectorXd::Constant(dof, 1.0));

  for (double t = 0; t < 1; t += dt)
  {
    // Get the current position and velocity of the system
    auto [q, qd] = getPositionAndVelocity(t, dof);
    // Measure the Joint torques
    Eigen::VectorXd tau_m = measureTorque(t,dof);
    // Update the observer with the new measurements
    observer.update(q, qd, tau_m);

    // Print the estimated external torques
    std::cout << "Estimated torques at time " << t << ": " << observer.estimatedTorques().transpose() << std::endl;
  }

  return 0;
}