#include <sdu_estimators/estimators/gradient_estimator.hpp>

namespace sdu_estimators::estimators
{
  GradientEstimator::~GradientEstimator() = default;
  GradientEstimator::GradientEstimator() = default;

  GradientEstimator::GradientEstimator(const float dt, const float gamma, const Eigen::VectorXd& theta_init)
  {
    this->dt = dt;
    this->gamma = gamma;
    this->theta_est = theta_init;
    this->theta_init = theta_init;
    this->p = theta_init.size();
  }

  void GradientEstimator::step(const Eigen::VectorXd& y, const Eigen::MatrixXd& phi)
  {
    // const int n = phi.cols();
    // const int m = phi.rows();

    // assert (y.cols() == 1) && (y.rows() == n) && (m == p);

    dtheta = gamma * phi * (
      y - phi.transpose() * theta_est
    );

    theta_est += dt * dtheta;
  }

  Eigen::VectorXd GradientEstimator::get_estimate()
  {
    return theta_est.reshaped(p, 1);
  }

  void GradientEstimator::reset()
  {
    theta_est = theta_init;
  }

}