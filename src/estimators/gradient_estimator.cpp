#include <sdu_estimators/estimators/gradient_estimator.hpp>
#include <iostream>

namespace sdu_estimators::estimators
{
  GradientEstimator::~GradientEstimator() = default;

  GradientEstimator::GradientEstimator(const float dt, const float gamma, const Eigen::VectorXd& theta_init)
    : GradientEstimator(dt, gamma, theta_init, 1.0f) {}

  GradientEstimator::GradientEstimator(const float dt, const float gamma, const Eigen::VectorXd& theta_init, const float r)
  {
    this->dt = dt;
    this->gamma = gamma;
    this->theta_est = theta_init;
    this->theta_init = theta_init;
    this->p = theta_init.size();
    this->r = r;
  }

  void GradientEstimator::step(const Eigen::VectorXd& y, const Eigen::MatrixXd& phi)
  {
    // const int n = phi.cols();
    // const int m = phi.rows();

    // assert (y.cols() == 1) && (y.rows() == n) && (m == p);

    y_err = y - phi.transpose() * theta_est;

    Eigen::VectorXd tmp1 = y_err.array().abs().pow(r);
    Eigen::VectorXd tmp2 = y_err.cwiseSign();

    std::cout << tmp1 << " " << tmp2 << std::endl;

    dtheta = gamma * phi * (
      tmp1.cwiseProduct(tmp2)
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