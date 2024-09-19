#include <iostream>
#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>

namespace sdu_estimators::parameter_estimators
{
  template <typename T, int32_t DIM_N, int32_t DIM_P>
  GradientEstimator<T, DIM_N, DIM_P>::~GradientEstimator() = default;

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  GradientEstimator<T, DIM_N, DIM_P>::GradientEstimator(const float dt, const float gamma,
                                                        const Eigen::Matrix<T, DIM_P, 1> & theta_init)
    : GradientEstimator(dt, gamma, theta_init, 1.0f) {}

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  GradientEstimator<T, DIM_N, DIM_P>::GradientEstimator(const float dt, const float gamma,
                                                        const Eigen::Matrix<T, DIM_P, 1> & theta_init, const float r)
  {
    this->dt = dt;
    this->gamma = gamma;
    this->theta_est = theta_init;
    this->theta_init = theta_init;
    this->p = theta_init.size();
    this->r = r;
  }

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void GradientEstimator<T, DIM_N, DIM_P>::step(const Eigen::Matrix<T, DIM_N, 1> &y,
                                                const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
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

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  Eigen::Vector<T, DIM_P> GradientEstimator<T, DIM_N, DIM_P>::get_estimate()
  {
    return theta_est.reshaped(p, 1);
  }

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void GradientEstimator<T, DIM_N, DIM_P>::reset()
  {
    theta_est = theta_init;
  }

}