#pragma once
#ifndef GRADIENT_ESTIMATOR_HPP
#define GRADIENT_ESTIMATOR_HPP

#include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>

namespace sdu_estimators::parameter_estimators
{
  /**
   * A simple gradient-based parameter estimator as described in e.g.,
   *   S. Sastry and M. Bodson, Adaptive Control: Stability, Convergence, and Robustness.
   *        USA: Prentice-Hall, Inc., 1989, isbn: 0130043265.
   *
   * The parameter \f$ \theta \f$ can be estimated by \f$ \hat{\theta} \f$ with the following update rule:
   *
   * \f$  \dot{\hat{\theta}}(t) = \gamma \phi(t) (y(t) - \phi^T(t) \hat{\theta}(t)), \f$
   *
   * where \f$ \gamma > 0 \f$ is a tuning parameter, \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output, \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{m \times n} \f$ is,
   * the regressor matrix and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^m \f$ is the parameter vector.
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class GradientEstimator : public ParameterEstimator<T, DIM_N, DIM_P>
  {
  public:
    GradientEstimator(float dt, const Eigen::Vector<T, DIM_P> gamma, const Eigen::Vector<T, DIM_P> & theta_init)
      : GradientEstimator(dt, gamma, theta_init, 1.0f)
    {
    }

    GradientEstimator(float dt, const Eigen::Vector<T, DIM_P> gamma, const Eigen::Matrix<T, DIM_P, 1> & theta_init, float r)
    {
      this->dt = dt;
      this->gamma = gamma;
      this->theta_est = theta_init;
      this->theta_init = theta_init;
      this->p = theta_init.size();
      this->r = r;
    }

    ~GradientEstimator()
    {
    }

    /**
     * @brief Step the execution of the estimator (must be called in a loop externally)
     */
    void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    {
      y_err = y - phi.transpose() * theta_est;

      Eigen::VectorXd tmp1 = y_err.array().abs().pow(r);
      Eigen::VectorXd tmp2 = y_err.cwiseSign();

      // std::cout << tmp1 << " " << tmp2 << std::endl;

      dtheta = gamma.asDiagonal() * phi * (
        tmp1.cwiseProduct(tmp2)
      );

      theta_est += dt * dtheta;
    }

    /**
     * @brief Get the current estimate of the parameter. Updates when the step function is called.
     */
    Eigen::Vector<T, DIM_P> get_estimate()
    {
      return theta_est.reshaped(DIM_P, 1);
    }

    /**
     * @brief Reset internal estimator variables
     */
    void reset()
    {
      theta_est = theta_init;
      // for (int i = 0; i < DIM_P; ++i)
      // {
      //   theta_est[i] = theta_init[i];
      // }
    }

  private:
    float dt{};
    float r{};
    Eigen::Vector<T, DIM_P> theta_est, theta_init, dtheta, gamma;
    Eigen::Vector<T, DIM_N> y_err;
    int p{}; // number of parameters
  };
}

#endif //GRADIENT_ESTIMATOR_HPP
