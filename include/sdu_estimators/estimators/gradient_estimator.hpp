#pragma once
#ifndef GRADIENT_ESTIMATOR_HPP
#define GRADIENT_ESTIMATOR_HPP

#include <sdu_estimators/estimators/estimator.hpp>

namespace sdu_estimators::estimators
{
  /**
   * A simple gradient-based parameter estimator as described in e.g.,
   *   S. Sastry and M. Bodson, Adaptive Control: Stability, Convergence, and Robustness.
   *        USA: Prentice-Hall, Inc., 1989, isbn: 0130043265.
   *
   *  The parameter \f$ \theta \f$ can be estimated by \f$ \hat{\theta} \f$ with the following update rule:
   *
   *  \f$  \dot{\hat{\theta}}(t) = \gamma \phi(t) (y(t) - \phi^T(t) \hat{\theta}(t)), \f$
   *
   *  where \f$ \gamma > 0 \f$ is a tuning parameter, \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output, \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{m \times n} \f$ is,
   * the regressor matrix and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^m \f$ is the parameter vector.
   */

  class GradientEstimator : public Estimator
  {
  public:
    GradientEstimator(float dt, float gamma, const Eigen::VectorXd & theta_init);
    GradientEstimator(float dt, float gamma, const Eigen::VectorXd & theta_init, float r);
    ~GradientEstimator() override;

    /**
     * @brief Step the execution of the estimator (must be called in a loop externally)
     */
    void step(const Eigen::VectorXd &y, const Eigen::MatrixXd &phi) override;

    /**
     * @brief Get the current estimate of the parameter. Updates when the step function is called.
     */
    Eigen::VectorXd get_estimate() override;

    /**
     * @brief Reset internal estimator variables
     */
    void reset() override;

  private:
    float dt{};
    float gamma{};
    float r{};
    Eigen::VectorXd theta_est, theta_init, dtheta, y_err;
    int p{}; // number of parameters
  };
}

#endif //GRADIENT_ESTIMATOR_HPP
