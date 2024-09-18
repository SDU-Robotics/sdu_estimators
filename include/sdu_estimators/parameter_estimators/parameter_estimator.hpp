#pragma once

#ifndef PARAMETERESTIMATOR_HPP
#define PARAMETERESTIMATOR_HPP

#include <Eigen/Dense>

namespace sdu_estimators::parameter_estimators
{

/**
 * This class provides a base class for the different parameter_estimators for estimating the parameter in the linear regression
 * equation (LRE) defined as:
 *
 * \f$ y(t) = \phi^T(t) \theta(t), \f$
 *
 * where \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output,
 * \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{m \times n} \f$ is the regressor matrix
 * and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^m \f$ is the parameter vector.
 *
 */

class ParameterEstimator
{
public:
  virtual ~ParameterEstimator() = default;

  /**
   * @brief Step the execution of the estimator (must be called in a loop externally)
   */
  virtual void step(const Eigen::VectorXd &y,
                    const Eigen::MatrixXd &phi) = 0;

  /**
   * @brief Get the current estimate of the parameter. Updates when the step function is called.
   */
  virtual Eigen::VectorXd get_estimate() = 0;

  /**
   * @brief Reset internal estimator variables
   */
  virtual void reset() = 0;
};

} //namespace sdu_estimators::parameter_estimators

#endif //PARAMETERESTIMATOR_HPP