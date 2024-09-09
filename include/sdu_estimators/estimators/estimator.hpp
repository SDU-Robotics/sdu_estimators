#pragma once

#ifndef ESTIMATOR_HPP
#define ESTIMATOR_HPP

#include <Eigen/Dense>

namespace sdu_estimators::estimators
{

/**
 * This class provides a base class for the different estimators for estimating the parameter in the linear regression
 * equation (LRE) defined as:
 *
 * \f$ y(t) = \phi^T(t) \theta(t), \f$
 *
 * where \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output, \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{m \times n} \f$ is,
 * the regressor matrix and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^m \f$ is the parameter vector.
 */

class Estimator
{
public:
  /**
   * @brief Step the execution of the estimator (must be called in a loop externally)
   */
  virtual void step(const Eigen::VectorXd &y,
                    const Eigen::MatrixXd &phi);

  /**
   * @brief Get the current estimate of the parameter. Updates when the step function is called.
   */
  virtual Eigen::VectorXd get_estimate();

  /**
   * @brief Reset internal estimator variables
   */
  virtual void reset();

  virtual ~Estimator() = default;
};

} //namespace sdu_estimators::estimators

#endif //ESTIMATOR_HPP
