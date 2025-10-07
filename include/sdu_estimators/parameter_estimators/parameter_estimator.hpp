#pragma once

#ifndef PARAMETERESTIMATOR_HPP
#define PARAMETERESTIMATOR_HPP

#include <cstdint>
#include <Eigen/Core>

namespace sdu_estimators::parameter_estimators
{

/**
 * This class provides a base class for the different parameter estimators for estimating the parameter in the linear regression
 * equation (LRE) defined as:
 *
 * \f$ y(t) = \phi^\intercal(t) \theta(t), \f$
 *
 * where \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output,
 * \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{p \times n} \f$ is the regressor matrix
 * and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^p \f$ is the parameter vector.
 *
 */
template <typename T, std::int32_t DIM_N, std::int32_t DIM_P>
class ParameterEstimator
{
public:
  virtual ~ParameterEstimator() = default;

  /**
   * @brief Step the execution of the estimator (must be called in a loop externally)
   *
   * @param y The output, \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$.
   * @param phi The regressor matrix, \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{p \times n} \f$.
   */
  virtual void step(const Eigen::Matrix<T, DIM_N, 1> &y,
                    const Eigen::Matrix<T, DIM_P, DIM_N> &phi) = 0;

  /**
   * @brief Get the current estimate of the parameter. Updates when the step function is called.
   *
   * @return The estimate of the parameter \f$ \hat{\theta} : \mathbb{R}_+ \to \mathbb{R}^p \f$.
   */
  virtual Eigen::Vector<T, DIM_P> get_estimate() = 0;

  /**
   * @brief Reset internal estimator variables
   */
  virtual void reset() = 0;
};

} //namespace sdu_estimators::parameter_estimators

#endif //PARAMETERESTIMATOR_HPP
