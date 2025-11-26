#pragma once
#ifndef GRADIENT_ESTIMATOR_SPHERE_HPP
#define GRADIENT_ESTIMATOR_SPHERE_HPP

#include <Eigen/Core>
#include <cstdint>

#include "sdu_estimators/math/riemannian_manifolds/sphere.hpp"
#include "sdu_estimators/parameter_estimators/parameter_estimator.hpp"
#include "sdu_estimators/utils.hpp"


namespace sdu_estimators::parameter_estimators
{
  /**
   *
   */

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  class GradientEstimatorSphere : public ParameterEstimator<T, DIM_N, DIM_P>
  {
   public:
    /**
     * @brief Constructor for the default gradient-based update rule.
     *
     * @param dt Sample time.
     * @param gamma \f$ \gamma \in \mathbb{R}^p \f$ is the vector of gains.
     * @param theta_init The initial value of the parameter estimate \f$ \hat{\theta}(0) \f$.
     */
    GradientEstimatorSphere(double dt, double gamma, const Eigen::Vector<T, DIM_P> &theta_init)
        : dt(dt),
          gamma(gamma),
          theta_est(theta_init),
          theta_init(theta_init),
          p(DIM_P)
    {
    }

    ~GradientEstimatorSphere()
    {
    }

    /**
     * @brief Step the execution of the estimator (must be called in a loop externally)
     */
    void step(const Eigen::Vector<T, DIM_N> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    // utils::IntegrationMethod method = utils::IntegrationMethod::Euler)
    {
      y_err = y - phi.transpose() * theta_est;

      Eigen::Vector<T, DIM_P> egrad = -phi * y_err;
      Eigen::Vector<T, DIM_P> rgrad = sphere_manifold.euclidean_to_riemannian_gradient(theta_est, egrad);

      Eigen::Vector<T, DIM_P> eta = -dt * gamma * rgrad;

      theta_est = sphere_manifold.retraction(theta_est, eta);
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
    }

   private:
    double dt, r, gamma;
    Eigen::Vector<T, DIM_P> theta_est, theta_init, dtheta;
    Eigen::Vector<T, DIM_N> y_err, y_old;
    Eigen::Matrix<T, DIM_P, DIM_N> phi_old;
    int p;  // number of parameters

    math::manifold::Sphere<T, DIM_P> sphere_manifold;
  };
}  // namespace sdu_estimators::parameter_estimators

#endif  // GRADIENT_ESTIMATOR_SPHERE_HPP
