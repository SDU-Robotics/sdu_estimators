#pragma once
#ifndef DREM_HPP
#define DREM_HPP

#include <Eigen/Core>
#include <Eigen/LU>

#include <cstdint>
#include <cmath>
#include <iostream>
#include <type_traits>

#include "sdu_estimators/parameter_estimators/parameter_estimator.hpp"
#include "sdu_estimators/regressor_extensions/regressor_extension.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

#include "sdu_estimators/typedefs.hpp"

namespace sdu_estimators::parameter_estimators
{
  /**
   * An implementation of dynamic regressor extension and mixing (DREM) as described in e.g.,
   * \verbatim embed:rst:inline :cite:`Aranovskiy2017` \endverbatim.
   *
   */

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  class DREM : public ParameterEstimator<T, DIM_N, DIM_P>
  {
    // static_assert(std::is_base_of_v<regressor_extensions::RegressorExtension, T_REG_EXT>,
    //   "T_REG_EXT must derive from regressor_extensions::RegressorExtension");

   public:
    DREM(
        float dt,
        const Eigen::Matrix<T, DIM_P, 1> &gamma,
        const Eigen::Matrix<T, DIM_P, 1> &theta_init,
        regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> *reg_ext,
        integrator::IntegrationMethod method = integrator::IntegrationMethod::Euler)
        : DREM(dt, gamma, theta_init, reg_ext, 1.0f, method)
    {
    }

    DREM(
        float dt,
        const Eigen::Matrix<T, DIM_P, 1> &gamma,
        const Eigen::Matrix<T, DIM_P, 1> &theta_init,
        regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> *reg_ext,
        float r,
        integrator::IntegrationMethod method = integrator::IntegrationMethod::Euler)
        : dt(dt),
          gamma(gamma),
          r(r),
          theta_est(theta_init),
          theta_init(theta_init),
          p(theta_init.size()),
          reg_ext(reg_ext),
          intg_method(method)
    {
    }

    ~DREM()
    {
    }

    /**
     * @brief Step the execution of the estimator (must be called in a loop externally)
     */
    void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    {
      reg_ext->step(y, phi);

      Eigen::Matrix<T, DIM_P, 1> y_f = reg_ext->getY();
      Eigen::Matrix<T, DIM_P, DIM_P> phi_f = reg_ext->getPhi();

      // Eigen::HouseholderQR<Eigen::Matrix<T, DIM_P, DIM_P>> qr(phi_f);

      Eigen::FullPivLU<Eigen::Matrix<T, DIM_P, DIM_P>> lu(phi_f);
      T Delta = lu.determinant();

      if (!std::isfinite(Delta))
        Delta = 0;
      
      Yvar = lu.solve(Delta * y_f);  // To compute phi_f^{-1} * Delta * y_f.

      auto get_dydt = [=](Eigen::Vector<T, DIM_P> theta_est_)
      {
        Eigen::Vector<T, DIM_P> y_err = Yvar - Delta * theta_est_;
        Eigen::Vector<T, DIM_P> tmp1 = y_err.array().abs().pow(r);
        Eigen::Vector<T, DIM_P> tmp2 = y_err.cwiseSign();

        Eigen::Vector<T, DIM_P> dtheta = gamma.asDiagonal() * Delta * tmp1.cwiseProduct(tmp2);
        return dtheta;
      };

      theta_est = integrator::Integrator<T, DIM_P, 1>::integrate(
        theta_est, 
        get_dydt, 
        dt,
        intg_method);
    }

    /**
     * @brief Get the current estimate of the parameter. Updates when the step function is called.
     */
    Eigen::Vector<T, DIM_P> get_estimate()
    {
      return theta_est;
    }

    /**
     * @brief Reset internal estimator variables
     */
    void reset()
    {
      theta_est = theta_init;
      reg_ext->reset();
    }

   private:
    float dt;
    Eigen::Matrix<T, DIM_P, 1> gamma;
    float r;
    Eigen::Matrix<T, DIM_P, 1> theta_est, theta_init, Yvar;
    Eigen::Matrix<T, DIM_N, 1> y_err;
    int p;  // number of parameters

    float y_err_i;

    // T_REG_EXT<T, DIM_N, DIM_P> reg_ext;
    // T_REG_EXT reg_ext;
    // regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P> reg_ext;
    regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> *reg_ext;

    integrator::IntegrationMethod intg_method;
  };
}  // namespace sdu_estimators::parameter_estimators

#endif  // DREM_HPP
