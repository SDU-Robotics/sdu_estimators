#pragma once
#ifndef CASCADED_DREM_HPP
#define CASCADED_DREM_HPP

#include <cstdint>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
#include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>

namespace sdu_estimators::parameter_estimators
{
    /**
     * @brief Cascaded DREM
     * 
     * An implementation of the cascaded DREM estimator, recently precented in 
     * \verbatim embed:rst:inline :cite:`Diget2026` \endverbatim
     */

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    class CascadedDREM // : public ParameterEstimator<T, DIM_N, DIM_P>
    {
    public:
        CascadedDREM(float dt, float a,
            parameter_estimators::utils::IntegrationMethod method = parameter_estimators::utils::IntegrationMethod::Euler) :
                dt(dt), a(a), intg_method(intg_method)
        {
            reg_ext_outer = new regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P>(dt, a, intg_method);
            reg_ext_inner = new regressor_extensions::Kreisselmeier<T, 2*DIM_N, 2*DIM_P>(dt, a, intg_method);
        }

        void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_N, 1> &dy, 
                  const Eigen::Matrix<T, DIM_P, DIM_N> &phi, const Eigen::Matrix<T, DIM_P, DIM_N> &dphi)
        {
            y_ext << y, dy;
            phi_ext << phi.transpose(), phi.transpose()*0,
                       dphi.transpose(), phi.transpose();

            reg_ext_outer->step(y, phi);
            reg_ext_inner->step(y_ext, phi_ext);

            // Inner loop where we find an estimate of dtheta_ext.
            Eigen::Matrix<T, 2*DIM_P, 1> y_ext_f = reg_ext_inner->getY();
            Eigen::Matrix<T, 2*DIM_P, 2*DIM_P> phi_ext_f = reg_ext_inner->getPhi();

            Eigen::FullPivLU<Eigen::Matrix<T, 2*DIM_P, 2*DIM_P>> lu_ext(phi_ext_f);
            T Delta_ext = lu_ext.determinant();

            if (!std::isfinite(Delta_ext))
                Delta_ext = 0;

            Yvar_ext = lu_ext.solve(Delta_ext * y_ext_f);
            
            if (Delta_ext > 1e-10)
                theta_est_ext = Yvar_ext / Delta_ext;    

            std::cout << "i\n" << theta_est_ext << std::endl;

            // 
            Eigen::Matrix<T, DIM_P, 1> y_f = reg_ext_outer->getY();
            Eigen::Matrix<T, DIM_P, DIM_P> phi_f = reg_ext_outer->getPhi();

            Eigen::FullPivLU<Eigen::Matrix<T, DIM_P, DIM_P>> lu(phi_f);
            T Delta = lu.determinant();

            if (!std::isfinite(Delta))
                Delta = 0;

            Yvar = lu.solve(Delta * y_f);  // To compute phi_f^{-1} * Delta * y_f.

            if (Delta > 1e-10)
                theta_est = Yvar / Delta;
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
            reg_ext_inner->reset();
        }

    private:
        regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P> * reg_ext_outer;
        regressor_extensions::Kreisselmeier<T, 2*DIM_N, 2*DIM_P> * reg_ext_inner;

        float dt;
        float a;

        utils::IntegrationMethod intg_method;

        // Variables for computation
        Eigen::Matrix<T, DIM_P, 1> theta_est, theta_init, Yvar;
        Eigen::Matrix<T, DIM_P*2, 1> theta_est_ext, Yvar_ext;

        Eigen::Matrix<T, DIM_N*2, 1> y_ext;
        Eigen::Matrix<T, DIM_P*2, DIM_N*2> phi_ext;
    };
}

#endif  // CASCADED_DREM_HPP