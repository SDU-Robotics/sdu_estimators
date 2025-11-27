#pragma once
#ifndef CASCADED_DREM_HPP
#define CASCADED_DREM_HPP

#include <cstdint>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

#include "sdu_estimators/parameter_estimators/parameter_estimator.hpp"
#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"
#include "sdu_estimators/integrator/integrator.hpp"


namespace sdu_estimators::parameter_estimators
{
    /**
     * @brief Cascaded DREM
     * 
     * An implementation of the cascaded DREM estimator, recently described in 
     * \verbatim embed:rst:inline :cite:`Diget2026` \endverbatim
     */

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    class CascadedDREM : public ParameterEstimator<T, DIM_N, DIM_P>
    {
    public:
        CascadedDREM(float dt, float a,
            integrator::IntegrationMethod method = integrator::IntegrationMethod::RK4) :
                dt(dt), a(a), intg_method(method)
        {
            reg_ext_second = new regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P>(dt, a, intg_method);
            reg_ext_first = new regressor_extensions::Kreisselmeier<T, 2*DIM_N, 2*DIM_P>(dt, a, intg_method);
            
            phi_ext.setZero();
            Hc_state.setZero();
            dHc_state.setZero();
            dHc_state_old.setZero();
            dy.setZero();
            dphi.setZero();
            theta_est.setZero();

            eps = 1e-10;
        }

        /**
         * @brief Step the execution of the estimator (must be called in a loop externally).
         */
        //void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_N, 1> &dy)
                  //const Eigen::Matrix<T, DIM_P, DIM_N> &phi, const Eigen::Matrix<T, DIM_P, DIM_N> &dphi) 
        void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
        {
            // Set up the extended regressor equation
            y_ext << y, dy;
            
            phi_ext(Eigen::seqN(0, DIM_P), Eigen::seqN(0, DIM_N)) = phi;
            phi_ext(Eigen::seqN(0, DIM_P), Eigen::seqN(DIM_N, DIM_N)) = dphi;
            phi_ext(Eigen::seqN(DIM_P, DIM_P), Eigen::seqN(DIM_N, DIM_N)) = phi;

            reg_ext_second->step(y, phi);
            reg_ext_first->step(y_ext, phi_ext);

            // Inner loop where we find an estimate of dtheta_ext.
            Eigen::Matrix<T, 2*DIM_P, 1> y_ext_f = reg_ext_first->getY();
            Eigen::Matrix<T, 2*DIM_P, 2*DIM_P> phi_ext_f = reg_ext_first->getPhi();

            Eigen::FullPivLU<Eigen::Matrix<T, 2*DIM_P, 2*DIM_P>> lu_ext(phi_ext_f);
            T Delta_ext = lu_ext.determinant();

            if (!std::isfinite(Delta_ext))
                Delta_ext = 0;

            Yvar_ext = lu_ext.solve(Delta_ext * y_ext_f);
            
            if (Delta_ext > eps)
                theta_est_ext = Yvar_ext / Delta_ext;

            // Now we have an estimate of the derivative of the parameter, theta
            // theta_est_ext = [hat_theta_ext, hat_dtheta_ext]
            dtheta_ext << theta_est_ext(Eigen::seqN(DIM_P, DIM_P));

            // 
            Eigen::Matrix<T, DIM_P, 1> y_f = reg_ext_second->getY();
            Eigen::Matrix<T, DIM_P, DIM_P> phi_f = reg_ext_second->getPhi(); 

            //
            Eigen::FullPivLU<Eigen::Matrix<T, DIM_P, DIM_P>> lu(phi_f);
            T Delta = lu.determinant();

            if (!std::isfinite(Delta))
                Delta = 0;

            auto get_dHc_state = [=](Eigen::Matrix<T, DIM_P, 1> Hc_state_)
            {
                Eigen::Matrix<T, DIM_P, 1> dHc_state = -a * Hc_state + phi_f * dtheta_ext;
                return dHc_state;
            };
            
            Hc_state = integrator::Integrator<T, DIM_P, 1>::integrate(
                Hc_state,
                get_dHc_state,
                dt,
                intg_method
            );
            Hc_out = -Hc_state;

            // adj(phi_f) = Delta * phi_f^{-1}
            Yvar = lu.solve(Delta * y_f);  // To compute phi_f^{-1} * Delta * y_f.
            Vvar_est = lu.solve(Delta * Hc_out);

            if (Delta > eps)
                theta_est = Yvar / Delta - Vvar_est / Delta;
        }

        /**
         * @brief Set the values of the derivative of the LRE components, y and phi.
         * 
         * This function should be called *before* step.
         */
        void set_dy_dphi(const Eigen::Matrix<T, DIM_N, 1> &dy, const Eigen::Matrix<T, DIM_P, DIM_N> &dphi)
        {
            // std::cout << dy << " " << dphi << std::endl;
            this->dy = dy;
            this->dphi = dphi;
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
            reg_ext_second->reset();
            reg_ext_first->reset();
            Hc_state.setZero();
            dHc_state.setZero();
            dHc_state_old.setZero();
        }

        /**
         * @brief Set the lower bound on the Delta value before dividing with it.
         */
        void set_eps(double eps)
        {
            this->eps = eps;
        }

    private:
        regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P> * reg_ext_second;
        regressor_extensions::Kreisselmeier<T, 2*DIM_N, 2*DIM_P> * reg_ext_first;

        float dt;
        float a;

        integrator::IntegrationMethod intg_method;

        // Variables for computation
        Eigen::Vector<T, DIM_P> theta_est, theta_init, Yvar, dtheta_ext, Vvar_est, 
            Hc_state, dHc_state, dHc_state_old, Hc_out;
        Eigen::Vector<T, DIM_P*2> theta_est_ext, Yvar_ext;

        Eigen::Vector<T, DIM_N*2> y_ext;
        Eigen::Matrix<T, DIM_P*2, DIM_N*2> phi_ext;

        Eigen::Vector<T, DIM_N> dy;
        Eigen::Matrix<T, DIM_P, DIM_N> dphi;

        double eps;
    };
}

#endif  // CASCADED_DREM_HPP