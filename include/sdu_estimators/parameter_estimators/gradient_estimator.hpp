#pragma once
#ifndef GRADIENT_ESTIMATOR_HPP
#define GRADIENT_ESTIMATOR_HPP

#include <cstdint>

#include "sdu_estimators/parameter_estimators/parameter_estimator.hpp"
#include "sdu_estimators/utils.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

#include "sdu_estimators/typedefs.hpp"

#include <Eigen/LU>


namespace sdu_estimators::parameter_estimators
{
    enum class GradientEstimatorType
    {
        Standard,
        Projected
    };

    /**
    * A simple gradient-based parameter estimator as described in e.g.,
    * \verbatim embed:rst:inline :cite:`Sastry1989` \endverbatim.
    *
    * The parameter \f$ \theta \f$ can be estimated by \f$ \hat{\theta} \f$ with the following update rule:
    *
    * \f{equation}{
    *    \dot{\hat{\theta}}(t) = \gamma \phi(t) \left( y(t) - \phi^\intercal(t) \hat{\theta}(t) \right),
    * \f}
    *
    * where \f$ \gamma > 0 \f$ is a tuning parameter, \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output, \f$ \phi :
    * \mathbb{R}_+ \to \mathbb{R}^{p \times n} \f$ is, the regressor matrix and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^p \f$
    * is the parameter vector.
    *
    */

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    class GradientEstimator : public ParameterEstimator<T, DIM_N, DIM_P>
    {
    public:
        /**
        * @brief Constructor for the default gradient-based update rule.
        *
        * @param dt Sample time.
        * @param gamma \f$ \gamma \in \mathbb{R}^p \f$ is the vector of gains.
        * @param theta_init The initial value of the parameter estimate \f$ \hat{\theta}(0) \f$.
        */
        GradientEstimator(
            float dt,
            const Eigen::Vector<T, DIM_P> gamma,
            const Eigen::Vector<T, DIM_P> &theta_init,
            integrator::IntegrationMethod method = integrator::IntegrationMethod::RK4)
            : GradientEstimator(dt, gamma, theta_init, 1.0f, method)
        {
        }

        /**
        * @brief Constructor for the gradient-based finite-time update rule.
        *
        * \f{equation}{
        *    \dot{\hat{\theta}}(t) = \gamma \phi(t) \lceil y(t) - \phi^\intercal(t) \hat{\theta}(t) \rfloor^r,
        * \f}
        *
        * where \f$ \lceil x \rfloor^r = \lvert x \rvert^r \text{sign}(x) \f$, \f$ r \in (0, 1) \f$.
        *
        * @param dt Sample time.
        * @param gamma \f$ \gamma \in \mathbb{R}^p \f$ is the vector of gains.
        * @param theta_init The initial value of the parameter estimate \f$ \hat{\theta}(0) \f$.
        * @param r The value of the coefficient, \f$ r \in (0,1) \f$.
        */
        GradientEstimator(
            float dt,
            const Eigen::Vector<T, DIM_P> gamma,
            const Eigen::Vector<T, DIM_P> &theta_init,
            float r,
            integrator::IntegrationMethod method = integrator::IntegrationMethod::RK4)
            : dt(dt),
            r(r),
            gamma(gamma),
            theta_est(theta_init),
            theta_init(theta_init),
            intg_method(method)
        {
            type = GradientEstimatorType::Standard;
            use_normalisation = false;
            normalisation_gamma = 1.;
        }

        ~GradientEstimator()
        {
        }

        /**
        * @brief Step the execution of the estimator (must be called in a loop externally)
        */
        void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
        // utils::IntegrationMethod method = utils::IntegrationMethod::Euler)
        {
            auto get_dydt = [=](Eigen::Vector<T, DIM_P> theta_est_)
            {
                Eigen::Vector<T, DIM_N> y_err = y - phi.transpose() * theta_est_;

                Eigen::Vector<T, DIM_N> tmp1 = y_err.array().abs().pow(r);
                Eigen::Vector<T, DIM_N> tmp2 = y_err.cwiseSign();

                // std::cout << tmp1 << " " << tmp2 << std::endl;

                Eigen::Vector<T, DIM_P> dtheta = gamma.asDiagonal() * phi * (tmp1.cwiseProduct(tmp2));

                if (type == GradientEstimatorType::Projected)
                {
                    for (int i = 0; i < DIM_P; ++i)
                    {
                        if ( ((theta_est_[i] <= theta_lower_bound_[i]) && (dtheta[i] < 0) ) 
                          || ((theta_est_[i] >= theta_upper_bound_[i]) && (dtheta[i] > 0) ) )
                        {
                            dtheta[i] = 0;
                        }
                    }
                }

                if (use_normalisation)
                {
                    Eigen::Matrix<T, DIM_P, DIM_P> phi_sqr = phi * phi.transpose();
                    
                    Eigen::Matrix<T, DIM_P, DIM_P> factor = 
                        Eigen::Matrix<T, DIM_P, DIM_P>::Identity() + normalisation_gamma * phi_sqr;

                    // std::cout << factor << std::endl;

                    dtheta = factor.inverse() * dtheta;
                }

                return dtheta;
            };

            theta_est = integrator::Integrator<T, DIM_P, 1>::integrate(
                theta_est, 
                get_dydt, 
                dt, 
                intg_method);

            // Cap estimate within bounds
            if (type == GradientEstimatorType::Projected)
            {
                for (int i = 0; i < DIM_P; ++i)
                {
                    if (theta_est[i] < theta_lower_bound_[i])
                        theta_est[i] = theta_lower_bound_[i];

                    else if (theta_est[i] > theta_upper_bound_[i])
                        theta_est[i] = theta_upper_bound_[i];
                }
            }
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
        }

        /**
        * @brief Set bounds on the estimated theta. If this function is called, the step-function uses projection.
        *
        * If the parameter lies in a set known a-priori \f$ \theta \in \Theta \f$, i.e., for every element 
        * \f$ i = 1, \dots, p \f$ of the parameter vector \f$ \theta \f$
        *
        * \f{equation}{
        *    \theta_i \in [\theta_i^-, \theta_i^+]
        * \f}
        *
        * To ensure that \f$ \hat{\theta}_i \f$ lies in this set, the update law is updated with:
        * 
        * \f{align}{
        *   \dot{\hat{\theta}}_i = 0 \quad &\text{if} \quad &\hat{\theta}_i \leq \theta_i^- 
                                                    &\quad \text{and} \quad \dot{\hat{\theta}}_i < 0 \\
        *                            \quad &\text{or} \quad &\hat{\theta}_i \geq \theta_i^+
        *                                           &\quad \text{and} \quad \dot{\hat{\theta}}_i > 0
        * \f}
        * 
        * I.e., if the parameter estimate is on the boundary it can only be updated to go within the set 
        * again or on the boundary -- not outside.
        * 
        * @param theta_lower_bound Lower bound on the parameter estimate, \f$ \theta_i^- \f$ for \f$ i = 1, \dots, p \f$.
        * @param theta_upper_bound Upper bound on the parameter estimate, \f$ \theta_i^+ \f$ for \f$ i = 1, \dots, p \f$.
        */
        void set_theta_bounds(Eigen::Vector<T, DIM_P> & theta_lower_bound, 
                              Eigen::Vector<T, DIM_P> & theta_upper_bound)
        {
            type = GradientEstimatorType::Projected;

            theta_lower_bound_ = theta_lower_bound;
            theta_upper_bound_ = theta_upper_bound;
            
            // Check if theta_init is within bounds
            for (int i = 0; i < DIM_P; ++i)
            {
                if (theta_init[i] <= theta_lower_bound[i])
                {
                    theta_init[i] = theta_lower_bound[i];
                }
                else if (theta_init[i] > theta_upper_bound[i])
                {
                    theta_init[i] = theta_upper_bound[i];
                }
            }

            theta_est = theta_init;
        }

        void enable_normalisation(float normalisation_gamma_)
        {
            use_normalisation = true;
            normalisation_gamma = normalisation_gamma_;
        }

        void disable_normalisation()
        {
            use_normalisation = false;
        }

    private:
        float dt;
        float r;
        Eigen::Vector<T, DIM_P> theta_est, theta_init, gamma;

        Eigen::Vector<T, DIM_P> theta_lower_bound_, theta_upper_bound_;

        integrator::IntegrationMethod intg_method;

        GradientEstimatorType type;

        bool use_normalisation;
        float normalisation_gamma;
    };
}  // namespace sdu_estimators::parameter_estimators

#endif  // GRADIENT_ESTIMATOR_HPP
