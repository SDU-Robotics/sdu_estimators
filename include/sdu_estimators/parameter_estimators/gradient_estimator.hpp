#pragma once
#ifndef GRADIENT_ESTIMATOR_HPP
#define GRADIENT_ESTIMATOR_HPP

#include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
#include <sdu_estimators/parameter_estimators/utils.hpp>

namespace sdu_estimators::parameter_estimators
{
  /**
   * A simple gradient-based parameter estimator as described in e.g.,
   * `:cite:p:"Sastry1989"`.
   *
   * Test `:role:"content of the role"`
   *
   * The parameter \f$ \theta \f$ can be estimated by \f$ \hat{\theta} \f$ with the following update rule:
   *
   * \f{equation}{
   *    \dot{\hat{\theta}}(t) = \gamma \phi(t) (y(t) - \phi^\intercal(t) \hat{\theta}(t)),
   * \f}
   *
   * where \f$ \gamma > 0 \f$ is a tuning parameter, \f$ y : \mathbb{R}_+ \to \mathbb{R}^n \f$ is the output, \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{m \times n} \f$ is,
   * the regressor matrix and \f$ \theta : \mathbb{R}_+ \to \mathbb{R}^m \f$ is the parameter vector.
   *
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
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
    GradientEstimator(float dt, const Eigen::Vector<T, DIM_P> gamma, const Eigen::Vector<T, DIM_P> & theta_init,
       utils::IntegrationMethod method = utils::IntegrationMethod::Euler)
      : GradientEstimator(dt, gamma, theta_init, 1.0f, method)
    {
    }
    //
    // *
    //      * \f$  \dot{\hat{\theta}}(t) = \gamma \phi(t) \lceil y(t) - \phi^\intercal(t) \hat{\theta}(t) \rfloor^r, \f$
    //      *
    //      * where \f$ \lceil x \rfloor^r = \lvert x \rvert^r \text{sign}(x) \f$, \f$r \in (0,1) \f$.

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
    GradientEstimator(float dt, const Eigen::Vector<T, DIM_P> gamma, const Eigen::Vector<T, DIM_P> & theta_init, float r,
      utils::IntegrationMethod method = utils::IntegrationMethod::Euler)
    {
      this->dt = dt;
      this->gamma = gamma;
      this->theta_est = theta_init;
      this->theta_init = theta_init;
      this->p = theta_init.size();
      this->r = r;

      this->y_old.setZero();
      this->phi_old.setZero();

      this->intg_method = method;
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
      if (intg_method == utils::IntegrationMethod::Euler)
      {
        y_err = y - phi.transpose() * theta_est;

        Eigen::Vector<T, DIM_N> tmp1 = y_err.array().abs().pow(r);
        Eigen::Vector<T, DIM_N> tmp2 = y_err.cwiseSign();

        // std::cout << tmp1 << " " << tmp2 << std::endl;

        dtheta = gamma.asDiagonal() * phi * (
          tmp1.cwiseProduct(tmp2)
        );

        theta_est += dt * dtheta;
      }
      else if (intg_method == utils::IntegrationMethod::Heuns)
      {
        /*
         * Heun's method
         *
         * ytilde_{i+1} = y_i + h * f(t_i, y_i)
         * y_{i + 1} = y_i + h/2 * (f(t_i, y_i) + f(t_{i+1}, ytilde_{i+1}))
         *
         * For parameter estimation
         *
         * thetatilde_{i+1} = theta_i + h * dtheta(y_i, phi_i, theta_i)
         * theta_{i+1} = theta_i + (h/2) * (dtheta(y_i, phi_i, theta_i) + dtheta(y_{i+1}, phi_{i+1}, thetatilde_{i+1}))
         */
        y_err = y_old - phi_old.transpose() * theta_est;
        Eigen::Vector<T, DIM_N> tmp1 = y_err.array().abs().pow(r);
        Eigen::Vector<T, DIM_N> tmp2 = y_err.cwiseSign();

        Eigen::Vector<T, DIM_P> dtheta_old = gamma.asDiagonal() * phi * tmp1.cwiseProduct(tmp2);
        Eigen::Vector<T, DIM_P> theta_tilde_new = theta_est + dt * dtheta_old;

        y_err = y - phi.transpose() * theta_tilde_new;
        tmp1 = y_err.array().abs().pow(r);
        tmp2 = y_err.cwiseSign();

        Eigen::Vector<T, DIM_P> dtheta_new = gamma.asDiagonal() * phi * tmp1.cwiseProduct(tmp2);

        theta_est = theta_est + (dt / 2.) * (dtheta_old + dtheta_new);

        y_old = y;
        phi_old = phi;
      }
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
    Eigen::Vector<T, DIM_N> y_old;
    Eigen::Matrix<T, DIM_P, DIM_N> phi_old;
    int p{}; // number of parameters

    utils::IntegrationMethod intg_method;
  };
}

#endif //GRADIENT_ESTIMATOR_HPP
