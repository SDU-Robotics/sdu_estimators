#pragma once
#ifndef DREM_HPP
#define DREM_HPP

#include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
// #include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>
#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>
#include <type_traits>

#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"

namespace sdu_estimators::parameter_estimators
{
  /**
   * An implementation of dynamic regressor extension and mixing (DREM) as described in e.g.,
   *  Aranovskiy, S., Bobtsov, A., Ortega, R., & Pyrkin, A. (2016, July).
   *    Parameters estimation via dynamic regressor extension and mixing.
   *    In 2016 American Control Conference (ACC) (pp. 6971-6976). IEEE.
   *
   *
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class DREM : public ParameterEstimator<T, DIM_N, DIM_P>
  {
  // static_assert(std::is_base_of_v<regressor_extensions::RegressorExtension, T_REG_EXT>,
  //   "T_REG_EXT must derive from regressor_extensions::RegressorExtension");

  public:
    DREM(float dt, const Eigen::Matrix<T, DIM_P, 1> & gamma, const Eigen::Matrix<T, DIM_P, 1> & theta_init,
      regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> * reg_ext)
      : DREM(dt, gamma, theta_init, reg_ext, 1.0f)
    {
    }


    DREM(float dt, const Eigen::Matrix<T, DIM_P, 1> & gamma, const Eigen::Matrix<T, DIM_P, 1> & theta_init,
      regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> * reg_ext, float r)
    {
      this->dt = dt;
      this->gamma = gamma;
      this->theta_est = theta_init;
      this->theta_init = theta_init;
      this->dtheta = theta_init * 0;
      this->p = theta_init.size();
      this->r = r;
      this->reg_ext = reg_ext;
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

      double Delta = phi_f.determinant();

      Eigen::MatrixXd phi_tmp = phi_f;
      float Yvar_i;

      for (int i = 0; i < DIM_P; ++i)
      {
        phi_tmp(Eigen::all, i) = y_f;

        Yvar_i = phi_tmp.determinant();

        y_err_i = Yvar_i - Delta * theta_est[i];
        dtheta[i] = gamma[i] * Delta * (pow(abs(y_err_i), r) * std::signbit(-y_err_i));

        phi_tmp(Eigen::all, i) = phi_f(Eigen::all, i);
      }

      theta_est += dt * dtheta;
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
    float dt{};
    Eigen::Matrix<T, DIM_P, 1> gamma;
    float r{};
    Eigen::Matrix<T, DIM_P, 1> theta_est, theta_init, dtheta;
    Eigen::Matrix<T, DIM_N, 1> y_err;
    int p{}; // number of parameters

    float y_err_i{};

    // T_REG_EXT<T, DIM_N, DIM_P> reg_ext;
    // T_REG_EXT reg_ext;
    // regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P> reg_ext;
    regressor_extensions::RegressorExtension<T, DIM_N, DIM_P> * reg_ext;
  };
}



#endif //DREM_HPP
