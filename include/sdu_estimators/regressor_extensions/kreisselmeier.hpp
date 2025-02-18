#pragma once

#ifndef KREISSELMEIER_HPP
#define KREISSELMEIER_HPP

#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>
#include <sdu_estimators/parameter_estimators/utils.hpp>

namespace sdu_estimators::regressor_extensions
{
  /**
   * Description
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class Kreisselmeier : public RegressorExtension<T, DIM_N, DIM_P>
  {
  public:
    Kreisselmeier(float dt, float ell,
      parameter_estimators::utils::IntegrationMethod method = parameter_estimators::utils::IntegrationMethod::Euler)
    {
      this->dt = dt;
      this->ell = ell;

      first_run = true;

      this->intg_method = method;

      this->dy_f_old.setZero();
      this->dphi_f_old.setZero();
    }

    ~Kreisselmeier()
    {

    }

    void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    {
      if (first_run)
      {
        this->y_f.setZero();
        this->phi_f.setZero();

        first_run = false;
      }

      dphi_f = -ell * this->phi_f + phi * phi.transpose();
      dy_f = -ell * this->y_f + phi * y;

      if (intg_method == parameter_estimators::utils::IntegrationMethod::Euler)
      {
        this->phi_f += dt * dphi_f;
        this->y_f += dt * dy_f;
      }
      else if (intg_method == parameter_estimators::utils::IntegrationMethod::Heuns)
      {
        this->phi_f += dt * (dphi_f + dphi_f_old) / 2.;
        this->y_f += dt * (dy_f + dy_f_old) / 2.;
      }

      dphi_f_old = dphi_f;
      dy_f_old = dy_f;
    }

    // Eigen::VectorXd getY() override; // get filtered states
    // Eigen::MatrixXd getPhi() override; // get filtered states

    void reset()
    {
      first_run = true;
      this->y_f *= 0;
      this->phi_f *= 0;
    }

  private:
    float dt{};
    float ell{};

    bool first_run{};

    parameter_estimators::utils::IntegrationMethod intg_method;

    Eigen::Matrix<T, DIM_P, 1> dy_f, dy_f_old; //
    Eigen::Matrix<T, DIM_P, DIM_P> dphi_f, dphi_f_old; //
};

} // sdu_estimators

#endif //KREISSELMEIER_HPP
