#pragma once

#ifndef KREISSELMEIER_HPP
#define KREISSELMEIER_HPP

#include "sdu_estimators/regressor_extensions/regressor_extension.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

#include <iostream>

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
      integrator::IntegrationMethod method = integrator::IntegrationMethod::RK4,
      bool constant_gain = false)
    {
      this->dt = dt;
      this->ell = ell;

      this->intg_method = method;

      this->constant_gain = constant_gain; // not used
      
      reset();
    }

    ~Kreisselmeier()
    {

    }

    void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
    {
      auto get_dphifdt = [=](Eigen::Matrix<T, DIM_P, DIM_P> phi_f_)
      {
        Eigen::Matrix<T, DIM_P, DIM_P> dphi_f;
        // if (constant_gain)
        // {
        //   // dphi_f = -ell * phi_f_ + ell * phi * phi.transpose();
        //   dphi_f = -ell * phi_f_ + phi * phi.transpose();
        // }
        // else
        // {
        dphi_f = -ell * phi_f_ + phi * phi.transpose();
        // }

        return dphi_f;
      };

      this->phi_f = integrator::Integrator<T, DIM_P, DIM_P>::integrate(
        this->phi_f,
        get_dphifdt,
        dt,
        intg_method
      );

      auto get_dyfdt = [=](Eigen::Matrix<T, DIM_P, 1> y_f_)
      {
        Eigen::Matrix<T, DIM_P, 1> dy_f;
        // if (constant_gain)
        // {
        //   // dy_f = -ell * y_f_ + ell * phi * y;
        //   dy_f = -ell * y_f_ + phi * y;
        // }
        // else
        // {
        dy_f = -ell * y_f_ + phi * y;
        // }

        return dy_f;
      };

      this->y_f = integrator::Integrator<T, DIM_P, 1>::integrate(
        this->y_f,
        get_dyfdt,
        dt,
        intg_method
      );
    }

    // Eigen::Matrix<T, DIM_P, 1> getY()
    // {
    //   Eigen::Matrix<T, DIM_P, 1> tmp = RegressorExtension<T, DIM_N, DIM_P>::getY();
    //   if (constant_gain)
    //     tmp = tmp / ell;

    //   return tmp;
    // }

    // // get filtered states
    // Eigen::Matrix<T, DIM_P, DIM_P> getPhi()
    // {
    //   Eigen::Matrix<T, DIM_P, DIM_P> tmp = RegressorExtension<T, DIM_N, DIM_P>::getPhi();
    //   if (constant_gain)
    //     tmp = tmp / ell;

    //   return tmp;
    // };

    // Eigen::VectorXd getY() override; // get filtered states
    // Eigen::MatrixXd getPhi() override; // get filtered states

    void reset()
    {
      this->phi_f.setZero();
      this->y_f.setZero();
    }

  private:
    float dt;
    float ell;
    bool constant_gain;

    integrator::IntegrationMethod intg_method;

    Eigen::Matrix<T, DIM_P, 1> dy_f; //
    Eigen::Matrix<T, DIM_P, DIM_P> dphi_f; //
};

} // sdu_estimators

#endif //KREISSELMEIER_HPP
