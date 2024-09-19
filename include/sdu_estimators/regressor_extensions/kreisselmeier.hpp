#pragma once

#ifndef KREISSELMEIER_HPP
#define KREISSELMEIER_HPP

#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>

namespace sdu_estimators::regressor_extensions
{
  /**
   * Description
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class Kreisselmeier : public RegressorExtension<T, DIM_N, DIM_P>
  {
  public:
    Kreisselmeier(float dt, float ell)
    {
      this->dt = dt;
      this->ell = ell;

      first_run = true;
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

      this->phi_f += dt * (
            -ell * this->phi_f + phi * phi.transpose()
          );

      this->y_f += dt * (
        -ell * this->y_f + phi * y
      );
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
};

} // sdu_estimators

#endif //KREISSELMEIER_HPP
