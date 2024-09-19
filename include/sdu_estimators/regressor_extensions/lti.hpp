#pragma once

#ifndef LTI_HPP
#define LTI_HPP

#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>

namespace sdu_estimators::regressor_extensions
{
  /**
   * Description
   */

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class LTI : public RegressorExtension<T, DIM_N, DIM_P>
  {
    static_assert(DIM_N == 1, "The LTI regressor extension only works with N == 1.");

  public:
    LTI(float dt, Eigen::Vector<T, DIM_P> & alpha, Eigen::Vector<T, DIM_P> & beta)
    {
      this->dt = dt;

      first_run = true;

      this->alpha = alpha;
      this->beta = beta;
    }

    ~LTI()
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

      for (int i = 0; i < DIM_P; ++i)
      {
        this->y_f(i, Eigen::all) += dt * (-beta[i] * this->y_f(i, Eigen::all) + alpha[i] * y.transpose());

        this->phi_f(i, Eigen::all) += dt * (-beta[i] * this->phi_f(i, Eigen::all) + alpha[i] * phi.transpose());
      }

      // std::cout << this->phi_f << std::endl;
      // this->y_f += dt * (
      //   -ell * this->y_f + phi * y
      // );
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

    bool first_run{};

    Eigen::Vector<T, DIM_P> alpha, beta;
};

} // sdu_estimators

#endif //LTI__HPP
