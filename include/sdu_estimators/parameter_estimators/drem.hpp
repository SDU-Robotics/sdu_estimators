#pragma once
#ifndef DREM_HPP
#define DREM_HPP

#include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
// #include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>
#include <iostream>
#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>
#include <type_traits>

#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"

#include "sdu_estimators/parameter_estimators/utils.hpp"
#include <cmath>

#include <Eigen/Sparse>

namespace sdu_estimators::parameter_estimators
{
  /**
   * An implementation of dynamic regressor extension and mixing (DREM) as described in e.g., 
   * \verbatim embed:rst:inline :cite:`Aranovskiy2017` \endverbatim.
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

      Eigen::HouseholderQR<Eigen::Matrix<T, DIM_P, DIM_P>> qr(phi_f);
      /*
      Eigen::SparseLU<Eigen::Matrix<T, DIM_P, DIM_P> > lu_solver;
      lu_solver.analyzePattern(phi_f);
      lu_solver.factorize(phi_f);

      double Delta = lu_solver.signDeterminant() * exp(lu_solver.logAbsDeterminant()); // phi_f.determinant();
      */

      /*
      Eigen::SparseLU<Eigen::SparseMatrix<T>, Eigen::COLAMDOrdering<int> >   solver;
      std::cout << "test" << std::endl;
      solver.analyzePattern(phi_f.sparseView());
      std::cout << "test2" << std::endl;
      // solver.factorize(phi_f.sparseView());
      std::cout << "test3" << std::endl;

      T Delta = solver.signDeterminant() * exp(solver.logAbsDeterminant()); // phi_f.determinant();
      // T Delta = exp(utils::logdet<Eigen::Matrix<T, DIM_P, DIM_P>>(phi_f, false));
      */

      Eigen::FullPivLU<Eigen::Matrix<T, DIM_P, DIM_P>> lu(phi_f);
      T Delta = lu.determinant();

      // double Delta = phi_f.determinant();
      // std::cout << "Delta " << Delta << std::endl;

      if (!std::isfinite(Delta))
        Delta = 0;

      // Eigen::Matrix<T, DIM_P, DIM_P> phi_tmp = phi_f;
      // float Yvar_i;

      Yvar = lu.solve(Delta * y_f); // To compute phi_f^{-1} * Delta * y_f.
      // adj(phi_f) = phi_f^{-1} * Delta
      // Eigen::MatrixXd Yvar = phi_f.inverse() * Delta * y_f;

      // Yvar = Delta * phi_f.inverse() * y_f;
      // for (int i = 0; i < DIM_P; ++i)
      // {
      //   if (Yvar[i] != Yvar[i])  // Element is nan
      //   {
      //     // Yvar[i] = 0;
      //     Yvar *= 0;
      //     break;
      //   }
      // }
      // adj(phi_f) = Delta * phi_f.inverse()

      Eigen::Vector<T, DIM_P> y_err = Yvar - Delta * theta_est;
      Eigen::Vector<T, DIM_P> tmp1 = y_err.array().abs().pow(r);
      Eigen::Vector<T, DIM_P> tmp2 = y_err.cwiseSign();

      dtheta = gamma.asDiagonal() * Delta * tmp1.cwiseProduct(tmp2);

      /*
      for (int i = 0; i < DIM_P; ++i)
      {
        phi_tmp(Eigen::all, i) = y_f;

        Yvar_i = phi_tmp.determinant();

        y_err_i = Yvar_i - Delta * theta_est[i];
        dtheta[i] = gamma[i] * Delta * (pow(abs(y_err_i), r) * std::signbit(-y_err_i));

        phi_tmp(Eigen::all, i) = phi_f(Eigen::all, i);
      } */

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
    Eigen::Matrix<T, DIM_P, 1> theta_est, theta_init, dtheta, Yvar;
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

