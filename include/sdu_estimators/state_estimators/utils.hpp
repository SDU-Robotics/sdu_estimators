#pragma once

#ifndef UTIL_HPP
#define UTIL_HPP

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace sdu_estimators::state_estimators::utils
{
  enum IntegrationMethod
  {
    Euler,
    EulerBackwards,
    Bilinear,
    Exact
  };

  inline void c2d(Eigen::MatrixXd & A, Eigen::MatrixXd & B,
           float Ts,
           Eigen::MatrixXd & Ad, Eigen::MatrixXd & Bd,
           IntegrationMethod method)
  {
    // Ad = A;
    // Bd = B;
    auto I = Eigen::MatrixXd::Identity(A.rows(), A.rows());

    switch (method)
    {
      case Euler:
        Ad = I + Ts * A;
        Bd = Ts * B;
        break;

      case EulerBackwards:
        Ad = (I - Ts * A).inverse();
        Bd = Ts * B;
        break;

      case Bilinear:  // Tustin method
        Ad = (I + Ts * A / 2.f) * (I - Ts * A / 2.f).inverse();
        Bd = Ts * B;
        break;

      case Exact:
        Ad = (Ts * A).exp();
        Bd = A.completeOrthogonalDecomposition().pseudoInverse() * (Ad - I) * B;
        break;
    }
  }
  //
  // extern "C"
  // {
  //   void greet_fortran(const char * name);
  // }
    // call SCLICOT SB01BD Fortran-code routine for pole placement
}



#endif //UTIL_HPP
