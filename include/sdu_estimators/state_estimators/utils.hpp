#pragma once

#ifndef UTIL_HPP
#define UTIL_HPP

#include <Eigen/Dense>
#include <iostream>
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

  template <typename T, int32_t DIM_Nx, int32_t DIM_Nu>
  void c2d(Eigen::Matrix<T, DIM_Nx, DIM_Nx> & A, Eigen::Matrix<T, DIM_Nx, DIM_Nu> & B,
           float Ts,
           Eigen::Matrix<T, DIM_Nx, DIM_Nx> & Ad, Eigen::Matrix<T, DIM_Nx, DIM_Nu> & Bd,
           IntegrationMethod method)
  {
    // Ad = A;
    // Bd = B;
    auto I = Eigen::Matrix<T, DIM_Nx, DIM_Nx>::Identity(A.rows(), A.rows());

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

  /**
   * @brief Compute the observability matrix for state space systems.
   *
   * @param A
   * @param C
   * @return
   */
  inline Eigen::MatrixXd obsv(Eigen::MatrixXd & A, Eigen::MatrixXd & C)
  {
    int n = A.rows();
    int ny = C.rows();

    Eigen::MatrixXd ob;
    std::cout << ob << std::endl;
    ob.resize(ny * n, n);
    ob.setZero();
    std::cout << ob << std::endl;

    ob(Eigen::seqN(0, ny), Eigen::all) = C;

    for (int i = 1; i < n; i++)
    {
      std::cout << i << std::endl;
      ob(Eigen::seqN(i * ny, ny), Eigen::all) =
        ob(Eigen::seqN((i - 1) * ny, ny), Eigen::all) * A;
    }

    return ob;
  }

  inline Eigen::MatrixXd obsvf(Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C, float tol = NAN)
  {
    int n = A.rows();
    int ny = C.rows();

    Eigen::MatrixXd ob;
    std::cout << ob << std::endl;
    ob.resize(ny * n, n);
    ob.setZero();
    std::cout << ob << std::endl;

    ob(Eigen::seqN(0, ny), Eigen::all) = C;

    for (int i = 1; i < n; i++)
    {
      std::cout << i << std::endl;
      ob(Eigen::seqN(i * ny, ny), Eigen::all) =
        ob(Eigen::seqN((i - 1) * ny, ny), Eigen::all) * A;
    }

    return ob;
  }

  /*
   * @brief Controllability staircase form.
   *
   * Written from the MATLAB function.
   */
  inline void ctrbf(Eigen::MatrixXd & A, Eigen::MatrixXd & B, Eigen::MatrixXd & C, float tol = NAN)
  {
    int ra = A.rows();
    int cb = B.cols();

    Eigen::MatrixXd ptjn1;
    ptjn1.resize(ra, ra);
    ptjn1.setIdentity();

    int deltajn1 = 0;

    Eigen::MatrixXd k;
    k.resize(1, ra);
    k.setZero();

    if (tol == NAN)
      tol = ra * A.norm() * DBL_EPSILON;

    // Begin major loop.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B);
    auto uj = svd.matrixU();
    auto sjdiag = svd.singularValues();
    auto sj = svd.singularValues().asDiagonal();
    auto vjT = svd.matrixV().transpose();

    int rsj = sj.rows();

    for (int jj = 0; jj < ra; ++jj)
    {
      // uj, sjdiag, vjT = np.linalg.svd(bjn1)

    }
  }
}



#endif //UTIL_HPP
