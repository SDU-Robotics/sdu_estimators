#pragma once

#ifndef SYMMETRICPOSITIVEDEFINITE_HPP
#define SYMMETRICPOSITIVEDEFINITE_HPP

#include <cstdint>
#include <iostream>
#include <unsupported/Eigen/MatrixFunctions>

#include "manifold.hpp"

namespace sdu_estimators::math::manifold
{

  /**
   * Point: A PSD n x n matrix.
   *
   * Vector: A n x n matrix.
   */
  template<typename T, int32_t n>
  class SymmetricPositiveDefinite : Manifold<T, Eigen::Matrix<T, n, n>, Eigen::Matrix<T, n, n>>
  {
   public:
    using point = Eigen::Matrix<T, n, n>;   // an PSD matrix
    using vector = Eigen::Matrix<T, n, n>;  // a matrix

    SymmetricPositiveDefinite()
    {
      // std::cout << "constructed" << std::endl;
    }

    T norm(point &point_, vector &vector_)
    {
      point AA = point_.colPivHouseholderQr().solve(vector_);
      T out = traceAA(AA);
      return out;
    }

    T dist(point &point_a, point &point_b)
    {
      point a_div_b = point_a.colPivHouseholderQr().solve(point_b);
      point a_div_b_logm = a_div_b.log();
      T out = traceAA(a_div_b_logm);
      return out;
    }

    /**
     * Projects vector to tangent space.
     * @param point_
     * @param vector_
     * @return A vector in the tangent space at the given point.
     */
    vector projection(point &point_, vector &vector_)
    {
      return symm(vector_);
    }

    /**
     * Converts from Euclidean gradient to Riemannian gradient.
     * @param point_
     * @param euclidean_gradient
     * @return Riemannian gradient.
     */
    vector euclidean_to_riemannian_gradient(point &point_, vector &euclidean_gradient)
    {
      point elem = point_ * euclidean_gradient * point_;
      return projection(point_, elem);
    }

    /**
     * Retraction from tangent space to the manifold.
     * @param point_
     * @param tangent_vector
     * @return
     */
    point retraction(point &point_, vector &tangent_vector)
    {
      point p_inv_tv = point_.colPivHouseholderQr().solve(tangent_vector);
      vector arg = point_ + tangent_vector + tangent_vector * p_inv_tv / 2.;
      return symm(arg);
    }

    /**
     * Exponential map on the manifold.
     * @param point_
     * @param tangent_vector
     * @return
     */
    point exp(point &point_, vector &tangent_vector)
    {
      point p_inv_tv = point_.colPivHouseholderQr().solve(tangent_vector);
      vector tmp = point_ * p_inv_tv.exp();
      return symm(tmp);
    }

    /**
     * Logarithmic map on the manifold.
     * @param point_a
     * @param point_b
     * @return
     */
    vector log(point &point_a, point &point_b)
    {
      point a_inv_b = point_a.colPivHouseholderQr().solve(point_b);
      vector tmp = point_a * a_inv_b.log();
      return symm(tmp);
    }

   private:
    point symm(point &X)
    {
      return 0.5 * (X + X.transpose());
    }

    T traceAB(point &A, point &B)
    {
      // Eigen::Vector<T, n * n> vecA = A.reshaped(1, n * n).transpose();
      point AB = A * B;
      T out = AB.trace();
      return out;
    }

    T traceAA(point &A)
    {
      return sqrt(traceAB(A, A));
    }
  };

#undef point
#undef vector
}  // namespace sdu_estimators::math::manifold

#endif  // SYMMETRICPOSITIVEDEFINITE_HPP
