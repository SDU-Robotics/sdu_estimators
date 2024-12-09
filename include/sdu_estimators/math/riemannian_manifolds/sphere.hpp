#pragma once

#ifndef SPHERE_HPP
#define SPHERE_HPP

#include <iostream>

#include "manifold.hpp"

namespace sdu_estimators::math::manifold
{
  #define point Eigen::Vector<T, DIM_N>
  #define vector Eigen::Vector<T, DIM_N>

  template <typename T, int32_t DIM_N>
  class Sphere : Manifold<T, point, vector>
  {
  public:
    Sphere()
    {
      // std::cout << "constructed" << std::endl;
    }

    T dist(point &point_a, point &point_b)
    {
      T chordal_distance = (point_a - point_b).norm();
      std::complex<T> tmp = 0.5 * chordal_distance;
      T d = std::real(2. * asin(tmp));
      // T d = std::real(2 * asin(0.5 * chordal_distance));
      return d;
    }

    /**
     * Projects vector to tangent space.
     * @param point_
     * @param vector_
     * @return A vector in the tangent space at the given point.
     */
    vector projection(point &point_, vector &vector_)
    {
      return vector_ - point_ * (point_.transpose() * vector_);
    }

    /**
     * Converts from Euclidean gradient to Riemannian gradient.
     * @param point_
     * @param euclidean_gradient
     * @return Riemannian gradient.
     */
    vector euclidean_to_riemannian_gradient(point &point_, vector &euclidean_gradient)
    {
      return projection(point_, euclidean_gradient);
    }

    /**
     * Retraction from tangent space to the manifold.
     * @param point_
     * @param tangent_vector
     * @return
     */
    point retraction(point &point_, vector &tangent_vector)
    {
      point y = point_ + tangent_vector;
      y.normalize();
      return y;
    }

    /**
     * Exponential map on the manifold.
     * @param point_
     * @param tangent_vector
     * @return
     */
    point exp(point &point_, vector &tangent_vector)
    {
      T nrm = tangent_vector.norm();
      T nrm_pi = nrm / M_PI;

      // vector y = point_ * cos(nrm) + tangent_vector * (sin(nrm_pi) / nrm_pi);
      vector y = point_ * cos(nrm) + tangent_vector * (sin(nrm) / nrm);
      y.normalize();
      return y;
    }

    /**
     * Logarithmic map on the manifold.
     * @param point_a
     * @param point_b
     * @return
     */
    vector log(point &point_a, point &point_b)
    {
      vector vab = point_b - point_a;
      vector v = projection(point_a, vab);
      T di = dist(point_a, point_b);

      T eps = std::numeric_limits<T>::epsilon();
      // std::cout << eps << std::endl;
      // std::cout << eps << std::endl;
      double factor = (di + eps) / (v.norm() + eps);

      return factor * v;
    }
  };

  #undef point
  #undef vector
}

#endif //SPHERE_HPP
