#pragma once

#ifndef MANIFOLD_HPP
#define MANIFOLD_HPP

#include <Eigen/Dense>

namespace sdu_estimators::math::manifold
{

  /**
  * This class provides a base class for a Riemannian manifold.
  * Since we just need it for parameter estimation, we only implement the following functions:
  *
  * - Metric
  * - Distance
  * - Exponential
  * - Logarithmic
  * - Retraction
  * - Euclidean gradient -> Riemannian gradient
  * - Projection
  */
  template <typename T, typename point, typename vector>
  class Manifold
  {
  public:
    virtual ~Manifold() = default;

   /**
    * The geodesic distance between two points on the manifold
    * @param point_a
    * @param point_b
    * @return distance
    */
    virtual T dist(point &point_a, point &point_b) = 0;

    /**
     * Projects vector to tangent space.
     * @param point_
     * @param vector_
     * @return A vector in the tangent space at the given point.
     */
    virtual vector projection(point &point_, vector &vector_) = 0;

    /**
     * Converts from Euclidean gradient to Riemannian gradient.
     * @param point_
     * @param euclidean_gradient
     * @return Riemannian gradient.
     */
    virtual vector euclidean_to_riemannian_gradient(point &point_, vector &euclidean_gradient) = 0;

    /**
     * Retraction from tangent space to the manifold.
     * @param point_
     * @param trangent_vector
     * @return
     */
    virtual point retraction(point &point_, vector &trangent_vector) = 0;

    /**
     * Exponential map on the manifold.
     * @param point_
     * @param tangent_vector
     * @return
     */
    virtual point exp(point &point_, vector &tangent_vector) = 0;

    /**
     * Logarithmic map on the manifold.
     * @param point_a
     * @param point_b
     * @return
     */
    virtual vector log(point &point_a, point &point_b) = 0;
  };
}

#endif //MANIFOLD_HPP