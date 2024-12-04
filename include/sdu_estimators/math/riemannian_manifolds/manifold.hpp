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
     * @return
     */
    virtual T dist(point &point_a, point &point_b) = 0;

  
    virtual vector projection(point &point_, vector &vector_) = 0;
  };
}

#endif //MANIFOLD_HPP