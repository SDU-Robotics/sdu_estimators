#pragma once

#ifndef REGRESSOR_EXTENSION_HPP
#define REGRESSOR_EXTENSION_HPP

#include <Eigen/Dense>

namespace sdu_estimators::regressor_extensions
{
  template <typename T, int32_t DIM_N, int32_t DIM_P>
  class RegressorExtension
  {
  public:
    // RegressorExtension() = default;

    virtual ~RegressorExtension() = default;

    virtual void step(const Eigen::Matrix<T, DIM_N, 1> &y,
                      const Eigen::Matrix<T, DIM_P, DIM_N> &phi) = 0;

    // get filtered states
    virtual Eigen::Matrix<T, DIM_P, 1> getY()
    {
      return y_f;
    }

    // get filtered states
    virtual Eigen::Matrix<T, DIM_P, DIM_P> getPhi()
    {
      return phi_f;
    };

    virtual void reset() = 0;

  protected:
    Eigen::Matrix<T, DIM_P, 1> y_f; // filtered y
    Eigen::Matrix<T, DIM_P, DIM_P> phi_f; // filtered phi
  };
}

#endif //REGRESSOR_EXTENSION_HPP
