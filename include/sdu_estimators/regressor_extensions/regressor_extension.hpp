#pragma once

#ifndef REGRESSOR_EXTENSION_HPP
#define REGRESSOR_EXTENSION_HPP

#include <Eigen/Dense>

namespace sdu_estimators::regressor_extensions
{
  class RegressorExtension
  {
    public:
      virtual ~RegressorExtension() = default;

      virtual void step(const Eigen::VectorXd &y,
                        const Eigen::MatrixXd &phi) = 0;

      virtual Eigen::VectorXd getY() = 0; // get filtered states
      virtual Eigen::MatrixXd getPhi() = 0; // get filtered states

      virtual void reset() = 0;
  };
}

#endif //REGRESSOR_EXTENSION_HPP
