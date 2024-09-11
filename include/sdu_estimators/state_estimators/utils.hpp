#pragma once
#include <Eigen/Dense>

#ifndef UTIL_HPP
#define UTIL_HPP

namespace sdu_estimators::state_estimators::utils
{
  void c2d(Eigen::MatrixXd & A, Eigen::MatrixXd & B,
           float Ts,
           Eigen::MatrixXd & Ad, Eigen::MatrixXd & Bd)
  {
    Ad = A;
    Bd = B;
  }
}

#endif //UTIL_HPP
