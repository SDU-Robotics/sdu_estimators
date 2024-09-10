#pragma once

#ifndef KREISSELMEIER_HPP
#define KREISSELMEIER_HPP

#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>

namespace sdu_estimators::regressor_extensions
{
  /**
   * Description
   */

  class Kreisselmeier : public RegressorExtension
  {
  public:
    Kreisselmeier(float dt, float ell);
    ~Kreisselmeier() override;

    void step(const Eigen::VectorXd &y,
                  const Eigen::MatrixXd &phi) override;

    Eigen::VectorXd getY() override; // get filtered states
    Eigen::MatrixXd getPhi() override; // get filtered states

    void reset() override;

  private:
    float dt{};
    float ell{};

    Eigen::VectorXd y_f; // filtered y
    Eigen::MatrixXd phi_f; // filtered phi

    bool first_run{};
};

} // sdu_estimators

#endif //KREISSELMEIER_HPP
