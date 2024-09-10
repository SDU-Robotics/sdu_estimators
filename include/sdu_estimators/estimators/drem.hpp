#pragma once
#ifndef DREM_HPP
#define DREM_HPP

#include <sdu_estimators/estimators/estimator.hpp>
#include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>

namespace sdu_estimators::estimators
{
  /**
   * An implementation of dynamic regressor extension and mixing (DREM) as described in e.g.,
   *  Aranovskiy, S., Bobtsov, A., Ortega, R., & Pyrkin, A. (2016, July).
   *    Parameters estimation via dynamic regressor extension and mixing.
   *    In 2016 American Control Conference (ACC) (pp. 6971-6976). IEEE.
   *
   *
   */

  class DREM : public Estimator
  {
  public:
    DREM(float dt, const Eigen::VectorXd & gamma, const Eigen::VectorXd & theta_init, float ell);
    DREM(float dt, const Eigen::VectorXd & gamma, const Eigen::VectorXd & theta_init, float ell, float r);
    ~DREM() override;

    /**
     * @brief Step the execution of the estimator (must be called in a loop externally)
     */
    void step(const Eigen::VectorXd &y, const Eigen::MatrixXd &phi) override;

    /**
     * @brief Get the current estimate of the parameter. Updates when the step function is called.
     */
    Eigen::VectorXd get_estimate() override;

    /**
     * @brief Reset internal estimator variables
     */
    void reset() override;

  private:
    float dt{};
    Eigen::VectorXd gamma;
    float r{};
    Eigen::VectorXd theta_est, theta_init, dtheta, y_err;
    int p{}; // number of parameters

    float y_err_i{};

    regressor_extensions::Kreisselmeier reg_ext;
  };
}



#endif //DREM_HPP
