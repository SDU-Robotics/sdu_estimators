#include <iostream>
#include <sdu_estimators/parameter_estimators/drem.hpp>
#include <sdu_estimators/regressor_extensions/kreisselmeier.hpp>

namespace sdu_estimators::parameter_estimators
{
  DREM::~DREM() = default;

  DREM::DREM(float dt, const Eigen::VectorXd& gamma, const Eigen::VectorXd& theta_init, float ell)
    : DREM(dt, gamma, theta_init, ell, 1.0f) {}

  DREM::DREM(float dt, const Eigen::VectorXd& gamma, const Eigen::VectorXd& theta_init, float ell, float r)
    : reg_ext(dt, ell)
  {
    this->dt = dt;
    this->gamma = gamma;
    this->theta_est = theta_init;
    this->theta_init = theta_init;
    this->dtheta = theta_init * 0;
    this->p = theta_init.size();
    this->r = r;

    std::cout << p << std::endl;
  }

  void DREM::step(const Eigen::VectorXd& y, const Eigen::MatrixXd& phi)
  {
    reg_ext.step(y, phi);

    Eigen::VectorXd y_f = reg_ext.getY();
    Eigen::MatrixXd phi_f = reg_ext.getPhi();

    double Delta = phi_f.determinant();

    Eigen::MatrixXd phi_tmp = phi_f;
    float Yvar_i;

    for (int i = 0; i < p; ++i)
    {
      phi_tmp(Eigen::all, i) = y_f;

      Yvar_i = phi_tmp.determinant();

      y_err_i = Yvar_i - Delta * theta_est[i];
      dtheta[i] = gamma[i] * Delta * (pow(abs(y_err_i), r) * std::signbit(-y_err_i));

      phi_tmp(Eigen::all, i) = phi_f(Eigen::all, i);
    }

    theta_est += dt * dtheta;
  }

  Eigen::VectorXd DREM::get_estimate()
  {
    return theta_est;
  }

  void DREM::reset()
  {
    theta_est *= 0;
    reg_ext.reset();
  }



}