#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "sdu_estimators/parameter_estimators/drem.hpp"
#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

using namespace sdu_estimators;

int main()
{
  float dt = 0.001;
  float tend = 50 / dt; // 10s

  Eigen::Matrix<double, 2, 1> gamma;
  gamma << 10,
           10;

  gamma *= 1;

  Eigen::Matrix<double, 2, 1> theta_init, theta_true;
  // theta_init.resize(2);
  // theta_true.resize(2);

  theta_init << 0,
                0;
  theta_true << 1,
                2;

  float ell = 0.95;
  sdu_estimators::regressor_extensions::Kreisselmeier<double, 1, 2> reg_ext(dt, ell);

  // Eigen::Vector<double, 2> alpha, beta;
  // alpha << 5, 10;
  // beta << 0.1, 0.5;
  // sdu_estimators::regressor_extensions::LTI<double, 1, 2> reg_ext(dt, alpha, beta);

  // std::cout << "test" << std::endl;
  // std::vector<int> d{0, 100};
  // sdu_estimators::regressor_extensions::Delay<double, 1, 2> reg_ext(d);

  std::cout << "test" << std::endl;

  float r = 1;
  integrator::IntegrationMethod intg_method = integrator::IntegrationMethod::RK4;
  sdu_estimators::parameter_estimators::DREM<double, 1, 2> DREM(dt, gamma, theta_init, &reg_ext, r, intg_method);
  // sdu_estimators::parameter_estimators::GradientEstimator grad_est(dt, gamma, theta_init);
  std::vector<Eigen::VectorXd> all_theta_est;
  Eigen::VectorXd y;
  Eigen::VectorXd phi;
  y.resize(1);
  phi.resize(2);

  float t;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;
    phi << std::sin(t),
           std::cos(t);
    y << phi.transpose() * theta_true;

    DREM.step(y, phi);
    Eigen::VectorXd tmp = DREM.get_estimate();

    // save data
    all_theta_est.push_back(tmp);
    // std::cout << tmp.transpose() << std::endl;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_DREM.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
            << "," << theta_true[0] << "," << theta_true[1] << std::endl;
  }

  outfile.close();
}
