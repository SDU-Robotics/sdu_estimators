#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "sdu_estimators/parameter_estimators/gradient_estimator.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

using namespace sdu_estimators;

#define DIM_N 4
#define DIM_P 2

int main()
{
  float dt = 0.001;
  float tend = 50 / dt; // 10s

  double gamma_ = 10;
  Eigen::Vector<double, 2> gamma = {gamma_, gamma_};
  float r = 1.;
  Eigen::Matrix<double, 2, 1> theta_init, theta_true;
  // theta_init.resize(2);
  // theta_true.resize(2);

  theta_init << 0,
                0;
  theta_true << 1,
                2;

  integrator::IntegrationMethod intg_method = integrator::IntegrationMethod::RK4;
  parameter_estimators::GradientEstimator<double, DIM_N, DIM_P> grad_est(dt, gamma, theta_init, r, intg_method);

  Eigen::Vector<double, 2> theta_lower_bound, theta_upper_bound;
  theta_lower_bound << 0, 0;
  theta_upper_bound << 10, 10;

  grad_est.set_theta_bounds(theta_lower_bound, theta_upper_bound);

  grad_est.enable_normalisation(1);

  // sdu_estimators::parameter_estimators::GradientEstimator grad_est(dt, gamma, theta_init);
  std::vector<Eigen::VectorXd> all_theta_est;
  Eigen::Vector<double, DIM_N> y;
  Eigen::Matrix<double, DIM_P, DIM_N> phi;

  float t;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;
    // phi << std::sin(t),
    //        std::cos(t);
    phi << 2.*std::cos(t), -std::cos(t+1.), 3.*std::cos(2.*t+1./2.), 2.*std::cos(t/3. + 1.),
            std::cos(2.*t), std::cos(t/2.), 2.*std::cos(3.*t/2. + 3./4.), -3.*std::cos(4.*t/3.);
    y << phi.transpose() * theta_true;

    grad_est.step(y, phi);
      // sdu_estimators::parameter_estimators::utils::IntegrationMethod::Euler);

    Eigen::VectorXd tmp = grad_est.get_estimate();

    // save data
    all_theta_est.push_back(tmp);
    // std::cout << tmp.transpose() << std::endl;
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_gradient_normalised.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
            << "," << theta_true[0] << "," << theta_true[1] << std::endl;
  }

  outfile.close();
}
