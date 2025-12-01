#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "sdu_estimators/parameter_estimators/gradient_estimator.hpp"
#include "sdu_estimators/integrator/integrator.hpp"
#include "sdu_estimators/math/persistency_of_excitation.hpp"

#define DIM_N 4
#define DIM_P 2

using namespace sdu_estimators;

int main()
{
  float dt = 0.001;
  float tend = 20 / dt; // 10s
  Eigen::Vector<double, 2> gamma = {1, 1};
  gamma *= 0.1;
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
  // sdu_estimators::parameter_estimators::GradientEstimator grad_est(dt, gamma, theta_init);
  std::vector<Eigen::VectorXd> all_theta_est;
  Eigen::Vector<double, DIM_N> y;
  Eigen::Matrix<double, DIM_P, DIM_N> phi;

  std::vector<Eigen::VectorXd> all_eig_vals;

  math::PersistencyOfExcitation<double, DIM_N, DIM_P> pe_calc(dt, 500);

  float t;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;
    // phi << std::sin(t) + 1,
    //        std::sin(t + 2) + 2;
    phi << 2.*std::cos(t), -std::cos(t+1.), 3.*std::cos(2.*t+1./2.), 2.*std::cos(t/3. + 1.),
            std::cos(2.*t), std::cos(t/2.), 2.*std::cos(3.*t/2. + 3./4.), -3.*std::cos(4.*t/3.);

    y << phi.transpose() * theta_true;

    grad_est.step(y, phi);
      // sdu_estimators::parameter_estimators::utils::IntegrationMethod::Euler);

    Eigen::VectorXd tmp = grad_est.get_estimate();

    // save data
    all_theta_est.push_back(tmp);
    
    // std::cout << tmp.transpose() << std::endl;

    // calculate PE
    pe_calc.step(phi);
    all_eig_vals.push_back(pe_calc.get_eigen_values());
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_gradient_PE.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2,eig_val_1,eig_val_2" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
            << "," << theta_true[0] << "," << theta_true[1] << "," 
            << all_eig_vals[i][0] << "," << all_eig_vals[i][1] << std::endl;
  }

  outfile.close();
}
