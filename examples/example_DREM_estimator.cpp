#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <sdu_estimators/sdu_estimators.hpp>
#include <sdu_estimators/estimators/drem.hpp>

#include <vector>

int main()
{
  float dt = 0.002;
  float tend = 50 / dt; // 10s

  Eigen::VectorXd gamma;
  gamma.resize(2);
  gamma << 10,
           10;

  float ell = 1;
  float r = 0.5;
  Eigen::VectorXd theta_init, theta_true;
  theta_init.resize(2);
  theta_true.resize(2);

  theta_init << 0,
                0;
  theta_true << 1,
                2;

  sdu_estimators::estimators::DREM DREM(dt, gamma, theta_init, ell, r);
  // sdu_estimators::estimators::GradientEstimator grad_est(dt, gamma, theta_init);
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
    std::cout << tmp.transpose() << std::endl;
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
