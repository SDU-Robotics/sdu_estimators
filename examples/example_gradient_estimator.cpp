#include <cmath>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <sdu_estimators/sdu_estimators.hpp>
#include <sdu_estimators/estimators/gradient_estimator.hpp>

#include <vector>

int main()
{
  float dt = 0.002;
  float tend = 50 / dt; // 10s
  float gamma = 1;
  Eigen::VectorXd theta_init, theta_true;
  theta_init.resize(2);
  theta_true.resize(2);

  theta_init << 0,
                0;
  theta_true << 1,
                2;

  sdu_estimators::estimators::GradientEstimator grad_est(dt, gamma, theta_init);
  std::vector<Eigen::VectorXd> all_theta_est;
  Eigen::VectorXd y;
  Eigen::VectorXd phi;
  y.resize(1);
  phi.resize(2);

  float t;

  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;
    phi << std::sin(t), std::cos(t);
    y << phi.transpose() * theta_true;

    grad_est.step(y, phi);
    Eigen::VectorXd tmp = grad_est.get_estimate();

    // save data
    all_theta_est.push_back(tmp);
    std::cout << tmp.transpose() << std::endl;
  }

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_gradient.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
            << "," << theta_true[0] << "," << theta_true[1] << std::endl;
  }

  outfile.close();
}
