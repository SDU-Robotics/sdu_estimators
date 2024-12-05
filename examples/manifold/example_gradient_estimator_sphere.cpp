#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/parameter_estimators/gradient_estimator_sphere.hpp>
#include <sdu_estimators/math/riemannian_manifolds/sphere.hpp>
#include <vector>

int main()
{
  double dt = 0.0001;
  double tend = 20 / dt; // 10s
  double gamma = 2;

  Eigen::Vector<double, 3> theta_init, theta_true;
  // theta_init.resize(2);
  // theta_true.resize(2);

  theta_init << 1,
                2,
                3;
  theta_init.normalize();
  theta_true << 0,
                0,
                1;

  sdu_estimators::parameter_estimators::GradientEstimatorSphere<double, 3> estimator(dt, gamma, theta_init);
//  // sdu_estimators::parameter_estimators::GradientEstimator grad_est(dt, gamma, theta_init);
  std::vector<Eigen::VectorXd> all_theta_est;
  Eigen::VectorXd y;
  Eigen::VectorXd phi;
  y.resize(1);
  phi.resize(3);

  sdu_estimators::math::manifold::Sphere<double, 3> sphere_manifold;
  std::vector<double> dist;

  float t;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;
    phi << std::sin(1 * t),
           std::cos(1 * t),
           0;

    // phi += 1*Eigen::Vector3d::Random();

    y << 0;

    estimator.step(y, phi);
      // sdu_estimators::parameter_estimators::utils::IntegrationMethod::Euler);

    Eigen::Vector<double, 3> tmp = estimator.get_estimate();

    // save data
    all_theta_est.push_back(tmp);
    // std::cout << tmp.transpose() << std::endl;

    dist.push_back(sphere_manifold.dist(tmp, theta_true));
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" << std::endl;
  std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_gradient_sphere.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_est_3,theta_act_1,theta_act_2,theta_act_3,dist" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1] << "," << all_theta_est[i][2]
            << "," << theta_true[0] << "," << theta_true[1] << "," << theta_true[2]
            << "," << dist[i] << std::endl;
  }

  outfile.close();
}
