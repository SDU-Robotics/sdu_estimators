#include <fstream>
#include <iostream>
#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>

/**
 * Model of the flexible gluing tip used in
 *
 */
int main()
{
  float fs = 500;
  float dt = 1. / fs;
  float tend = 50. / dt;

  Eigen::Matrix<double, 2, 1> theta_init, theta_true;
  theta_init << 0, 0;
  theta_true << 1, 2; // rho, E

  Eigen::Vector<double, 2> gamma = {1e-4, 0.5};

  sdu_estimators::parameter_estimators::GradientEstimator<double, 4, 2> grad_est(dt, gamma, theta_init);

  float A = 1; // section area [m^2]
  float l = 5; // link length [m]
  float I = 1; // section inertia
  float g = 9.82; // gravitational constant [m/s^2]

  Eigen::Matrix<double, 2, 4> phi;
  Eigen::Vector<double, 4> q, dq, ddq, y;

  std::vector<Eigen::VectorXd> all_theta_est;
  std::vector<Eigen::VectorXd> all_y;
  std::vector<Eigen::MatrixXd> all_phi;

  float t;
  for (int i = 0; i < tend; ++i)
  {
    t = i * dt;

    // movement, probably not realistic at all
    ddq.setRandom();
    dq += dt * ddq;
    q += dt * dq;

    // build regressor matrix
    phi(0, 0) = A * ddq[0] * l +A * ddq[2] * l / 2. + A * g * l;
    phi(0, 1) = 0.;
    phi(0, 2) = A * ddq[0] * l / 2. + 13. * A * ddq[2] * l/35. + A * g * l / 2.;
    phi(0, 3) = - A * ddq[0] * pow(l, 2) / 12. - 11. * A * ddq[2] * pow(l, 2) / 210. -
                          A * g * pow(l, 2) / 12.;

    phi(1, 0) = 0.;
    phi(1, 1) = A * q[1] / l;
    phi(1, 2) = - I * (12. * q[3] / pow(l, 2) - 24. * q[2] / pow(l, 3)) / 2.;
    phi(1, 3) = I * (8 * q[3] / l - 12 * q[2] / pow(l, 2)) / 2;

    y << phi.transpose() * theta_true;

    // std::cout << y << std::endl;

    grad_est.step(y, phi);
    all_theta_est.push_back(grad_est.get_estimate());

    all_y.push_back(y);
    all_phi.push_back(phi);
  }

  std::ofstream outfile;
  outfile.open ("data_beam.csv");

  outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2,y1,y2,y3,y4" << std::endl;

  for (int i = 0; i < tend; ++i)
  {
    outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
            << "," << theta_true[0] << "," << theta_true[1];

    for (auto & elem : all_y[i])
      outfile << "," << elem;

    outfile << std::endl;
  }


  return 1;
}