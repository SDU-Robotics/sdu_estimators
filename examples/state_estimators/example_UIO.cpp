#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/state_estimators/state_space_model.hpp>
#include <sdu_estimators/state_estimators/utils.hpp>
#include <vector>

#include "sdu_estimators/state_estimators/unknown_input_observer.hpp"

int main()
{
  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> B;
  Eigen::Matrix<double, 2, 2> C;
  Eigen::Matrix<double, 2, 1> D;
  Eigen::Matrix<double, 2, 1> E;
  Eigen::Matrix<double, 2, 2> B_alt;
  Eigen::Matrix<double, 2, 2> D_alt;

  Eigen::Matrix<double, 2, 2> F;
  Eigen::Matrix<double, 2, 2> Tmat;
  Eigen::Matrix<double, 2, 2> K;
  Eigen::Matrix<double, 2, 2> H;

  A << 0, 1,
       0, -10;
  std::cout << "A\n" << A << std::endl;

  B << 0,
       10;
  std::cout << "B\n" << B << std::endl;

  C.setIdentity();
  std::cout << "C\n" << C << std::endl;

  D.setZero();
  std::cout << "D\n" << D << std::endl;

  E << 0,
       1;
  std::cout << "E\n" << E << std::endl;

  B_alt << B, E;
  // B_alt.leftCols(B.cols()) = B;
  // B_alt.rightCols(E.cols()) = E;
  std::cout << "B_alt\n" << B_alt << std::endl;

  D_alt.setZero();
  // D_alt(Eigen::seqN(0, 2), Eigen::all) = D;
  D_alt.leftCols(D.cols()) = D;
  std::cout << "D_alt\n" << D_alt << std::endl;

  // UIO matrices
  F << -2,  0,
        0, -1;

  Tmat << 1, 0,
       0, 0;

  K << 2, 1,
       0, 0;

  H << 0, 0,
       0, 1;

  float Ts = 0.001;
  float tstart = 0;
  float tstop = 50;
  int N = (tstop - tstart) / Ts;

  sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Euler;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::EulerBackwards;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Bilinear;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Exact;

  sdu_estimators::state_estimators::StateSpaceModel<double, 2, 2, 2> sys(A, B_alt, C, D_alt, Ts, method);
  sys.reset();

  std::cout << "Ad\n" << sys.getAd() << std::endl;
  std::cout << "Bd\n" << sys.getBd() << std::endl;

  sdu_estimators::state_estimators::UnknownInputObserver<double, 2, 1, 2> obs(F, Tmat, B, K, H, Ts, method);

  std::cout << "obs Ad \n" << obs.getAd() << std::endl;
  std::cout << "obs Bd \n" << obs.getBd() << std::endl;

  std::vector<Eigen::Vector<double, 2>> all_states;
  std::vector<Eigen::Vector<double, 2>> all_est_states;
  Eigen::VectorXd tmp1, tmp2;

  Eigen::Vector<double, 2> uu, y;
  Eigen::Vector<double, 1> u, d;

  for (int i = 0; i < N; ++i)
  {
    // std::cout << i << std::endl;
    u << 0;
    d << sin(i * Ts);
    uu << u, d;

    // std::cout << "uu\n" << uu << std::endl;

    sys.update(uu);
    // std::cout << F << std::endl;

    tmp1 = sys.get_state();
    // std::cout << "tmp\n" << tmp << std::endl;

    all_states.push_back(tmp1);

    y = sys.get_output();
    // Eigen::VectorXd u_;
    // u_.resize(1);
    // u_[0] = u[1];

    obs.update(y, u);

    // tmp2 = obs.get_state();
    tmp2 = obs.get_state_estimate();
    all_est_states.push_back(tmp2);

    // std::cout << y << std::endl;
  }

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_UIO.csv");

  outfile << "timestamp,actual_x_1,actual_x_2,estimated_x_1,estimated_x_2,x_err_1,x_err_2" << std::endl;

  for (int i = 0; i < N; ++i)
  {
    outfile << i * Ts << "," << all_states[i][0] << "," << all_states[i][1] <<
      "," << all_est_states[i][0] << "," << all_est_states[i][1] <<
      "," << all_states[i][0] - all_est_states[i][0] <<
      "," << all_states[i][1] - all_est_states[i][1] << std::endl;
  }

  outfile.close();
}
