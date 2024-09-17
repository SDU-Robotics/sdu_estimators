#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/sdu_estimators.hpp>
#include <sdu_estimators/state_estimators/state_space_model.hpp>
#include <sdu_estimators/state_estimators/utils.hpp>
#include <vector>

#include "sdu_estimators/state_estimators/luenberger_observer.hpp"

int main()
{
  Eigen::MatrixXd A, B, C, D, E, B_alt, D_alt, L;

  A.resize(2, 2);
  A << 0, 1,
       0, -10;
  std::cout << "A\n" << A << std::endl;

  B.resize(2, 1);
  B << 0,
       10;
  std::cout << "B\n" << B << std::endl;

  C.resize(2, 2);
  C.setIdentity();
  std::cout << "C\n" << C << std::endl;

  D.resize(2, 1);
  D.setZero();
  std::cout << "D\n" << D << std::endl;

  E.resize(2, 1);
  E << 0,
       1;
  std::cout << "E\n" << E << std::endl;

  B_alt.resize(B.rows(), B.cols() + E.cols());
  // B_alt << B, E;
  B_alt.leftCols(B.cols()) = B;
  B_alt.rightCols(E.cols()) = E;
  std::cout << "B_alt\n" << B_alt << std::endl;

  D_alt.resize(D.rows(), 2 * D.cols());
  D_alt.setZero();
  // D_alt(Eigen::seqN(0, 2), Eigen::all) = D;
  D_alt.leftCols(D.cols()) = D;
  std::cout << "D_alt\n" << D_alt << std::endl;

  L.resize(2, 2);
  L << 2, 1,
       0, -9;
  std::cout << "L\n" << L << std::endl;

  float Ts = 0.002;
  float tstart = 0;
  float tstop = 25;
  int N = (tstop - tstart) / Ts;

  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Euler;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::EulerBackwards;
  sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Bilinear;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Exact;

  sdu_estimators::state_estimators::StateSpaceModel sys(A, B_alt, C, D_alt, Ts, method);
  sys.reset();

  std::cout << "Ad\n" << sys.getAd() << std::endl;
  std::cout << "Bd\n" << sys.getBd() << std::endl;

  sdu_estimators::state_estimators::LuenbergerObserver obs(A, B, C, L, Ts, method);

  std::cout << "obs Ad \n" << obs.getAd() << std::endl;
  std::cout << "obs Bd \n" << obs.getBd() << std::endl;

  std::vector<Eigen::Vector<double, 2>> all_states;
  std::vector<Eigen::Vector<double, 2>> all_est_states;
  Eigen::VectorXd tmp, u, y, uu;

  u.resize(1);
  uu.resize(2);

  for (int i = 0; i < N; ++i)
  {
    // std::cout << i << std::endl;
    u << 0;
    uu << u, sin(i * Ts);

    // std::cout << "uu\n" << uu << std::endl;

    sys.update(uu);
    // std::cout << F << std::endl;

    tmp = sys.get_state();
    // std::cout << "tmp\n" << tmp << std::endl;

    all_states.push_back(tmp);

    y = sys.get_output();
    // Eigen::VectorXd u_;
    // u_.resize(1);
    // u_[0] = u[1];

    obs.update(y, u);

    tmp = obs.get_state();
    all_est_states.push_back(tmp);

    // std::cout << y << std::endl;
  }

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_luenberger.csv");

  outfile << "timestamp,actual_x_1,actual_x_2,estimated_x_1,estimated_x_2" << std::endl;

  for (int i = 0; i < N; ++i)
  {
    outfile << i * Ts << "," << all_states[i][0] << "," << all_states[i][1] <<
      "," << all_est_states[i][0] << "," << all_est_states[i][1] << std::endl;
  }

  outfile.close();
}
