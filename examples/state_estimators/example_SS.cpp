#include <Eigen/Core>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/state_estimators/state_space_model.hpp>
#include <vector>
#include <sdu_estimators/state_estimators/utils.hpp>

int main()
{
  float c, k, m;
  c = 4;
  k = 2;
  m = 20;

  Eigen::Vector<double, 1> F;
  // F.resize(1);
  F << 5;

  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> B;
  Eigen::Matrix<double, 1, 2> C;
  Eigen::Matrix<double, 1, 1> D;

  // A.resize(2, 2);
  A << 0, 1,
       -k/m, -c/m;

  // B.resize(2, 1);
  B << 0,
       1/m;

  // C.resize(1, 2);
  C << 1, 0;

  // D.resize(1, 1);
  D << 0;

  float Ts = 0.002;
  float tstart = 0;
  float tstop = 60;
  int N = (tstop - tstart) / Ts;

  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Euler;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::EulerBackwards;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Bilinear;
  sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Exact;

  sdu_estimators::state_estimators::StateSpaceModel<double, 2, 1, 1> sys(A, B, C, D, Ts, method);
  sys.reset();

  std::cout << sys.getAd() << std::endl;
  std::cout << sys.getBd() << std::endl;

  std::vector<Eigen::Vector<double, 2>> all_states;
  Eigen::VectorXd tmp;

  for (int i = 0; i < N; ++i)
  {
    // std::cout << i << std::endl;
    sys.update(F);
    // std::cout << F << std::endl;

    tmp = sys.get_state();
    all_states.push_back(tmp);

    // std::cout << tmp << std::endl;
  }

  // Write all_theta_est to file
  std::ofstream outfile;
  outfile.open ("data_SS.csv");

  outfile << "timestamp,x1,x2" << std::endl;

  for (int i = 0; i < N; ++i)
  {
    outfile << i * Ts << "," << all_states[i][0] << "," << all_states[i][1] << std::endl;
  }

  outfile.close();
}
