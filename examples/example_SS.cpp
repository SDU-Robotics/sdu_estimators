#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/state_estimators/state_space_model.hpp>
#include <sdu_estimators/sdu_estimators.hpp>
#include <vector>
#include <sdu_estimators/state_estimators/utils.hpp>

int main()
{
  Eigen::MatrixXd A, B, C, D;

  A.resize(2, 2);
  A << 0, 1,
       -1, -1;

  B.resize(2, 1);
  B << 0,
       1;

  C.resize(1, 2);
  C << 1, 0;

  D.resize(1, 1);
  D << 0;

  float Ts = 0.002;

  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Euler;
  // sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::EulerBackwards;
  sdu_estimators::state_estimators::utils::IntegrationMethod method = sdu_estimators::state_estimators::utils::Bilinear;

  sdu_estimators::state_estimators::StateSpaceModel SS(A, B, C, D, Ts, method);

  std::cout << SS.getAd() << std::endl;
  std::cout << SS.getBd() << std::endl;
}
