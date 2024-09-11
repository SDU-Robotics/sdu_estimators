#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/state_estimators/state_space_model.hpp>
#include <sdu_estimators/sdu_estimators.hpp>
#include <vector>

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

  float Ts = 0.5;

  sdu_estimators::state_estimators::StateSpaceModel SS(A, B, C, D, Ts);

  std::cout << SS.getAd() << std::endl;
}
