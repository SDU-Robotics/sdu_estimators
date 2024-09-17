#include <Eigen/Dense>
#include <iostream>
#include <sdu_estimators/state_estimators/utils.hpp>

int main()
{
  Eigen::MatrixXd A, C;

  A.resize(2, 2);
  A << 1, 2,
       3, 4;

  std::cout << A << std::endl;

  C.resize(1, 2);
  C << 1, 2;

  std::cout << C << std::endl;

  auto obsv_mat = sdu_estimators::state_estimators::utils::obsv(A, C);

  std::cout << "obsv_mat\n" << obsv_mat << std::endl;

  return 1;
}