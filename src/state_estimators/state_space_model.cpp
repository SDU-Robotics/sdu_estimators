#include "sdu_estimators/state_estimators/state_space_model.hpp"
// #include "sdu_estimators/state_estimators/utils.hpp"

namespace sdu_estimators::state_estimators
{

  StateSpaceModel::StateSpaceModel(Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& C, Eigen::MatrixXd& D,
    float Ts, utils::IntegrationMethod method)
    : A(A), B(B), C(C), D(D), Ts(Ts)
  {
    state_size = A.rows();
    input_size = B.cols();
    output_size = C.rows();

    c2d(A, B, Ts, Ad, Bd, method);
  }

  void StateSpaceModel::update(Eigen::VectorXd& u)
  {
  }

  Eigen::VectorXd StateSpaceModel::get_output()
  {
  }

  Eigen::VectorXd StateSpaceModel::get_state()
  {
  }
  Eigen::MatrixXd StateSpaceModel::getAd()
  {
    return Ad;
  }
  Eigen::MatrixXd StateSpaceModel::getBd()
  {
    return Bd;
  }

  StateSpaceModel::~StateSpaceModel()
  {
  }
}  // namespace sdu_estimators::state_estimators