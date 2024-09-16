#include "sdu_estimators/state_estimators/state_space_model.hpp"
// #include "sdu_estimators/state_estimators/utils.hpp"

namespace sdu_estimators::state_estimators
{

  /**
   * @brief Constructor for continouos time model.
   *
   * @param A
   * @param B
   * @param C
   * @param D
   * @param Ts
   * @param method
   */
  StateSpaceModel::StateSpaceModel(Eigen::MatrixXd& A, Eigen::MatrixXd& B, Eigen::MatrixXd& C, Eigen::MatrixXd& D,
    float Ts, utils::IntegrationMethod method)
    : A(A), B(B), C(C), D(D), Ts(Ts)
  {
    state_size = A.rows();
    input_size = B.cols();
    output_size = C.rows();

    c2d(A, B, Ts, Ad, Bd, method);

    x.resize(state_size);
    x.setZero();
  }

  /**
   * @brief Constructor for discrete model.
   *
   * @param Ad
   * @param Bd
   * @param C
   * @param D
   */
  StateSpaceModel::StateSpaceModel(Eigen::MatrixXd& Ad, Eigen::MatrixXd& Bd, Eigen::MatrixXd& C, Eigen::MatrixXd& D)
  : Ad(Ad), Bd(Bd), C(C), D(D)
  {
    state_size = Ad.rows();
    input_size = Bd.cols();
    output_size = C.rows();

    A.setZero(Ad.rows(), Ad.cols());
    B.setZero(Bd.rows(), Bd.cols());

    x.resize(state_size);
    x.setZero();

    Ts = 0;
  }

  void StateSpaceModel::update(Eigen::VectorXd& u)
  {
    x = Ad * x + Bd * u;
    y = C * x + D * u;
  }

  Eigen::VectorXd StateSpaceModel::get_output()
  {
    return y;
  }

  Eigen::VectorXd StateSpaceModel::get_state()
  {
    return x;
  }

  Eigen::MatrixXd StateSpaceModel::getAd()
  {
    return Ad;
  }

  Eigen::MatrixXd StateSpaceModel::getBd()
  {
    return Bd;
  }

  void StateSpaceModel::reset()
  {
    x.setZero();
  }

  StateSpaceModel::~StateSpaceModel()
  {
  }
}  // namespace sdu_estimators::state_estimators