#include "sdu_estimators/state_estimators/luenberger_observer.hpp"
#include <iostream>

namespace sdu_estimators::state_estimators
{
  //  LuenbergerObserver::LuenbergerObserver(
  //     Eigen::MatrixXd& A,
  //     Eigen::MatrixXd& B,
  //     Eigen::MatrixXd& C,
  //     Eigen::VectorXd& poles,
  //     float Ts,
  //     utils::IntegrationMethod method)
  // {
  // }

  /**
   *
   * @param A Continuous A-matrix.
   * @param B
   * @param C
   * @param L Discrete time gain matrix.
   * @param Ts
   * @param method
   */
  LuenbergerObserver::LuenbergerObserver(
      Eigen::MatrixXd& A,
      Eigen::MatrixXd& B,
      Eigen::MatrixXd& C,
      Eigen::MatrixXd& L,
      float Ts,
      utils::IntegrationMethod method)
      : StateSpaceModel()
  {
    Eigen::MatrixXd newA = A - L * C;

    Eigen::MatrixXd newB(B.rows(), B.cols() + L.cols()), newD;
    newB << B, L;

    // std::cout << "newB\n" << newB << std::endl;
    newD = newB.replicate(1, 1);
    newD.setZero();

    this->StateSpaceModel::~StateSpaceModel();                 // destroy the base class
    new (this) StateSpaceModel(newA, newB, C, newD, Ts, method);  // overwrites the base class storage with a new instance
    // new (this) StateSpaceModel(newA, newB, C, newD);  // overwrites the base class storage with a new instance
    // std::cout << "Bd\n" << this->getBd() << std::endl;
    StateSpaceModel::reset();

    yhat_old.resize(C.rows());
    yhat_old.setZero();
  }

  void LuenbergerObserver::update(Eigen::VectorXd& y, Eigen::VectorXd& u)
  {
    // Eigen::VectorXd err = y - yhat_old;

    // std::cout << "err" << err << std::endl;

    Eigen::VectorXd uu(u.size() + y.size());
    uu << u, y;

    // std::cout << uu << std::endl;

    StateSpaceModel::update(uu);

    // Eigen::VectorXd yhat = StateSpaceModel::get_output();
    // yhat_old = get_output();
  }

}  // namespace sdu_estimators::state_estimators