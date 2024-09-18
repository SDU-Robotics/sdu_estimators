#include "sdu_estimators/state_estimators/unknown_input_observer.hpp"
#include <stdexcept>

namespace sdu_estimators::state_estimators
{
  // UnknownInputObserver::UnknownInputObserver(
  //     Eigen::MatrixXd& A,
  //     Eigen::MatrixXd& B,
  //     Eigen::MatrixXd& C,
  //     Eigen::MatrixXd& E,
  //     Eigen::VectorXd& poles,
  //     float Ts,
  //     utils::IntegrationMethod method)
  // {
  //   // Design procedure
  //   // 1. Check rank condition for E and CE.
  //   Eigen::MatrixXd CE = C * E;
  //   if (Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(E).rank() !=
  //       Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(CE).rank())
  //   {
  //     std::logic_error("E and CE should have the same rank.");
  //   }
  //
  //   // 2. Compute UIO matrices H, T, A1
  //   H = E * (CE.transpose() * CE).inverse() * CE.transpose();
  //   T = Eigen::MatrixXd::Identity(H.rows(), H.rows()) - H * C;
  //   A1 = T * A;
  //
  //   // 3. Check observability of (C, A1)
  //   Eigen::MatrixXd ob = utils::obsv(A1, C);
  //   int n1 = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(ob).rank();
  //
  //   if (n1 == A.rows())
  //   {
  //     // TODO: (A1, C) is observable, and K1 can be computed by pole placement
  //     // TODO: Implement some kind of pole placement, or figure out how to use SB01BD.f.
  //   }
  //   else
  //   {
  //     /*
  //      * It is not observable, so some other measures has to be taken.
  //      * Construct a transformation matrix P for the observable canonical decomposition.
  //      *
  //      *  - Define the n1 independent row vectors of ob as p1.T, ..., pn1.T.
  //      *  - Define additional n - n1 [n = self.A.shape[0]] independent row vector pn1+1.T, ..., pn.T,
  //      *    and construct a non-singular matrix.
  //      *
  //      *        P = [p1 ... pn1, pn+1 ... pn]
  //      *
  //      */
  //   }
  // }

  /**
   * @brief The unknown input observer.
   *    The matrices F, T, K and H should be found following a design procedure in e.g.,
   *    MATLAB or Python.
   *
   * @param F
   * @param T
   * @param B
   * @param K
   * @param H
   * @param Ts
   * @param method
   */
  UnknownInputObserver::UnknownInputObserver(
      Eigen::MatrixXd& F,
      Eigen::MatrixXd& T,
      Eigen::MatrixXd& B,
      Eigen::MatrixXd& K,
      Eigen::MatrixXd& H,
      float Ts,
      utils::IntegrationMethod method)
  {
    Eigen::MatrixXd newA = F, newB, newC, newD;

    newB.resize(T.rows(), B.cols() + K.cols());
    newB << T * B, K;

    newC.resize(F.rows(), F.rows());
    newC.setIdentity();

    newD.resize(B.rows(), B.cols() + H.cols());
    newD << B * 0, H;

    this->StateSpaceModel::~StateSpaceModel();                 // destroy the base class
    new (this) StateSpaceModel(newA, newB, newC, newD, Ts, method);
  }

  /**
   *
   * @param y
   * @param u
   */
  void UnknownInputObserver::update(Eigen::VectorXd& y, Eigen::VectorXd& u)
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