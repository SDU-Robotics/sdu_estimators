#ifndef UNKNOWN_INPUT_OBSERVER_HPP
#define UNKNOWN_INPUT_OBSERVER_HPP
#include "state_space_model.hpp"

namespace sdu_estimators::state_estimators
{
  class UnknownInputObserver : public StateSpaceModel
  {
  public:
    UnknownInputObserver(Eigen::MatrixXd & A,
                         Eigen::MatrixXd & B,
                         Eigen::MatrixXd & C,
                         Eigen::MatrixXd & E,
                         Eigen::VectorXd & poles,
                         float Ts,
                         utils::IntegrationMethod method);

    void update(Eigen::VectorXd & y, Eigen::VectorXd & u);

  private:
    Eigen::VectorXd yhat_old;
    Eigen::MatrixXd H, T, A1;
  };
}

#endif //UNKNOWN_INPUT_OBSERVER_HPP
