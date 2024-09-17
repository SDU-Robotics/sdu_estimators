#ifndef LUENBERGER_OBSERVER_HPP
#define LUENBERGER_OBSERVER_HPP
#include "state_space_model.hpp"

namespace sdu_estimators::state_estimators
{
  class LuenbergerObserver : public StateSpaceModel
  {
  public:
    // LuenbergerObserver(Eigen::MatrixXd & A,
    //                    Eigen::MatrixXd & B,
    //                    Eigen::MatrixXd & C,
    //                    Eigen::VectorXd & poles,
    //                    float Ts,
    //                    utils::IntegrationMethod method);
    LuenbergerObserver(Eigen::MatrixXd & A,
                       Eigen::MatrixXd & B,
                       Eigen::MatrixXd & C,
                       Eigen::MatrixXd & L,
                       float Ts,
                       utils::IntegrationMethod method);

    void update(Eigen::VectorXd & y, Eigen::VectorXd & u);

  private:
    Eigen::VectorXd yhat_old;
  };
}

#endif //LUENBERGER_OBSERVER_HPP
