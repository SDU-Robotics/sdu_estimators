#ifndef LUENBERGER_OBSERVER_HPP
#define LUENBERGER_OBSERVER_HPP
#include "state_space_model.hpp"

namespace sdu_estimators::state_estimators
{
  class LuenbergerObserver : public StateSpaceModel
  {
  public:
    /**
     * @brief Continous
     *
     * @param A
     * @param B
     * @param C
     * @param L
     * @param Ts
     * @param method
     */
    LuenbergerObserver(Eigen::MatrixXd & A,
                       Eigen::MatrixXd & B,
                       Eigen::MatrixXd & C,
                       Eigen::MatrixXd & L,
                       float Ts,
                       utils::IntegrationMethod method);

    void update(Eigen::VectorXd & y, Eigen::VectorXd & u);

    Eigen::VectorXd get_state_estimate()
    {
      return StateSpaceModel::get_state();
    }

  private:
    Eigen::VectorXd yhat_old;
  };
}

#endif //LUENBERGER_OBSERVER_HPP
