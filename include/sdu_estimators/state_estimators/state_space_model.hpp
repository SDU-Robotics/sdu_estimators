#ifndef STATE_SPACE_MODEL_HPP
#define STATE_SPACE_MODEL_HPP

#include <Eigen/Dense>
#include "utils.hpp"

namespace sdu_estimators::state_estimators
{
  class StateSpaceModel
  {
  public:
    StateSpaceModel(Eigen::MatrixXd & A,
                    Eigen::MatrixXd & B,
                    Eigen::MatrixXd & C,
                    Eigen::MatrixXd & D,
                    float Ts,
                    utils::IntegrationMethod method);

    void update(Eigen::VectorXd & u);

    Eigen::VectorXd get_output();

    Eigen::VectorXd get_state();

    Eigen::MatrixXd getAd();

    Eigen::MatrixXd getBd();

    ~StateSpaceModel();

  private:
    int state_size{};
    int input_size{};
    int output_size{};
    Eigen::MatrixXd A, B, C, D, Ad, Bd;
    Eigen::VectorXd x, y;
    float Ts;

    void check_inputs();
  };
}

#endif //STATE_SPACE_MODEL_HPP
