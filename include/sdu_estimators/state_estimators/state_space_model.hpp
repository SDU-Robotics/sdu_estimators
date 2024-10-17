#ifndef STATE_SPACE_MODEL_HPP
#define STATE_SPACE_MODEL_HPP

#include <Eigen/Dense>
#include "sdu_estimators/state_estimators/utils.hpp"

namespace sdu_estimators::state_estimators
{
  template <typename T, int32_t DIM_Nx, int32_t DIM_Nu, int32_t DIM_Ny>
  class StateSpaceModel
  {
  public:
    StateSpaceModel();

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
    StateSpaceModel(Eigen::Matrix<T, DIM_Nx, DIM_Nx> & A,
                    Eigen::Matrix<T, DIM_Nx, DIM_Nu> & B,
                    Eigen::Matrix<T, DIM_Ny, DIM_Nx> & C,
                    Eigen::Matrix<T, DIM_Ny, DIM_Nu> & D,
                    float Ts,
                    utils::IntegrationMethod method)
                      : A(A), B(B), C(C), D(D), Ts(Ts)
    {
      utils::c2d<T, DIM_Nx, DIM_Nu>(A, B, Ts, Ad, Bd, method);
      reset();
    }

    /**
     * @brief Constructor for discrete model.
     *
     * @param Ad
     * @param Bd
     * @param C
     * @param D
     */
    StateSpaceModel(Eigen::Matrix<T, DIM_Nx, DIM_Nx> & Ad,
                    Eigen::Matrix<T, DIM_Nx, DIM_Nu> & Bd,
                    Eigen::Matrix<T, DIM_Ny, DIM_Nx> & C,
                    Eigen::Matrix<T, DIM_Ny, DIM_Nu> & D)
                      : Ad(Ad), Bd(Bd), C(C), D(D)
    {
      this->A.setZero();
      this->B.setZero();

      reset();
    }

    void update(Eigen::Vector<T, DIM_Nu> & u)
    {
      x = Ad * x + Bd * u;
      y = C * x + D * u;
    }

    Eigen::Vector<T, DIM_Ny> get_output()
    {
      return y;
    }

    Eigen::Vector<T, DIM_Nx> get_state()
    {
      return x;
    }

    Eigen::Matrix<T, DIM_Nx, DIM_Nx> getAd()
    {
      return Ad;
    }

    Eigen::Matrix<T, DIM_Nx, DIM_Nu> getBd()
    {
      return Bd;
    }

    void reset()
    {
      x.setZero();
      y.setZero();
    }

    ~StateSpaceModel()
    {}

  private:
    int state_size{};
    int input_size{};
    int output_size{};
    Eigen::Matrix<T, DIM_Nx, DIM_Nx> A, Ad;
    Eigen::Matrix<T, DIM_Nx, DIM_Nu> B, Bd;
    Eigen::Matrix<T, DIM_Ny, DIM_Nx> C;
    Eigen::Matrix<T, DIM_Ny, DIM_Nu> D;
    Eigen::Vector<T, DIM_Nx> x;
    Eigen::Vector<T, DIM_Ny> y;
    float Ts;

    void check_inputs();
  };
}

#endif //STATE_SPACE_MODEL_HPP
