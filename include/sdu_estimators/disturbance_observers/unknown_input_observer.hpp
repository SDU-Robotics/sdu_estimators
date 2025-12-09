#ifndef UNKNOWN_INPUT_OBSERVER_HPP
#define UNKNOWN_INPUT_OBSERVER_HPP

#include "sdu_estimators/state_estimators/state_space_model.hpp"
#include "sdu_estimators/state_estimators/utils.hpp"

#include "sdu_estimators/typedefs.hpp"

namespace sdu_estimators::disturbance_observers
{
  template <typename T, int32_t DIM_Nx, int32_t DIM_Nu, int32_t DIM_Ny>
  class UnknownInputObserver
  {
  public:

    /**
     * @brief The unknown input observer.
     *    The matrices F, T, K and H should be found following a design procedure in e.g.,
     *    MATLAB or Python.
     *
     * @param F
     * @param Tmat
     * @param B
     * @param K
     * @param H
     * @param Ts
     * @param method
     */
    UnknownInputObserver(
      Eigen::Matrix<T, DIM_Nx, DIM_Nx> & F,
      Eigen::Matrix<T, DIM_Nx, DIM_Nx> & Tmat,
      Eigen::Matrix<T, DIM_Nx, DIM_Nu> & B,
      Eigen::Matrix<T, DIM_Nx, DIM_Ny> & K,
      Eigen::Matrix<T, DIM_Nx, DIM_Ny> & H,
      float Ts,
      sdu_estimators::state_estimators::utils::IntegrationMethod method
    )
    {
      Eigen::Matrix<T, DIM_Nx, DIM_Nx> newA, newC;
      Eigen::Matrix<T, DIM_Nx, DIM_Nu + DIM_Ny> newB, newD;

      newA << F;
      newB << Tmat * B, K;
      newC.setIdentity();
      newD << B * 0, H;

      sys = new sdu_estimators::state_estimators::StateSpaceModel(newA, newB, newC, newD, Ts, method);
    }

    void update(Eigen::Vector<T, DIM_Ny> & y, Eigen::Vector<T, DIM_Nu> & u)
    {
      Eigen::Vector<T, DIM_Nu + DIM_Ny> uu;
      uu << u, y;

      sys->update(uu);
    }

    Eigen::VectorXd get_state_estimate()
    {
      return sys->get_output();
    }

    Eigen::Matrix<T, DIM_Nx, DIM_Nx> getAd()
    {
      return sys->getAd();
    }

    Eigen::Matrix<T, DIM_Nx, DIM_Nu + DIM_Ny> getBd()
    {
      return sys->getBd();
    }


   private:
    Eigen::VectorXd yhat_old;
    Eigen::MatrixXd H, Tmat, A1;

    sdu_estimators::state_estimators::StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny> * sys;
  };
}

#endif //UNKNOWN_INPUT_OBSERVER_HPP
