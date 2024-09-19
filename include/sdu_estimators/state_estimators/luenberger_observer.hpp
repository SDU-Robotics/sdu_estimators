#ifndef LUENBERGER_OBSERVER_HPP
#define LUENBERGER_OBSERVER_HPP
#include "state_space_model.hpp"

namespace sdu_estimators::state_estimators
{
  template <typename T, int32_t DIM_Nx, int32_t DIM_Nu, int32_t DIM_Ny>
  class LuenbergerObserver : public StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>
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
    LuenbergerObserver(Eigen::Matrix<T, DIM_Nx, DIM_Nx> & A,
                       Eigen::Matrix<T, DIM_Nx, DIM_Nu> & B,
                       Eigen::Matrix<T, DIM_Ny, DIM_Nx> & C,
                       Eigen::Matrix<T, DIM_Nx, DIM_Ny> & L,
                       float Ts,
                       utils::IntegrationMethod method)
    {
      Eigen::Matrix<T, DIM_Nx, DIM_Nx> newA = A - L * C;

      Eigen::Matrix<T, DIM_Nx, DIM_Nu + DIM_Ny> newB, newD;
      newB << B, L;

      // std::cout << "newB\n" << newB << std::endl;
      newD = newB;
      newD.setZero();

      this->StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>::~StateSpaceModel();                 // destroy the base class
      new (this) StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>(newA, newB, C, newD, Ts, method);  // overwrites the base class storage with a new instance
      // new (this) StateSpaceModel(newA, newB, C, newD);  // overwrites the base class storage with a new instance
      // std::cout << "Bd\n" << this->getBd() << std::endl;
      StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>::reset();
    }

    void update(Eigen::Vector<T, DIM_Ny> & y, Eigen::Vector<T, DIM_Nu> & u)
    {
      Eigen::Vector<T, DIM_Nu + DIM_Ny> uu;
      uu << u, y;

      StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>::update(uu);
    }

    Eigen::Vector<T, DIM_Nx> get_state_estimate()
    {
      return StateSpaceModel<T, DIM_Nx, DIM_Nu + DIM_Ny, DIM_Ny>::get_state();
    }
  };
}

#endif //LUENBERGER_OBSERVER_HPP
