#ifndef STATE_SPACE_DISTURBANCE_OBSERVER_HPP
#define STATE_SPACE_DISTURBANCE_OBSERVER_HPP

#include "sdu_estimators/integrator/integrator.hpp"

#include "sdu_estimators/typedefs.hpp"

namespace sdu_estimators::disturbance_observers
{
  /**
   * DIM_Nx: Dimension of state
   * DIM_Nu: Dimension of input
   * DIM_Nd: Dimension of disturbance
   */
  template <typename T, int32_t DIM_Nx, int32_t DIM_Nu, int32_t DIM_Nd>
  class StateSpaceDisturbanceObserver
  {
  public:

    /**
     * @brief A linear disturbance observer (DOB) on state space form.
     * The system is on the form:
     *  dot(x) = A x + B u + D d
     *  y = C x
     *
     * And the observer is on the form:
     *  dot(z) = -Ld * D ( z + L x) - Ld (A x + B u)
     *  hat(d) = z + Ld x
     *
     * As can be seen, it assumes knowledge of the states.
     * As of now, it only works with the Euler integrator.
     *
     * @param A
     * @param B
     * @param D
     * @param Ld
     * @param L
     * @param dt
     * @param intg_method: Integration method
     */
    StateSpaceDisturbanceObserver(
      const Eigen::Matrix<T, DIM_Nx, DIM_Nx> A,
      const Eigen::Matrix<T, DIM_Nx, DIM_Nu> B,
      const Eigen::Matrix<T, DIM_Nx, DIM_Nd> D,
      const Eigen::Matrix<T, DIM_Nd, DIM_Nx> Ld,
      const Eigen::Matrix<T, DIM_Nd, DIM_Nx> L,
      float dt,
      sdu_estimators::integrator::IntegrationMethod intg_method
    ) : A(A), B(B), D(D), Ld(Ld), L(L), dt(dt), intg_method(intg_method)
    {
      reset();
    }

    void step(const Eigen::Vector<T, DIM_Nx> & x, const Eigen::Vector<T, DIM_Nu> & u)
    {
      d_est = z + Ld * x;
      
      auto get_dydt = [=](Eigen::Vector<T, DIM_Nd> z_)
      {
        Eigen::Vector<T, DIM_Nd> dz = -Ld * D * (z_ + L * x) - Ld * (A * x + B * u);

        // std::cout << "x: " << x << ", z_: " << z_ << ", u: " << u << ", dz:" << dz << std::endl;

        return dz;
      };

      z = sdu_estimators::integrator::Integrator<T, DIM_Nd, 1>::integrate(
        z,
        get_dydt,
        dt,
        intg_method
      );
      
      // Eigen::Vector<T, DIM_Nd> dz = -Ld * D * (z + L * x) - Ld * (A * x + B * u);
      

      // z = z + dt * dz;
      
      // std::cout << "dt " << dt << std::endl;
      // std::cout << "z \n" << z << std::endl;

      
    }

    Eigen::Vector<T, DIM_Nd> get_disturbance_estimate()
    {
      return d_est;
    }

    void reset()
    {
      z.setZero();
      d_est.setZero();
    }

    ~StateSpaceDisturbanceObserver() {}

   private:
      Eigen::Matrix<T, DIM_Nx, DIM_Nx> A;
      Eigen::Matrix<T, DIM_Nx, DIM_Nu> B;
      Eigen::Matrix<T, DIM_Nx, DIM_Nd> D;
      Eigen::Matrix<T, DIM_Nd, DIM_Nx> Ld;
      Eigen::Matrix<T, DIM_Nd, DIM_Nx> L;
      float dt;
      sdu_estimators::integrator::IntegrationMethod intg_method;

      // Internal state and disturbance estimate:
      Eigen::Vector<T, DIM_Nd> z, d_est;
  };
}

#endif //STATE_SPACE_DISTURBANCE_OBSERVER_HPP
