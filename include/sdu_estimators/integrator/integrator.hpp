#pragma once

#ifndef INTEGRAGOR_HPP
#define INTEGRAGOR_HPP

#include <Eigen/Dense>
#include <functional>
#include <iostream>

namespace sdu_estimators::integrator {
    /**
     * @brief Utility functions for integrating different methods.
     *  
     * Inspired by the implementation of the integrators in 
     * https://github.com/sfwa/ukf/blob/master/include/UKF/Integrator.h
     */
    template <typename T, int32_t DIM_N, int32_t DIM_P>
    class IntegratorEuler
    {        
        public:
            using State = Eigen::Matrix<T, DIM_N, DIM_P>;

            static State integrate(
                const State & y,
                const std::function<State(
                    const State&
                )> & get_dydt,
                float delta)
            {
                return y + delta * get_dydt(y);
            }
    };

    template <typename T, int32_t DIM_N, int32_t DIM_P>
    class IntegratorRK2
    {
        public:
            using State = Eigen::Matrix<T, DIM_N, DIM_P>;

            static State integrate(
                const State & y,
                const std::function<State(
                    const State&
                )> & get_dydt,
                float delta)
            {
                State k1 = delta * get_dydt(y);
                State k2 = delta * get_dydt(y + 0.5 * k1);
                return y + k2;
            }
    };

    template <typename T, int32_t DIM_N, int32_t DIM_P>
    class IntegratorRK4
    {
        public:
            using State = Eigen::Matrix<T, DIM_N, DIM_P>;
            
            static State integrate(
                const State & y,
                const std::function<State(
                    const State&
                )> & get_dydt,
                float delta)
            {
                State k1 = delta * get_dydt(y);
                State k2 = delta * get_dydt(y + 0.5 * k1);
                State k3 = delta * get_dydt(y + 0.5 * k2);
                State k4 = delta * get_dydt(y + k3);

                return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.;
            }
    };
}

#endif  //INTEGRAGOR_HPP