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
        static Eigen::Matrix<T, DIM_N, DIM_P> integrate(
            float t, 
            const Eigen::Matrix<T, DIM_N, DIM_P> & y,
            const std::function<Eigen::Matrix<T, DIM_N, DIM_P>(
                double,
                const Eigen::Matrix<T, DIM_N, DIM_P>&,
                const Eigen::Matrix<T, DIM_N, DIM_P>&
            )> & get_dydt,
            const Eigen::Matrix<T, DIM_N, DIM_P> & u,
            float delta)
        {
            return y + delta * get_dydt(t, y, u);
        }
    };

    template <typename T, int32_t DIM_N, int32_t DIM_P>
    class IntegratorRK2
    {
        public:
        static Eigen::Matrix<T, DIM_N, DIM_P> integrate(
            float t, 
            const Eigen::Matrix<T, DIM_N, DIM_P> & y,
            const std::function<Eigen::Matrix<T, DIM_N, DIM_P>(
                double,
                const Eigen::Matrix<T, DIM_N, DIM_P>&,
                const Eigen::Matrix<T, DIM_N, DIM_P>&
            )> & get_dydt,
            const Eigen::Matrix<T, DIM_N, DIM_P> & u,
            float delta)
        {
            Eigen::Matrix<T, DIM_N, DIM_P> k1 = delta * get_dydt(t, y, u);
            Eigen::Matrix<T, DIM_N, DIM_P> k2 = delta * get_dydt(t + 0.5 * delta, y + 0.5 * k1, u);
            return y + k2;
        }
    };

    template <typename T, int32_t DIM_N, int32_t DIM_P>
    class IntegratorRK4
    {
        public:
        static Eigen::Matrix<T, DIM_N, DIM_P> integrate(
            float t, 
            const Eigen::Matrix<T, DIM_N, DIM_P> & y,
            const std::function<Eigen::Matrix<T, DIM_N, DIM_P>(
                double,
                const Eigen::Matrix<T, DIM_N, DIM_P>&,
                const Eigen::Matrix<T, DIM_N, DIM_P>&
            )> & get_dydt,
            const Eigen::Matrix<T, DIM_N, DIM_P> & u,
            float delta)
        {
            Eigen::Matrix<T, DIM_N, DIM_P> k1 = delta * get_dydt(t, y, u);
            Eigen::Matrix<T, DIM_N, DIM_P> k2 = delta * get_dydt(t + 0.5 * delta, y + 0.5 * k1, u);
            Eigen::Matrix<T, DIM_N, DIM_P> k3 = delta * get_dydt(t + 0.5 * delta, y + 0.5 * k2, u);
            Eigen::Matrix<T, DIM_N, DIM_P> k4 = delta * get_dydt(t + delta, y + k3, u);

            return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.;
        }
    };
}

#endif  //INTEGRAGOR_HPP