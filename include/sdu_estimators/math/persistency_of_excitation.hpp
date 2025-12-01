#pragma once

#include <functional>
#ifndef PERSISTENCY_OF_EXCITATION_HPP
#define PERSISTENCY_OF_EXCITATION_HPP

#include <Eigen/Core>
#include <cstdint>
#include <deque>
#include <numeric>

#include "sdu_estimators/integrator/integrator.hpp"

namespace sdu_estimators::math 
{
    /**
     * @brief A class for computing the persistency of excitation integral online. 
     *
     * The definition can be seen in e.g.,
     * \verbatim embed:rst:inline :cite:`Sastry1989` \endverbatim.
     * 
     * **Defintion** (Persistency of Excitation):
     * 
     * A function \f$ \phi : \mathbb{R}_+ \to \mathbb{R}^{n} \f$ is *persistently exciting*, 
     * noted as \f$ (\gamma, \beta) \f$-PE, if there exists \f$ \gamma, \beta > 0 \f$ such that
     * 
     * \f{equation}{
     *      \int_{t}^{t + \gamma} \phi(\tau) \phi^\intercal(\tau) \, \mathrm{d}\tau 
     *      \geq \beta I_n , 
     *      \qquad t \geq 0,
     * \f}
     *
     * where \f$ I \in \mathbb{R}^{n \times n} \f$ is the identity matrix.
     * 
     * As of now, the closed integral above is computed using the extended trapezoidal rule
     * \verbatim embed:rst:inline :cite:`Press2007` \endverbatim.
     *
     * @tparam T 
     * @tparam DIM_N 
     * @tparam DIM_P 
     */
    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    class PersistencyOfExcitation
    {
        /**
         * @brief 
         * 
         */
        public:
            PersistencyOfExcitation(float dt, int N) 
                : dt(dt), N(N)
            {
                // initialize circular array
                Eigen::Matrix<T, DIM_P, DIM_P> zeromat;
                zeromat.setZero();

                for (int i = 0; i < N; ++i) 
                {
                    circular_array.push_back(zeromat);
                }

                integral_sum.setZero();
            }

            ~PersistencyOfExcitation(){};

            void step(const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
            {
                // Remove one element of the circular array.
                integral_sum -= circular_array.at(0);
                circular_array.pop_front();

                // Push element of the integral to the circular array.
                Eigen::Matrix<T, DIM_P, DIM_P> elem;
                elem << phi * phi.transpose();
                circular_array.push_back(elem);
                integral_sum += elem;

                // std::cout << "integral_sum\n" << integral_sum << std::endl;

                /** Calculate the integral using the extended Trapezoidal rule.
                 * \f{equation}{
                 *      \int_{t}^{t + \gamma} \phi(\tau) \phi^\intercal(\tau) \, \mathrm{d}\tau 
                 *      \geq \beta_1 I_n , 
                 *      \qquad t \geq 0,
                 * \f}
                 */ 
                // Eigen::Matrix<T, DIM_P, DIM_P> integral_sum = 
                //     dt * std::reduce(circular_array.begin(), circular_array.end());

                Eigen::Matrix<T, DIM_P, DIM_P> integral_sum_internal;
                integral_sum_internal = dt * integral_sum;

                // Remove 1/2 of the last and the first element from the sum.
                integral_sum_internal -= 0.5 * dt * circular_array.at(0);
                integral_sum_internal -= 0.5 * dt * circular_array.at(N-1);

                // std::cout << "integral_sum_internal" << std::endl;
                // std::cout << integral_sum_internal << std::endl;

                // Compute the eigenvalues
                eig_vals = integral_sum_internal.eigenvalues().real();

                // std::cout << "eig_vals\n" << eig_vals << std::endl;
            }

            /**
             * @brief Return the vector of eigenvalues of the current value of the PE-integral.
             * 
             * @return Eigen::Vector<T, DIM_P> A vector of eigenvalues sorted in descending order.
             */
            Eigen::Vector<T, DIM_P> get_eigen_values()
            {
                std::sort(eig_vals.begin(), eig_vals.end(), std::greater<T>());
                return eig_vals;
            }

        private:
            float dt;
            int N;
            integrator::IntegrationMethod method;

            Eigen::Matrix<T, DIM_P, DIM_P> integral_sum;
            std::deque<Eigen::Matrix<T, DIM_P, DIM_P>> circular_array;

            Eigen::Vector<T, DIM_P> eig_vals;
    };
}

#endif // PERSISTENCY_OF_EXCITATION_HPP