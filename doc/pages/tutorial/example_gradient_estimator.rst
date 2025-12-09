.. _example_gradient_estimator:

***************************
Example: Gradient Estimator
***************************

Consider the linear regression equation (LRE) :cite:`Sastry1989`:

.. math::

    y(t) = \phi^\intercal(t) \theta
    

.. tabs::

    .. code-tab:: c++

        #include <Eigen/Core>
        #include <chrono>
        #include <cmath>
        #include <fstream>
        #include <iostream>
        #include <vector>

        #include "sdu_estimators/parameter_estimators/gradient_estimator.hpp"
        #include "sdu_estimators/integrator/integrator.hpp"

        using namespace sdu_estimators;

        #define DIM_N 4
        #define DIM_P 2

        int main()
        {
            float dt = 0.001;
            float tend = 50 / dt; // 50s

            Eigen::Vector<double, DIM_P> gamma = {1, 1};
            gamma = 0.1 * gamma;
            float r = 0.5;
            Eigen::Vector<double, DIM_P> theta_init, theta_true;

            theta_init << 0,
                          0;
            theta_true << 1,
                          2;

            integrator::IntegrationMethod intg_method = integrator::IntegrationMethod::RK4;
            parameter_estimators::GradientEstimator<double, DIM_N, DIM_P> solver(dt, gamma, theta_init, r, intg_method);
            std::vector<Eigen::VectorXd> all_theta_est;
            Eigen::Vector<double, DIM_N> y;
            Eigen::Vector<double, DIM_P, DIM_N> phi;

            float t;

            for (int i = 0; i < tend; ++i)
            {
                t = i * dt;
                phi << 2.*std::cos(t), -std::cos(t+1.), 3.*std::cos(2.*t+1./2.), 2.*std::cos(t/3. + 1.),
                    std::cos(2.*t), std::cos(t/2.), 2.*std::cos(3.*t/2. + 3./4.), -3.*std::cos(4.*t/3.);
                y << phi.transpose() * theta_true;

                grad_est.step(y, phi);

                Eigen::VectorXd tmp = solver.get_estimate();

                // save data
                all_theta_est.push_back(tmp);
            }

            // Write all_theta_est to file
            std::ofstream outfile;
            outfile.open ("data_gradient.csv");

            outfile << "timestamp,theta_est_1,theta_est_2,theta_act_1,theta_act_2" << std::endl;

            for (int i = 0; i < tend; ++i)
            {
                outfile << i * dt << "," << all_theta_est[i][0] << "," << all_theta_est[i][1]
                        << "," << theta_true[0] << "," << theta_true[1] << std::endl;
            }

            outfile.close();
        }

    .. code-tab:: py

        #!/usr/bin/env python3

        import sdu_estimators
        import numpy as np
        import matplotlib.pyplot as plt

        def main():
            dt = 0.001
            tend = int(20 / dt)

            gamma = 0.1 * np.array([[1], [1]]).flatten()

            r = 1

            theta_init = np.zeros((1, 2)).flatten()
            theta_true = np.array([[1.], [2.]]).flatten()

            method = sdu_estimators.integrator.IntegrationMethod.RK4
            print(method)

            solver = sdu_estimators.parameter_estimators.GradientEstimator_4x2(dt, gamma, theta_init, r, method)           

            all_theta_est = np.zeros((tend, 2))

            tvec = []

            for i in range(tend):
                t = i * dt
                tvec.append(t)
                phi = np.array([[2*np.cos(t), -np.cos(t+1), 3*np.cos(2*t+1/2.), 2*np.cos(t/3.+1)],
                                [np.cos(2*t), np.cos(t/2.), 2*np.cos(3*t/2.+3/4.), -3*np.cos(4*t/3.)]])
                y = phi.T @ theta_true

                solver.step(y.flatten(), phi)

                all_theta_est[i, :] = solver.get_estimate()

            print(all_theta_est)

            plt.figure()
            plt.plot(tvec, all_theta_est, label="Estimated")
            plt.hlines(theta_true, tvec[0], tvec[-1], linestyles='dashed', label='Actual')
            plt.legend()
            plt.grid()

            plt.show()


        if __name__ == "__main__":
            main()

