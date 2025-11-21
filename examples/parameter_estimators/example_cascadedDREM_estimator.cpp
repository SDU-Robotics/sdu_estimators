#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sdu_estimators/parameter_estimators/cascaded_drem.hpp>
#include <vector>

#define DIM_N 4
#define DIM_P 2

using namespace sdu_estimators;

int main()
{
    float dt = 0.001;
    float tend = 30 / dt; 

    Eigen::Matrix<double, DIM_P, 1> theta_init, theta_true, dtheta;
    // theta_init.resize(2);
    // theta_true.resize(2);

    theta_init << 0, 0;
    // theta_true << 1,
    //               2;

    theta_true << 3, 0;

    float a = 10;

    sdu_estimators::parameter_estimators::utils::IntegrationMethod method = 
        sdu_estimators::parameter_estimators::utils::IntegrationMethod::Trapezoidal;

    sdu_estimators::parameter_estimators::CascadedDREM<double, DIM_N, DIM_P> solver(dt, a, method);
    sdu_estimators::parameter_estimators::CascadedDREM<double, DIM_N, DIM_P> solver_standard(dt, a, method);

    // sdu_estimators::parameter_estimators::GradientEstimator grad_est(dt, gamma, theta_init);
    std::vector<Eigen::VectorXd> all_theta_est;
    std::vector<Eigen::VectorXd> all_theta_est_standard;

    std::vector<Eigen::VectorXd> all_theta_true;
    Eigen::VectorXd y, dy;
    Eigen::MatrixXd phi, dphi;
    y.resize(DIM_N);
    dy.resize(DIM_N);    
    phi.resize(DIM_P, DIM_N);
    dphi.resize(DIM_P, DIM_N);

    float t;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for (int i = 0; i < tend; ++i)
    {
        t = i * dt;

        dtheta << theta_true(1),
                (2. - pow(theta_true(0), 2)) * theta_true(1) / 3. - theta_true(0);

        theta_true += dt * dtheta;

        phi << 2.*std::cos(t), -std::cos(t+1.), 3.*std::cos(2.*t+1./2.), 2.*std::cos(t/3. + 1.),
            std::cos(2.*t), std::cos(t/2.), 2.*std::cos(3.*t/2. + 3./4.), -3.*std::cos(4.*t/3.);

        y << phi.transpose() * theta_true;

        dphi << -2.*std::sin(t), std::sin(t + 1.), -6.*std::sin(2.*t + 1./2.), -(2.*std::sin(t/3. + 1.))/3.,
            -2.*std::sin(2.*t), -std::sin(t/2.)/2., -3.*std::sin((3.*t)/2. + 3./4.), 4.*std::sin((4.*t)/3.);

        dy << dphi.transpose() * theta_true + phi.transpose() * dtheta;

        solver.set_dy_dphi(dy, dphi);
        solver.step(y, phi);

        // solver_standard.set_dy_dphi(dy, dphi); // no derivatives means the cascade simplifies to standard DREM
        solver_standard.step(y, phi);

        Eigen::VectorXd tmp = solver.get_estimate();
        Eigen::VectorXd tmp_standard = solver_standard.get_estimate();

        // save data
        all_theta_est.push_back(tmp);
        all_theta_est_standard.push_back(tmp_standard);
        all_theta_true.push_back(theta_true);
        
        // std::cout << tmp.transpose() << std::endl;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << "[s]" <<
    // std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]"
            << std::endl;

    // Write all_theta_est to file
    std::ofstream outfile;
    outfile.open("data_cascadedDREM.csv");

    outfile << "timestamp,theta_est_1,theta_est_2,theta_est_standard_1,theta_est_standard_2,theta_act_1,theta_act_2" << std::endl;

    for (int i = 0; i < tend; ++i)
    {
    outfile << i * dt << "," 
            << all_theta_est[i][0] << "," << all_theta_est[i][1] << "," 
            << all_theta_est_standard[i][0] << "," << all_theta_est_standard[i][1] << "," 
            << all_theta_true[i][0] << "," << all_theta_true[i][1] << std::endl;
    }

    outfile.close();
}
