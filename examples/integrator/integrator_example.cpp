#include <Eigen/Dense>
#include "sdu_estimators/integrator/integrator.hpp"

#include <vector>
#include <fstream>
#include <iostream>


using namespace sdu_estimators;

using Vector = Eigen::Matrix<double, 2, 1>;

Vector get_dydt(double t, Vector y, Vector u)
{
    Vector out;

    out(0) = y(1);
    out(1) = (2. - pow(y(0), 2)) * y(1) / 3. - y(0);
    
    return out;
}

int main()
{
    double dt = 0.1;
    double end_time = 12;
    int steps = end_time / dt;

    std::cout << "steps " << steps << std::endl;

    std::vector<Vector> theta_euler, theta_rk2, theta_rk4;

    Vector theta0; 
    theta0 << 3, 0;
    theta_euler.push_back(theta0);
    theta_rk2.push_back(theta0);
    theta_rk4.push_back(theta0);

    Vector u;
    u.setZero();

    float t;

    for (int i = 0; i < steps; ++i) 
    {
        t = dt * i;

        theta_euler.push_back(
            integrator::IntegratorEuler<double, 2, 1>::integrate(t, 
                theta_euler.back(), 
                get_dydt, 
                u, dt)
        );

        theta_rk2.push_back(
            integrator::IntegratorRK2<double, 2, 1>::integrate(t, 
                theta_rk2.back(), 
                get_dydt, 
                u, dt)
        );

        theta_rk4.push_back(
            integrator::IntegratorRK4<double, 2, 1>::integrate(t, 
                theta_rk4.back(), 
                get_dydt, 
                u, dt)
        );
    }

    // Write all_theta_est to file
    std::ofstream outfile;
    outfile.open ("data_integrator.csv");

    outfile << "timestamp,theta_1_euler,theta_2_euler,theta_1_rk2,theta_2_rk2,theta_1_rk4,theta_2_rk4" << std::endl;

    for (int i = 0; i < steps; ++i)
    {
        outfile << i * dt << "," 
            << theta_euler[i][0] << "," << theta_euler[i][1] << ","
            << theta_rk2[i][0] << "," << theta_rk2[i][1] << ","
            << theta_rk4[i][0] << "," << theta_rk4[i][1]
            << std::endl;
    }

    outfile.close();
    std::cout << "printed result to file" << std::endl;
}