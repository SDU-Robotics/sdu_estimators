#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <sdu_estimators/disturbance_observers/momentum_observer.hpp>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832
#endif

/**
 * @brief Dummy robot model for testing the momentum observer.
 */
class RobotModel
{
    public:
        RobotModel(int dof) : dof(dof)
        {
        }

        Eigen::MatrixXd get_inertia_matrix(const Eigen::VectorXd& q) const
        {
            return Eigen::MatrixXd::Identity(dof, dof);
        }

        Eigen::MatrixXd get_coriolis(const Eigen::VectorXd& q, const Eigen::VectorXd& qd) const
        {
            return Eigen::MatrixXd::Identity(dof, dof);
        }

        Eigen::VectorXd get_gravity(const Eigen::VectorXd& q) const
        {
            return Eigen::MatrixXd::Identity(dof, 1);
        }

        Eigen::VectorXd get_friction(const Eigen::VectorXd& qd) const
        {
            return Eigen::MatrixXd::Identity(dof, 1);
        }

    private:
        int dof;
};

/**
 * @brief Two-link robot model for testing the momentum observer
 * 
 */
class TwoLinkRobot : RobotModel
{
    public:
        TwoLinkRobot(double l1, double l2, double m1, double m2, 
                    double Il1, double Il2, double a1, double a2) :
                    l1(l1), 
                    l2(l2), 
                    m1(m1), 
                    m2(m2), 
                    Il1(Il1), 
                    Il2(Il2), 
                    a1(a1), 
                    a2(a2),
                    RobotModel(2)
        {
            g = 9.82;
        }

        Eigen::MatrixXd get_inertia_matrix(const Eigen::VectorXd& q) const
        {
            double b11 = Il1 + m1 * pow(l1, 2) + Il2 + m2 * (pow(a1, 2) + pow(l2, 2) + 2 * a1 * l2 * cos(q[1])),
                   b12 = Il2 + m2 * (pow(l2, 2) + a1 * l2 * cos(q[1])),
                   b22 = Il2 + m2 * pow(l2, 2);

            Eigen::Matrix<double, 2, 2> B;

            B << b11, b12,
                 b12, b22;

            return B;
        }

        Eigen::MatrixXd get_coriolis(const Eigen::VectorXd& q, const Eigen::VectorXd& qd) const
        {
            double h = -m1 * a1 * l2 * sin(q[1]);

            Eigen::Matrix<double, 2, 2> C;

            C << h * qd[1], h * (qd[0] + qd[1]),
                 -h * qd[1], 0;

            return C;
        }

        Eigen::VectorXd get_gravity(const Eigen::VectorXd& q) const
        {
            Eigen::Matrix<double, 2, 1> grav;

            double g0 = (m1 * l1 + m2 * a1) * g * cos(q[0]) + m2 * l2 * g * cos(q[0] + q[1]),
                   g1 = m2 * l2 * g * cos(q[0] + q[1]);

            grav << g0, g1;

            return grav;
        }

        Eigen::VectorXd get_friction(const Eigen::VectorXd& qd) const
        {
            return Eigen::Matrix<double, 2, 1>::Zero();
        }

    
    private:
        double l1, l2, m1, m2, Il1, Il2, a1, a2, g;
};

/**
 * @brief Placeholder function to simulate getting joint positions and velocities.
 * @param t
 * @param size
 * @return std::pair<Eigen::VectorXd, Eigen::VectorXd>
 */
std::pair<Eigen::VectorXd, Eigen::VectorXd> getPositionAndVelocity(double t, size_t size)
{
    Eigen::VectorXd q(size);
    Eigen::VectorXd qd(size);

    // create a sinus path for all joints, let the speed be 0.25 radians per
    // second
    for (size_t i = 0; i < size; i++)
    {
    q[i] = 0.7 * M_PI * sin(t * M_PI + i);
    qd[i] = 0.5 * M_PI * cos(t * M_PI + i);
    }
    return { q, qd };
}

/**
 * @brief Placeholder function to simulate measuring joint torques.
 *
 * @param t
 * @param size
 * @return Eigen::VectorXd
 */
Eigen::VectorXd measureTorque(double t, size_t size)
{
    Eigen::VectorXd tau(size);

    // create a sinus path for all joints, let the speed be 0.25 radians per
    // second
    for (size_t i = 0; i < size; i++)
    {
        tau[i] = -3 * M_PI * M_PI * cos(t * M_PI + i);
    }
    return tau;
}

int main()
{
    double dt = 0.01;
    size_t dof = 2;

    double l1 = 0.5,
           l2 = 0.5,
           m1 = 50.,
           m2 = 50.,
          Il1 = 10.,
          Il2 = 10.,
           a1 = 1.,
           a2 = 1.;

    sdu_estimators::disturbance_observers::MomentumObserver observer(
        std::make_shared<TwoLinkRobot>(l1, l2, m1, m2, Il1, Il2, a1, a2), 
        dt, Eigen::VectorXd::Constant(dof, 1.0));

    int IMax = 1.0 / dt;
    for (int i = 0; i < IMax; ++i)
    {
        double t = i * dt;
        // Get the current position and velocity of the system
        auto [q, qd] = getPositionAndVelocity(t, dof);
        // Measure the Joint torques
        Eigen::VectorXd tau_m = measureTorque(t, dof);
        // Update the observer with the new measurements
        observer.update(q, qd, tau_m);

        // Print the estimated external torques
        std::cout << "Estimated torques at time " << t << ": " << observer.estimatedTorques().transpose() << std::endl;
    }

    return 0;
}