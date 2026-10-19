#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/disturbance_observers/momentum_observer.hpp"

#include "sdu_estimators/disturbance_observers/state_space_disturbance_observer.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
    template <typename T, int32_t DIM_Nx, int32_t DIM_Nu, int32_t DIM_Nd>
    void nb_StateSpaceDisturbanceObserver(nb::module_ & m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_Nx) + "x" + std::to_string(DIM_Nu) + "x" + std::to_string(DIM_Nd);

        using Class = disturbance_observers::StateSpaceDisturbanceObserver<T, DIM_Nx, DIM_Nu, DIM_Nd>;

        std::string nbclass_name = std::string("StateSpaceDisturbanceObserver") + typestr;

        // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
        nb::class_<Class>(m, nbclass_name.c_str())
            .def(
                nb::init<
                    const Eigen::Matrix<T, DIM_Nx, DIM_Nx>,
                    const Eigen::Matrix<T, DIM_Nx, DIM_Nu>,
                    const Eigen::Matrix<T, DIM_Nx, DIM_Nd>,
                    const Eigen::Matrix<T, DIM_Nd, DIM_Nx>,
                    const Eigen::Matrix<T, DIM_Nd, DIM_Nx>,
                    float,
                    integrator::IntegrationMethod>(),
                nb::arg("A"),
                nb::arg("B"),
                nb::arg("D"),
                nb::arg("Ld"),
                nb::arg("L"),
                nb::arg("dt"),
                nb::arg("integration_method"))
            .def("get_disturbance_estimate", &Class::get_disturbance_estimate)
            .def("step", &Class::step, nb::arg("x"), nb::arg("u"));
    }

    void nb_disturbance_observers(nb::module_ & m)
    {
        nb::module_ m_dist_obs = m.def_submodule("disturbance_observers", "Submodule containing definitions for disturbance observers.");

        m_dist_obs.doc() = "The disturbance observer submodule";

        nb::class_<sdu_estimators::disturbance_observers::MomentumObserver>(m, "MomentumObserver")
        .def(
            nb::init<
                std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>,
                std::function<Eigen::MatrixXd(const Eigen::VectorXd&, const Eigen::VectorXd&)>,
                std::function<Eigen::VectorXd(const Eigen::VectorXd&)>,
                std::function<Eigen::VectorXd(const Eigen::VectorXd&)>,
                double,
                const Eigen::VectorXd&>(),
            nb::arg("get_inertia_matrix"),
            nb::arg("get_coriolis"),
            nb::arg("get_gravity"),
            nb::arg("get_friction"),
            nb::arg("dt"),
            nb::arg("K"))
        .def("reset", &sdu_estimators::disturbance_observers::MomentumObserver::reset)
        .def(
            "update",
            (void(sdu_estimators::disturbance_observers::MomentumObserver::*)(
                const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)) &
                sdu_estimators::disturbance_observers::MomentumObserver::update,
            nb::arg("q"),
            nb::arg("qd"),
            nb::arg("tau"))
        .def(
            "update",
            (void(sdu_estimators::disturbance_observers::MomentumObserver::*)(
                const std::vector<double>&, const std::vector<double>&, const std::vector<double>&)) &
                sdu_estimators::disturbance_observers::MomentumObserver::update,
            nb::arg("q"),
            nb::arg("qd"),
            nb::arg("tau_m"))
        .def("estimatedTorques", &sdu_estimators::disturbance_observers::MomentumObserver::estimatedTorques)
        .def(
            "getAccEstimate",
            &sdu_estimators::disturbance_observers::MomentumObserver::getAccEstimate,
            nb::arg("q"),
            nb::arg("qd"),
            nb::arg("tau"))
        .def("zeroExternalFT", &sdu_estimators::disturbance_observers::MomentumObserver::zeroExternalFT);

    nb_StateSpaceDisturbanceObserver<double, 2, 2, 2>(m_dist_obs);
    nb_StateSpaceDisturbanceObserver<double, 2, 1, 1>(m_dist_obs);
    nb_StateSpaceDisturbanceObserver<double, 2, 1, 2>(m_dist_obs);
  }
}
