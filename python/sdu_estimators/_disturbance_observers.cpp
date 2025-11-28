#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/disturbance_observers/momentum_observer.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
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
  }
}
