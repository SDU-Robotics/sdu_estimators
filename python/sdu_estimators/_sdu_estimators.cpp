#include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

#include <sdu_estimators/sdu_estimators.hpp>
// #include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
// #include <sdu_estimators/parameter_estimators/drem.hpp>
#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>

namespace nb = nanobind;

namespace sdu_estimators
{
  NB_MODULE(_sdu_estimators, m)
  {
    m.doc() = "Python Bindings for sdu_estimators";
    m.def("add_one", &add_one, "Increments an integer value");



    // nb::class_<parameter_estimators::DREM>(m, "DREM")
    //   .def(nb::init<float, Eigen::VectorXd &, Eigen::VectorXd &, float>())
    //   .def(nb::init<float, const Eigen::VectorXd &, const Eigen::VectorXd &, float, float>());

    //   .def(nb::init<float, const Eigen::VectorXd &, const Eigen::VectorXd &, float>())
    //   .def(nb::init<float, const Eigen::VectorXd &, const Eigen::VectorXd &, float, float>());
    // nb::class_<parameter_estimators::DREM, parameter_estimators::ParameterEstimator>(m, "DREM")
    //   .def(nb::init<float, const Eigen::VectorXd &, const Eigen::VectorXd &, float>())
    //   .def(nb::init<float, const Eigen::VectorXd &, const Eigen::VectorXd &, float, float>())
    //   .def("step", &parameter_estimators::DREM::step,
    //     "Step the execution of the estimator (must be called in a loop externally).",
    //     nb::arg("y"), nb::arg("phi"))
    //   .def("get_estimate", &parameter_estimators::DREM::get_estimate,
    //     "Get the current estimate of the parameter. Updates when the step function is called")
    //   .def("reset", &parameter_estimators::DREM::reset);
  }
}  // namespace sdu_estimators