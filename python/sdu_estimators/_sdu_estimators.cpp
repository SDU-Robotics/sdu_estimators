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

    // nb::class_<parameter_estimators::GradientEstimator<double, 1, 2>, 
    //            shared_ptr<parameter_estimators::GradientEstimator<double, 1, 2>>,
    //            parameter_estimators::ParameterEstimator>(m, "GradientEstimator")
    nb::class_<parameter_estimators::GradientEstimator<double, 1, 2>>(m, "GradientEstimator")
      .def(nb::init<float, const Eigen::Vector<double, 2>, const Eigen::Vector<double, 2>>())
      .def(nb::init<float, const Eigen::Vector<double, 2>, const Eigen::Vector<double, 2>, float>())
      .def("get_estimate", &parameter_estimators::GradientEstimator<double, 1, 2>::get_estimate)
      .def("step", &parameter_estimators::GradientEstimator<double, 1, 2>::step);

    // void step(const Eigen::Matrix<T, DIM_N, 1> &y, const Eigen::Matrix<T, DIM_P, DIM_N> &phi)
  }
}  // namespace sdu_estimators