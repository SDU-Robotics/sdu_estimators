#include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

#include <sdu_estimators/sdu_estimators.hpp>
// #include <sdu_estimators/parameter_estimators/parameter_estimator.hpp>
#include <sdu_estimators/parameter_estimators/drem.hpp>
#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>
#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>

namespace nb = nanobind;

#define DIM_N_1 1

#define DIM_P_2 2
#define DIM_P_3 3


namespace sdu_estimators
{
  NB_MODULE(_sdu_estimators, m)
  {
    m.doc() = "Python Bindings for sdu_estimators";
    m.def("add_one", &add_one, "Increments an integer value");

    // nb::class_<parameter_estimators::GradientEstimator<double, 1, 2>, 
    //            shared_ptr<parameter_estimators::GradientEstimator<double, 1, 2>>,
    //            parameter_estimators::ParameterEstimator>(m, "GradientEstimator")

    nb::class_<parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_2>>(m, "GradientEstimator")
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_2>, const Eigen::Vector<double, DIM_P_2>>())
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_2>, const Eigen::Vector<double, DIM_P_2>, float>())
      .def("get_estimate", &parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_2>::get_estimate)
      .def("step", &parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_2>::step);

    nb::class_<parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_3>>(m, "GradientEstimator")
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_3>, const Eigen::Vector<double, DIM_P_3>>())
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_3>, const Eigen::Vector<double, DIM_P_3>, float>())
      .def("get_estimate", &parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_3>::get_estimate)
      .def("step", &parameter_estimators::GradientEstimator<double, DIM_N_1, DIM_P_3>::step);

    // nb::class_<parameter_estimators::DREM<double, 1, 2>>(m, "DREM")
    //   .def(nb::init<float, const Eigen::Vector<double, 2>, const Eigen::Vector<double, 2>,
    //     regressor_extensions::RegressorExtension<double, 1, 2>*>())
    //   .def(nb::init<float, const Eigen::Vector<double, 2>, const Eigen::Vector<double, 2>, float,
    //     regressor_extensions::RegressorExtension<double, 1, 2>>())
    //   .def("get_estimate", &parameter_estimators::DREM<double, 1, 2>::get_estimate)
    //   .def("step", &parameter_estimators::DREM<double, 1, 2>::step);

    // 1x2
    nb::class_<regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>>(m, "RegressorExtension_1x2")
      .def("getY", &regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>::getY)
      .def("getPhi", &regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>::getPhi);

    nb::class_<regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_2>,
               regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>>(m, "Kreisselmeier_1x2")
      .def(nb::init<float, float>())
      .def("step", &regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_2>::step)
      .def("reset", &regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_2>::reset);

    nb::class_<parameter_estimators::DREM<double, DIM_N_1, DIM_P_2>>(m, "DREM_1x2")
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_2>, const Eigen::Vector<double, DIM_P_2>,
                regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>* >())
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_2>, const Eigen::Vector<double, DIM_P_2>,
                regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_2>*, float >())
      .def("get_estimate", &parameter_estimators::DREM<double, DIM_N_1, DIM_P_2>::get_estimate)
      .def("step", &parameter_estimators::DREM<double, DIM_N_1, DIM_P_2>::step);

    // 1x3
    nb::class_<regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>>(m, "RegressorExtension_1x3")
      .def("getY", &regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>::getY)
      .def("getPhi", &regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>::getPhi);

    nb::class_<regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_3>,
               regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>>(m, "Kreisselmeier_1x3")
      .def(nb::init<float, float>())
      .def("step", &regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_3>::step)
      .def("reset", &regressor_extensions::Kreisselmeier<double, DIM_N_1, DIM_P_3>::reset);

    nb::class_<parameter_estimators::DREM<double, DIM_N_1, DIM_P_3>>(m, "DREM_1x3")
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_3>, const Eigen::Vector<double, DIM_P_3>,
                regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>* >())
      .def(nb::init<float, const Eigen::Vector<double, DIM_P_3>, const Eigen::Vector<double, DIM_P_3>,
                regressor_extensions::RegressorExtension<double, DIM_N_1, DIM_P_3>*, float >())
      .def("get_estimate", &parameter_estimators::DREM<double, DIM_N_1, DIM_P_3>::get_estimate)
      .def("step", &parameter_estimators::DREM<double, DIM_N_1, DIM_P_3>::step);



    /* Unfortunately, it seems the above, quite ugly, way of binding has to be done for the number of parameters and 
     * you desire.
     * TODO: Figure out a nicer way of binding. Maybe generate text using a Python script.
     */ 
  }
}  // namespace sdu_estimators