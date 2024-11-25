#include <nanobind/nanobind.h>
// #include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

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
  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void nb_GradientEstimator(nb::module_ m, const std::string & typestr)
  {
    using Class = parameter_estimators::GradientEstimator<T, DIM_N, DIM_P>;
    // using ClassParent = parameter_estimators::ParameterEstimator<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("GradientEstimator") + typestr;

    // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
    nb::class_<Class>(m, nbclass_name.c_str())
      .def(nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>>(),
        nb::arg("dt"), nb::arg("gamma"), nb::arg("theta_init"))
      .def(nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>, float>(),
        nb::arg("dt"), nb::arg("gamma"), nb::arg("theta_init"), nb::arg("r"))
      .def(nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>, float, parameter_estimators::utils::IntegrationMethod>(),
        nb::arg("dt"), nb::arg("gamma"), nb::arg("theta_init"), nb::arg("r"), nb::arg("integration_method"))
      .def("get_estimate", &Class::get_estimate)
    .def("step", &Class::step, nb::arg("y"), nb::arg("phi"));
      // .def("step", &Class::step, nb::arg("y"), nb::arg("phi"), nb::arg("method"));
  }

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void nb_RegressorExtension(nb::module_ m, const std::string & typestr)
  {
    using Class = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("RegressorExtension") + typestr;

    nb::class_<Class>(m, nbclass_name.c_str())
      .def("getY", &Class::getY)
      .def("getPhi", &Class::getPhi);
  }

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void nb_Kreisselmeier(nb::module_ m, const std::string & typestr)
  {
    using Class = regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P>;
    using ClassParent = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("Kreisselmeier") + typestr;

    nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
      .def(nb::init<float, float>(),
        nb::arg("dt"), nb::arg("ell"))
      .def("step", &Class::step,
        nb::arg("y"), nb::arg("phi"))
      .def("reset", &Class::reset);
  }

  template <typename T, int32_t DIM_N, int32_t DIM_P>
  void nb_DREM(nb::module_ m, const std::string & typestr)
  {
    using Class = parameter_estimators::DREM<T, DIM_N, DIM_P>;
    // using ClassParent = parameter_estimators::ParameterEstimator<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("DREM") + typestr;

    // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
    nb::class_<Class>(m, nbclass_name.c_str())
      .def(nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>,
                regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>* >(),
                nb::arg("dt"), nb::arg("gamma"), nb::arg("theta_init"), nb::arg("reg_ext"))
      .def(nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>,
                regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>*, float >(),
                nb::arg("dt"), nb::arg("gamma"), nb::arg("theta_init"), nb::arg("reg_ext"), nb::arg("r"))
      .def("get_estimate", &Class::get_estimate)
      .def("step", &Class::step,
        nb::arg("y"), nb::arg("phi"));
  }


  NB_MODULE(_sdu_estimators, m)
  {
    m.doc() = "Python Bindings for sdu_estimators";

    nb::enum_<parameter_estimators::utils::IntegrationMethod>(m, "IntegrationMethod")
      .value("Euler", parameter_estimators::utils::IntegrationMethod::Euler)
      .value("Heuns", parameter_estimators::utils::IntegrationMethod::Heuns);

    nb_GradientEstimator<double, 1, 2>(m, "_1x2");
    nb_GradientEstimator<double, 1, 3>(m, "_1x3");

    nb_RegressorExtension<double, 1, 2>(m, "_1x2");
    nb_RegressorExtension<double, 1, 3>(m, "_1x3");

    nb_Kreisselmeier<double, 1, 2>(m, "_1x2");
    nb_Kreisselmeier<double, 1, 3>(m, "_1x3");

    nb_DREM<double, 1, 2>(m, "_1x2");
    nb_DREM<double, 1, 3>(m, "_1x3");
  }
}  // namespace sdu_estimators