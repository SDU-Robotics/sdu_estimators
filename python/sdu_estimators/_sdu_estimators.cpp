#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include <sdu_estimators/parameter_estimators/drem.hpp>
#include <sdu_estimators/parameter_estimators/cascaded_drem.hpp>
#include <sdu_estimators/parameter_estimators/gradient_estimator.hpp>
#include <sdu_estimators/parameter_estimators/gradient_estimator_sphere.hpp>

// Regressor Extenders
#include <sdu_estimators/regressor_extensions/regressor_extension.hpp>

// Disturbance Observers
#include <sdu_estimators/disturbance_observers/momentum_observer.hpp>

// manifold
#include <cstdint>
#include <sdu_estimators/math/riemannian_manifolds/sphere.hpp>

namespace nb = nanobind;

namespace sdu_estimators
{
//   template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
//   void nb_GradientEstimator(nb::module_ m, const std::string& typestr)
//   {
//     using Class = parameter_estimators::GradientEstimator<T, DIM_N, DIM_P>;
//     // using ClassParent = parameter_estimators::ParameterEstimator<T, DIM_N, DIM_P>;
//     std::string nbclass_name = std::string("GradientEstimator") + typestr;

//     // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
//     nb::class_<Class>(m, nbclass_name.c_str())
//         .def(
//             nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>>(),
//             nb::arg("dt"),
//             nb::arg("gamma"),
//             nb::arg("theta_init"))
//         .def(
//             nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>, float>(),
//             nb::arg("dt"),
//             nb::arg("gamma"),
//             nb::arg("theta_init"),
//             nb::arg("r"))
//         .def(
//             nb::init<
//                 float,
//                 const Eigen::Vector<T, DIM_P>,
//                 const Eigen::Vector<T, DIM_P>,
//                 float,
//                 utils::IntegrationMethod>(),
//             nb::arg("dt"),
//             nb::arg("gamma"),
//             nb::arg("theta_init"),
//             nb::arg("r"),
//             nb::arg("integration_method"))
//         .def("get_estimate", &Class::get_estimate)
//         .def("step", &Class::step, nb::arg("y"), nb::arg("phi"));
//     // .def("step", &Class::step, nb::arg("y"), nb::arg("phi"), nb::arg("method"));
//   }

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  void nb_RegressorExtension(nb::module_ m, const std::string& typestr)
  {
    using Class = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("RegressorExtension") + typestr;

    nb::class_<Class>(m, nbclass_name.c_str()).def("getY", &Class::getY).def("getPhi", &Class::getPhi);
  }

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  void nb_Kreisselmeier(nb::module_ m, const std::string& typestr)
  {
    using Class = regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P>;
    using ClassParent = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("Kreisselmeier") + typestr;

    nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
        .def(nb::init<float, float>(), nb::arg("dt"), nb::arg("ell"))
        .def(
            nb::init<float, float, utils::IntegrationMethod>(),
            nb::arg("dt"),
            nb::arg("ell"),
            nb::arg("integration_method"))
        .def("step", &Class::step, nb::arg("y"), nb::arg("phi"))
        .def("reset", &Class::reset);
  }

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  void nb_DREM(nb::module_ m, const std::string& typestr)
  {
    using Class = parameter_estimators::DREM<T, DIM_N, DIM_P>;
    // using ClassParent = parameter_estimators::ParameterEstimator<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("DREM") + typestr;

    // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
    nb::class_<Class>(m, nbclass_name.c_str())
        .def(
            nb::init<
                float,
                const Eigen::Vector<T, DIM_P>,
                const Eigen::Vector<T, DIM_P>,
                regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>*>(),
            nb::arg("dt"),
            nb::arg("gamma"),
            nb::arg("theta_init"),
            nb::arg("reg_ext"))
        .def(
            nb::init<
                float,
                const Eigen::Vector<T, DIM_P>,
                const Eigen::Vector<T, DIM_P>,
                regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>*,
                float>(),
            nb::arg("dt"),
            nb::arg("gamma"),
            nb::arg("theta_init"),
            nb::arg("reg_ext"),
            nb::arg("r"))
        .def(
            nb::init<
                float,
                const Eigen::Vector<T, DIM_P>,
                const Eigen::Vector<T, DIM_P>,
                regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>*,
                float,
                utils::IntegrationMethod>(),
            nb::arg("dt"),
            nb::arg("gamma"),
            nb::arg("theta_init"),
            nb::arg("reg_ext"),
            nb::arg("r"),
            nb::arg("integration_method"))
        .def("get_estimate", &Class::get_estimate)
        .def("step", &Class::step, nb::arg("y"), nb::arg("phi"));
  }

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  void nb_CascadedDREM(nb::module_ m, const std::string& typestr)
  {
    using Class = parameter_estimators::CascadedDREM<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("CascadedDREM") + typestr;

    nb::class_<Class>(m, nbclass_name.c_str())
        .def(
            nb::init<
                float,
                float,
                utils::IntegrationMethod
            >(),
            nb::arg("dt"),
            nb::arg("a"),
            nb::arg("integration_method"))
        .def("get_estimate", &Class::get_estimate)
        .def("set_dy_dphi", &Class::set_dy_dphi, nb::arg("dy"), nb::arg("dphi"))
        .def("step", &Class::step, nb::arg("y"), nb::arg("phi"));
  }

  template<typename T, std::int32_t DIM_N>
  void nb_Sphere(nb::module_ m, const std::string& typestr)
  {
    using Class = math::manifold::Sphere<T, DIM_N>;
    std::string nbclass_name = std::string("Sphere") + typestr;

    nb::class_<Class>(m, nbclass_name.c_str())
        .def(nb::init<>())
        .def("dist", &Class::dist, nb::arg("point_a"), nb::arg("point_b"))
        .def("projection", &Class::projection, nb::arg("point"), nb::arg("vector"))
        .def(
            "euclidean_to_riemannian_gradient",
            &Class::euclidean_to_riemannian_gradient,
            nb::arg("point"),
            nb::arg("euclidean_gradient"))
        .def("retraction", &Class::retraction, nb::arg("point"), nb::arg("tangent_vector"))
        .def("exp", &Class::exp, nb::arg("point"), nb::arg("tangent_vector"))
        .def("log", &Class::log, nb::arg("point_a"), nb::arg("point_b"));
  }

  template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
  void nb_GradientEstimatorSphere(nb::module_ m, const std::string& typestr)
  {
    using Class = parameter_estimators::GradientEstimatorSphere<T, DIM_N, DIM_P>;
    std::string nbclass_name = std::string("GradientEstimatorSphere") + typestr;

    nb::class_<Class>(m, nbclass_name.c_str())
        .def(
            nb::init<double, double, const Eigen::Vector<T, DIM_P>>(),
            nb::arg("dt"),
            nb::arg("gamma"),
            nb::arg("theta_init"))
        .def("step", &Class::step, nb::arg("y"), nb::arg("phi"))
        .def("get_estimate", &Class::get_estimate)
        .def("reset", &Class::reset);
  }

  NB_MODULE(_sdu_estimators, m)
  {
    m.doc() = "Python Bindings for sdu_estimators";

    nb::module_ m_param_ests = m.def_submodule("parameter_estimators", "Submodule containing general parameter estimators for problems on LRE-form.");
    nb::module_ m_param_ests_utils = m_param_ests.def_submodule("utils", "");

    nb::module_ m_reg_ext = m.def_submodule("regressor_extensions", "Submodule containing regressor extensions for LRE.");
    nb::module_ m_math = m.def_submodule("math", "Submodule containing math utilities.");
    nb::module_ m_riemann = m_math.def_submodule("riemannian_manifolds", "Submodule containing definitions for a selection of Riemannian manifolds.");
    nb::module_ m_dist_obs = m.def_submodule("disturbance_observers", "Submodule containing definitions for disturbance observers.");

    // nb::enum_<utils::IntegrationMethod>(m_param_ests_utils, "IntegrationMethod")
    //     .value("Euler", utils::IntegrationMethod::Euler)
    //     .value("Trapezoidal", utils::IntegrationMethod::Trapezoidal)
    //     .export_values();

    // // Parameter Estimators
    // nb_GradientEstimator<double, 1, 1>(m_param_ests, "_1x1");
    // nb_GradientEstimator<double, 1, 2>(m_param_ests, "_1x2");
    // nb_GradientEstimator<double, 1, 3>(m_param_ests, "_1x3");
    // nb_GradientEstimator<double, 3, 6>(m_param_ests, "_3x6");

    nb_GradientEstimatorSphere<double, 1, 3>(m_param_ests, "_1x3");

    nb_DREM<double, 1, 1>(m_param_ests, "_1x1");
    nb_DREM<double, 1, 2>(m_param_ests, "_1x2");
    nb_DREM<double, 1, 3>(m_param_ests, "_1x3");
    nb_DREM<double, 3, 6>(m_param_ests, "_3x6");

    nb_CascadedDREM<double, 4, 2>(m_param_ests, "_4x2");
    
    // Regressor Extensions
    nb_RegressorExtension<double, 1, 1>(m_reg_ext, "_1x1");
    nb_RegressorExtension<double, 1, 2>(m_reg_ext, "_1x2");
    nb_RegressorExtension<double, 1, 3>(m_reg_ext, "_1x3");
    nb_RegressorExtension<double, 3, 6>(m_reg_ext, "_3x6");

    nb_Kreisselmeier<double, 1, 1>(m_reg_ext, "_1x1");
    nb_Kreisselmeier<double, 1, 2>(m_reg_ext, "_1x2");
    nb_Kreisselmeier<double, 1, 3>(m_reg_ext, "_1x3");
    nb_Kreisselmeier<double, 3, 6>(m_reg_ext, "_3x6");

    // Riemannian Manifolds
    nb_Sphere<double, 3>(m_riemann, "_3");

    // Disturbance Observers
    nb::class_<sdu_estimators::disturbance_observers::MomentumObserver>(m_dist_obs, "MomentumObserver")
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
}  // namespace sdu_estimators