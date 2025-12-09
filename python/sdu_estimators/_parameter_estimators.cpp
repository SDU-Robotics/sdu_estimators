#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/parameter_estimators/drem.hpp"
#include "sdu_estimators/parameter_estimators/cascaded_drem.hpp"
#include "sdu_estimators/parameter_estimators/gradient_estimator.hpp"
#include "sdu_estimators/parameter_estimators/gradient_estimator_sphere.hpp"

#include "sdu_estimators/integrator/integrator.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    void nb_GradientEstimator(nb::module_ & m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

        using Class = parameter_estimators::GradientEstimator<T, DIM_N, DIM_P>;
        // using ClassParent = parameter_estimators::ParameterEstimator<T, DIM_N, DIM_P>;
        std::string nbclass_name = std::string("GradientEstimator") + typestr;

        // nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
        nb::class_<Class>(m, nbclass_name.c_str())
            .def(
                nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>&>(),
                nb::arg("dt"),
                nb::arg("gamma"),
                nb::arg("theta_init"))
            .def(
                nb::init<float, const Eigen::Vector<T, DIM_P>, const Eigen::Vector<T, DIM_P>&, float>(),
                nb::arg("dt"),
                nb::arg("gamma"),
                nb::arg("theta_init"),
                nb::arg("r"))
            .def(
                nb::init<
                    float,
                    const Eigen::Vector<T, DIM_P>,
                    const Eigen::Vector<T, DIM_P>,
                    float,
                    integrator::IntegrationMethod>(),
                nb::arg("dt"),
                nb::arg("gamma"),
                nb::arg("theta_init"),
                nb::arg("r"),
                nb::arg("integration_method"))
            .def("get_estimate", &Class::get_estimate)
            .def("step", &Class::step, nb::arg("y"), nb::arg("phi"));
    }

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    void nb_DREM(nb::module_ & m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

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
                    integrator::IntegrationMethod>(),
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
    void nb_CascadedDREM(nb::module_ & m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

        using Class = parameter_estimators::CascadedDREM<T, DIM_N, DIM_P>;
        std::string nbclass_name = std::string("CascadedDREM") + typestr;

        nb::class_<Class>(m, nbclass_name.c_str())
            .def(
                nb::init<
                    float,
                    float,
                    integrator::IntegrationMethod
                >(),
                nb::arg("dt"),
                nb::arg("a"),
                nb::arg("integration_method"))
            .def("get_estimate", &Class::get_estimate)
            .def("set_dy_dphi", &Class::set_dy_dphi, nb::arg("dy"), nb::arg("dphi"))
            .def("step", &Class::step, nb::arg("y"), nb::arg("phi"))
            .def("set_eps", &Class::set_eps, nb::arg("eps"));
    }

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    void nb_GradientEstimatorSphere(nb::module_ & m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

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


    void nb_parameter_estimators(nb::module_ & m)
    {
        nb::module_ m_param_ests = m.def_submodule("parameter_estimators", "Submodule containing general parameter estimators for problems on LRE-form.");

        m_param_ests.doc() = "The parameter estimation submodule";
        
        nb_GradientEstimator<double, 1, 1>(m_param_ests);
        nb_GradientEstimator<double, 1, 2>(m_param_ests);
        nb_GradientEstimator<double, 1, 3>(m_param_ests);
        nb_GradientEstimator<double, 3, 6>(m_param_ests);
        nb_GradientEstimator<double, 4, 2>(m_param_ests);

        nb_DREM<double, 1, 1>(m_param_ests);
        nb_DREM<double, 1, 2>(m_param_ests);
        nb_DREM<double, 1, 3>(m_param_ests);
        nb_DREM<double, 3, 6>(m_param_ests);

        nb_CascadedDREM<double, 4, 2>(m_param_ests);

        nb_GradientEstimatorSphere<double, 1, 3>(m_param_ests);
    }
}