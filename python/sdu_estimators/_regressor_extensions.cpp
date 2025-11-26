#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/regressor_extensions/regressor_extension.hpp"
#include "sdu_estimators/regressor_extensions/kreisselmeier.hpp"
#include "sdu_estimators/integrator/integrator.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    void nb_RegressorExtension(nb::module_ m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

        using Class = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
        std::string nbclass_name = std::string("RegressorExtension") + typestr;

        nb::class_<Class>(m, nbclass_name.c_str()).def("getY", &Class::getY).def("getPhi", &Class::getPhi);
    }

    template<typename T, std::int32_t DIM_N, std::int32_t DIM_P>
    void nb_Kreisselmeier(nb::module_ m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

        using Class = regressor_extensions::Kreisselmeier<T, DIM_N, DIM_P>;
        using ClassParent = regressor_extensions::RegressorExtension<T, DIM_N, DIM_P>;
        std::string nbclass_name = std::string("Kreisselmeier") + typestr;

        nb::class_<Class, ClassParent>(m, nbclass_name.c_str())
            .def(nb::init<float, float>(), nb::arg("dt"), nb::arg("ell"))
            .def(
                nb::init<float, float, integrator::IntegrationMethod>(),
                nb::arg("dt"),
                nb::arg("ell"),
                nb::arg("integration_method"))
            .def("step", &Class::step, nb::arg("y"), nb::arg("phi"))
            .def("reset", &Class::reset);
    }

    void nb_regressor_extensions(nb::module_ m)
    {
        m.doc() = "The regressor extensions submodule";
        
        nb_RegressorExtension<double, 1, 1>(m);
        nb_RegressorExtension<double, 1, 2>(m);
        nb_RegressorExtension<double, 1, 3>(m);
        nb_RegressorExtension<double, 3, 6>(m);

        nb_Kreisselmeier<double, 1, 1>(m);
        nb_Kreisselmeier<double, 1, 2>(m);
        nb_Kreisselmeier<double, 1, 3>(m);
        nb_Kreisselmeier<double, 3, 6>(m);
    }
}