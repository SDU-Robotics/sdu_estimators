#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/integrator/integrator.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
    template <typename T, int32_t DIM_N, int32_t DIM_P>
    void nb_IntegratorClass(nb::module_ m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N) + "x" + std::to_string(DIM_P);

        using Class = integrator::Integrator<T, DIM_N, DIM_P>;
        using State = Eigen::Matrix<T, DIM_N, DIM_P>;

        std::string nbclass_name = std::string("integrate") + typestr;

        m.def(nbclass_name.c_str(),
            &Class::integrate,
            nb::arg("state"),
            nb::arg("get_dydt"),
            nb::arg("delta"),
            nb::arg("method"));

        // nb::class_<Class>(m, nbclass_name.c_str())
        //     .def_static("integrate",
        //         &Class::integrate,
        //         nb::arg("state"),
        //         nb::arg("get_dydt"),
        //         nb::arg("delta"),
        //         nb::arg("method")
        //     );

        // nb::class_<Class>(m, nbclass_name.c_str())
        //     .def("integrate", 
        //         (State(Class::*)(
        //             const State &,
        //             const std::function<State(const State &)>,
        //             float,
        //             integrator::IntegrationMethod)) &
        //             Class::integrate,
        //         nb::arg("state"),
        //         nb::arg("get_dydt"),
        //         nb::arg("delta"),
        //         nb::arg("method"));
    }

    void nb_integrator(nb::module_ m)
    {
        nb::enum_<integrator::IntegrationMethod>(m, "IntegrationMethod")
            .value("Euler", integrator::IntegrationMethod::Euler)
            .value("RK2", integrator::IntegrationMethod::RK2)
            .value("RK4", integrator::IntegrationMethod::RK4);
            
        nb_IntegratorClass<double, 2, 1>(m);
    }
}