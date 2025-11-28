#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace sdu_estimators
{
    void nb_parameter_estimators(nb::module_&);
    void nb_regressor_extensions(nb::module_&);
    void nb_riemannian_manifolds(nb::module_&);
    void nb_disturbance_observers(nb::module_&);
    void nb_integrator(nb::module_&);

    NB_MODULE(_sdu_estimators, m)
    {
        m.doc() = "Python Bindings for sdu_estimators";       

        // Run the binding code for the parameter estimators.
        nb_parameter_estimators(m);
        nb_regressor_extensions(m);
        nb_riemannian_manifolds(m);
        nb_disturbance_observers(m);
        nb_integrator(m);
    }
}