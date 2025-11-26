#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace sdu_estimators
{
    void nb_parameter_estimators(nb::module_);
    void nb_regressor_extensions(nb::module_);
    void nb_riemannian_manifolds(nb::module_);
    void nb_disturbance_observers(nb::module_);
    void nb_integrator(nb::module_);

    NB_MODULE(_sdu_estimators, m)
    {
        m.doc() = "Python Bindings for sdu_estimators";

        nb::module_ m_param_ests = m.def_submodule("parameter_estimators", "Submodule containing general parameter estimators for problems on LRE-form.");
        nb::module_ m_reg_ext = m.def_submodule("regressor_extensions", "Submodule containing regressor extensions for LRE.");
        nb::module_ m_math = m.def_submodule("math", "Submodule containing math utilities.");
        nb::module_ m_riemann = m_math.def_submodule("riemannian_manifolds", "Submodule containing definitions for a selection of Riemannian manifolds.");
        nb::module_ m_dist_obs = m.def_submodule("disturbance_observers", "Submodule containing definitions for disturbance observers.");
        nb::module_ m_intg = m.def_submodule("integrator", "Submodule containing definitions for disturbance observers.");

        // Run the binding code for the parameter estimators.
        nb_parameter_estimators(m_param_ests);
        nb_regressor_extensions(m_reg_ext);
        nb_riemannian_manifolds(m_riemann);
        nb_disturbance_observers(m_dist_obs);
        nb_integrator(m_intg);
    }
}