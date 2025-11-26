#include <cstdint>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h> 
#include <nanobind/stl/vector.h>

// Parameter Estimators
#include "sdu_estimators/math/riemannian_manifolds/manifold.hpp"
#include "sdu_estimators/math/riemannian_manifolds/sphere.hpp"
#include "sdu_estimators/math/riemannian_manifolds/symmetric_positive_definite.hpp"

namespace nb = nanobind;

namespace sdu_estimators 
{
    template<typename T, std::int32_t DIM_N>
    void nb_Sphere(nb::module_ m)
    {
        std::string typestr;
        typestr = "_" + std::to_string(DIM_N);

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

    void nb_riemannian_manifolds(nb::module_ m)
    {
        m.doc() = "The Riemannain manifold submodule";

        nb_Sphere<double, 3>(m);
    }
}