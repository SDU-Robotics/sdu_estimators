#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sdu_estimators/sdu_estimators.hpp"

namespace py = pybind11;

namespace sdu_estimators {

PYBIND11_MODULE(_sdu_estimators, m)
{
  m.doc() = "Python Bindings for sdu_estimators";
  m.def("add_one", &add_one, "Increments an integer value");
}

} // namespace sdu_estimators
