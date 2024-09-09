#include <nanobind/nanobind.h>

#include <sdu_estimators/sdu_estimators.hpp>

namespace nb = nanobind;

namespace sdu_estimators
{
  NB_MODULE(_sdu_estimators, m)
  {
    m.doc() = "Python Bindings for sdu_estimators";
    m.def("add_one", &add_one, "Increments an integer value");
  }

}  // namespace sdu_estimators
