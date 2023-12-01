// File: util.coo
// Python bindings util classes/functions

#include "python/util.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>

#include "util/stop_token.h"

namespace py = pybind11;

namespace hpts::bindings {

// Expose numeric limits to python, which has no understanding of a max size
struct NumericLimits {
    static constexpr int int_min = std::numeric_limits<int>::min();
    static constexpr int int_max = std::numeric_limits<int>::max();
    static constexpr float float_min = std::numeric_limits<float>::min();
    static constexpr float float_max = std::numeric_limits<float>::max();
    static constexpr double double_min = std::numeric_limits<double>::min();
    static constexpr double double_max = std::numeric_limits<double>::max();
};

void declare_util(py::module &m) {
    py::class_<StopToken, std::shared_ptr<StopToken>>(m, "StopToken")
        .def(py::init<>())
        .def("stop", &StopToken::stop)
        .def("stop_requested", &StopToken::stop_requested);
    py::class_<NumericLimits>(m, "NumericLimits")
        .def_readonly_static("I_MIN", &NumericLimits::int_min)
        .def_readonly_static("I_MAX", &NumericLimits::int_max)
        .def_readonly_static("F_MIN", &NumericLimits::float_min)
        .def_readonly_static("F_MAX", &NumericLimits::float_max)
        .def_readonly_static("D_MIN", &NumericLimits::double_min)
        .def_readonly_static("D_MAX", &NumericLimits::double_max);
}

}    // namespace hpts::bindings
