// File: common.cpp
// Python bindings for the environments

#include "python/common.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/observation.h"
#include "common/state_loader.h"
#include "env/boxworld/boxworld_base.h"
#include "env/craftworld/craftworld_base.h"
#include "env/rnd/rnd_simple.h"
#include "env/sokoban/sokoban_base.h"

namespace py = pybind11;

namespace hpts::bindings {

void declare_load_problems(py::module &m) {
    m.def("load_problems_rnd_simple", &load_problems<env::rnd::RNDSimpleState>);
    m.def("load_problems_boxworld", &load_problems<env::bw::BoxWorldBaseState>);
    m.def("load_problems_craftworld", &load_problems<env::cw::CraftWorldBaseState>);
    m.def("load_problems_sokoban", &load_problems<env::sokoban::SokobanBaseState>);
}

ObservationShape init_observation_shape(int c, int h, int w) {
    return {c, h, w};
}

void declare_observation_shape(py::module &m) {
    py::class_<ObservationShape>(m, "ObservationShape")
        .def(py::init(&init_observation_shape))
        .def("__copy__", [](const ObservationShape &self) { return ObservationShape(self); })
        .def("__deepcopy__", [](const ObservationShape &self, py::dict) { return ObservationShape(self); })
        .def(py::self == py::self)    // NOLINT (misc-redundant-expression)
        .def(py::self != py::self)    // NOLINT (misc-redundant-expression)
        .def("flat_size", &ObservationShape::flat_size)
        .def_readwrite("c", &ObservationShape::c)
        .def_readwrite("h", &ObservationShape::h)
        .def_readwrite("w", &ObservationShape::w);
}

void declare_common(py::module &m) {
    declare_load_problems(m);
    declare_observation_shape(m);
}

}    // namespace hpts::bindings
