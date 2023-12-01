// File: env.cpp
// Python bindings for the environments

#include "python/env.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "env/boxworld/boxworld_base.h"
#include "env/craftworld/craftworld_base.h"
#include "env/rnd/rnd_simple.h"
#include "env/simple_env.h"
#include "env/sokoban/sokoban_base.h"

namespace py = pybind11;

namespace hpts::bindings {

template <env::SimpleEnv T>
void declare_simple_environment(py::module &m, const std::string &pyclass_name) {
    py::class_<T>(m, pyclass_name.c_str())
        .def(py::init<const std::string &>())
        .def("__copy__", [](const T &self) { return T(self); })
        .def("__deepcopy__", [](const T &self, py::dict) { return T(self); })
        .def("__hash__", [](const T &self) { return self.get_hash(); })
        .def("__repr__",
             [](const T &self) {
                 std::stringstream stream;
                 stream << self;
                 return stream.str();
             })
        .def(py::self == py::self)    // NOLINT (misc-redundant-expression)
        .def(py::self != py::self)    // NOLINT (misc-redundant-expression)
        .def("apply_action", [](T &self, std::size_t action) { self.apply_action(action); })
        .def("get_observation",
             [](const T &self) {
                 py::array_t<float> out = py::cast(self.get_observation());
                 return out;
             })
        .def("observation_shape",
             [](const T &self) {
                 const auto obs_shape = self.observation_shape();
                 std::array<int, 3> obs_shape_{obs_shape.c, obs_shape.h, obs_shape.w};
                 py::array_t<int> out = py::cast(obs_shape_);
                 return out;
             })
        .def("child_actions", [](const T &self) { return self.child_actions(); })
        .def("get_heuristic", [](const T &self) { return self.get_heuristic(); })
        .def("get_hash", [](const T &self) { return self.get_hash(); })
        .def("is_terminal", &T::is_terminal)
        .def("is_solution", &T::is_solution)
        .def_readonly_static("name", &T::name)
        .def_readonly_static("num_actions", &T::num_actions);
}

void declare_environments(py::module &m) {
    declare_simple_environment<env::rnd::RNDSimpleState>(m, "RNDSimpleState");
    declare_simple_environment<env::bw::BoxWorldBaseState>(m, "BoxWorldState");
    declare_simple_environment<env::cw::CraftWorldBaseState>(m, "CraftWorldState");
    declare_simple_environment<env::sokoban::SokobanBaseState>(m, "SokobanState");
}

}    // namespace hpts::bindings
