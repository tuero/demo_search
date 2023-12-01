// File: astar.h
// Python bindings for astar algorithm

#ifndef HPTS_PYTHON_ASTAR_H_
#define HPTS_PYTHON_ASTAR_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_astar(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_ASTAR_H_
