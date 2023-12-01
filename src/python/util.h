// File: util.h
// Python bindings util classes/functions

#ifndef HPTS_PYTHON_UTIL_H_
#define HPTS_PYTHON_UTIL_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_util(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_UTIL_H_
