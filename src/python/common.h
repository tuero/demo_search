// File: common.h
// Python bindings common classes/functions

#ifndef HPTS_PYTHON_COMMON_H_
#define HPTS_PYTHON_COMMON_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_common(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_ENV_H_
