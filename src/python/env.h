// File: env.h
// Python bindings for the environments

#ifndef HPTS_PYTHON_ENV_H_
#define HPTS_PYTHON_ENV_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_environments(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_ENV_H_
