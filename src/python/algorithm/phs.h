// File: phs.h
// Python bindings for phs algorithm

#ifndef HPTS_PYTHON_PHS_H_
#define HPTS_PYTHON_PHS_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_phs(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_PHS_H_
