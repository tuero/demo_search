// File: twoheaded_convnet.h
// Python bindings for twoheaded convnet model evaluator used in search inference

#ifndef HPTS_PYTHON_MODEL_EVALUATOR_TWOHEADED_CONVNET_H_
#define HPTS_PYTHON_MODEL_EVALUATOR_TWOHEADED_CONVNET_H_

#include <pybind11/pybind11.h>

namespace hpts::bindings {

void declare_model_evaluator_twoheaded_convnet(pybind11::module &m);

}    // namespace hpts::bindings

#endif    // HPTS_PYTHON_MODEL_EVALUATOR_TWOHEADED_CONVNET_H_
