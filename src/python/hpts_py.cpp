// File: hpts_py.cpp
// Entry point for pybind module declarations

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "python/algorithm/astar.h"
#include "python/algorithm/phs.h"
#include "python/common.h"
#include "python/env.h"
#include "python/model/policy_convnet.h"
#include "python/model/twoheaded_convnet.h"
#include "python/util.h"

PYBIND11_MODULE(_hptspy, handle) {
    handle.doc() = "Hierarchical Policy Tree Search module docs.";

    auto handle_env = handle.def_submodule("_env", "HPTS Environments.");
    hpts::bindings::declare_environments(handle_env);

    auto handle_astar = handle.def_submodule("_astar", "HPTS A*.");
    hpts::bindings::declare_astar(handle_astar);

    auto handle_phs = handle.def_submodule("_phs", "HPTS PHS.");
    hpts::bindings::declare_phs(handle_phs);

    auto handle_common = handle.def_submodule("_common", "HPTS Common.");
    hpts::bindings::declare_common(handle_common);

    auto handle_model = handle.def_submodule("_model", "HPTS Model.");
    hpts::bindings::declare_model_evaluator_policy_convnet(handle_model);
    hpts::bindings::declare_model_evaluator_twoheaded_convnet(handle_model);

    auto handle_util = handle.def_submodule("_util", "HPTS Common.");
    hpts::bindings::declare_util(handle_util);
}
