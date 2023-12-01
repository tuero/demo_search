// File: phs.cpp
// Python bindings for phs algorithm

#include "python/algorithm/phs.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>

#include "algorithm/phs/phs.h"
#include "common/signaller.h"
#include "env/boxworld/boxworld_base.h"
#include "env/craftworld/craftworld_base.h"
#include "env/rnd/rnd_simple.h"
#include "env/sokoban/sokoban_base.h"
#include "model/policy_convnet/policy_convnet_wrapper.h"          // For inference input/output types
#include "model/twoheaded_convnet/twoheaded_convnet_wrapper.h"    // For inference input/output types
#include "util/thread_pool.h"
#include "util/utility.h"

namespace py = pybind11;

namespace hpts::bindings {

using namespace env;
using namespace model;
using namespace model::wrapper;

template <typename EnvT, typename PHSEvaluatorT>
void declare_phs_search_input(py::module &m, const std::string &pyclass_name) {
    using SearchInput = algorithm::phs::SearchInput<EnvT, PHSEvaluatorT>;
    py::class_<SearchInput>(m, pyclass_name.c_str())
        .def(py::init<const std::string &, const EnvT &, int, std::shared_ptr<StopToken>, std::shared_ptr<PHSEvaluatorT>>())
        .def("__copy__", [](const SearchInput &self) { return SearchInput(self); })
        .def("__deepcopy__", [](const SearchInput &self, py::dict) { return SearchInput(self); })
        .def_readwrite("puzzle_name", &SearchInput::puzzle_name)
        .def_readwrite("state", &SearchInput::state)
        .def_readwrite("search_budget", &SearchInput::search_budget)
        .def_readwrite("stop_token", &SearchInput::stop_token)
        .def_readwrite("model_eval", &SearchInput::model_eval);
}

template <typename T>
void declare_phs_search_output(py::module &m, const std::string &pyclass_name) {
    using SearchOutput = algorithm::phs::SearchOutput<T>;
    py::class_<SearchOutput>(m, pyclass_name.c_str())
        .def("__copy__", [](const SearchOutput &self) { return SearchOutput(self); })
        .def("__deepcopy__", [](const SearchOutput &self, py::dict) { return SearchOutput(self); })
        .def_readwrite("puzzle_name", &SearchOutput::puzzle_name)
        .def_readwrite("solution_found", &SearchOutput::solution_found)
        .def_readwrite("solution_cost", &SearchOutput::solution_cost)
        .def_readwrite("num_expanded", &SearchOutput::num_expanded)
        .def_readwrite("num_generated", &SearchOutput::num_generated)
        .def_readwrite("solution_prob", &SearchOutput::solution_prob)
        .def_readwrite("solution_log_prob", &SearchOutput::solution_log_prob)
        .def_readwrite("solution_path_states", &SearchOutput::solution_path_states)
        .def_readwrite("solution_path_observations", &SearchOutput::solution_path_observations)
        .def_readwrite("solution_path_actions", &SearchOutput::solution_path_actions)
        .def_readwrite("solution_path_costs", &SearchOutput::solution_path_costs);
}

template <typename EnvT, typename PHSEvaluatorT>
void _declare_phs(py::module &m, const std::string &model_eval_name, const std::string &env_name) {
    using SearchInput = algorithm::phs::SearchInput<EnvT, PHSEvaluatorT>;
    using SearchOutput = algorithm::phs::SearchOutput<EnvT>;
    const std::string suffix = model_eval_name + std::string("_") + env_name;
    declare_phs_search_input<EnvT, PHSEvaluatorT>(m, (std::string("phs_search_input_") + suffix).c_str());
    // m.def((std::string("phs_") + suffix).c_str(), &algorithm::phs::search<EnvT, PHSEvaluatorT>);
    m.def((std::string("phs_") + suffix).c_str(), [](const SearchInput &problem) {
        signal_installer(problem.stop_token);
        return algorithm::phs::search<EnvT, PHSEvaluatorT>(problem);
    });
    m.def((std::string("phs_batched_") + suffix).c_str(), [](const std::vector<SearchInput> &problems, std::size_t num_threads) {
        ThreadPool<SearchInput, SearchOutput> pool(num_threads);
        std::vector<SearchOutput> results;
        if (problems.size() > 0) {
            signal_installer(problems[0].stop_token);
        }
        for (const auto &batch : split_to_batch(problems, num_threads)) {
            auto result = pool.run(algorithm::phs::search<EnvT, PHSEvaluatorT>, batch);
            results.insert(results.end(), std::make_move_iterator(result.begin()), std::make_move_iterator(result.end()));
        }
        return results;
    });
}

template <typename EnvT>
void _declare_phs(py::module &m, const std::string &env_name) {
    declare_phs_search_output<EnvT>(m, (std::string("phs_search_output_") + env_name).c_str());
    _declare_phs<EnvT, ModelEvaluator<PolicyConvNetWrapperLevin>>(m, "policy_convnet", env_name);
    _declare_phs<EnvT, ModelEvaluator<TwoHeadedConvNetWrapperLevin>>(m, "twoheaded_convnet", env_name);
}

void declare_phs(py::module &m) {
    _declare_phs<rnd::RNDSimpleState>(m, "rnd_simple");
    _declare_phs<bw::BoxWorldBaseState>(m, "boxworld");
    _declare_phs<cw::CraftWorldBaseState>(m, "craftworld");
    _declare_phs<sokoban::SokobanBaseState>(m, "sokoban");
}

}    // namespace hpts::bindings
