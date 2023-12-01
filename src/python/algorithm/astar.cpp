// File: astar.cpp
// Python bindings for astar algorithm

#include "python/algorithm/astar.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>

#include "algorithm/astar/astar.h"
#include "common/signaller.h"
#include "env/boxworld/boxworld_base.h"
#include "env/craftworld/craftworld_base.h"
#include "env/rnd/rnd_simple.h"
#include "env/sokoban/sokoban_base.h"
#include "util/thread_pool.h"
#include "util/utility.h"

namespace py = pybind11;

namespace hpts::bindings {

template <typename T>
void declare_astar_search_input(py::module &m, const std::string &pyclass_name) {
    using SearchInput = algorithm::astar::SearchInputNoModel<T>;
    py::class_<SearchInput>(m, pyclass_name.c_str())
        .def(py::init<const std::string &, const T &, int, std::shared_ptr<StopToken>>())
        .def("__copy__", [](const SearchInput &self) { return SearchInput(self); })
        .def("__deepcopy__", [](const SearchInput &self, py::dict) { return SearchInput(self); })
        .def_readwrite("puzzle_name", &SearchInput::puzzle_name)
        .def_readwrite("state", &SearchInput::state)
        .def_readwrite("search_budget", &SearchInput::search_budget)
        .def_readwrite("stop_token", &SearchInput::stop_token);
}

template <typename T>
void declare_astar_search_output(py::module &m, const std::string &pyclass_name) {
    using SearchOutput = algorithm::astar::SearchOutput<T>;
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

template <typename T>
void _declare_astar(py::module &m, const std::string &env_name) {
    using SearchInput = algorithm::astar::SearchInputNoModel<T>;
    using SearchOutput = algorithm::astar::SearchOutput<T>;
    declare_astar_search_input<T>(m, (std::string("astar_search_input_") + env_name).c_str());
    declare_astar_search_output<T>(m, (std::string("astar_search_output_") + env_name).c_str());
    // m.def((std::string("astar_") + env_name).c_str(), &algorithm::astar::search<T>);
    m.def((std::string("astar_") + env_name).c_str(), [](const SearchInput &problem) {
        signal_installer(problem.stop_token);
        return algorithm::astar::search<T>(problem);
    });
    m.def((std::string("astar_batched_") + env_name).c_str(),
          [](const std::vector<SearchInput> &problems, std::size_t num_threads) {
              ThreadPool<SearchInput, SearchOutput> pool(num_threads);
              std::vector<SearchOutput> results;
              if (problems.size() > 0) {
                  signal_installer(problems[0].stop_token);
              }
              for (const auto &batch : split_to_batch(problems, num_threads)) {
                  auto result = pool.run(algorithm::astar::search<T>, batch);
                  results.insert(results.end(), std::make_move_iterator(result.begin()), std::make_move_iterator(result.end()));
              }
              return results;
          });
}

void declare_astar(py::module &m) {
    _declare_astar<env::rnd::RNDSimpleState>(m, "rnd_simple");
    _declare_astar<env::bw::BoxWorldBaseState>(m, "boxworld");
    _declare_astar<env::cw::CraftWorldBaseState>(m, "craftworld");
    _declare_astar<env::sokoban::SokobanBaseState>(m, "sokoban");
}

}    // namespace hpts::bindings
