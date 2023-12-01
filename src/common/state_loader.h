// File: state_loader.h
// Description: Loads the problems and create input for search process

#ifndef HPTS_COMMON_STATE_LOADER_H_
#define HPTS_COMMON_STATE_LOADER_H_

#include <absl/strings/str_format.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <tuple>

#include "common/types.h"
#include "model/model_evaluator.h"
#include "util/stop_token.h"
#include "util/thread_pool.h"

namespace hpts {

// Concept for environment type requiring constructor with string param
template <typename T>
concept StringConstructable = requires(T t, const std::string &s) { T(s); };

/**
 * Load states for the given problem
 * @param config The config which defines path locations
 * @return tuple of vector of initialized starting states for each problem, and problem strings
 */
template <StringConstructable T>
[[nodiscard]] auto load_problems(const std::string &path, std::size_t max_instances = std::numeric_limits<std::size_t>::max(),
                                 std::size_t num_threads = 1) -> std::tuple<std::vector<T>, std::vector<std::string>> {
    std::vector<T> problems;
    std::vector<std::string> problem_strs;
    std::size_t problem_counter = 0;

    std::ifstream file(path);
    if (!file.is_open()) {
        SPDLOG_ERROR("Problem file {:s} cannot be opened.", path);
        std::exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Only grab max_instances if set
        if (problem_counter >= max_instances) {
            break;
        }
        // Line starting with ; is sometimes used as a description for the levelset
        if (line[0] == ';') {
            continue;
        }
        problem_strs.push_back(line);
        // problems.push_back(line);
        ++problem_counter;
    }
    file.close();

    if (problem_counter == 0) {
        SPDLOG_ERROR("No problems found in {:s}.", path);
        std::exit(1);
    }

    ThreadPool<std::string, T> pool(num_threads);
    problems = pool.run([](const std::string &s) -> T { return T(s); }, problem_strs);

    return {problems, problem_strs};
}

/**
 * Create search inputs from loaded states
 * @param config The config which defines path locations
 * @param problems The loaded states
 * @param stop_token Stop token to signal if stopping abruptly
 * @params args ModelEvaluators required for search algorithm
 * @return vector of search inputs, used as input to search algorithms
 */
template <typename T, typename... Args>
[[nodiscard]] auto create_search_inputs(const std::vector<T> &problems, int search_budget, StopToken *stop_token,
                                        std::shared_ptr<Args>... args) -> std::vector<SearchInput<T, Args...>> {
    std::vector<SearchInput<T, Args...>> search_inputs;
    int problem_number = -1;
    for (const auto &problem : problems) {
        search_inputs.emplace_back(absl::StrFormat("puzzle_%d", ++problem_number), problem, search_budget, stop_token, args...);
    }
    return search_inputs;
}

}    // namespace hpts

#endif    // HPTS_COMMON_STATE_LOADER_H_
