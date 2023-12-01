// File: test_runner.h
// Description: Generic test runner

#ifndef HPTS_ALGORITHM_TEST_RUNNER_H_
#define HPTS_ALGORITHM_TEST_RUNNER_H_

#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <concepts>
#include <filesystem>
#include <functional>
#include <random>

#include "common/logging.h"
#include "common/types.h"
#include "util/concepts.h"
#include "util/metrics_tracker.h"
#include "util/stop_token.h"
#include "util/thread_pool.h"
#include "util/timer.h"

namespace hpts::algorithm {

// Concepts for search algorithm output to satisfy requirements for training
template <typename T>
concept IsTestInput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.search_budget } -> std::same_as<int &>;
};

template <typename T, typename EnvT>
concept IsTestOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.solution_prob } -> std::same_as<double &>;
    { t.solution_path_states } -> std::same_as<std::vector<EnvT> &>;
};

inline void export_file_run(std::vector<std::string> solution_path_state_strs, const std::string &output_path,
                            const std::string &puzzle_name, const std::string &run_type) {
    std::filesystem::create_directories(absl::StrFormat("%s/paths/%s/", output_path, run_type));
    std::string full_path(absl::StrFormat("%s/paths/%s/%s.txt", output_path, run_type, puzzle_name));
    std::ofstream export_file(full_path, std::ofstream::trunc | std::ofstream::out);
    for (auto const &state_str : solution_path_state_strs) {
        export_file << state_str << "---" << std::endl;
    }
    export_file.close();
}

template <typename EnvT, typename SearchInputT, typename SearchOutputT>
    requires IsTestInput<SearchInputT> && IsTestOutput<SearchOutputT, EnvT>
void run_test_levels(const std::vector<SearchInputT> &problems, std::function<SearchOutputT(const SearchInputT &)> algorithm,
                     int num_threads, int search_budget, double time_budget, const std::string &output_path,
                     std::shared_ptr<StopToken> stop_token, int max_iterations = std::numeric_limits<int>::max()) {
    // Create thread pool
    ThreadPool<SearchInputT, SearchOutputT> pool(num_threads);

    int bootstrap_iter = 0;
    int total_expanded = 0;
    int total_generated = 0;
    double total_cost = 0;
    int budget = search_budget;
    std::vector<SearchInputT> outstanding_problems = problems;
    std::mt19937 rng(0);

    // Create metrics logger + directory
    const std::string metrics_path = absl::StrCat(output_path, "/metrics");
    std::filesystem::create_directories(metrics_path);
    MetricsTracker metrics_tracker(output_path, "test");

    // Run pool
    Timer timer(time_budget);
    timer.start();

    while (!timer.is_timeout() && !outstanding_problems.empty() && bootstrap_iter < max_iterations) {
        ++bootstrap_iter;
        SPDLOG_INFO("Bootstrap iteration: {:d} of {:d}, budget: {:d}", bootstrap_iter, max_iterations, budget);
        SPDLOG_INFO("Remaining unsolved problems: {:d}, remaining time: {:.2f}", outstanding_problems.size(),
                    timer.get_time_remaining());

        // Update problem instance budget
        for (auto &p : outstanding_problems) {
            p.search_budget = budget;
            if constexpr (HasRNG<SearchInputT>) {
                p.rng = rng;
            }
        }

        std::vector<SearchInputT> unsolved_problems;
        auto batched_input = split_to_batch(outstanding_problems, num_threads);
        for (const auto &batch : batched_input) {
            std::vector<SearchOutputT> results = pool.run(algorithm, batch);
            for (int i = 0; i < (int)results.size(); ++i) {
                const SearchOutputT &res = results[i];
                metrics_tracker.add_problem_row({bootstrap_iter, res.puzzle_name, res.solution_cost, res.solution_prob,
                                                 res.num_expanded, res.num_generated, budget});
                if (res.solution_found) {
                    std::vector<std::string> solution_path_state_strs;
                    for (const auto &s : res.solution_path_states) {
                        solution_path_state_strs.push_back(s.to_str());
                    }
                    export_file_run(solution_path_state_strs, output_path, res.puzzle_name, "test");
                    total_cost += res.solution_cost;
                } else {
                    // add problem back for another attempt
                    unsolved_problems.push_back(batch[i]);
                }
                total_expanded += res.num_expanded;
                total_generated += res.num_generated;
            }
            metrics_tracker.save();
            log_flush();
        }

        metrics_tracker.save();
        log_flush();
        outstanding_problems = unsolved_problems;

        // Unconditionally double budget
        budget *= (budget > 0) ? 2 : 1;
        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exiting test iteration.");
            break;
        }
    }

    double duration_seconds = timer.get_duration();
    metrics_tracker.save();
    SPDLOG_INFO("Total time: {:.2f}(s), total exp: {:d}, total gen: {:d}, total cost: {:.2f}", duration_seconds, total_expanded,
                total_generated, total_cost);
}

}    // namespace hpts::algorithm

#endif    // HPTS_ALGORITHM_TEST_RUNNER_H_
