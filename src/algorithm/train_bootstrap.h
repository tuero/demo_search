// File: train_bootstrap.h
// Description: Generic train runner based on the bootstrap method

#ifndef HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_
#define HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_

#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <algorithm>
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
#include "util/utility.h"

namespace hpts::algorithm {

// Concepts for search algorithm output to satisfy requirements for training
template <typename T>
concept IsTrainInput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.search_budget } -> std::same_as<int &>;
};

template <typename T>
concept IsTrainOutput = requires(T t, const T ct) {
    { t.puzzle_name } -> std::same_as<std::string &>;
    { t.solution_found } -> std::same_as<bool &>;
    { t.solution_cost } -> std::same_as<double &>;
    { t.num_expanded } -> std::same_as<int &>;
    { t.num_generated } -> std::same_as<int &>;
    { t.solution_prob } -> std::same_as<double &>;
};

// Concept for learning handler to satisfy requirements to interface with training
template <typename T, typename SearchOutput>
concept IsLearningHandler = requires(T t, std::vector<SearchOutput> &&results, std::mt19937 &rng) {
    { t.init() } -> std::same_as<void>;
    { t.log_status() } -> std::same_as<void>;
    { t.process_data(std::move(results), rng) } -> std::same_as<void>;
    { t.learning_step(rng, makeval<std::size_t>(), makeval<std::size_t>()) } -> std::same_as<void>;
    { t.terminate() } -> std::same_as<void>;
    { t.checkpoint(makeval<long long int>()) } -> std::same_as<void>;
};

struct TrainingConfig {
    int seed;
    std::size_t num_threads;
    std::size_t bootstrap_batch_multiplier;
    int initial_search_budget;                      // Initial search budget, subjet to change from bootstrap process
    double time_budget;                             // Max time in seconds to train
    int max_iterations;                             // Max number of passes over the training set to train
    double validation_solved_ratio;                 // Percentage of validation set solved to checkpoint
    long long int checkpoint_expansion_interval;    // Checkpoint every amount of expansions (rounded)
    std::string output_path;
};

constexpr long long int CHECKPOINT_ALL_TRAIN_SOLVED = -2;
constexpr long long int CHECKPOINT_ALL_VALIDATE_SOLVED = -3;
constexpr long long int CHECKPOINT_RATIO_VALIDATE_SOLVED = -4;

template <typename SearchInputT, typename SearchOutputT, typename LearningHandler>
    requires IsTrainInput<SearchInputT> && IsTrainOutput<SearchOutputT> && IsLearningHandler<LearningHandler, SearchOutputT>
void run_train_levels(std::vector<SearchInputT> &problems_train, std::vector<SearchInputT> &problems_validate,
                      LearningHandler &learning_handler, std::function<SearchOutputT(const SearchInputT &)> algorithm,
                      const TrainingConfig &config, std::shared_ptr<StopToken> stop_token) {
    // Create thread pool
    ThreadPool<SearchInputT, SearchOutputT> pool(config.num_threads);

    // Init values
    std::unordered_set<std::string> solved_set_train;
    std::unordered_set<std::string> solved_set_validate;
    int bootstrap_iter = 0;
    long long int total_expanded = 0;
    long long int total_generated = 0;
    long long int interval_expanded = 0;
    bool has_checkpointed_validation = false;
    int search_budget = config.initial_search_budget;
    std::size_t outstanding_problems_train = problems_train.size();
    std::size_t outstanding_problems_validate = problems_validate.size();
    std::mt19937 rng(config.seed);

    // Any initialization required by the learning handler
    learning_handler.init();
    learning_handler.checkpoint(0);

    // Create metrics logger + directory
    const std::string metrics_path = absl::StrCat(config.output_path, "/metrics");
    std::filesystem::create_directories(metrics_path);
    MetricsTracker metrics_tracker_train(config.output_path, "train");
    MetricsTracker metrics_tracker_validate(config.output_path, "validate");

    // Run pool
    Timer timer(config.time_budget);
    timer.start();

    // @note: Add another condition for early exit
    while (!timer.is_timeout() && bootstrap_iter < config.max_iterations) {
        ++bootstrap_iter;
        int prev_solved_train = (int)solved_set_train.size();
        outstanding_problems_train = problems_train.size() - solved_set_train.size();
        outstanding_problems_validate = problems_validate.size() - solved_set_validate.size();
        SPDLOG_INFO("Bootstrap iteration: {:d} of {:d}, budget: {:d}", bootstrap_iter, config.max_iterations, search_budget);
        SPDLOG_INFO("Remaining unsolved problems: Train = {:d}, Validate = {:d}, remaining time: {:.2f}",
                    outstanding_problems_train, outstanding_problems_validate, timer.get_time_remaining());
        metrics_tracker_train.add_iteration_row({bootstrap_iter, outstanding_problems_train, timer.get_duration()});
        metrics_tracker_train.save();
        metrics_tracker_validate.save();

        // Update problem instance budget
        for (auto &p : problems_train) {
            p.search_budget = search_budget;
        }

        // Shuffle for each pass over and split into batches
        std::shuffle(problems_train.begin(), problems_train.end(), rng);
        auto batched_input = split_to_batch(problems_train, config.num_threads * config.bootstrap_batch_multiplier);

        // Consider each batch of problems
        int batch_idx = -1;
        for (auto &batch : batched_input) {
            SPDLOG_INFO("Iteration: {:d}, Batch {:d} of {:d}, remaining time: {:.2f}", bootstrap_iter, ++batch_idx,
                        batched_input.size(), timer.get_time_remaining());
            // Some algorithms need a source of RNG (training data determination, etc.)
            for (auto &batch_item : batch) {
                if constexpr (HasRNG<SearchInputT>) {
                    batch_item.rng = rng;
                }
            }
            learning_handler.log_status();
            // Guard on moved results
            {
                // Run pool on problems
                std::vector<SearchOutputT> results = pool.run(algorithm, batch);

                // Get metrics
                for (const auto &res : results) {
                    total_expanded += res.num_expanded;
                    total_generated += res.num_generated;
                    interval_expanded += res.num_expanded;
                    metrics_tracker_train.add_problem_row({bootstrap_iter, res.puzzle_name, res.solution_cost, res.solution_prob,
                                                           res.num_expanded, res.num_generated, search_budget});

                    if (res.solution_found) {
                        solved_set_train.insert(res.puzzle_name);
                    }
                }

                // Send results to store in learner
                learning_handler.process_data(std::move(results), rng);
            }

            // Check if checkpoint step reached
            if (interval_expanded >= config.checkpoint_expansion_interval) {
                learning_handler.checkpoint(total_expanded / config.checkpoint_expansion_interval);
                interval_expanded -= config.checkpoint_expansion_interval;
            }

            if (stop_token->stop_requested()) {
                SPDLOG_INFO("Stop requested, exiting batch loop.");
                break;
            }

            // Update Models
            learning_handler.learning_step(rng, problems_train.size(), problems_train.size() - solved_set_train.size());

            // Check if we timeout
            if (timer.is_timeout()) {
                break;
            }
        }

        // Run on validation set
        for (auto &batch : split_to_batch(problems_validate, config.num_threads * config.bootstrap_batch_multiplier)) {
            // Some algorithms need a source of RNG (training data determination, etc.)
            for (auto &batch_item : batch) {
                if constexpr (HasRNG<SearchInputT>) {
                    batch_item.rng = rng;
                }
            }
            for (const auto &res : pool.run(algorithm, batch)) {
                metrics_tracker_validate.add_problem_row({bootstrap_iter, res.puzzle_name, res.solution_cost, res.solution_prob,
                                                          res.num_expanded, res.num_generated, search_budget});
                if (res.solution_found) {
                    solved_set_validate.insert(res.puzzle_name);
                }
            }
        }

        metrics_tracker_train.save();
        metrics_tracker_validate.save();
        log_flush();

        // If this iteration is the one which fully solved, checkpoint
        if (solved_set_train.size() == problems_train.size() && outstanding_problems_train > 0) {
            learning_handler.checkpoint(CHECKPOINT_ALL_TRAIN_SOLVED);
        }

        // If this iteration is the one which fully solved, checkpoint
        if (solved_set_validate.size() == problems_validate.size() && outstanding_problems_validate > 0) {
            learning_handler.checkpoint(CHECKPOINT_ALL_VALIDATE_SOLVED);
        }
        if (solved_set_validate.size() >= problems_validate.size() * config.validation_solved_ratio && !has_checkpointed_validation) {
            has_checkpointed_validation = true;
            learning_handler.checkpoint(CHECKPOINT_RATIO_VALIDATE_SOLVED);
        }

        // No new puzzles solved, update budget
        if (prev_solved_train == (int)solved_set_train.size() && outstanding_problems_train > 0) {
            search_budget *= 2;
        }
        if (stop_token->stop_requested()) {
            SPDLOG_INFO("Stop requested, exting training iteration loop.");
            break;
        }
    }
    learning_handler.terminate();
}

}    // namespace hpts::algorithm

#endif    // HPTS_ALGORITHM_TRAIN_BOOTSTRAP_H_
