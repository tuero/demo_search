// File: train.h
// Description: PHS* model learning handler

#ifndef HPTS_ALGORITHM_PHS_TRAIN_H_
#define HPTS_ALGORITHM_PHS_TRAIN_H_

#include <spdlog/spdlog.h>

#include <memory>

#include "algorithm/phs/phs.h"
#include "model/model_evaluator.h"
#include "util/concepts.h"
#include "util/replay_buffer.h"
#include "util/utility.h"
#include "util/zip.h"

namespace hpts::algorithm::phs {

constexpr int BATCH_SAMPLE_MULTIPLIER = 4;

template <typename EnvT, model::IsModelEvaluator PHSEvaluatorT>
class LearningHandler {
    using LearningInputT = PHSEvaluatorT::LearningInput;
    using PolicyLevinLearningInputT = model::wrapper::PolicyConvNetWrapperLevin::LearningInput;
    using PolicyPolicyGradientLearningInputT = model::wrapper::PolicyConvNetWrapperPolicyGradient::LearningInput;
    using PolicyPHSLearningInputT = model::wrapper::PolicyConvNetWrapperPHS::LearningInput;
    using TwoHeadedLevinLearningInputT = model::wrapper::TwoHeadedConvNetWrapperLevin::LearningInput;
    using TwoHeadedPolicyGradientLearningInputT = model::wrapper::TwoHeadedConvNetWrapperPolicyGradient::LearningInput;
    using TwoHeadedPHSLearningInputT = model::wrapper::TwoHeadedConvNetWrapperPHS::LearningInput;

public:
    LearningHandler(std::shared_ptr<PHSEvaluatorT> model_eval, int capacity, int batch_size, int grad_steps, double base_reward,
                    double discount)
        : model_eval(std::move(model_eval)),
          buffer(capacity, batch_size * BATCH_SAMPLE_MULTIPLIER),
          batch_size(batch_size),
          grad_steps(grad_steps),
          base_reward(base_reward),
          discount(discount) {}

    void init() {
        // Start off with models in sync
        model_eval->checkpoint_and_sync_without_optimizer(-1);
    }

    void log_status() {
        SPDLOG_INFO("Buffer size: {:d}", buffer.count());
    }

    void process_data(std::vector<SearchOutput<EnvT>>&& results, [[maybe_unused]] std::mt19937& rng) {
        for (const auto& result : results) {
            if (!result.solution_found) {
                continue;
            }
            // Create learning input based on model type
            if constexpr (std::is_same_v<LearningInputT, PolicyLevinLearningInputT>) {
                for (auto&& [obs, action] : zip(result.solution_path_observations, result.solution_path_actions)) {
                    training_samples.emplace_back(std::move(obs), action, result.num_expanded);
                }
            } else if constexpr (std::is_same_v<LearningInputT, PolicyPolicyGradientLearningInputT>) {
                for (auto&& [obs, action, cost] :
                     zip(result.solution_path_observations, result.solution_path_actions, result.solution_path_costs)) {
                    const double r = base_reward * std::pow(discount, cost - 1);
                    training_samples.emplace_back(std::move(obs), action, r);
                }
            } else if constexpr (std::is_same_v<LearningInputT, PolicyPHSLearningInputT>) {
                for (auto&& [obs, action] : zip(result.solution_path_observations, result.solution_path_actions)) {
                    training_samples.emplace_back(std::move(obs), action, result.solution_cost, result.num_expanded,
                                                  result.solution_log_prob);
                }
            } else if constexpr (std::is_same_v<LearningInputT, TwoHeadedLevinLearningInputT>) {
                for (auto&& [obs, action, cost] :
                     zip(result.solution_path_observations, result.solution_path_actions, result.solution_path_costs)) {
                    training_samples.emplace_back(std::move(obs), action, cost, result.num_expanded);
                }
            } else if constexpr (std::is_same_v<LearningInputT, TwoHeadedPolicyGradientLearningInputT>) {
                for (auto&& [obs, action, cost] :
                     zip(result.solution_path_observations, result.solution_path_actions, result.solution_path_costs)) {
                    const double r = base_reward * std::pow(discount, cost - 1);
                    training_samples.emplace_back(std::move(obs), action, cost, r);
                }
            } else if constexpr (std::is_same_v<LearningInputT, TwoHeadedPHSLearningInputT>) {
                for (auto&& [obs, action, cost] :
                     zip(result.solution_path_observations, result.solution_path_actions, result.solution_path_costs)) {
                    training_samples.emplace_back(std::move(obs), action, cost, result.solution_cost, result.num_expanded,
                                                  result.solution_log_prob);
                }
            } else {
                static_assert(bool_value<false, LearningInputT>::value, "Unsupported LearningInputT");
            }
        }
    }

    void learning_step(std::mt19937& rng, [[maybe_unused]] std::size_t num_problems,
                       [[maybe_unused]] std::size_t outstanding_problems) {
        if (training_samples.size() == 0) {
            return;
        }

        auto device_manager = model_eval->get_device_manager();
        auto model = device_manager->Get(batch_size, 0);
        for (int i = 0; i < grad_steps; ++i) {
            std::shuffle(training_samples.begin(), training_samples.end(), rng);
            auto batched_input = split_to_batch(training_samples, batch_size);
            double loss = 0;
            for (auto& batch_item : batched_input) {
                std::vector<LearningInputT> batch;
                for (auto& sample_item : batch_item) {
                    batch.push_back(sample_item);
                }
                loss += model->Learn(batch);
            }
            SPDLOG_INFO("Loss: {:f}", loss / batched_input.size());
        }
        training_samples.clear();

        // Checkpoint updated model and sync other inference models
        device_manager->checkpoint_and_sync_without_optimizer(-1);
    }

    void checkpoint(long long int step) {
        model_eval->save_checkpoint_without_optimizer(step);
    }

    void terminate() {
        // save models
        model_eval->save_checkpoint(-1);
    }

private:
    std::shared_ptr<PHSEvaluatorT> model_eval;
    ReplayBuffer<LearningInputT> buffer;
    std::vector<LearningInputT> training_samples;

    int batch_size;
    int grad_steps;
    double base_reward;
    double discount;
};

}    // namespace hpts::algorithm::phs

#endif    // HPTS_ALGORITHM_PHS_TRAIN_H_
