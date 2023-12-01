// File: twoheaded_convnet_multi_wrapper.h
// Description: Convnet wrapper for multiple policy + heuristic nets

#include "model/twoheaded_convnet/twoheaded_convnet_multi_wrapper.h"

// NOLINTBEGIN
#include <absl/container/flat_hash_map.h>
#include <absl/strings/str_cat.h>
// NOLINTEND

#include <spdlog/spdlog.h>

#include <cmath>
#include <filesystem>
#include <ostream>
#include <sstream>

#include "model/loss_functions.h"
#include "model/torch_util.h"
#include "util/zip.h"

namespace hpts::model::wrapper {

TwoHeadedConvNetMultiWrapperBase::TwoHeadedConvNetMultiWrapperBase(const TwoHeadedConvNetConfig& config, int num_models,
                                                                   double learning_rate, double l2_weight_decay,
                                                                   const std::string& device, const std::string& output_path,
                                                                   const std::string& checkpoint_base_name)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      config(config),
      input_flat_size(config.observation_shape.flat_size()) {
    for (int i = 0; i < num_models; ++i) {
        models_.emplace_back(config.observation_shape, config.num_actions, config.resnet_channels, config.resnet_blocks,
                             config.policy_channels, config.heuristic_channels, config.policy_mlp_layers,
                             config.heuristic_mlp_layers, config.use_batchnorm);
        model_optimizers_.emplace_back(models_.back()->parameters(),
                                       torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay));
        models_.back()->to(torch_device_);
    }
};

void TwoHeadedConvNetMultiWrapperBase::print() const {
    std::ostringstream oss;
    std::ostream& os = oss;
    os << *(models_[0]);
    SPDLOG_INFO("{:s}", oss.str());
    std::size_t num_params = 0;
    for (const auto& p : models_[0]->parameters()) {
        num_params += p.numel();
    }
    SPDLOG_INFO("Number of parameters: {:d}", num_params);
    SPDLOG_INFO("Number of models: {:d}", models_.size());
}

auto TwoHeadedConvNetMultiWrapperBase::SaveCheckpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);

    for (std::size_t i = 0; i < models_.size(); ++i) {
        torch::save(models_[i], absl::StrCat(full_path, "_", i, ".pt"));
        torch::save(model_optimizers_[i], absl::StrCat(full_path, "_", i, "-optimizer.pt"));
    }
    return full_path;
}
auto TwoHeadedConvNetMultiWrapperBase::SaveCheckpointWithoutOptimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);

    for (std::size_t i = 0; i < models_.size(); ++i) {
        torch::save(models_[i], absl::StrCat(full_path, "_", i, ".pt"));
    }
    return full_path;
}

void TwoHeadedConvNetMultiWrapperBase::LoadCheckpoint(const std::string& path) {
    for (std::size_t i = 0; i < models_.size(); ++i) {
        if (!std::filesystem::exists(absl::StrCat(path, "_", i, ".pt")) ||
            !std::filesystem::exists(absl::StrCat(path, "_", i, "-optimizer.pt"))) {
            SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
            std::exit(1);
        }

        torch::load(models_[i], absl::StrCat(path, "_", i, ".pt"), torch_device_);
        torch::load(model_optimizers_[i], absl::StrCat(path, "_", i, "-optimizer.pt"), torch_device_);
    }
}
void TwoHeadedConvNetMultiWrapperBase::LoadCheckpointWithoutOptimizer(const std::string& path) {
    for (std::size_t i = 0; i < models_.size(); ++i) {
        if (!std::filesystem::exists(absl::StrCat(path, "_", i, ".pt"))) {
            SPDLOG_ERROR("path {:s} does not contain model", path);
            std::exit(1);
        }
        torch::load(models_[i], absl::StrCat(path, "_", i, ".pt"), torch_device_);
    }
}

auto TwoHeadedConvNetMultiWrapperBase::Inference(std::vector<InferenceInput>& batch) -> std::vector<InferenceOutput> {
    const int batch_size = static_cast<int>(batch.size());
    absl::flat_hash_map<int, std::vector<int>> mapping_input;

    for (auto&& [i, batch_item] : enumerate(batch)) {
        mapping_input[batch_item.subgoal].push_back(static_cast<int>(i));
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat);
    std::vector<InferenceOutput> inference_output(batch_size);
    for (const auto& [subgoal_id, indices] : mapping_input) {
        const int collection_size = static_cast<int>(indices.size());
        torch::Tensor input_observations = torch::empty({collection_size, input_flat_size}, options);
        for (int i = 0; i < collection_size; ++i) {
            auto& batch_item = batch[indices[i]];
            input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options);
        }

        // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
        input_observations = input_observations.to(torch_device_);
        input_observations = input_observations.reshape(
            {collection_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

        // Put model in eval mode for inference + scoped no_grad
        models_[subgoal_id]->eval();
        const torch::NoGradGuard no_grad;

        // Run inference
        const auto model_output = models_[subgoal_id]->forward(input_observations);
        const auto logits_output = model_output.logits.to(torch::kDouble).to(torch::kCPU);
        const auto policy_output = model_output.policy.to(torch::kDouble).to(torch::kCPU);
        const auto log_policy_output = model_output.log_policy.to(torch::kDouble).to(torch::kCPU);
        const auto heuristic_output = model_output.heuristic.to(torch::kDouble).to(torch::kCPU);

        for (int i = 0; i < collection_size; ++i) {
            inference_output[indices[i]] = {tensor_to_vec<double>(logits_output[i]), tensor_to_vec<double>(policy_output[i]),
                                            tensor_to_vec<double>(log_policy_output[i]), heuristic_output[i].item<double>()};
        }
    }
    return inference_output;
}

auto TwoHeadedConvNetMultiWrapperLevin::Learn(std::vector<LearningInput>& batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    absl::flat_hash_map<int, std::vector<int>> mapping_input;

    for (int i = 0; i < batch_size; ++i) {
        mapping_input[batch[i].subgoal].push_back(i);
    }

    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    double total_loss = 0;

    for (const auto& [subgoal_id, indices] : mapping_input) {
        const int collection_size = static_cast<int>(indices.size());
        torch::Tensor input_observations = torch::empty({collection_size, input_flat_size}, options_float);
        torch::Tensor target_actions = torch::empty({collection_size, 1}, options_long);
        torch::Tensor target_costs = torch::empty({collection_size, 1}, options_float);
        torch::Tensor expandeds = torch::empty({collection_size, 1}, options_float);

        for (int i = 0; i < collection_size; ++i) {
            auto& batch_item = batch[indices[i]];
            input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
            target_actions[i] = batch_item.target_action;
            target_costs[i] = static_cast<double>(batch_item.target_cost_to_goal);
            expandeds[i] = static_cast<float>(batch_item.solution_expanded);
        }

        // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
        input_observations = input_observations.to(torch_device_);
        target_actions = target_actions.to(torch_device_);
        target_costs = target_costs.to(torch_device_);
        expandeds = expandeds.to(torch_device_);
        input_observations = input_observations.reshape(
            {collection_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

        // Put model in train mode for learning
        models_[subgoal_id]->train();
        models_[subgoal_id]->zero_grad();

        // Get model output
        auto model_output = models_[subgoal_id]->forward(input_observations);

        const torch::Tensor loss = (expandeds * loss::cross_entropy_loss(model_output.logits, target_actions, false) +
                                    loss::mean_squared_error_loss(model_output.heuristic, target_costs, false))
                                       .mean();
        total_loss += loss.item<double>() * collection_size;

        // Optimize model
        loss.backward();
        model_optimizers_[subgoal_id].step();
    }

    return total_loss / batch_size;
}

auto TwoHeadedConvNetMultiWrapperPolicyGradient::Learn(std::vector<LearningInput>& batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    absl::flat_hash_map<int, std::vector<int>> mapping_input;

    for (int i = 0; i < batch_size; ++i) {
        mapping_input[batch[i].subgoal].push_back(i);
    }

    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);
    double total_loss = 0;

    for (const auto& [subgoal_id, indices] : mapping_input) {
        const int collection_size = static_cast<int>(indices.size());
        torch::Tensor input_observations = torch::empty({collection_size, input_flat_size}, options_float);
        torch::Tensor target_actions = torch::empty({collection_size, 1}, options_long);
        torch::Tensor target_costs = torch::empty({collection_size, 1}, options_float);
        torch::Tensor rewards = torch::empty({batch_size, 1}, options_float);

        for (int i = 0; i < collection_size; ++i) {
            auto& batch_item = batch[indices[i]];
            input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
            target_actions[i] = batch_item.target_action;
            target_costs[i] = static_cast<double>(batch_item.target_cost_to_goal);
            rewards[i] = static_cast<float>(batch_item.reward);
        }

        // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
        input_observations = input_observations.to(torch_device_);
        target_actions = target_actions.to(torch_device_);
        target_costs = target_costs.to(torch_device_);
        rewards = rewards.to(torch_device_);
        input_observations = input_observations.reshape(
            {collection_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

        // Put model in train mode for learning
        models_[subgoal_id]->train();
        models_[subgoal_id]->zero_grad();

        // Get model output
        auto model_output = models_[subgoal_id]->forward(input_observations);

        const torch::Tensor loss = loss::policy_gradient_loss(model_output.logits, target_actions, rewards) +
                                   loss::mean_squared_error_loss(model_output.heuristic, target_costs);
        total_loss += loss.item<double>() * collection_size;

        // Optimize model
        loss.backward();
        model_optimizers_[subgoal_id].step();
    }

    return total_loss / batch_size;
}

}    // namespace hpts::model::wrapper
