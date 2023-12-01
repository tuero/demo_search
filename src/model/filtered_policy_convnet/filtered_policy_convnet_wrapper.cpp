// File: filtered_policy_convnet_wrapper.h
// Description: Convnet wrapper for filtered policy convnet

#include "model/filtered_policy_convnet/filtered_policy_convnet_wrapper.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
// NOLINTEND
#include <spdlog/spdlog.h>

#include <cmath>
#include <filesystem>
#include <ostream>
#include <sstream>

#include "model/loss_functions.h"
#include "model/torch_util.h"

namespace hpts::model::wrapper {

FilteredPolicyConvNetWrapper::FilteredPolicyConvNetWrapper(const FilteredPolicyConvNetConfig& config, double learning_rate,
                                                           double l2_weight_decay, const std::string& device,
                                                           const std::string& output_path,
                                                           const std::string& checkpoint_base_name)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      model_(config.observation_shape, config.num_actions, config.resnet_channels, config.resnet_blocks, config.policy_channels,
             config.policy_mlp_layers, config.use_batchnorm),
      model_optimizer_(model_->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)),
      config(config),
      input_flat_size(config.observation_shape.flat_size()) {
    model_->to(torch_device_);
};

void FilteredPolicyConvNetWrapper::print() const {
    std::ostringstream oss;
    std::ostream& os = oss;
    os << *model_;
    SPDLOG_INFO("{:s}", oss.str());
    std::size_t num_params = 0;
    for (const auto& p : model_->parameters()) {
        num_params += p.numel();
    }
    SPDLOG_INFO("Number of parameters: {:d}", num_params);
}

auto FilteredPolicyConvNetWrapper::SaveCheckpoint(int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));
    return full_path;
}

void FilteredPolicyConvNetWrapper::LoadCheckpoint(const std::string& path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt")) || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
        std::exit(1);
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"), torch_device_);
}

auto FilteredPolicyConvNetWrapper::Inference(std::vector<std::any>& batch) -> std::vector<std::any> {
    const int batch_size = static_cast<int>(batch.size());

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options);
    torch::Tensor input_filters =
        torch::zeros({batch_size * config.num_actions, 2 * config.observation_shape.h * config.observation_shape.w}, options);
    for (int i = 0; i < batch_size; ++i) {
        auto batch_item = std::any_cast<FilteredPolicyConvNetInferenceInput&>(batch[i]);
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options);
    }
    // for ()

    // Copy for each number of actions
    // (B, C*H*W) -> (B * A, C*H*W)
    input_observations = torch::repeat_interleave(input_observations, config.num_actions, 0);

    // Reshape to expected size for network (batch_size * A, flat) -> (batch_size * A, c + 2, h, w)
    input_observations = input_observations.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size * config.num_actions, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});
    input_filters =
        input_filters.reshape({batch_size * config.num_actions, 2, config.observation_shape.h, config.observation_shape.w});

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    // Run inference
    const auto model_output = model_->forward(input_observations);
    const auto logits_output = model_output.logits.to(torch::kDouble).to(torch::kCPU);
    const auto policy_output = model_output.policy.to(torch::kDouble).to(torch::kCPU);
    const auto log_policy_output = model_output.log_policy.to(torch::kDouble).to(torch::kCPU);
    std::vector<std::any> inference_output;
    for (int i = 0; i < batch_size; ++i) {
        inference_output.push_back(std::make_any<FilteredPolicyConvNetInferenceOutput>(
            tensor_to_vec<double>(logits_output[i]), tensor_to_vec<double>(policy_output[i]),
            tensor_to_vec<double>(log_policy_output[i])));
    }
    return inference_output;
}

auto FilteredPolicyConvNetWrapper::Learn(std::vector<std::any>& batch) -> double {
    const int batch_size = static_cast<int>(batch.size());
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({batch_size, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor rewards = torch::empty({batch_size, 1}, options_float);

    for (int i = 0; i < batch_size; ++i) {
        auto batch_item = std::any_cast<FilteredPolicyConvNetLearningInput&>(batch[i]);
        input_observations[i] = torch::from_blob(batch_item.observation.data(), {input_flat_size}, options_float);
        target_actions[i] = static_cast<int>(batch_item.target_action);
        rewards[i] = static_cast<float>(batch_item.reward);
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    rewards = rewards.to(torch_device_);
    input_observations = input_observations.reshape(
        {batch_size, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    const auto model_output = model_->forward(input_observations);
    const torch::Tensor loss = policy_gradient_loss(model_output.logits, target_actions, rewards);
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

}    // namespace hpts::model::wrapper
