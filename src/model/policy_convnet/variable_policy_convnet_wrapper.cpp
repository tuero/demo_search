// File: variable_policy_convnet_wrapper.h
// Description: Convnet wrapper for policy convnet which groups a collection of observations into a single policy

#include "model/policy_convnet/variable_policy_convnet_wrapper.h"

// NOLINTBEGIN
#include <absl/strings/str_cat.h>
// NOLINTEND
#include <spdlog/spdlog.h>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <ostream>
#include <sstream>

#include "model/loss_functions.h"
#include "model/torch_util.h"
#include "util/zip.h"

namespace hpts::model::wrapper {

VariablePolicyConvNetWrapperBase::VariablePolicyConvNetWrapperBase(const PolicyConvNetConfig& config, double learning_rate,
                                                                   double l2_weight_decay, const std::string& device,
                                                                   const std::string& output_path,
                                                                   const std::string& checkpoint_base_name)
    : BaseModelWrapper(device, output_path, checkpoint_base_name),
      // Batch-wise heuristic net as a polciy net
      model_(config.observation_shape, config.resnet_channels, config.resnet_blocks, config.policy_channels,
             config.policy_mlp_layers, config.use_batchnorm),
      model_optimizer_(model_->parameters(), torch::optim::AdamOptions(learning_rate).weight_decay(l2_weight_decay)),
      config(config),
      input_flat_size(config.observation_shape.flat_size()) {
    model_->to(torch_device_);
};

void VariablePolicyConvNetWrapperBase::print() const {
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

auto VariablePolicyConvNetWrapperBase::SaveCheckpoint(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    torch::save(model_optimizer_, absl::StrCat(full_path, "-optimizer.pt"));
    return full_path;
}
auto VariablePolicyConvNetWrapperBase::SaveCheckpointWithoutOptimizer(long long int step) -> std::string {
    // create directory for model
    std::filesystem::create_directories(path_);
    std::string full_path = absl::StrCat(path_, checkpoint_base_name_, "checkpoint-", step);
    SPDLOG_INFO("Checkpointing model to {:s}.pt", full_path);
    torch::save(model_, absl::StrCat(full_path, ".pt"));
    return full_path;
}

void VariablePolicyConvNetWrapperBase::LoadCheckpoint(const std::string& path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt")) || !std::filesystem::exists(absl::StrCat(path, "-optimizer.pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model and/or optimizer", path);
        std::exit(1);
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
    torch::load(model_optimizer_, absl::StrCat(path, "-optimizer.pt"), torch_device_);
}
void VariablePolicyConvNetWrapperBase::LoadCheckpointWithoutOptimizer(const std::string& path) {
    if (!std::filesystem::exists(absl::StrCat(path, ".pt"))) {
        SPDLOG_ERROR("path {:s} does not contain model", path);
        std::exit(1);
    }
    torch::load(model_, absl::StrCat(path, ".pt"), torch_device_);
}

auto VariablePolicyConvNetWrapperBase::Inference(std::vector<InferenceInput>& batch) -> std::vector<InferenceOutput> {
    const int N = std::accumulate(batch.begin(), batch.end(), 0,
                                  [&](std::size_t lhs, const InferenceInput& rhs) { return lhs + rhs.observations.size(); });
    // const int batch_size = std::accumulate(
    //     batch.begin(), batch.end(), 0, [&](std::size_t lhs, const InferenceInput& rhs) { return lhs + rhs.observations.size();
    //     });

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    // (B * A, C*W*H)
    torch::Tensor input_observations = torch::empty({N, input_flat_size}, options);
    std::size_t idx = 0;
    for (auto& batch_item : batch) {
        for (auto&& [obs_idx, obs] : enumerate(batch_item.observations)) {
            const auto i = static_cast<int>(idx + obs_idx);
            input_observations[i] = torch::from_blob(obs.data(), {input_flat_size}, options);
        }
        idx += batch_item.observations.size();
    }

    // Reshape to expected size for network (B * A, flat) -> (B * A, C, H, W)
    input_observations = input_observations.to(torch_device_);
    input_observations =
        input_observations.reshape({N, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

    // Put model in eval mode for inference + scoped no_grad
    model_->eval();
    const torch::NoGradGuard no_grad;

    // Run inference  (B * A, C, H, W) -> (B * A, 1)
    torch::Tensor model_output = model_->forward(input_observations);    // (B * A, C, H, W) -> (B * A, 1)

    // Collect back the original number of inputs per batch item, and create policy
    idx = 0;
    std::vector<InferenceOutput> inference_output;
    for (const auto& batch_item : batch) {
        const auto slice_size = batch_item.observations.size();
        const auto output_slice = model_output.index({torch::indexing::Slice(idx, idx + slice_size)}).flatten();
        const auto logits_output = output_slice.to(torch::kDouble).to(torch::kCPU);
        const auto policy_output = torch::softmax(output_slice, 0).to(torch::kDouble).to(torch::kCPU);
        const auto log_policy_output = torch::log_softmax(output_slice, 0).to(torch::kDouble).to(torch::kCPU);
        inference_output.emplace_back(tensor_to_vec<double>(logits_output), tensor_to_vec<double>(policy_output),
                                      tensor_to_vec<double>(log_policy_output));
        idx += slice_size;
    }
    return inference_output;
}

auto VariablePolicyConvNetWrapperLevin::Learn(std::vector<LearningInput>& batch) -> double {
    const auto batch_size = static_cast<int>(batch.size());
    const int N = std::accumulate(batch.begin(), batch.end(), 0,
                                  [&](std::size_t lhs, const LearningInput& rhs) { return lhs + rhs.observations.size(); });
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({N, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);

    int idx = 0;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        for (auto&& [obs_idx, obs] : enumerate(batch_item.observations)) {
            const auto i = static_cast<int>(idx + obs_idx);
            input_observations[i] = torch::from_blob(obs.data(), {input_flat_size}, options_float);
        }
        const auto i = static_cast<int>(batch_idx);    // stop torch from complaining about narrowing conversions
        target_actions[i] = batch_item.target_action;
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
        idx += static_cast<int>(batch_item.observations.size());
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    input_observations =
        input_observations.reshape({N, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    torch::Tensor model_output = model_->forward(input_observations);    // (B * A, C, H, W) -> (B * A, 1)

    // Collect back the original number of inputs per batch item, and create policy
    idx = 0;
    std::vector<torch::Tensor> losses;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        const auto slice_size = static_cast<int>(batch_item.observations.size());
        const auto logits =
            model_output.index({torch::indexing::Slice(idx, idx + slice_size)}).view({1, -1});    // (1, slice_size)
        losses.push_back(loss::cross_entropy_loss(logits, target_actions[static_cast<int>(batch_idx)].view({1, 1}), false));
        idx += slice_size;
    }

    const torch::Tensor loss = (expandeds * torch::cat(losses)).mean();
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

auto VariablePolicyConvNetWrapperPolicyGradient::Learn(std::vector<LearningInput>& batch) -> double {
    const auto batch_size = static_cast<int>(batch.size());
    const int N = std::accumulate(batch.begin(), batch.end(), 0,
                                  [&](std::size_t lhs, const LearningInput& rhs) { return lhs + rhs.observations.size(); });
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({N, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor rewards = torch::empty({batch_size, 1}, options_float);

    int idx = 0;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        for (auto&& [obs_idx, obs] : enumerate(batch_item.observations)) {
            const auto i = static_cast<int>(idx + obs_idx);
            input_observations[i] = torch::from_blob(obs.data(), {input_flat_size}, options_float);
        }
        const auto i = static_cast<int>(batch_idx);    // stop torch from complaining about narrowing conversions
        target_actions[i] = batch_item.target_action;
        rewards[i] = static_cast<float>(batch_item.reward);
        idx += static_cast<int>(batch_item.observations.size());
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    rewards = rewards.to(torch_device_);
    input_observations =
        input_observations.reshape({N, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    torch::Tensor model_output = model_->forward(input_observations);    // (B * A, C, H, W) -> (B * A, 1)

    // Collect back the original number of inputs per batch item, and create policy
    idx = 0;
    std::vector<torch::Tensor> losses;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        const auto slice_size = static_cast<int>(batch_item.observations.size());
        const auto logits =
            model_output.index({torch::indexing::Slice(idx, idx + slice_size)}).view({1, -1});    // (1, slice_size)
        losses.push_back(loss::policy_gradient_loss(logits, target_actions[static_cast<int>(batch_idx)].view({1, 1}),
                                                    rewards[static_cast<int>(batch_idx)].view({1, 1}), false));
        idx += slice_size;
    }

    const torch::Tensor loss = torch::cat(losses).mean();
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

auto VariablePolicyConvNetWrapperPHS::Learn(std::vector<LearningInput>& batch) -> double {
    const auto batch_size = static_cast<int>(batch.size());
    const int N = std::accumulate(batch.begin(), batch.end(), 0,
                                  [&](std::size_t lhs, const LearningInput& rhs) { return lhs + rhs.observations.size(); });
    const auto options_float = torch::TensorOptions().dtype(torch::kFloat);
    const auto options_long = torch::TensorOptions().dtype(torch::kLong);

    // Create tensor from raw flat array
    // torch::from_blob requires a pointer to non-const and doesn't take ownership
    torch::Tensor input_observations = torch::empty({N, input_flat_size}, options_float);
    torch::Tensor target_actions = torch::empty({batch_size, 1}, options_long);
    torch::Tensor depths = torch::empty({batch_size, 1}, options_float);
    torch::Tensor expandeds = torch::empty({batch_size, 1}, options_float);
    torch::Tensor log_pis = torch::empty({batch_size, 1}, options_float);

    int idx = 0;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        for (auto&& [obs_idx, obs] : enumerate(batch_item.observations)) {
            const auto i = static_cast<int>(idx + obs_idx);
            input_observations[i] = torch::from_blob(obs.data(), {input_flat_size}, options_float);
        }
        const auto i = static_cast<int>(batch_idx);    // stop torch from complaining about narrowing conversions
        target_actions[i] = batch_item.target_action;
        depths[i] = static_cast<float>(batch_item.solution_cost);
        expandeds[i] = static_cast<float>(batch_item.solution_expanded);
        log_pis[i] = static_cast<float>(batch_item.solution_log_pi);
        idx += static_cast<int>(batch_item.observations.size());
    }

    // Reshape to expected size for network (batch_size, flat) -> (batch_size, c, h, w)
    input_observations = input_observations.to(torch_device_);
    target_actions = target_actions.to(torch_device_);
    depths = depths.to(torch_device_);
    expandeds = expandeds.to(torch_device_);
    log_pis = log_pis.to(torch_device_);
    input_observations =
        input_observations.reshape({N, config.observation_shape.c, config.observation_shape.h, config.observation_shape.w});

    // Put model in train mode for learning
    model_->train();
    model_->zero_grad();

    // Get model output
    torch::Tensor model_output = model_->forward(input_observations);    // (B * A, C, H, W) -> (B * A, 1)

    // Collect back the original number of inputs per batch item, and create policy
    idx = 0;
    std::vector<torch::Tensor> losses;
    for (auto&& [batch_idx, batch_item] : enumerate(batch)) {
        const auto slice_size = static_cast<int>(batch_item.observations.size());
        const auto logits =
            model_output.index({torch::indexing::Slice(idx, idx + slice_size)}).view({1, -1});    // (1, slice_size)
        losses.push_back(loss::phs_loss(
            logits, target_actions[static_cast<int>(batch_idx)].view({1, 1}), depths[static_cast<int>(batch_idx)].view({1, 1}),
            expandeds[static_cast<int>(batch_idx)].view({1, 1}), log_pis[static_cast<int>(batch_idx)].view({1, 1}), false));
        idx += slice_size;
    }

    const torch::Tensor loss = torch::cat(losses).mean();
    auto loss_value = loss.item<double>();

    // Optimize model
    loss.backward();
    model_optimizer_.step();

    return loss_value;
}

}    // namespace hpts::model::wrapper
