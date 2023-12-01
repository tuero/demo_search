// File: filtered_policy_convnet_wrapper.h
// Description: Convnet wrapper for filtered policy convnet

#ifndef HPTS_WRAPPER_FILTERED_POLICY_CONVNET_H_
#define HPTS_WRAPPER_FILTERED_POLICY_CONVNET_H_

#include "common/types.h"
#include "model/base_model_wrapper.h"
#include "model/filtered_policy_convnet/filtered_policy_convnet.h"

namespace hpts::model::wrapper {

struct FilteredPolicyConvNetConfig {
    ObservationShape observation_shape;
    int num_actions;
    int resnet_channels;
    int resnet_blocks;
    int policy_channels;
    std::vector<int> policy_mlp_layers;
    bool use_batchnorm;
};

struct FilteredPolicyConvNetInferenceInput {
    Observation observation;
    std::vector<int> top_filters;
    std::vector<int> bottom_filters;
};

struct FilteredPolicyConvNetInferenceOutput {
    std::vector<double> logits;
    std::vector<double> policy;
    std::vector<double> log_policy;
};

struct FilteredPolicyConvNetLearningInput {
    Observation observation;
    std::vector<int> top_filters;
    std::vector<int> bottom_filters;
    std::size_t target_action;
    double target_cost_to_goal;
    double reward;
};

class FilteredPolicyConvNetWrapper : public BaseModelWrapper {
public:
    FilteredPolicyConvNetWrapper(const FilteredPolicyConvNetConfig& config, double learning_rate, double l2_weight_decay,
                                 const std::string& device, const std::string& output_path,
                                 const std::string& checkpoint_base_name = "");

    void print() const override;

    auto SaveCheckpoint(int step = -1) -> std::string override;
    auto SaveCheckpointWithoutOptimizer(int step = -1) -> std::string override;

    void LoadCheckpoint(const std::string& path) override;
    void LoadCheckpointWithoutOptimizer(const std::string& path) override;

    [[nodiscard]] auto Inference(std::vector<std::any>& batch) -> std::vector<std::any> override;
    auto Learn(std::vector<std::any>& batch) -> double override;

private:
    network::FilteredPolicyConvNet model_;
    torch::optim::Adam model_optimizer_;
    FilteredPolicyConvNetConfig config;
    int input_flat_size;
};

}    // namespace hpts::model::wrapper

#endif    // HPTS_WRAPPER_FILTERED_POLICY_CONVNET_H_
